###################################################################
##################### modified from SMPLify-X  ####################
################## convert 3D keypoints to SMPLX  #################
###################################################################
###################################################################

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import json
import numpy as np
import torch
import sys
import time
import smplx
import traceback



from cmd_parser import parse_config
from utils import JointMapper
from prior import create_prior
from fit_single_frame import fit_single_frame, _LOWER_BODY_POSE_DOFS

torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
import numpy as np, random as _random
np.random.seed(42)
_random.seed(42)
torch.use_deterministic_algorithms(True, warn_only=True)

try:
    import cPickle as pickle
except ImportError:
    import pickle

def main(**args):
    #############################
    ###### load gpu device ######
    #############################
    # if args["gpu_id"] is not None:
    #     os.environ['CUDA_VISIBLE_DEVICES'] = str(args["gpu_id"])
    #     print(f"Using GPU: {args["gpu_id"]}")

    ##############################
    ###### load floate tyoe ######
    ##############################
    float_dtype = args['float_dtype']
    if float_dtype == 'float64':
        dtype = torch.float64
    elif float_dtype == 'float32':
        dtype = torch.float32
    else:
        raise ValueError('Unknown float type {}, exiting!'.format(float_dtype))

    #######################
    ###### load cuda ######
    #######################
    use_cuda = args.get('use_cuda', True)
    if use_cuda and not torch.cuda.is_available():
        raise ValueError('CUDA is not available, exiting!')

    start = time.time()
    #######################
    ###### load data ######
    #######################
    _t_load = time.time()
    if args["dataset"] == 'ADT':
        from data_parser import ADT
        print('adt')
        dataset_obj = ADT(sequence_path=args["data_folder"], **args)
        sequence_name = os.path.basename(args["data_folder"].rstrip('/'))
    elif args["dataset"] == 'custom':
        from data_parser import CustomDataset
        print('custom')
        dataset_obj = CustomDataset(data_path=args["data_folder"], **args)
        sequence_name = os.path.basename(args["data_folder"].rstrip('/'))
    else:
        raise ValueError('Unknown dataset: {}'.format(args["dataset"]))
    print(f"[timing] dataset loaded in {time.time()-_t_load:.2f}s  ({len(dataset_obj)} frames)")

    ########################################
    ###### load SMPLX model and priors #####
    ########################################
    joint_mapper = JointMapper(dataset_obj.get_model2data())
    model_params = dict(model_path=args["model_folder"],
                        joint_mapper=joint_mapper,
                        create_global_orient=True,
                        create_body_pose=not args["use_vposer"],
                        create_betas=True,
                        create_left_hand_pose=True,
                        create_right_hand_pose=True,
                        create_expression=True,
                        create_jaw_pose=True,
                        create_leye_pose=True,
                        create_reye_pose=True,
                        create_transl=True,
                        dtype=dtype,
                        **args)
    # load gender models
    if args["model_type"] == 'smplh' and args["gender"] == "neutral":
        raise ValueError('SMPL-H has no gender-neutral model')
    else:
        body_model = smplx.create(**model_params)
    # load priors
    use_hands = args["use_hands"]
    use_face = args["use_face"]
    body_pose_prior = create_prior(
        prior_type=args["body_prior_type"],
        dtype=dtype,
        **args)
    jaw_prior, expr_prior = None, None
    if use_face:
        jaw_prior = create_prior(
            prior_type=args["jaw_prior_type"],
            dtype=dtype,
            **args)
        expr_prior = create_prior(
            prior_type=args["expr_prior_type"],
            dtype=dtype, **args)
    left_hand_prior, right_hand_prior = None, None
    if use_hands:
        lhand_args = args.copy()
        lhand_args['num_gaussians'] = args["num_pca_comps"]
        left_hand_prior = create_prior(
            prior_type=args["left_hand_prior_type"],
            dtype=dtype,
            use_left_hand=True,
            **lhand_args)
        rhand_args = args.copy()
        rhand_args['num_gaussians'] = args["num_pca_comps"]
        right_hand_prior = create_prior(
            prior_type=args["right_hand_prior_type"],
            dtype=dtype,
            use_right_hand=True,
            **rhand_args)
    shape_prior = create_prior(
        prior_type=args["shape_prior_type"],
        dtype=dtype, **args)
    angle_prior = create_prior(prior_type='angle', dtype=dtype)

    #######################
    ###### to device ######
    #######################
    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    body_model = body_model.to(device=device)
    body_pose_prior = body_pose_prior.to(device=device)
    angle_prior = angle_prior.to(device=device)
    shape_prior = shape_prior.to(device=device)
    if use_face:
        expr_prior = expr_prior.to(device=device)
        jaw_prior = jaw_prior.to(device=device)
    if use_hands:
        left_hand_prior = left_hand_prior.to(device=device)
        right_hand_prior = right_hand_prior.to(device=device)

    # A weight for every joint of the model
    joint_weights = dataset_obj.get_joint_weights().to(device=device, dtype=dtype)
    # Add a fake batch dimension for broadcasting
    joint_weights.unsqueeze_(dim=0)

    ####################################
    ###### Create the search tree ######
    ####################################
    search_tree = None
    pen_distance = None
    filter_faces = None
    if args["interpenetration"]:
        from mesh_intersection.bvh_search_tree import BVH
        import mesh_intersection.loss as collisions_loss
        from mesh_intersection.filter_faces import FilterFaces

        assert use_cuda, 'Interpenetration term can only be used with CUDA'
        assert torch.cuda.is_available(), \
            'No CUDA Device! Interpenetration term can only be used' + \
            ' with CUDA'
        search_tree = BVH(max_collisions=args["max_collisions"])
        pen_distance = \
            collisions_loss.DistanceFieldPenetrationLoss(
                sigma=args["df_cone_height"], point2plane=args["point2plane"],
                vectorized=True, penalize_outside=args["penalize_outside"])
        if args["part_segm_fn"]:
            # Read the part segmentation
            part_segm_fn = os.path.expandvars(args["part_segm_fn"])
            with open(part_segm_fn, 'rb') as faces_parents_file:
                face_segm_data = pickle.load(faces_parents_file,
                                            encoding='latin1')
            faces_segm = face_segm_data['segm']
            faces_parents = face_segm_data['parents']
            # Create the module used to filter invalid collision pairs
            filter_faces = FilterFaces(
                faces_segm=faces_segm, faces_parents=faces_parents,
                ign_part_pairs=args["ign_part_pairs"]).to(device=device)

    ####################################
    ###### fit sequence and store ######
    ####################################
    # Optionally initialize (and freeze) betas from an upstream estimate
    # (e.g. SMPLer-X). When set, the existing freeze path in fit_single_frame
    # takes over on every frame: betas are kept fixed at this value.
    # silhouette_cameras: dict {logical_cam_name: {K, D, R, T, image_size}},
    # injected by fitter_pipeline.py (keys are sorted logical names e.g. FC1, FC2, GB, ...).
    # None when main.py is run directly without silhouette support.
    silhouette_cameras = args.get('silhouette_cameras', None)
    if silhouette_cameras is not None:
        print(f"Using {len(silhouette_cameras)} silhouette cameras: {list(silhouette_cameras.keys())}")

    mask_folder = args.get('mask_folder', None)
    cam_names = sorted(silhouette_cameras.keys()) if silhouette_cameras is not None else []
    n_views = len(cam_names)

    # smpler_init: list of per-frame dicts (or None) from fitter_pipeline.
    # Each dict has 'body_pose' (63,) and 'global_orient' (3,) in world frame,
    # fused across camera views.  Used to warm-start pose_embedding and global_orient.
    init_bps, init_gos, init_trs, init_betas = dataset_obj.get_init_body(init_poses=True)
    init_left_hand_poses  = dataset_obj.get_init_hand_poses('left')
    init_right_hand_poses = dataset_obj.get_init_hand_poses('right')

    global_betas = None
    prev_left_hand_pose  = None
    prev_right_hand_pose = None
    prev_body_pose = None
    ref_lower_body    = None  # frame-0 lower body DOFs, pinned for all subsequent frames
    ref_global_orient = None  # frame-0 global_orient, pinned for all subsequent frames
    ref_translation = None  # frame-0 global_orient, pinned for all subsequent frames
    if init_betas is not None:
        init_arr = np.asarray(init_betas, dtype=np.float64 if float_dtype == 'float64' else np.float32).flatten()
        nb = body_model.num_betas
        if len(init_arr) > nb:
            init_arr = init_arr[:nb]
        elif len(init_arr) < nb:
            init_arr = np.pad(init_arr, (0, nb - len(init_arr)))
        global_betas = torch.as_tensor(init_arr, dtype=dtype, device=device).reshape(1, -1)
        preview = global_betas.detach().cpu().numpy().flatten()[:5].round(3).tolist()
        print(f"Using injected betas (shape={list(global_betas.shape)}, original={len(np.asarray(init_betas).flatten())}): {preview} ...")
    if not os.path.exists(os.path.join(args['output_folder'], sequence_name, 'meshes')):
        os.makedirs(os.path.join(args['output_folder'], sequence_name, 'meshes'))
    smplx_stored_path = os.path.join(args['output_folder'], sequence_name, 'body_smplx.json')
    mesh_stored_path = os.path.join(args['output_folder'], sequence_name, 'meshes')
    failed_frames = []
    _timing_frames = []   # list of {frame, fit_s, mesh_s}
    with open(smplx_stored_path, 'w') as f:
        prev_body_pose = None
        for idx, data in enumerate(dataset_obj):
            try:
                if idx > 24:
                  break
                print('Fitting frame {}/{} ...'.format(idx+1, len(dataset_obj)))

                gt_silhouettes = None
                frame_args = args.copy()
                if idx == 0:
                    frame_args['maxiters'] = args['maxiters'] * 3

                # Pass frame-0 references so fit_single_frame can pin lower body
                # and global_orient for all subsequent frames.
                if ref_lower_body is not None:
                    frame_args['lower_body_ref']    = ref_lower_body
                    frame_args['global_orient_ref'] = ref_global_orient
                    frame_args['translation_ref']   = ref_translation

                # Per-frame SMPLer-X body pose: frame-0 initializer AND per-frame
                # prior anchor (used in the loss the same way as the temporal term).
                frame_args['init_global_orient'] = (
                      init_gos[idx]
                      if init_gos is not None and idx < len(init_gos)
                      else None)

                frame_args['init_transl'] = (
                      init_trs[idx]
                      if init_trs is not None and idx < len(init_trs)
                      else None)


                frame_args['init_body_pose'] = (
                    init_bps[idx]
                    if init_bps is not None and idx < len(init_bps)
                    else None)

                # Per-frame WiLoR hand pose warm-start
                frame_args['init_left_hand_pose'] = (
                    init_left_hand_poses[idx]
                    if init_left_hand_poses is not None and idx < len(init_left_hand_poses)
                    else None)
                frame_args['init_right_hand_pose'] = (
                    init_right_hand_poses[idx]
                    if init_right_hand_poses is not None and idx < len(init_right_hand_poses)
                    else None)

                # Per-frame inner face landmarks (dlib 17-67 → SMPLX static 51) for
                # the optional face refinement stage. These are the same triangulated
                # points the dataset already loads as face_data, so we reuse them
                # directly rather than a separate head landmark file. face_data is
                # None when use_face is False; a landmark with non-positive confidence
                # is NaN'd so the refinement's ~isnan validity mask skips it instead
                # of pulling the fit toward the origin.
                gt_face_landmarks = None
                if dataset_obj.face_data is not None and idx < len(dataset_obj.face_data):
                    fl = dataset_obj.face_data[idx][17:68]          # (51, 4): xyz + conf
                    lmks = fl[:, :3].astype(np.float32).copy()
                    lmks[fl[:, 3] <= 0] = np.nan
                    gt_face_landmarks = torch.from_numpy(lmks).to(device=device, dtype=dtype)

                _t_fit = time.time()
                body_dict, body_mesh = fit_single_frame(
                                data,
                                frame_idx=idx,
                                global_betas=global_betas,
                                prev_body_pose=prev_body_pose,
                                prev_left_hand_pose=prev_left_hand_pose,
                                prev_right_hand_pose=prev_right_hand_pose,
                                search_tree=search_tree,
                                pen_distance=pen_distance,
                                filter_faces=filter_faces,
                                body_model=body_model,
                                joint_weights=joint_weights,
                                dtype=dtype,
                                shape_prior=shape_prior,
                                expr_prior=expr_prior,
                                body_pose_prior=body_pose_prior,
                                left_hand_prior=left_hand_prior,
                                right_hand_prior=right_hand_prior,
                                jaw_prior=jaw_prior,
                                angle_prior=angle_prior,
                                gt_silhouettes=gt_silhouettes,
                                gt_face_landmarks=gt_face_landmarks,
                                device=device,
                                **frame_args)
                _dt_fit = time.time() - _t_fit

                # Save frame-0 lower body and global_orient as fixed references.
                if idx == 0:
                    ref_lower_body = body_model.body_pose.data[0, _LOWER_BODY_POSE_DOFS].clone().cpu()
                    ref_global_orient = body_model.global_orient.data.clone().cpu()
                    ref_translation = body_model.translation.data.clone().cpu()

                # update body dict and temporal consistent body pose
                body_dict['frame_idx'] = idx
                global_betas = torch.tensor(body_dict['betas'], device=device)
                prev_body_pose = torch.tensor(body_dict['body_pose'], device=device)
                prev_left_hand_pose = torch.tensor(body_dict['left_hand_pose'], device=device)
                prev_right_hand_pose = torch.tensor(body_dict['right_hand_pose'], device=device)

                # store results
                f.write(json.dumps(body_dict) + '\n')
                f.flush()
                _dt_mesh = 0.0
                if body_mesh is not None:
                  mesh_stored_path = os.path.join(args["output_folder"], sequence_name, "meshes", f"{idx:06d}_fit.obj")
                  _t_mesh = time.time()
                  body_mesh.export(mesh_stored_path)
                  _dt_mesh = time.time() - _t_mesh

                _timing_frames.append({'frame': idx, 'fit_s': round(_dt_fit, 3), 'mesh_s': round(_dt_mesh, 3)})
                print(f"[timing/frame {idx}] fit={_dt_fit:.2f}s  mesh_export={_dt_mesh:.3f}s")
            except Exception as e:
                print('Fitting sequence {} failed at frame {} with error: {}'.format(
                    sequence_name, idx, e))
                traceback.print_exc()
                failed_frames.append(idx)
                continue

    f.close()

    if _timing_frames:
        _fit_times  = [r['fit_s']  for r in _timing_frames]
        _mesh_times = [r['mesh_s'] for r in _timing_frames]
        _timing_summary = {
            'sequence': sequence_name,
            'n_frames': len(_timing_frames),
            'fit_avg_s':        round(sum(_fit_times)  / len(_fit_times),  3),
            'fit_max_s':        round(max(_fit_times),  3),
            'fit_total_s':      round(sum(_fit_times),  3),
            'mesh_export_avg_s': round(sum(_mesh_times) / len(_mesh_times), 3),
            'mesh_export_total_s': round(sum(_mesh_times), 3),
            'frames': _timing_frames,
        }
        _timing_path = os.path.join(args['output_folder'], sequence_name, 'timing.json')
        with open(_timing_path, 'w') as _tf:
            json.dump(_timing_summary, _tf, indent=2)
        print(f"[timing summary]  fit avg={_timing_summary['fit_avg_s']}s  max={_timing_summary['fit_max_s']}s"
              f"  mesh_export avg={_timing_summary['mesh_export_avg_s']}s"
              f"  → {_timing_path}")

    elapsed = time.time() - start
    time_msg = time.strftime('%H hours, %M minutes, %S seconds',
                             time.gmtime(elapsed))
    print('Processing the sequence took: {}'.format(time_msg))
    print('Failed {} frames: '.format(len(failed_frames)))
    if len(failed_frames) > 0:
        with open(os.path.join(args['output_folder'], sequence_name, 'failed_frames.txt'), 'w') as f:
            for item in failed_frames:
                f.write("%s\n" % item)


if __name__ == "__main__":
    args = parse_config()
    main(**args)
