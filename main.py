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
import cv2
import numpy as np
import torch
import time
import smplx
import traceback



from cmd_parser import parse_config
from utils import JointMapper, aa_nearest
from prior import create_prior
from fit_single_frame import fit_single_frame, _LOWER_BODY_POSE_DOFS, _STATIC_POSE_DOFS

torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
import random as _random
np.random.seed(42)
_random.seed(42)
torch.use_deterministic_algorithms(True, warn_only=True)

try:
    import cPickle as pickle
except ImportError:
    import pickle


def _load_gt_silhouettes(mask_folder, cam_names, frame_idx, person_id, device):
    """Per-camera binary silhouettes for one frame, aligned with `cam_names`.

    Masks live at {mask_folder}/{cam_name}/f{frame_idx:05d}.png with pixel
    values 0=person0, 1=person1, 255=background. Returns a list (same order as
    `cam_names`) of (H, W) float32 tensors — 1.0 where this person, else 0.0 —
    or None for any view whose mask file is missing. The fitting helper skips
    None views and all-zero masks (person not visible in that view).
    """
    sils = []
    for cam_name in cam_names:
        mpath = os.path.join(mask_folder, cam_name, f'f{frame_idx:05d}.png')
        m = cv2.imread(mpath, cv2.IMREAD_UNCHANGED)
        if m is None:
            sils.append(None)
            continue
        sil = (m == person_id).astype(np.float32)
        sils.append(torch.from_numpy(sil).to(device=device, dtype=torch.float32))
    return sils


def main(**args):
    float_dtype = args['float_dtype']
    if float_dtype == 'float64':
        dtype = torch.float64
    elif float_dtype == 'float32':
        dtype = torch.float32
    else:
        raise ValueError('Unknown float type {}, exiting!'.format(float_dtype))

    use_cuda = args.get('use_cuda', True)
    if use_cuda and not torch.cuda.is_available():
        raise ValueError('CUDA is not available, exiting!')

    start = time.time()
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
    # Optional injected betas (SMPLer-X): frozen in fit_single_frame when set.
    silhouette_cameras = args.get('silhouette_cameras', None)
    if silhouette_cameras is not None:
        print(f"Using {len(silhouette_cameras)} silhouette cameras: {list(silhouette_cameras.keys())}")

    rtmo_folder = args.get('rtmo_folder', None)
    mv_rtmo = {}                                  # {cam_name: per-frame array}
    if rtmo_folder is not None and silhouette_cameras is not None:
        for cam_name in sorted(silhouette_cameras.keys()):
            _p = os.path.join(rtmo_folder, f'{cam_name}_rtmo.npy')
            if os.path.isfile(_p):
                mv_rtmo[cam_name] = np.load(_p, allow_pickle=True)
            else:
                print(f"  [mv2d] no RTMO for {cam_name} ({_p})")
        print(f"Loaded multi-view RTMO for {sorted(mv_rtmo.keys())}")


    # Per-frame SMPLer-X init: body_pose / global_orient / transl / betas (world frame).
    init_bps, init_gos, init_trs, init_betas = dataset_obj.get_init_body(init_poses=True)
    init_left_hand_poses  = dataset_obj.get_init_hand_poses('left')
    init_right_hand_poses = dataset_obj.get_init_hand_poses('right')

    global_betas = None
    prev_left_hand_pose  = None
    prev_right_hand_pose = None
    prev_body_pose = None
    ref_lower_body    = None  # frame-0 static DOFs (legs + spine), pinned for all later frames
    ref_global_orient = None  # frame-0 global_orient, anchored for all later frames
    ref_translation   = None  # frame-0 translation, anchored for all later frames
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
        prev_global_orient = None
        prev_translation = None
        prev_saved_go = None   # for axis-angle unwrap of the saved global_orient trajectory
        for idx, data in enumerate(dataset_obj):
            try:
                if idx > 24:
                  break
                print('Fitting frame {}/{} ...'.format(idx+1, len(dataset_obj)))
                frame_args = args.copy()

                gt_silhouettes = None
                mask_folder = args.get('mask_folder', None)
                if mask_folder is not None and silhouette_cameras is not None:
                    gt_silhouettes = _load_gt_silhouettes(
                        mask_folder, sorted(silhouette_cameras.keys()), idx,
                        args.get('mask_person_id', args.get('person_id', 0)),
                        device)

                # GB 2D keypoints for this frame/person (COCO-17), for the GB stage.
                mv_kp2d = {}
                _pid = args.get('person_id', 0)
                for cam_name, arr in mv_rtmo.items():
                    if idx >= len(arr):
                        continue
                    _fr  = arr[idx]
                    _det = _fr.get(_pid) if isinstance(_fr, dict) else None
                    if isinstance(_det, dict) and 'keypoints' in _det:
                        _k = torch.as_tensor(np.asarray(_det['keypoints'],       dtype=np.float32), device=device)
                        _c = torch.as_tensor(np.asarray(_det['keypoint_scores'], dtype=np.float32), device=device)
                        mv_kp2d[cam_name] = (_k, _c)
                frame_args['mv_kp2d'] = mv_kp2d

                if idx == 0:
                    frame_args['maxiters'] = int(args['maxiters'] * 1.5)

                # Pass frame-0 references so fit_single_frame can pin the static parts (legs +
                # spine pose, global_orient, translation) to frame 0 for all subsequent frames.
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

                # Per-frame inner face landmarks (51, dlib 17-67); occluded -> NaN (skipped).
                gt_face_landmarks = None
                if dataset_obj.face_data is not None and idx < len(dataset_obj.face_data):
                    fl = dataset_obj.face_data[idx][17:68]          # (51, 4): xyz + conf
                    lmks = fl[:, :3].astype(np.float32).copy()
                    lmks[fl[:, 3] <= 0] = np.nan
                    gt_face_landmarks = torch.from_numpy(lmks).to(device=device, dtype=dtype)

                _t_fit = time.time()
                output_model = fit_single_frame(
                                data,
                                frame_idx=idx,
                                global_betas=global_betas,
                                prev_body_pose=prev_body_pose,
                                prev_global_orient=prev_global_orient,
                                prev_translation=prev_translation,
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
                                gt_face_landmarks=gt_face_landmarks,
                                gt_silhouettes=gt_silhouettes,
                                device=device,
                                **frame_args)
                _dt_fit = time.time() - _t_fit

                body_pose = output_model.body_pose.detach()
                output = output_model(return_verts=args.get('save_mesh', True), body_pose=body_pose)

                # Unwrap the saved global_orient so the trajectory stays continuous across the
                # axis-angle pi-boundary (same rotation, no spurious ~2*pi sign flip). Mesh is
                # unaffected; this only cleans the numbers for downstream smoothing / export.
                _go_save = output.global_orient.detach().reshape(1, 3)
                if prev_saved_go is not None:
                    _go_save = aa_nearest(_go_save, prev_saved_go)
                prev_saved_go = _go_save.clone()

                body_dict ={"frame_idx": idx,
                            "betas": output.betas.detach().cpu().numpy().tolist()[0],
                            "body_pose": output.body_pose.detach().cpu().numpy().tolist()[0],
                            "left_hand_pose": output.left_hand_pose.detach().cpu().numpy().tolist()[0],
                            "right_hand_pose": output.right_hand_pose.detach().cpu().numpy().tolist()[0],
                            "expression": output.expression.detach().cpu().numpy().tolist()[0],
                            "jaw_pose": output_model.jaw_pose.detach().cpu().numpy().tolist()[0],
                            "leye_pose": output_model.leye_pose.detach().cpu().numpy().tolist()[0],
                            "reye_pose": output_model.reye_pose.detach().cpu().numpy().tolist()[0],
                            "global_orient": _go_save.cpu().numpy().tolist()[0],
                            "transl": output.transl.detach().cpu().numpy().tolist()[0]}

                if args.get('save_mesh', True):
                    vertices = output.vertices.detach().cpu().numpy().squeeze()
                    import trimesh
                    body_mesh = trimesh.Trimesh(vertices, output_model.faces, process=False)
                else:
                    body_mesh = None

                # update body dict and temporal consistent body pose
                global_betas         = torch.tensor(body_dict['betas'], device=device)
                prev_body_pose       = torch.tensor(body_dict['body_pose'], device=device)
                prev_global_orient   = torch.tensor(body_dict['global_orient'], device=device)
                prev_translation     = torch.tensor(body_dict['transl'], device=device)
                prev_left_hand_pose  = torch.tensor(body_dict['left_hand_pose'], device=device)
                prev_right_hand_pose = torch.tensor(body_dict['right_hand_pose'], device=device)

                # Capture frame 0 as the FIXED reference
                if idx == 0:
                    ref_lower_body    = prev_body_pose[_STATIC_POSE_DOFS].clone()
                    ref_global_orient = prev_global_orient.clone()
                    ref_translation   = prev_translation.clone()

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
