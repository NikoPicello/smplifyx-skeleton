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
from fit_single_frame import fit_single_frame

torch.backends.cudnn.enabled = False

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
    if args["dataset"] == 'ADT':
        from data_parser import ADT
        dataset_obj = ADT(sequence_path=args["data_folder"], **args)
        sequence_name = os.path.basename(args["data_folder"].rstrip('/'))
    elif args["dataset"] == 'custom':
        from data_parser import CustomDataset
        dataset_obj = CustomDataset(sequence_path=args["data_folder"], **args)
        sequence_name = os.path.basename(args["data_folder"].rstrip('/'))
    else:
        raise ValueError('Unknown dataset: {}'.format(args["dataset"]))

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
    else:
        device = torch.device('cpu')
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

    global_betas = None
    prev_pose_embedding = None
    init_betas = args.get('init_betas', None)
    if init_betas is not None:
        global_betas = torch.as_tensor(init_betas, dtype=dtype, device=device).reshape(1, -1)
        preview = global_betas.detach().cpu().numpy().flatten()[:5].round(3).tolist()
        print(f"Using injected betas (shape={list(global_betas.shape)}): {preview} ...")
    if not os.path.exists(os.path.join(args['output_folder'], sequence_name, 'meshes')):
        os.makedirs(os.path.join(args['output_folder'], sequence_name, 'meshes'))
    smplx_stored_path = os.path.join(args['output_folder'], sequence_name, 'body_smplx.json')
    mesh_stored_path = os.path.join(args['output_folder'], sequence_name, 'meshes')
    failed_frames = []
    with open(smplx_stored_path, 'w') as f:
        for idx, data in enumerate(dataset_obj):
            try:
                print('Fitting frame {}/{} ...'.format(idx+1, len(dataset_obj)))

                # Load per-frame silhouette masks — one per camera view.
                # Layout: mask_folder/{logical_cam_name}/f{idx:05d}.png
                # Pixel values: 0 = person 0, 1 = person 1, 255 = background.
                gt_silhouettes = None
                print(mask_folder)
                if mask_folder is not None and n_views > 0:
                    import cv2
                    person_id = args.get('mask_person_id', 0)
                    gt_silhouettes = []
                    for cam_name in cam_names:
                        mask_path = os.path.join(mask_folder, cam_name, f'f{idx:05d}.png')
                        print(mask_path)
                        if os.path.exists(mask_path):
                            label_map = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                            binary = (label_map == person_id).astype(np.float32)
                            gt_silhouettes.append(torch.from_numpy(binary))
                        else:
                            gt_silhouettes.append(None)

                global_betas, body_dict, body_mesh, prev_pose_embedding = fit_single_frame(
                                data,
                                frame_idx=idx,
                                global_betas=global_betas,
                                prev_pose_embedding=prev_pose_embedding,
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
                                **args)
                # store results
                body_dict['frame_idx'] = idx
                f.write(json.dumps(body_dict) + '\n')
                f.flush()
                mesh_stored_path = os.path.join(args["output_folder"], sequence_name, "meshes", f"{idx:06d}_fit.obj")
                body_mesh.export(mesh_stored_path)
            except Exception as e:
                print('Fitting sequence {} failed at frame {} with error: {}'.format(
                    sequence_name, idx, e))
                traceback.print_exc()
                failed_frames.append(idx)
                continue

    f.close()
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
