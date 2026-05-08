# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


import time
try:
    import cPickle as pickle
except ImportError:
    import pickle

# import sys
import os
import os.path as osp

import numpy as np
import torch

from tqdm import tqdm

from collections import defaultdict

# import cv2
# import PIL.Image as pil_img

from optimizers import optim_factory

import fitting
from fitting import SMPLifyLoss
from human_body_prior.tools.model_loader import load_vposer

# from mesh_intersection.bvh_search_tree import BVH
# import mesh_intersection.loss as collisions_loss
# from mesh_intersection.filter_faces import FilterFaces

##############################
###### fit single frame ######
##############################
def fit_single_frame(
                    keypoints,
                    frame_idx,
                    global_betas,
                    search_tree,
                    pen_distance,
                    filter_faces,
                    body_model,
                    joint_weights,
                    body_pose_prior,
                    jaw_prior,
                    left_hand_prior,
                    right_hand_prior,
                    shape_prior,
                    expr_prior,
                    angle_prior,
                    person_id,
                    prev_pose_embedding=None,
                    use_cuda=True,
                    vposer_latent_dim=32,
                    batch_size=1,
                    dtype=torch.float32,
                    **kwargs):
    assert batch_size == 1, 'PyTorch L-BFGS only supports batch_size == 1'
    device = torch.device('cuda') if use_cuda else torch.device('cpu')

    #######################################################################
    ###### Prepare the weights for the different optimization stages ######
    #######################################################################
    data_weights = kwargs["data_weights"]  # default: [20, 20, 20, 20, 20]  large weights for 3D keypoints
    body_pose_prior_weights = kwargs["body_pose_prior_weights"]  # default: [4.04e0, 4.04e0, 57.4e-2, 4.78e-2, 4.78e-2], small weights for 3D keypoints to fit better
    use_hands = kwargs["use_hands"]  # default: True
    if use_hands:
        hand_pose_prior_weights = kwargs["hand_pose_prior_weights"]  # default: [4.04e0, 4.04e0, 57.4e-2, 4.78e-2, 4.78e-2], small weights for 3D keypoints to fit better
        hand_joints_weights = kwargs["hand_joints_weights"]  # default: [0.0, 0.0, 0.0, 0.1, 2.0]
    shape_weights = kwargs["shape_weights"]  # default: [1e2, 5e1, 1e1, 0.5e1, 0.5e1]
    use_face = kwargs["use_face"]
    if use_face:
        jaw_pose_prior_weights = map(lambda x: map(float, x.split(',')),
                                        kwargs["jaw_pose_prior_weights"])
        jaw_pose_prior_weights = [list(w) for w in jaw_pose_prior_weights]
        expr_weights = kwargs["expr_weights"]  # default: [1e2, 5e1, 1e1, 0.5e1, 0.5e1]
        face_joints_weights = kwargs["face_joints_weights"]  # default: [0.0, 0.0, 0.0, 0.0, 2.0]
    arm_joints_weights = kwargs["arm_joints_weights"] ##### ADDED
    coll_loss_weights = kwargs["coll_loss_weights"]  # default: [0.0, 0.0, 0.0, 0.01, 1.0]
    silhouette_weights = kwargs.get("silhouette_weights", None)

    ################################
    ###### Prepare the VPoser ######
    ################################
    gt_face_landmarks = kwargs.get("gt_face_landmarks", None)

    use_vposer = kwargs["use_vposer"]  # default: True
    vposer, pose_embedding = [None, ] * 2
    if use_vposer:
        pose_embedding = torch.zeros([batch_size, 32],
                                     dtype=dtype, device=device,
                                     requires_grad=True)
        if prev_pose_embedding is not None:
            with torch.no_grad():
                pose_embedding.copy_(prev_pose_embedding.to(device=device, dtype=dtype))
        vposer_ckpt = osp.expandvars(kwargs["vposer_ckpt"])
        vposer, _ = load_vposer(vposer_ckpt, vp_model='snapshot')
        vposer = vposer.to(device=device)
        vposer.eval()
        body_mean_pose = torch.zeros([batch_size, vposer_latent_dim],
                                     dtype=dtype)
    else:
        body_mean_pose = body_pose_prior.get_mean().detach().cpu()

    #######################################
    ###### prepare the keypoint data ######
    #######################################
    keypoint_data = torch.tensor(keypoints, dtype=dtype)
    gt_joints = keypoint_data[:, :, :3].to(device=device, dtype=dtype)
    # per-frame validity: joints with any NaN coordinate have no data this frame
    valid_mask = (~torch.isnan(gt_joints).any(dim=-1)).float()  # (1, num_joints)
    gt_joints = torch.nan_to_num(gt_joints, nan=0.0)

    #################################################################
    ###### Weights used for the pose prior and the shape prior ######
    #################################################################
    opt_weights_dict = {'data_weight': data_weights,
                        'body_pose_weight': body_pose_prior_weights,
                        'shape_weight': shape_weights,
                        'arm_weight': arm_joints_weights} #### ADDED
    if use_face:
        opt_weights_dict['face_weight'] = face_joints_weights
        opt_weights_dict['expr_prior_weight'] = expr_weights
        opt_weights_dict['jaw_prior_weight'] = jaw_pose_prior_weights
    if use_hands:
        opt_weights_dict['hand_weight'] = hand_joints_weights
        opt_weights_dict['hand_prior_weight'] = hand_pose_prior_weights
    if kwargs["interpenetration"]:
        opt_weights_dict['coll_loss_weight'] = coll_loss_weights
    if silhouette_weights is not None:
        opt_weights_dict['silhouette_weight'] = silhouette_weights
    keys = opt_weights_dict.keys()
    opt_weights = [dict(zip(keys, vals)) for vals in
                   zip(*(opt_weights_dict[k] for k in keys
                         if opt_weights_dict[k] is not None))]
    for weight_list in opt_weights:
        for key in weight_list:
            weight_list[key] = torch.tensor(weight_list[key],
                                            device=device,
                                            dtype=dtype)

    #################################
    ###### Create fitting loss ######
    #################################
    # gt_silhouettes is a list of (H, W) tensors, one per camera view (None if mask missing)
    gt_silhouettes = kwargs.get("gt_silhouettes", None)
    sil_cameras = []
    if gt_silhouettes is not None and silhouette_weights is not None:
        # silhouette_cameras is a dict {logical_cam_name: {K,D,R,T,image_size}}
        silhouette_cameras = kwargs.get("silhouette_cameras", None)
        if silhouette_cameras is not None:
            for cam_name in sorted(silhouette_cameras.keys()):
                sil_cameras.append(
                    fitting.build_camera_tensors(silhouette_cameras[cam_name], device))
        else:
            print("Warning: gt_silhouettes provided but silhouette_cameras is missing — skipping silhouette term.")

    # Load SMPLX static face landmark data (51 inner dlib landmarks via
    # barycentric coords). Used when gt_face_landmarks is provided.
    lmk_faces_idx, lmk_bary_coords = None, None
    if gt_face_landmarks is not None and kwargs.get('model_type', 'smplx') == 'smplx':
        _gender = kwargs.get('gender', 'neutral').upper()
        _smplx_npz = osp.join(osp.expandvars(kwargs['model_folder']),
                              'smplx', f'SMPLX_{_gender}.npz')
        if osp.isfile(_smplx_npz):
            _d = np.load(_smplx_npz, allow_pickle=True)
            lmk_faces_idx  = _d['lmk_faces_idx']   # (51,)
            lmk_bary_coords = _d['lmk_bary_coords']  # (51, 3)

    loss = SMPLifyLoss(joint_weights=joint_weights,
                               pose_embedding=pose_embedding,
                               body_pose_prior=body_pose_prior,
                               shape_prior=shape_prior,
                               angle_prior=angle_prior,
                               expr_prior=expr_prior,
                               left_hand_prior=left_hand_prior,
                               right_hand_prior=right_hand_prior,
                               jaw_prior=jaw_prior,
                               pen_distance=pen_distance,
                               search_tree=search_tree,
                               tri_filtering_module=filter_faces,
                               cameras=sil_cameras if sil_cameras else None,
                               body_faces=body_model.faces_tensor,
                               lmk_faces_idx=lmk_faces_idx,
                               lmk_bary_coords=lmk_bary_coords,
                               dtype=dtype,
                               **kwargs)
    loss = loss.to(device=device)

    #############################
    ###### Fitting Process ######
    #############################
    with fitting.FittingMonitor(**kwargs) as monitor:
        # Initialize transl from the pelvis keypoint (joint 0) so the optimizer
        # starts the body at the right world-space position rather than at the
        # model origin.  Fall back to centroid of all valid joints if pelvis is NaN.
        pelvis_3d = gt_joints[0, 0]  # (3,) world-space pelvis from triangulation
        if torch.isnan(pelvis_3d).any():
            valid_j = valid_mask[0].bool()
            pelvis_3d = gt_joints[0, valid_j].mean(dim=0) if valid_j.any() else pelvis_3d
        transl_init = pelvis_3d.detach().cpu().unsqueeze(0)  # (1, 3)

        if frame_idx == 0 and global_betas is None:
            # First frame: reset everything to zero, then set transl.
            body_model.reset_params(transl=transl_init)
            if use_vposer:
                with torch.no_grad():
                    pose_embedding.fill_(0)
        else:
            # Subsequent frames: only update transl and betas.
            # Do NOT call reset_params (it zeros global_orient, hand poses, etc.).
            # global_orient, hand poses, expression carry over from the previous frame.
            with torch.no_grad():
                body_model.transl.data.copy_(transl_init.to(device=device, dtype=dtype))
                if global_betas is not None:
                    body_model.betas.data.copy_(global_betas.to(device=device, dtype=dtype))
            # Shape is already estimated — freeze it so it doesn't drift.
            body_model.betas.requires_grad_(False)

        # Frames > 0 have a good warm-started estimate: skip the coarse stages
        # (0 and 1) that are only needed to bootstrap from scratch.
        stage_start = 0 if frame_idx == 0 else 2

        for opt_idx, curr_weights in enumerate(tqdm(opt_weights[stage_start:], desc='Stage')):
            body_params = list(body_model.parameters())
            final_params = list(filter(lambda x: x.requires_grad, body_params))
            if use_vposer:
                final_params.append(pose_embedding)
            body_optimizer, body_create_graph = optim_factory.create_optimizer(final_params, **kwargs)
            body_optimizer.zero_grad()

            curr_weights['bending_prior_weight'] = (3.17e-1 * curr_weights['body_pose_weight'])
            if use_hands:
                joint_weights[:, 21:] = curr_weights['hand_weight']
            joint_weights[:, 5:13] = curr_weights['arm_weight'] ##### ADDED
            joint_weights = joint_weights * valid_mask  # zero out joints with no data this frame
            # if use_hands:
            #     joint_weights[:, 25:67] = curr_weights['hand_weight'] -> ORIGINAL
            if use_face:
                joint_weights[:, 67:] = curr_weights['face_weight']
            loss.reset_loss_weights(curr_weights)

            closure = monitor.create_fitting_closure(
                body_optimizer, body_model,
                gt_joints=gt_joints,
                joint_weights=joint_weights,
                loss=loss, create_graph=body_create_graph,
                use_vposer=use_vposer, vposer=vposer,
                pose_embedding=pose_embedding,
                return_verts=True, return_full_pose=True,
                gt_silhouettes=gt_silhouettes,
                gt_face_landmarks=gt_face_landmarks)

            true_stage_idx = stage_start + opt_idx
            final_loss_val = monitor.run_fitting(
                body_optimizer,
                closure, final_params,
                body_model,
                pose_embedding=pose_embedding, vposer=vposer,
                use_vposer=use_vposer,
                stage_idx=true_stage_idx)

            # Visualise silhouette alignment after this stage
            if loss.use_silhouette and gt_silhouettes is not None:
                with torch.no_grad():
                    vis_pose = vposer.decode(
                        pose_embedding, output_type='aa').view(1, -1) if use_vposer else None
                    vis_out = body_model(return_verts=True, body_pose=vis_pose)
                cam_names = sorted(kwargs.get('silhouette_cameras', {}).keys()) or None
                loss.visualize_stage(vis_out.vertices, gt_silhouettes,
                                     stage_idx=true_stage_idx, frame_idx=frame_idx,
                                     cam_names=cam_names, out_dir=f"./tmp/sil_vis_{person_id}")

    #############################################
    ###### Head-only refinement stage        ######
    #############################################
    # After VPoser optimization, the face orientation is often wrong because
    # rotating the head via pose_embedding disturbs all other body joints,
    # creating a local minimum where face_lmk stays large (~9cm error).
    # Fix: decode the final VPoser pose, then directly optimize only the
    # neck (body_pose[33:36]) and head (body_pose[42:45]) DOFs with face
    # landmark loss only — no joint_loss, no VPoser prior competing.
    if (use_vposer and loss.use_face_landmarks
            and gt_face_landmarks is not None):
        with torch.no_grad():
            refined_body_pose = vposer.decode(
                pose_embedding, output_type='aa').view(1, -1).clone()  # (1, 63)

        # neck = joints-1=11 → pose[33:36], head = joints-1=14 → pose[42:45]
        _NECK = slice(33, 36)
        _HEAD = slice(42, 45)
        neck_head = torch.cat([
            refined_body_pose[0, _NECK],
            refined_body_pose[0, _HEAD],
        ]).clone().detach().requires_grad_(True)  # (6,)
        rest_pose = refined_body_pose.clone().detach()  # (1, 63) stays frozen

        for p in body_model.parameters():
            p.requires_grad_(False)
        body_model.jaw_pose.requires_grad_(True)

        _face_w = torch.tensor(20.0, dtype=dtype, device=device)
        _jaw_w  = torch.tensor(1.0,  dtype=dtype, device=device)

        head_optim = torch.optim.LBFGS(
            [neck_head, body_model.jaw_pose],
            lr=kwargs.get('lr', 1.2), max_iter=20,
            line_search_fn='strong_wolfe')

        def _head_closure():
            head_optim.zero_grad()
            bp = rest_pose.clone()
            bp[0, _NECK] = neck_head[:3]
            bp[0, _HEAD] = neck_head[3:]
            out = body_model(return_verts=True, body_pose=bp, return_full_pose=True)

            verts_h = out.vertices[0]
            tri_v   = verts_h[loss.body_faces_lmk[loss.lmk_faces_idx]]
            lmk_pos = (tri_v * loss.lmk_bary_coords.unsqueeze(-1)).sum(dim=1)
            valid   = ~torch.isnan(gt_face_landmarks).any(dim=-1)
            gt_lmks = torch.nan_to_num(gt_face_landmarks, nan=0.0)
            f_loss  = ((gt_lmks - lmk_pos).pow(2) * valid.unsqueeze(-1)).sum() * _face_w ** 2
            j_loss  = torch.sum(loss.jaw_prior(out.jaw_pose.mul(_jaw_w)))
            total   = f_loss + j_loss
            total.backward()
            print(f"  [head] face_lmk={f_loss.item():.2f}  jaw={j_loss.item():.2f}")
            return total

        for _ in range(30):
            head_optim.step(_head_closure)

        with torch.no_grad():
            refined_body_pose[0, _NECK] = neck_head[:3].detach()
            refined_body_pose[0, _HEAD] = neck_head[3:].detach()

        for p in body_model.parameters():
            p.requires_grad_(True)
        # Re-freeze betas if they were frozen (frame_idx > 0 path)
        if frame_idx != 0 or global_betas is not None:
            body_model.betas.requires_grad_(False)
    else:
        refined_body_pose = None

    #############################################
    ###### Save Meshes and Body Parameters ######
    #############################################
    body_pose = vposer.decode(
        pose_embedding,
        output_type='aa').view(1, -1) if use_vposer else None
    if refined_body_pose is not None:
        body_pose = refined_body_pose

    model_type = kwargs["model_type"]  # default: 'smplx'
    append_wrists = model_type == 'smpl' and use_vposer
    if append_wrists:
            wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                        dtype=body_pose.dtype,
                                        device=body_pose.device)
            body_pose = torch.cat([body_pose, wrist_pose], dim=1)

    model_output = body_model(return_verts=True, body_pose=body_pose)
    vertices = model_output.vertices.detach().cpu().numpy().squeeze()

    import trimesh
    out_mesh = trimesh.Trimesh(vertices, body_model.faces, process=False)

    body_dict ={"betas": body_model.betas.detach().cpu().numpy().tolist()[0],
                "body_pose": body_pose.detach().cpu().numpy().tolist()[0],
                "global_orient": body_model.global_orient.detach().cpu().numpy().tolist()[0],
                "transl": body_model.transl.detach().cpu().numpy().tolist()[0]}

    final_embedding = pose_embedding.detach().clone() if use_vposer else None
    return body_model.betas.data.clone(), body_dict, out_mesh, final_embedding

