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
                               body_faces=body_model.faces_tensor if sil_cameras else None,
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

        if frame_idx == 0 and global_betas == None:
            new_params = defaultdict(body_pose=body_mean_pose, transl=transl_init)
            body_model.reset_params(**new_params)
        else: # fix shape parameters after first frame; transl warm-starts from prev
            new_params = defaultdict(betas=global_betas, body_pose=body_mean_pose,
                                     transl=transl_init)
            body_model.reset_params(**new_params)
            body_model.betas.requires_grad = False
        if use_vposer and prev_pose_embedding is None:
            with torch.no_grad():
                pose_embedding.fill_(0)

        # five stages of optimization
        for opt_idx, curr_weights in enumerate(tqdm(opt_weights, desc='Stage')):
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
            joint_weights.mul_(valid_mask)  # zero out joints with no data this frame
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
                gt_silhouettes=gt_silhouettes)

            final_loss_val = monitor.run_fitting(
                body_optimizer,
                closure, final_params,
                body_model,
                pose_embedding=pose_embedding, vposer=vposer,
                use_vposer=use_vposer)

    #############################################
    ###### Save Meshes and Body Parameters ######
    #############################################
    body_pose = vposer.decode(
        pose_embedding,
        output_type='aa').view(1, -1) if use_vposer else None

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

