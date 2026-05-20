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
from fitting import SMPLifyLoss, _reset_lbfgs_history
from human_body_prior.tools.model_loader import load_vposer

# from mesh_intersection.bvh_search_tree import BVH
# import mesh_intersection.loss as collisions_loss
# from mesh_intersection.filter_faces import FilterFaces

apply_refinement = True

# SMPL-X body_pose is (1, 63): 21 joints × 3 axis-angle DOFs.
# Joint order within body_pose (each joint = 3 DOFs):
#   0:l_hip 1:r_hip 2:spine1 3:l_knee 4:r_knee 5:spine2
#   6:l_ankle 7:r_ankle 8:spine3 9:l_foot 10:r_foot 11:neck
#   12:l_collar 13:r_collar 14:head 15:l_shoulder 16:r_shoulder
#   17:l_elbow 18:r_elbow 19:l_wrist 20:r_wrist
# Joint order: l_hip(0), r_hip(1), spine1(2), l_knee(3), r_knee(4),
#              spine2(5), l_ankle(6), r_ankle(7), spine3(8),
#              l_foot(9), r_foot(10), neck(11), ...
# Freeze hips, knees, ankles, feet — person is seated.
_LOWER_BODY_POSE_DOFS = [
    0, 1, 2,   # left_hip
    3, 4, 5,   # right_hip
    6, 7, 8,
    9, 10, 11, # left_knee
    12, 13, 14,# right_knee
    15, 16, 17,
    18, 19, 20,# left_ankle
    21, 22, 23,# right_ankle
    27, 28, 29,# left_foot
    30, 31, 32,# right_foot
]

def _jacobian_ik(body_model, gt_joints, valid_mask, device, dtype, kwargs):
    """Levenberg-Marquardt Jacobian IK for warm-started frames.
    Solves for global_orient, body_pose (upper body only), and transl.
    Returns the final joint residual norm (used for quality / fallback check).
    """
    n_iters   = int(kwargs.get('ik_niters',   10))
    lm_lambda = float(kwargs.get('ik_lambda',  1.0))
    delta_tol = float(kwargs.get('ik_delta_tol', 1e-4))

    # Row mask: zero out NaN joints so they don't drive the solve
    valid_flat = valid_mask.view(-1).repeat_interleave(3)          # (N*3,)

    # Stacked param layout: [global_orient(3) | body_pose(63) | transl(3)] = 69
    n_params    = 69
    frozen_cols = [3 + d for d in _LOWER_BODY_POSE_DOFS]          # body_pose lower-body DOFs

    # Temporal anchor: Tikhonov regularization of body_pose toward the pose at
    # IK-call time (= previous frame's final pose on non-LBFGS frames).
    # Adds rows [α·I_pose; α·(θ_prev − θ_curr)] to the augmented system each
    # iteration, penalising cumulative drift from the previous-frame pose.
    ik_temporal_w = float(kwargs.get('ik_temporal_weight', 0.0))
    if ik_temporal_w > 0.0:
        prev_bp_flat = body_model.body_pose.detach().clone().reshape(-1)   # (63,)
        I_pose_aug   = torch.zeros(63, n_params, device=device, dtype=dtype)
        I_pose_aug[:, 3:66] = torch.eye(63, device=device, dtype=dtype) * ik_temporal_w

    for _i in range(n_iters):
        go = body_model.global_orient.detach()   # (1, 3)
        bp = body_model.body_pose.detach()       # (1, 63)
        tr = body_model.transl.detach()          # (1, 3)

        def fwd(go_, bp_, tr_):
            return body_model(body_pose=bp_, global_orient=go_, transl=tr_,
                              return_verts=False).joints.reshape(-1)

        J_go, J_bp, J_tr = torch.autograd.functional.jacobian(
            fwd, (go, bp, tr), strict=False, strategy='forward-mode', vectorize=True)
        N3 = J_go.shape[0]
        J  = torch.cat([J_go.reshape(N3, -1),
                        J_bp.reshape(N3, -1),
                        J_tr.reshape(N3, -1)], dim=1)              # (N*3, 69)

        with torch.no_grad():
            cur_joints = fwd(go, bp, tr)                           # (N*3,)
        r = gt_joints.reshape(-1) - cur_joints                     # (N*3,)

        # Apply validity mask to rows
        J = J * valid_flat.unsqueeze(1)
        r = r * valid_flat

        # Freeze lower-body columns
        J[:, frozen_cols] = 0.0

        # Levenberg-Marquardt damping: augment [J; λI] x = [r; 0]
        J_aug = torch.cat([J,
                           lm_lambda * torch.eye(n_params, device=device, dtype=dtype)], dim=0)
        r_aug = torch.cat([r, torch.zeros(n_params, device=device, dtype=dtype)], dim=0)

        # Temporal anchor: penalise cumulative drift of body_pose from the
        # pose at IK-call time (= previous frame's result).
        if ik_temporal_w > 0.0:
            anchor_res = ik_temporal_w * (prev_bp_flat - bp.reshape(-1))  # (63,)
            J_aug = torch.cat([J_aug, I_pose_aug], dim=0)
            r_aug = torch.cat([r_aug, anchor_res],  dim=0)

        delta = torch.linalg.lstsq(J_aug, r_aug.unsqueeze(1)).solution.squeeze(1)

        delta_norm = delta.norm().item()
        if torch.isnan(delta).any():
            print(f"  [IK] NaN in delta at iter {_i+1}, stopping early")
            break

        with torch.no_grad():
            body_model.global_orient.data.add_(delta[:3].view(1, 3))
            new_bp = bp + delta[3:66].view(1, 63)
            new_bp[:, _LOWER_BODY_POSE_DOFS] = bp[:, _LOWER_BODY_POSE_DOFS]
            body_model.body_pose.data.copy_(new_bp)
            body_model.transl.data.add_(delta[66:69].view(1, 3))

        print(f"  [IK] iter={_i+1:3d}  residual={r.norm().item():.4f}  |delta|={delta_norm:.5f}")
        if delta_norm < delta_tol:
            print(f"  [IK] converged (|delta|={delta_norm:.2e} < tol={delta_tol:.2e})")
            break

    # Final residual after all updates
    with torch.no_grad():
        final_r = (gt_joints.reshape(-1) -
                   body_model(return_verts=False).joints.reshape(-1)) * valid_flat
    return final_r.norm().item()


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
                    prev_left_hand_pose=None,
                    prev_right_hand_pose=None,
                    prev_refined_upper_pose=None,
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
        # body_mean_pose = torch.zeros([batch_size, vposer_latent_dim],
        #                               dtype=dtype)
    # else:
      # body_mean_pose = body_pose_prior.get_mean().detach().cpu()

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

        lbfgs_interval = int(kwargs.get('lbfgs_rerun_interval', 100))
        _do_lbfgs = (frame_idx == 0) or (frame_idx % lbfgs_interval == 0)

        if frame_idx == 0:
            # First frame: reset everything to zero, then set transl.
            body_model.reset_params(transl=transl_init)
            init_body_pose     = kwargs.get('init_body_pose',     None)
            init_global_orient = kwargs.get('init_global_orient', None)

            if use_vposer:
                with torch.no_grad():
                    pose_embedding.fill_(0)

            # INIT BETAS
            if global_betas is not None:
                with torch.no_grad():
                    body_model.betas.data.copy_(global_betas.to(device=device, dtype=dtype))
                body_model.betas.requires_grad_(False)

            # INIT BODY POSE
            if init_body_pose is not None:
                bp_t = torch.tensor(init_body_pose, dtype=dtype, device=device).reshape(1, 63)
                with torch.no_grad():
                    if use_vposer:
                        z = vposer.encode(bp_t)
                        pose_embedding.data.copy_(z.mean)
                    else:
                        body_model.body_pose.data.copy_(bp_t)

            # INIT GLOBAL ORIENT
            if init_global_orient is not None:
                go_t = torch.tensor(init_global_orient, dtype=dtype, device=device).reshape(1, 3)
                with torch.no_grad():
                    body_model.global_orient.data.copy_(go_t)
        else:
            body_model.betas.requires_grad_(False)

        # Warm-start hand poses: blend previous frame's optimized pose with the
        # current WiLoR estimate.  Alpha controls how much weight goes to the
        # previous frame (0 = pure WiLoR, 1 = pure carry-over).
        hand_prev_alpha = float(kwargs.get('hand_prev_alpha', 0.5))
        if use_hands:
            init_lh = kwargs.get('init_left_hand_pose',  None)
            init_rh = kwargs.get('init_right_hand_pose', None)
            with torch.no_grad():
                if init_lh is not None:
                    lh_t = torch.tensor(init_lh, dtype=dtype, device=device).reshape(1, -1)
                    if prev_left_hand_pose is not None:
                        lh_t = hand_prev_alpha * prev_left_hand_pose.to(device=device, dtype=dtype) \
                               + (1.0 - hand_prev_alpha) * lh_t
                    body_model.left_hand_pose.data.copy_(lh_t)
                elif prev_left_hand_pose is not None:
                    # No WiLoR for this frame — carry previous pose directly.
                    body_model.left_hand_pose.data.copy_(
                        prev_left_hand_pose.to(device=device, dtype=dtype))
                if init_rh is not None:
                    rh_t = torch.tensor(init_rh, dtype=dtype, device=device).reshape(1, -1)
                    if prev_right_hand_pose is not None:
                        rh_t = hand_prev_alpha * prev_right_hand_pose.to(device=device, dtype=dtype) \
                               + (1.0 - hand_prev_alpha) * rh_t
                    body_model.right_hand_pose.data.copy_(rh_t)
                elif prev_right_hand_pose is not None:
                    body_model.right_hand_pose.data.copy_(
                        prev_right_hand_pose.to(device=device, dtype=dtype))

        # Hard-pin lower body DOFs and global_orient to the frame-0 reference.
        # global_orient is the main cause of legs rotating (whole body drifts);
        # lower body DOFs can drift on LBFGS-rerun frames where they aren't masked.
        # Applied before optimization so IK/LBFGS linearise at the right point.
        _lb_ref = kwargs.get('lower_body_ref', None)
        _go_ref = kwargs.get('global_orient_ref', None)
        if _lb_ref is not None or _go_ref is not None:
            with torch.no_grad():
                if _lb_ref is not None:
                    body_model.body_pose.data[0, _LOWER_BODY_POSE_DOFS] = \
                        _lb_ref.to(device=device, dtype=dtype)
                if _go_ref is not None:
                    body_model.global_orient.data.copy_(
                        _go_ref.to(device=device, dtype=dtype).reshape(1, 3))

        if not _do_lbfgs:
            # Freeze transl for the IK path — IK updates it via .data directly,
            # so requires_grad is irrelevant for IK, but freezing keeps it out of
            # the direct refinement optimizer that follows.
            body_model.transl.requires_grad_(False)
            # for ji in [13, 14, 15, 16, 17, 18, 19, 20]:
            #     valid_mask[:, ji] = 0.0
            # valid_mask[:, 22:37] = 0.0
            # valid_mask[:, 38:]   = 0.0
            ik_valid_mask = valid_mask.clone()
            for ji in [13, 14, 15, 16, 17, 18, 19, 20]:
                ik_valid_mask[:, ji] = 0.0
            # ik_valid_mask[:, 22:37] = 0.0   # left finger joints — not controllable by IK params
            # ik_valid_mask[:, 38:]   = 0.0   # right finger joints


            _jacobian_ik(body_model, gt_joints, ik_valid_mask, device, dtype, kwargs)
            # Mirror the joint_weights setup done by the last LBFGS stage so
            # the direct refinement below uses the same weight scale.
            _last_w = opt_weights[-1]
            if use_hands:
                joint_weights[:, 21:] = _last_w['hand_weight']
            joint_weights[:, 5:13] = _last_w['arm_weight']
            joint_weights = joint_weights * valid_mask
            if use_face:
                joint_weights[:, 67:] = _last_w['face_weight']
        else:
            body_model.transl.requires_grad_(True)
            for opt_idx, curr_weights in enumerate(tqdm(opt_weights[:], desc='Stage')):
                body_params = list(body_model.parameters())
                final_params = list(filter(lambda x: x.requires_grad, body_params))
                if use_vposer:
                    final_params.append(pose_embedding)
                body_optimizer, body_create_graph = optim_factory.create_optimizer(final_params, **kwargs)
                body_optimizer.zero_grad()

                curr_weights['bending_prior_weight'] = (3.17e-1 * curr_weights['body_pose_weight'])
                if use_hands:
                    joint_weights[:, 21:] = curr_weights['hand_weight']
                joint_weights[:, 5:13] = curr_weights['arm_weight']
                joint_weights = joint_weights * valid_mask
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

                true_stage_idx = opt_idx
                final_loss_val = monitor.run_fitting(
                    body_optimizer,
                    closure, final_params,
                    body_model,
                    pose_embedding=pose_embedding, vposer=vposer,
                    use_vposer=use_vposer,
                    stage_idx=true_stage_idx,
                    frame_idx=frame_idx)

                # if loss.use_silhouette and gt_silhouettes is not None:
                #     with torch.no_grad():
                #         vis_pose = vposer.decode(
                #             pose_embedding, output_type='aa').view(1, -1) if use_vposer else None
                #         vis_out = body_model(return_verts=True, body_pose=vis_pose)
                #     cam_names = sorted(kwargs.get('silhouette_cameras', {}).keys()) or None
                #     loss.visualize_stage(vis_out.vertices, gt_silhouettes,
                #                          stage_idx=true_stage_idx, frame_idx=frame_idx,
                #                          cam_names=cam_names, out_dir=f"./tmp/sil_vis_{person_id}")

    #############################################
    ###### Direct body-pose refinement stage ######
    #############################################
    # VPoser is biased toward standing poses (AMASS training set), which
    # causes compensation artifacts when fitting seated subjects. Fix:
    # decode the converged VPoser pose to an explicit (1, 63) body_pose
    # tensor, then optimize all DOFs directly — joint data + face landmarks
    # drive the pose, a weak L2 prior prevents implausible angles.
    # This also fixes head orientation (face_lmk competes with nothing).
    if apply_refinement:  # run direct refinement regardless of use_vposer
        with torch.no_grad():
            if use_vposer:
                refined_body_pose = vposer.decode(
                    pose_embedding, output_type='aa').view(1, -1).clone()  # (1, 63)
            else:
                refined_body_pose = body_model.body_pose.detach().clone()  # (1, 63)

        _JOINT_DOF_MAP = {
            'spine1'         : range(6, 9),
            'spine2'         : range(15, 18),
            'spine3'         : range(24, 27),
            'neck'           : range(33, 36),
            'left_collar'    : range(36, 39),
            'right_collar'   : range(39, 42),
            'head'           : range(42, 45),
            'left_shoulder'  : range(45, 48),
            'right_shoulder' : range(48, 51)
        }
        _default_joints = ['neck', 'head', 'left_shoulder', 'right_shoulder']
        _refine_joints = kwargs.get(f'direct_refine_joints_p{person_id}', _default_joints)
        _free_dofs = [d for name in _refine_joints for d in _JOINT_DOF_MAP[name]]
        _free_idxs = torch.tensor(_free_dofs, device=device)
        _frozen_mask = torch.ones(63, dtype=torch.bool, device=device)
        _frozen_mask[_free_idxs] = False
        _frozen_idxs = _frozen_mask.nonzero(as_tuple=True)[0]

        upper_pose_direct = refined_body_pose[0, _free_idxs].clone().detach().requires_grad_(True)
        lower_pose_frozen = refined_body_pose[0, _frozen_idxs].detach()
        # Temporal anchor for the refinement stage (only meaningful for frames > 0)
        upper_pose_anchor = refined_body_pose[0, _free_idxs].clone().detach()
        jaw_pose_anchor   = body_model.jaw_pose.detach().clone()

        for p in body_model.parameters():
            p.requires_grad_(False)
        body_model.jaw_pose.requires_grad_(True)

        _d_pose_w = torch.tensor(0.15,  dtype=dtype, device=device)
        _d_data_w = torch.tensor(15.0, dtype=dtype, device=device)
        _d_face_w = torch.tensor(20.0, dtype=dtype, device=device)
        _d_jaw_w  = torch.tensor(1.0,  dtype=dtype, device=device)
        # Intra-frame: prevent direct refinement from straying far from the IK result.
        _d_temp_w = torch.tensor(5.0 if frame_idx > 0 else 0.0, dtype=dtype, device=device)
        # Cross-frame: anchor to previous frame's final refined upper pose.
        # This is the main guard against per-frame explosions propagating forward.
        # Per-person tuning: cross_temp_weight_p0 / cross_temp_weight_p1 in yaml.
        prev_upper_free = None
        _cross_w_val = float(kwargs.get(f'cross_temp_weight_p{person_id}',
                                        kwargs.get('cross_temp_weight', 20.0)))
        if frame_idx > 0 and prev_refined_upper_pose is not None:
            prev_upper_free = prev_refined_upper_pose[_free_idxs].to(device=device, dtype=dtype)
            _d_cross_w = torch.tensor(6.0, dtype=dtype, device=device)
        else:
            _d_cross_w = torch.tensor(0.0, dtype=dtype, device=device)


        direct_optim = torch.optim.LBFGS(
            [upper_pose_direct, body_model.jaw_pose],
            lr=kwargs.get('lr', 1.2), max_iter=10,
            line_search_fn='strong_wolfe')

        # Only joints that are kinematic descendants of neck (neck(3), both arm
        # chains(5-12)). Legs and spine are frozen — their residuals cannot be
        # reduced by neck/head DOFs and only pollute the gradient.
        # Head joint (4) is intentionally excluded: gt index 4 is the centroid of
        # all 68 face landmarks, which sits in front of and below the SMPLX head
        # skeletal joint. Including it in jloss pulls the neck forward (downward
        # tilt). floss (face landmark loss) handles head orientation correctly.
        _upper_body_mask = torch.zeros_like(joint_weights)
        _upper_body_mask[:, 3] = 1.0    # neck
        _upper_body_mask[:, 5:13] = 1.0  # left arm(5-8), right arm(9-12)

        def _direct_closure():
            direct_optim.zero_grad()
            with torch.no_grad():
                upper_pose_direct.data.clamp_(-torch.pi, torch.pi)
            bp = torch.zeros(1, 63, dtype=dtype, device=device)
            bp[0, _free_idxs]   = upper_pose_direct
            bp[0, _frozen_idxs] = lower_pose_frozen
            out = body_model(return_verts=True, body_pose=bp,
                             return_full_pose=True)

            proj = out.joints
            w    = (joint_weights * valid_mask * _upper_body_mask).unsqueeze(-1)
            jdiff = loss.robustifier(gt_joints - proj)
            jloss = (w ** 2 * jdiff).sum() * _d_data_w ** 2

            ploss = upper_pose_direct.pow(2).sum() * _d_pose_w ** 2

            floss = torch.tensor(0.0, device=device, dtype=dtype)
            if loss.use_face_landmarks and gt_face_landmarks is not None:
                verts_d = out.vertices[0]
                tri_v   = verts_d[loss.body_faces_lmk[loss.lmk_faces_idx]]
                lmk_pos = (tri_v * loss.lmk_bary_coords.unsqueeze(-1)).sum(dim=1)
                valid_f = ~torch.isnan(gt_face_landmarks).any(dim=-1)
                gt_lmks = torch.nan_to_num(gt_face_landmarks, nan=0.0)
                floss   = ((gt_lmks - lmk_pos).pow(2) * valid_f.unsqueeze(-1)
                           ).sum() * _d_face_w ** 2

            jploss = torch.sum(loss.jaw_prior(out.jaw_pose.mul(_d_jaw_w)))

            tloss = ((upper_pose_direct - upper_pose_anchor).pow(2).sum()
                     + (body_model.jaw_pose - jaw_pose_anchor).pow(2).sum()
                     ) * _d_temp_w ** 2

            # Cross-frame anchor: penalise distance from previous frame's refined pose.
            closs = torch.tensor(0.0, device=device, dtype=dtype)
            if prev_upper_free is not None:
                closs = (upper_pose_direct - prev_upper_free).pow(2).sum() * _d_cross_w ** 2

            total = jloss + ploss + floss + jploss + tloss + closs
            total.backward()
            pose_grad_norm = upper_pose_direct.grad.norm().item() if upper_pose_direct.grad is not None else 0.0
            jaw_grad_norm  = body_model.jaw_pose.grad.norm().item() if body_model.jaw_pose.grad is not None else 0.0
            n_clamped = ((upper_pose_direct.data.abs() >= torch.pi - 1e-4).sum().item())
            print(f"  [direct] joint={jloss.item():.2f}  pose={ploss.item():.2f}"
                  f"  face={floss.item():.2f}  jaw={jploss.item():.2f}"
                  f"  temp={tloss.item():.2f}  cross={closs.item():.2f}"
                  f"  |grad_pose|={pose_grad_norm:.4f}  |grad_jaw|={jaw_grad_norm:.4f}  clamped={n_clamped}")
            return total

        # for _ in range(5):
        #     direct_optim.step(_direct_closure)
        for step_i in range(3):
            pose_before = upper_pose_direct.data.clone()
            jaw_before  = body_model.jaw_pose.data.clone()
            direct_optim.step(_direct_closure)
            pose_delta = (upper_pose_direct.data - pose_before).norm().item()
            jaw_delta  = (body_model.jaw_pose.data - jaw_before).norm().item()
            print(f"  [direct] step={step_i}  Δpose={pose_delta:.6f}  Δjaw={jaw_delta:.6f}")


        with torch.no_grad():
            refined_body_pose = torch.zeros(1, 63, dtype=dtype, device=device)
            refined_body_pose[0, _free_idxs]   = upper_pose_direct.detach()
            refined_body_pose[0, _frozen_idxs] = lower_pose_frozen

            # Write refined pose back into the model so the save step picks it up.
            if use_vposer:
                z_refined = vposer.encode(refined_body_pose)
                pose_embedding.data.copy_(z_refined.mean)
            else:
                body_model.body_pose.data.copy_(refined_body_pose)

        for p in body_model.parameters():
            p.requires_grad_(True)
        if frame_idx != 0:
            body_model.betas.requires_grad_(False)
            body_model.transl.requires_grad_(False)
    else:
        refined_body_pose = None

    ################################################
    ###### Hand pose direct refinement         ######
    ################################################
    # if use_hands:
    if False:
        for p in body_model.parameters():
            p.requires_grad_(False)
        body_model.left_hand_pose.requires_grad_(True)
        body_model.right_hand_pose.requires_grad_(True)

        # Previous-frame anchor (cross-frame temporal smoothness)
        lh_anchor = prev_left_hand_pose.to(device=device, dtype=dtype) \
                    if prev_left_hand_pose is not None else None
        rh_anchor = prev_right_hand_pose.to(device=device, dtype=dtype) \
                    if prev_right_hand_pose is not None else None

        # WiLoR anchor for this frame: the raw per-frame estimate before blending.
        # Falls back to a small neutral L2 prior if WiLoR has no estimate.
        _wilor_lh_raw = kwargs.get('init_left_hand_pose',  None)
        _wilor_rh_raw = kwargs.get('init_right_hand_pose', None)
        wilor_lh = (torch.tensor(_wilor_lh_raw, dtype=dtype, device=device).reshape(1, -1)
                    if _wilor_lh_raw is not None else None)
        wilor_rh = (torch.tensor(_wilor_rh_raw, dtype=dtype, device=device).reshape(1, -1)
                    if _wilor_rh_raw is not None else None)

        _h_data_w  = torch.tensor(float(kwargs.get('hand_data_weight',  30.0)), dtype=dtype, device=device)
        _h_prior_w = torch.tensor(float(kwargs.get('hand_refine_prior_weight', 1.5)), dtype=dtype, device=device)
        # WiLoR pose is disabled by default: single-view estimators tend to predict fist-like
        # poses under occlusion/ambiguity. Use hand_wilor_weight > 0 only to experiment.
        _h_wilor_w = torch.tensor(float(kwargs.get('hand_wilor_weight', 0.5)), dtype=dtype, device=device)
        _h_cross_w = torch.tensor(
            float(kwargs.get('hand_cross_temp_weight', 1.0)) if frame_idx > 0 else 0.0,
            dtype=dtype, device=device)

        _hand_mask = torch.zeros_like(joint_weights)
        _hand_mask[:, 21:] = 1.0

        hand_optim = torch.optim.LBFGS(
            [body_model.left_hand_pose, body_model.right_hand_pose],
            lr=kwargs.get('lr', 1.0), max_iter=20,
            line_search_fn='strong_wolfe')

        _hand_closure_called = [0]

        def _hand_closure():
            hand_optim.zero_grad()
            out = body_model(return_verts=False)
            w = (joint_weights * valid_mask * _hand_mask).unsqueeze(-1)
            if _hand_closure_called[0] == 0:
                print(f"  [hand_dbg] gt_joints.shape={list(gt_joints.shape)}  out.joints.shape={list(out.joints.shape)}")
                print(f"  [hand_dbg] valid_hand={valid_mask[0, 21:].sum().item():.0f}/{valid_mask.shape[1]-21}  w_sum={w.sum().item():.4f}")
                print(f"  [hand_dbg] gt_lh[21:24]={gt_joints[0, 21:24, :].tolist()}")
                print(f"  [hand_dbg] gt_rh[37:40]={gt_joints[0, 37:40, :].tolist()}")
            _hand_closure_called[0] += 1
            # jdiff = loss.robustifier(gt_joints - out.joints
            jdiff = (gt_joints - out.joints).pow(2)
            hloss = (w ** 2 * jdiff).sum() * _h_data_w ** 2

            # Hand pose prior: same prior object used in LBFGS stages (L2 toward neutral
            # in current config). Weak weight — just prevents unconstrained configurations.
            hprior_loss = (torch.sum(loss.left_hand_prior(body_model.left_hand_pose))
                           + torch.sum(loss.right_hand_prior(body_model.right_hand_pose))
                           ) * _h_prior_w ** 2

            # WiLoR pose anchor: optional, off by default.
            # Enable by setting hand_wilor_weight > 0 in config.
            wilor_loss = torch.tensor(0.0, device=device, dtype=dtype)
            if _h_wilor_w.item() > 0:
                if wilor_lh is not None:
                    wilor_loss = wilor_loss + (body_model.left_hand_pose - wilor_lh).pow(2).sum() * _h_wilor_w ** 2
                if wilor_rh is not None:
                    wilor_loss = wilor_loss + (body_model.right_hand_pose - wilor_rh).pow(2).sum() * _h_wilor_w ** 2

            # Cross-frame anchor: previous frame's optimized pose. More reliable than WiLoR
            # because it was already driven by multi-view keypoints.
            closs_h = torch.tensor(0.0, device=device, dtype=dtype)
            if lh_anchor is not None:
                closs_h = closs_h + (body_model.left_hand_pose - lh_anchor).pow(2).sum() * _h_cross_w ** 2
            if rh_anchor is not None:
                closs_h = closs_h + (body_model.right_hand_pose - rh_anchor).pow(2).sum() * _h_cross_w ** 2

            total = hloss + hprior_loss + wilor_loss + closs_h
            total.backward()
            lh_grad_norm = body_model.left_hand_pose.grad.norm().item() if body_model.left_hand_pose.grad is not None else 0.0
            rh_grad_norm = body_model.right_hand_pose.grad.norm().item() if body_model.right_hand_pose.grad is not None else 0.0
            print(f"  [hand_refine] data={hloss.item():.2f}  prior={hprior_loss.item():.2f}"
                  f"  wilor={wilor_loss.item():.2f}  cross={closs_h.item():.2f}"
                  f"  |grad_lh|={lh_grad_norm:.4f}  |grad_rh|={rh_grad_norm:.4f}")

            print(f"  [hand_refine] data={hloss.item():.2f}  prior={hprior_loss.item():.2f}"
                  f"  wilor={wilor_loss.item():.2f}  cross={closs_h.item():.2f}")
            return total

        # for _ in range(5):
        #     hand_optim.step(_hand_closure)
        for step_i in range(4):
            lh_before = body_model.left_hand_pose.data.clone()
            rh_before = body_model.right_hand_pose.data.clone()
            hand_optim.step(_hand_closure)
            lh_delta = (body_model.left_hand_pose.data - lh_before).norm().item()
            rh_delta = (body_model.right_hand_pose.data - rh_before).norm().item()
            print(f"  [hand_refine] step={step_i}  Δlh={lh_delta:.6f}  Δrh={rh_delta:.6f}")
            if lh_delta > 0.1 or rh_delta > 0.1:
               _reset_lbfgs_history(hand_optim)



        for p in body_model.parameters():
            p.requires_grad_(False)

        # Re-apply pin after optimization to catch LBFGS-rerun drift.
        if _lb_ref is not None or _go_ref is not None:
            with torch.no_grad():
                if _lb_ref is not None:
                    body_model.body_pose.data[0, _LOWER_BODY_POSE_DOFS] = \
                        _lb_ref.to(device=device, dtype=dtype)
                if _go_ref is not None:
                    body_model.global_orient.data.copy_(
                        _go_ref.to(device=device, dtype=dtype).reshape(1, 3))

    #############################################
    ###### Save Meshes and Body Parameters ######
    #############################################
    if use_vposer:
        body_pose = vposer.decode(pose_embedding, output_type='aa').view(1, -1)
    else:
        body_pose = body_model.body_pose.detach()

    model_type = kwargs["model_type"]  # default: 'smplx'
    append_wrists = model_type == 'smpl' and use_vposer
    if append_wrists:
            wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                        dtype=body_pose.dtype,
                                        device=body_pose.device)
            body_pose = torch.cat([body_pose, wrist_pose], dim=1)

    model_output = body_model(return_verts=True, body_pose=body_pose if use_vposer else None)
    vertices = model_output.vertices.detach().cpu().numpy().squeeze()

    import trimesh
    out_mesh = trimesh.Trimesh(vertices, body_model.faces, process=False)

    body_dict ={"betas": body_model.betas.detach().cpu().numpy().tolist()[0],
                "body_pose": body_pose.detach().cpu().numpy().tolist()[0],
                "global_orient": body_model.global_orient.detach().cpu().numpy().tolist()[0],
                "transl": body_model.transl.detach().cpu().numpy().tolist()[0]}

    final_embedding = pose_embedding.detach().clone() if use_vposer else None
    final_lh = body_model.left_hand_pose.data.clone() if use_hands else None
    final_rh = body_model.right_hand_pose.data.clone() if use_hands else None
    # Return the final full body_pose (63,) for use as cross-frame anchor next frame.
    final_refined_upper = body_model.body_pose.data.clone()[0]  # (63,)
    return body_model.betas.data.clone(), body_dict, out_mesh, final_embedding, final_lh, final_rh, final_refined_upper

