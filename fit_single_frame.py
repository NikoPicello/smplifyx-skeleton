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

import cv2
# import PIL.Image as pil_img

from optimizers import optim_factory

import fitting
from fitting import SMPLifyLoss
from human_body_prior.tools.model_loader import load_vposer

_LOWER_BODY_POSE_DOFS = [
    0, 1, 2,   # left_hip
    3, 4, 5,   # right_hip
    9, 10, 11, # left_knee
    12, 13, 14,# right_knee
    18, 19, 20,# left_ankle
    21, 22, 23,# right_ankle
    27, 28, 29,# left_foot
    30, 31, 32,# right_foot
]
SEATED_HIP_X  = -1.1
SEATED_KNEE_X =  1.3

# Seated leg template. The legs carry no reliable keypoint data (knees/ankles are
# in joints_to_ign), so the lower body is *prescribed*, not fit: seeded before
# optimization and hard-set after it. Add more lower-body DOFs here (e.g. hip
# ab/adduction 1,2,4,5) if a pose needs them — values are axis-angle radians.
_SEATED_POSE = {0: SEATED_HIP_X, 3: SEATED_HIP_X, 9: SEATED_KNEE_X, 12: SEATED_KNEE_X}

# Rigid (Kabsch) frame-0 placement config. In-file constants (not plumbed
# through kwargs/config while iterating).
#   mode: 'kabsch'      -> R from Kabsch + transl solve
#         'orient_init' -> keep SMPLer-X global_orient, solve transl only
#         'off'         -> skip placement entirely
_RIGID_INIT_MODE = 'off'
# Rotation set: shoulders<->hips span the torso vertically (well-conditioned pitch/
# roll), nose/ears sit off the coronal plane (pin forward lean), and the hips are
# rigid to the pelvis so they fix yaw/roll directly instead of through the (possibly
# wrong) spine pose. Hips added once GB triangulation made them reliable.
_RIGID_INIT_JOINTS = [0, 3, 4, 5, 6, 11, 12]   # nose, ears, shoulders, hips
# Translation = the hips alone (they ARE the pelvis), so the root lands exactly on
# the observed hips — no levitation when torso length / betas don't match reality.
_TRANSL_INIT_JOINTS = [11, 12]                 # hips
_RIGID_INIT_HEAD_DOWNWEIGHT = 0.5              # head joints turn independently of the pelvis


def _apply_seated_legs(body_model):
    """In-place hard-set of the seated leg template onto body_pose (no grad)."""
    with torch.no_grad():
        for dof, val in _SEATED_POSE.items():
            body_model.body_pose.data[0, dof] = val


# Reference translation captured at frame 0 (per person). Subsequent frames are
# shifted back toward it to suppress per-frame positional (root) jitter.
_TRANSL_REF = {}  # {person_id: (1,3) reference transl tensor}

def _jacobian_ik(body_model, gt_joints, valid_mask, device, dtype, kwargs):
    """Levenberg-Marquardt Jacobian IK for warm-started frames.
    Solves for body_pose (upper body only), or body_pose + global_orient + transl,
    depending on the ik_update_global_transl kwarg.
    Returns the final joint residual norm (used for quality / fallback check).
    """
    n_iters   = int(kwargs.get('ik_niters',   10))
    lm_lambda = float(kwargs.get('ik_lambda',  1.0))
    delta_tol = float(kwargs.get('ik_delta_tol', 1e-4))
    update_global_transl = True

    num_joints = valid_mask.shape[1]
    ik_joint_w = torch.ones(num_joints, device=device, dtype=dtype)
    ik_joint_w[6]  = 0.8
    ik_joint_w[7]  = 0.15
    ik_joint_w[8]  = 2.
    ik_joint_w[10] = 0.8
    ik_joint_w[11] = 0.15
    ik_joint_w[12] = 2.

    valid_flat = (valid_mask.view(-1) * ik_joint_w).repeat_interleave(3)  # (N*3,)

    if update_global_transl:
        n_params    = 69
        bp_offset   = 3   # body_pose starts at col 3
    else:
        n_params    = 63
        bp_offset   = 0

    ik_lb_reg_w = float(kwargs.get('ik_lower_body_reg_weight', 15.0))
    _lb_cols = [d + bp_offset for d in _LOWER_BODY_POSE_DOFS]
    n_lb = len(_lb_cols)
    I_lb_aug = torch.zeros(n_lb, n_params, device=device, dtype=dtype)
    for _row, _col in enumerate(_lb_cols):
        I_lb_aug[_row, _col] = ik_lb_reg_w
    _lb_dof_idx = torch.tensor(_LOWER_BODY_POSE_DOFS, device=device)
    _lb_flat_anchor = body_model.body_pose.detach().reshape(-1)[_lb_dof_idx]  # fixed pre-IK reference

    ik_go_reg_w = float(kwargs.get('ik_global_orient_reg_weight', 15.0))
    if update_global_transl and ik_go_reg_w > 0.0:
        go_anchor = body_model.global_orient.detach().clone().reshape(-1)   # (3,)
        I_go_aug  = torch.zeros(3, n_params, device=device, dtype=dtype)
        I_go_aug[:, :3] = torch.eye(3, device=device, dtype=dtype) * ik_go_reg_w

    ik_temporal_w = float(kwargs.get('ik_temporal_weight', 0.0))
    if ik_temporal_w > 0.0:
        prev_bp_flat = body_model.body_pose.detach().clone().reshape(-1)   # (63,)
        I_pose_aug   = torch.zeros(63, n_params, device=device, dtype=dtype)
        I_pose_aug[:, bp_offset:bp_offset + 63] = torch.eye(63, device=device, dtype=dtype) * ik_temporal_w

    _t0_ik = time.perf_counter()
    for _i in range(n_iters):
        go = body_model.global_orient.detach()   # (1, 3)
        bp = body_model.body_pose.detach()       # (1, 63)
        tr = body_model.transl.detach()          # (1, 3)

        def fwd(go_, bp_, tr_):
            return body_model(body_pose=bp_, global_orient=go_, transl=tr_,
                              return_verts=False).joints.reshape(-1)

        _t_jac = time.perf_counter()
        J_go, J_bp, J_tr = torch.autograd.functional.jacobian(
            fwd, (go, bp, tr), strict=False, strategy='forward-mode', vectorize=True)
        if device.type == 'cuda': torch.cuda.synchronize()
        _dt_jac = time.perf_counter() - _t_jac
        N3 = J_bp.shape[0]

        if update_global_transl:
            J = torch.cat([J_go.reshape(N3, -1),
                           J_bp.reshape(N3, -1),
                           J_tr.reshape(N3, -1)], dim=1)  # (N*3, 69)
        else:
            J = J_bp.reshape(N3, -1)                      # (N*3, 63)

        with torch.no_grad():
            cur_joints = fwd(go, bp, tr)                           # (N*3,)
        r = gt_joints.reshape(-1) - cur_joints                     # (N*3,)

        # Apply validity mask to rows
        J = J * valid_flat.unsqueeze(1)
        r = r * valid_flat

        # Levenberg-Marquardt damping: augment [J; λI] x = [r; 0]
        J_aug = torch.cat([J,
                           lm_lambda * torch.eye(n_params, device=device, dtype=dtype)], dim=0)
        r_aug = torch.cat([r, torch.zeros(n_params, device=device, dtype=dtype)], dim=0)

        # Lower-body soft regularization: anchor toward the pre-IK state (previous frame's result).
        _curr_lb = bp.reshape(-1)[_lb_dof_idx]
        _r_lb = ik_lb_reg_w * (_lb_flat_anchor - _curr_lb)
        J_aug = torch.cat([J_aug, I_lb_aug], dim=0)
        r_aug = torch.cat([r_aug, _r_lb], dim=0)

        # Global-orient anchor: resist spinning the whole body across iterations.
        if update_global_transl and ik_go_reg_w > 0.0:
            go_anchor_res = ik_go_reg_w * (go_anchor - go.reshape(-1))   # (3,)
            J_aug = torch.cat([J_aug, I_go_aug], dim=0)
            r_aug = torch.cat([r_aug, go_anchor_res], dim=0)

        # Temporal anchor: penalise cumulative drift of body_pose from the
        # pose at IK-call time (= previous frame's result).
        if ik_temporal_w > 0.0:
            anchor_res = ik_temporal_w * (prev_bp_flat - bp.reshape(-1))  # (63,)
            J_aug = torch.cat([J_aug, I_pose_aug], dim=0)
            r_aug = torch.cat([r_aug, anchor_res],  dim=0)

        _t_lstsq = time.perf_counter()
        delta = torch.linalg.lstsq(J_aug, r_aug.unsqueeze(1)).solution.squeeze(1)
        _dt_lstsq = time.perf_counter() - _t_lstsq

        delta_norm = delta.norm().item()
        if torch.isnan(delta).any():
            print(f"  [IK] NaN in delta at iter {_i+1}, stopping early")
            break

        with torch.no_grad():
            if update_global_transl:
                body_model.global_orient.data.copy_(go + delta[:3].view(1, 3))
                body_model.body_pose.data.copy_(bp + delta[3:66].view(1, 63))
                # body_model.transl.data.copy_(tr + delta[66:].view(1, 3))
            else:
                body_model.body_pose.data.copy_(bp + delta.view(1, 63))

        print(f"  [IK] iter={_i+1:3d}  residual={r.norm().item():.4f}  |delta|={delta_norm:.5f}"
              f"  t_jac={_dt_jac:.3f}s  t_lstsq={_dt_lstsq:.3f}s")
        if delta_norm < delta_tol:
            print(f"  [IK] converged (|delta|={delta_norm:.2e} < tol={delta_tol:.2e})")
            break

    # Final residual after all updates
    with torch.no_grad():
        final_r = (gt_joints.reshape(-1) -
                   body_model(return_verts=False).joints.reshape(-1)) * valid_flat
    print(f"  [timing/IK] total={time.perf_counter()-_t0_ik:.2f}s  iters={_i+1}")
    return final_r.norm().item()


##############################
###### fit single frame ######
##############################
def fit_single_frame(
                    input_data,
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
                    prev_body_pose=None,
                    prev_left_hand_pose=None,
                    prev_right_hand_pose=None,
                    use_cuda=True,
                    vposer_latent_dim=32,
                    batch_size=1,
                    dtype=torch.float32,
                    device='cpu',
                    **kwargs):
    assert batch_size == 1, 'PyTorch L-BFGS only supports batch_size == 1'

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
    coll_loss_weights = kwargs["coll_loss_weights"]  # default: [0.0, 0.0, 0.0, 0.01, 1.0]
    silhouette_weights = kwargs.get("silhouette_weights", None)

    ################################
    ###### Prepare the VPoser ######
    ################################
    use_vposer = kwargs["use_vposer"]  # default: False
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

    #######################################
    ###### prepare the keypoint data ######
    #######################################
    kp_data = torch.tensor(input_data, dtype=dtype).to(device=device)
    gt_joints = kp_data[:, :, :3]
    conf = kp_data[:, :, 3]
    # Per-frame confidence gate: use a keypoint only when its detection confidence
    # clears joint_conf_threshold. This lets sometimes-reliable joints (the hips)
    # drop out frame-by-frame when occluded/badly triangulated, yet be exploited
    # when confident — instead of a blanket joints_to_ign. threshold 0 == old (conf>0).
    conf_thr = float(kwargs.get('joint_conf_threshold', 0.0))
    valid_mask = (conf > 0).float()
    # if conf_thr > 0:
    #   valid_mask[:, :17] = valid_mask[:, :17] * (conf[:, :17] >= conf_thr).float()
    gt_joints = torch.nan_to_num(gt_joints, nan=0.0)
    conf = torch.clamp(conf, 0., 1.0)
    per_frame_w = valid_mask * conf

    # Face landmarks for the head-refinement stage are taken straight from the input
    # keypoints (not a separate kwarg). The dataset stacks per item:
    #   [0:17] body | [17:38] Lhand | [38:59] Rhand | [59:127] face (68 dlib).
    # The loss uses the 51 INNER landmarks (dlib 17-67); occluded ones (conf<=0) are
    # flagged NaN so floss's ~isnan validity mask skips them instead of pulling to 0.
    gt_face_landmarks = None
    if use_face:
        _fb = 17 + (2 * 21 if use_hands else 0)      # start of the face block
        _inner = kp_data[0, _fb + 17:_fb + 68, :]    # (51, 4) xyz + conf
        _lmk = _inner[:, :3].clone()
        _lmk[_inner[:, 3] <= 0] = float('nan')
        gt_face_landmarks = _lmk

    if frame_idx == 0:
        print(f"[conf] hips L/R={conf[0,11].item():.2f}/{conf[0,12].item():.2f}"
              f"  thr={conf_thr}  hips_used={valid_mask[0,11].item():.0f}/{valid_mask[0,12].item():.0f}")


    #################################################################
    ###### Weights used for the pose prior and the shape prior ######
    #################################################################
    temporal_weights = kwargs.get('temporal_weights', [0.0] * len(data_weights))
    lower_body_temporal_weights = kwargs.get('lower_body_temporal_weights', [0.0] * len(data_weights))
    smpler_pose_weights = kwargs.get('smpler_pose_weights', [0.0] * len(data_weights))
    opt_weights_dict = {'data_weight': data_weights,
                        'body_pose_weight': body_pose_prior_weights,
                        'shape_weight': shape_weights,
                        'temporal_weight': temporal_weights,
                        'lower_body_temporal_weight': lower_body_temporal_weights,
                        'smpler_pose_weight': smpler_pose_weights}
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
        # _gender = 'neutral' # kwargs.get('gender', 'neutral').upper()
        _gender = 'female' if person_id == 0 else 'male'
        _smplx_npz = osp.join(osp.expandvars(kwargs['model_folder']),
                              'smplx', f'SMPLX_{_gender.upper()}.npz')
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
    _t_frame = time.perf_counter()
    _stage_times = []
    with fitting.FittingMonitor(**kwargs) as monitor:
        # Initialize transl from the pelvis so the optimizer starts the body at the
        # right world-space positioun rather than the model origin. coco17 has NO
        # pelvis joint — index 0 is the NOSE (out of scope), and seeding transl from
        # it placed the body ~0.5 m too high. Always use the hip midpoint (coco17
        # joints 11,12): the detector localizes the hips well enough even when they
        # are occluded (low confidence). Average only the conf>0 hips so a single
        # zero-confidence hip can't drag the mean to the origin; if both are absent,
        # fall back to the mean of all valid joints.
        # Seed transl from the hip midpoint, but only the hips that clear the
        # confidence gate (valid_mask). A noisy/occluded hip must not seed transl;
        # if neither hip is confident, fall back to the centroid of the valid joints.
        _hip_ok = valid_mask[0, [11, 12]].bool()
        if _hip_ok.any():
            pelvis_3d = gt_joints[0, [11, 12]][_hip_ok].mean(dim=0)
        else:
            _vj = valid_mask[0].bool()
            pelvis_3d = (gt_joints[0, _vj].mean(dim=0) if _vj.any()
                         else torch.zeros(3, device=device, dtype=gt_joints.dtype))
        transl_init = pelvis_3d.detach().cpu().unsqueeze(0)  # (1, 3)

        lbfgs_interval = int(kwargs.get('lbfgs_rerun_interval', 100))
        _do_lbfgs = True # (frame_idx == 0) or (frame_idx % lbfgs_interval == 0)
        _apply_hand_refinement = True # bool(kwargs.get('apply_hand_refinement', True))  # (frame_idx != 0)
        _apply_head_refinement = True # bool(kwargs.get('apply_head_refinement', True))  # (frame_idx != 0)

        if frame_idx == 0:
            # First frame: reset everything to zero.
            # body_model.reset_params(transl=transl_init)
            init_body_pose     = kwargs.get('init_body_pose',     None)
            init_global_orient = kwargs.get('init_global_orient', None)
            init_transl        = kwargs.get('init_transl',        None)

            if use_vposer:
                with torch.no_grad():
                    pose_embedding.fill_(0)

            # INIT BETAS
            if global_betas is not None:
                with torch.no_grad():
                    body_model.betas.data.copy_(global_betas.to(device=device, dtype=dtype))

            # INIT BODY POSE
            if init_body_pose is not None:
                bp_t = torch.tensor(init_body_pose, dtype=dtype, device=device).reshape(1, 63)
                with torch.no_grad():
                    body_model.body_pose.data.copy_(bp_t)

            # Seed the seated leg template — applied after any init_body_pose so it
            # always overrides the rest-pose zeros on those DOFs.
            _apply_seated_legs(body_model)

            # INIT GLOBAL ORIENT
            if init_global_orient is not None:
                go_t = torch.tensor(init_global_orient, dtype=dtype, device=device).reshape(1, 3)
                with torch.no_grad():
                    body_model.global_orient.data.copy_(go_t)

            # INIT TRANSL (from the SMPLer-X fusion root triangulation)
            if init_transl is not None:
                tr_t = torch.tensor(init_transl, dtype=dtype, device=device).reshape(1, 3)
                with torch.no_grad():
                    body_model.transl.data.copy_(tr_t)

            # ---- Rigid (weighted Kabsch) placement of global_orient + transl ----
            # Place/orient the body from RELIABLE joints only (well-triangulated
            # torso/head landmarks). Hips/knees/ankles are excluded: their 3D is
            # noisy and would corrupt rotation and translation alike.
            #
            # Two-step, pelvis-free:
            #   (1) Kabsch on mean-centered points -> global rotation R (independent
            #       of the pelvis pivot, so we never need the canonical pelvis).
            #   (2) with global_orient=R fixed, transl = weighted mean of
            #       (gt - model_joint); exact, since transl only shifts the body.
            #
            # "Head not aligned with hips": R is solved against the *canonical posed*
            # joints, which already bake in the SMPLer-X body_pose set just above.
            # A leaned torso / turned head is then present in BOTH X (canonical) and
            # Y (gt) and cancels -> R recovers the pelvis rotation, not the head's.
            # Leakage into the pelvis happens only if the init body_pose's relative
            # configuration is wrong; mitigate with rigid_init_head_downweight (lean
            # on the shoulders) + a soft global_orient anchor (not 'frozen') so the
            # free spine absorbs residual torso lean during LBFGS.
            #
            # Mode/joints/head-weight live in the module constants above.
            _rigid_mode = _RIGID_INIT_MODE
            if _rigid_mode != 'off':
                _RIGID_JOINTS = _RIGID_INIT_JOINTS                 # nose, ears, shoulders, hips
                _HEAD_JOINTS  = {0, 1, 2, 3, 4}                    # turn with the head, not the pelvis
                _head_dw      = _RIGID_INIT_HEAD_DOWNWEIGHT
                with torch.no_grad():
                    _go0 = torch.zeros_like(body_model.global_orient)
                    _tr0 = torch.zeros_like(body_model.transl)
                    J_can = body_model(global_orient=_go0, transl=_tr0,
                                       return_verts=False).joints[0]      # (J, 3) canonical
                    _idx = torch.tensor(_RIGID_JOINTS, device=device)
                    _w   = valid_mask[0, _idx] * conf[0, _idx]
                    for _k, _j in enumerate(_RIGID_JOINTS):               # lean rotation on shoulders
                        if _j in _HEAD_JOINTS:
                            _w[_k] = _w[_k] * _head_dw
                    _ok    = _w > 0
                    _n_ok  = int(_ok.sum().item())

                    if _rigid_mode == 'kabsch' and _n_ok >= 3:
                        X = J_can[_idx][_ok].double().cpu().numpy()       # canonical
                        Y = gt_joints[0, _idx][_ok].double().cpu().numpy()  # world targets
                        w = _w[_ok].double().cpu().numpy(); wsum = w.sum()
                        Xb = (w[:, None] * X).sum(0) / wsum
                        Yb = (w[:, None] * Y).sum(0) / wsum
                        H  = (w[:, None] * (X - Xb)).T @ (Y - Yb)
                        U, _, Vt = np.linalg.svd(H)
                        d = np.sign(np.linalg.det(Vt.T @ U.T))            # reflection guard
                        R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
                        aa, _ = cv2.Rodrigues(R)                          # (3, 1) axis-angle
                        body_model.global_orient.data.copy_(
                            torch.tensor(aa.reshape(1, 3), dtype=dtype, device=device))
                        print(f"[kabsch] R from {_n_ok} joints (head_dw={_head_dw})")
                    elif _rigid_mode == 'kabsch':
                        print(f"[kabsch] only {_n_ok} reliable joints (<3) — "
                              f"keeping SMPLer-X global_orient, transl-only seed")

                    # Step 2: translation from the HIPS (the pelvis anchor) with
                    # global_orient now fixed -> the root lands on the observed hips,
                    # so a torso/betas mismatch can't lift the body (no levitation).
                    # Fall back to the full rigid set, then any valid joint, if the
                    # hips are missing this frame.
                    J_rot  = body_model(transl=_tr0, return_verts=False).joints[0]
                    _tidx  = torch.tensor(_TRANSL_INIT_JOINTS, device=device)
                    _tw    = valid_mask[0, _tidx] * conf[0, _tidx]
                    _tok   = _tw > 0
                    if _tok.any():
                        _twv = _tw[_tok].unsqueeze(-1)
                        _t   = (_twv * (gt_joints[0, _tidx][_tok] - J_rot[_tidx][_tok])).sum(0) / _twv.sum()
                        print(f"[kabsch] transl from {int(_tok.sum())} hip joint(s)")
                    elif _n_ok >= 1:
                        _wt = _w[_ok].unsqueeze(-1)
                        _t  = (_wt * (gt_joints[0, _idx][_ok] - J_rot[_idx][_ok])).sum(0) / _wt.sum()
                        print("[kabsch] no hips — transl from rigid set")
                    else:
                        _vj = valid_mask[0].bool()                        # nothing reliable: any valid joint
                        _t  = ((gt_joints[0, _vj] - J_rot[_vj]).mean(0) if _vj.any()
                               else torch.zeros(3, device=device, dtype=dtype))
                        print("[kabsch] no rigid joints — transl from valid-joint centroid")
                    body_model.transl.data.copy_(_t.reshape(1, 3))
                    print(f"[kabsch] mode={_rigid_mode} "
                          f"transl={_t.detach().cpu().numpy().round(3).tolist()}")

            body_model.betas.requires_grad_(True)
            body_model.transl.requires_grad_(True)
            # global_orient has NO prior, so when left free it absorbs torso lean by
            # rolling/tilting the whole body — dragging the dataless legs/pelvis off
            # to the side. 'frozen' keeps it at the init (legs stay congruent, the
            # spine does the lean); 'anchored' lets it fine-tune under a soft L2 pull.
            _go_mode = kwargs.get('global_orient_mode', 'free')
            body_model.global_orient.requires_grad_(_go_mode != 'frozen')
            # Anchor target for 'anchored' mode (frame 0 → the init we just set).
            global_orient_anchor = body_model.global_orient.detach().clone()
        else:
            body_model.transl.requires_grad_(False)
            body_model.betas.requires_grad_(False)
            _go_mode = kwargs.get('global_orient_mode', 'free')
            _go_ref  = kwargs.get('global_orient_ref', None)
            if _go_ref is not None and _go_mode in ('frozen', 'anchored'):
                with torch.no_grad():
                    body_model.global_orient.data.copy_(_go_ref.to(device=device, dtype=dtype))
            body_model.global_orient.requires_grad_(_go_mode != 'frozen')
            global_orient_anchor = (_go_ref.to(device=device, dtype=dtype)
                                    if _go_ref is not None
                                    else body_model.global_orient.detach().clone())

        # Warm-start hand poses: blend previous frame's optimized pose with the
        # current WiLoR estimate.  Alpha controls how much weight goes to the
        # previous frame (0 = pure WiLoR, 1 = pure carry-over).
        hand_prev_alpha = float(kwargs.get('hand_prev_alpha', 1.))
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
        with torch.no_grad():
          if _lb_ref is not None:
            body_model.body_pose.data[0, _LOWER_BODY_POSE_DOFS] = \
              _lb_ref.to(device=device, dtype=dtype)

        if not _do_lbfgs:
            # Freeze transl for the IK path — IK updates it via .data directly,
            # so requires_grad is irrelevant for IK, but freezing keeps it out of
            # the direct refinement optimizer that follows.
            ik_valid_mask = valid_mask.clone()
            for ji in [13, 14, 15, 16, 17, 18, 19, 20]:
                ik_valid_mask[:, ji] = 0.0
                # ik_valid_mask[:, 22:37] = 0.0   # left finger joints — not controllable by IK params
                # ik_valid_mask[:, 38:]   = 0.0   # right finger joints


            _t_ik = time.perf_counter()
            _jacobian_ik(body_model, gt_joints, ik_valid_mask, device, dtype, kwargs)
            _stage_times.append(('IK', time.perf_counter() - _t_ik))
            # Mirror the joint_weights setup done by the last LBFGS stage so
            # the direct refinement below uses the same weight scale.
            _last_w = opt_weights[-1]
            if use_hands:
                joint_weights[:, 21:] = _last_w['hand_weight']
            joint_weights = joint_weights * valid_mask
            if use_face:
                joint_weights[:, 67:] = _last_w['face_weight']
        else:
            # ---- Temporal & SMPLer-X anchors (consumed inside the loss closure) ----
            # prev_body_pose drives the temporal loss; it is None on frame 0 so the
            # term stays silent there. Reshape to (1, 63) on the model's device/dtype
            # so it broadcasts against body_model_output.body_pose inside SMPLifyLoss.
            prev_body_pose_t = None
            if prev_body_pose is not None:
                prev_body_pose_t = prev_body_pose.to(device=device, dtype=dtype).reshape(1, -1)
            # Per-frame SMPLer-X pose anchor (only active if an init_body_pose was
            # supplied for this frame; None otherwise -> smpler term stays silent).
            smpler_pose_t = None
            _init_bp = kwargs.get('init_body_pose', None)
            if _init_bp is not None:
                smpler_pose_t = torch.tensor(_init_bp, dtype=dtype, device=device).reshape(1, -1)

            for opt_idx, curr_weights in enumerate(tqdm(opt_weights[:4], desc='Stage')):
                final_params = [p for p in body_model.parameters() if p.requires_grad]
                if use_vposer:
                    final_params.append(pose_embedding)
                body_optimizer, body_create_graph = optim_factory.create_optimizer(final_params, **kwargs)

                curr_weights['bending_prior_weight'] = (3.17e-1 * curr_weights['body_pose_weight'])
                if use_hands:
                    joint_weights[:, 17:] = curr_weights['hand_weight']
                if use_face:
                    joint_weights[:, 59:] = curr_weights['face_weight']
                # Each hip is a single sparse keypoint (effective weight = conf), so it
                # loses to the dense upper body + spine/pose priors and stays unfit even
                # when well triangulated. hip_weight boosts (>1) or trims (<1) them.
                joint_weights[:, 11:13] = float(kwargs.get('hip_weight', 1.0))
                # Fold per-keypoint confidence into the static joint weights for this
                # stage: per_frame_w = valid_mask * conf, so zero-confidence joints get
                # zero weight and the rest scale by their detection confidence. The loss
                # squares this (weights ** 2) before multiplying the joint residual.
                stage_weights = joint_weights * per_frame_w
                loss.reset_loss_weights(curr_weights)

                _go_anchor_w = (float(kwargs.get('global_orient_weight', 0.0))
                                if kwargs.get('global_orient_mode', 'free') == 'anchored'
                                else 0.0)
                closure = monitor.create_fitting_closure(
                    body_optimizer, body_model,
                    gt_joints=gt_joints,
                    joint_weights=stage_weights,
                    loss=loss, create_graph=body_create_graph,
                    use_vposer=use_vposer, vposer=vposer,
                    pose_embedding=pose_embedding,
                    return_verts=True, return_full_pose=True,
                    prev_body_pose=prev_body_pose_t,
                    smpler_body_pose=smpler_pose_t,
                    global_orient_ref=global_orient_anchor,
                    global_orient_weight=_go_anchor_w)

                _t_stage = time.perf_counter()
                final_loss = monitor.run_fitting(body_optimizer, closure, final_params,
                                                 body_model, pose_embedding=pose_embedding,
                                                 vposer=vposer, use_vposer=use_vposer,
                                                 stage_idx=opt_idx, frame_idx=frame_idx)
                if device.type == 'cuda': torch.cuda.synchronize()
                _dt_stage = time.perf_counter() - _t_stage
                _stage_times.append((f'LBFGS_s{opt_idx}', _dt_stage))
                with torch.no_grad():
                    _jw = body_model(return_verts=False).joints
                    _wr = ((gt_joints[0, [9, 10]] - _jw[0, [9, 10]])
                           * valid_mask[0, [9, 10]].unsqueeze(-1)).norm(dim=-1)
                    _hp = ((gt_joints[0, [11, 12]] - _jw[0, [11, 12]])
                           * valid_mask[0, [11, 12]].unsqueeze(-1)).norm(dim=-1)
                print(f"  [timing/frame {frame_idx}] LBFGS stage {opt_idx}: {_dt_stage:.2f}s"
                      f"  loss={final_loss:.4f}  wrist_resid={_wr[0].item():.3f}/{_wr[1].item():.3f}"
                      f"  hip_resid={_hp[0].item():.3f}/{_hp[1].item():.3f}")



    #############################################
    ###### head refinement stage ######
    #############################################
    # VPoser is biased toward standing poses (AMASS training set), which
    # causes compensation artifacts when fitting seated subjects. Fix:
    # decode the converged VPoser pose to an explicit (1, 63) body_pose
    # tensor, then optimize all DOFs directly — joint data + face landmarks
    # drive the pose, a weak L2 prior prevents implausible angles.
    # This also fixes head orientation (face_lmk competes with nothing).
    _t_head = time.perf_counter()
    if _apply_head_refinement:  # run head refinement regardless of use_vposer
        with torch.no_grad():
            if use_vposer:
                refined_body_pose = vposer.decode(
                    pose_embedding, output_type='aa').view(1, -1).clone()  # (1, 63)
            else:
                refined_body_pose = body_model.body_pose.detach().clone()  # (1, 63)

        # body_pose is (1,63) = 21 joints x 3 axis-angle DOFs (order per the header
        # comment). Map every joint name -> its 3 DOF columns so a config can
        # declare exactly which joints the head refinement is allowed to move.
        _JOINT_DOF_MAP = {name: range(3 * k, 3 * k + 3) for k, name in enumerate([
            'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee', 'spine2',
            'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot', 'neck',
            'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist'])}

        # Which joints the refinement may move (one set, shared by both persons):
        # head_refine_joints from config, else a sensible upper-body default.
        # Unknown names are dropped with a warning so a typo can't crash the run.
        _default_refine = ['neck', 'head', 'left_shoulder', 'right_shoulder',
                           'left_collar', 'right_collar', 'spine1', 'spine3']
        _refine_joints = kwargs.get('head_refine_joints') or _default_refine
        _bad = [n for n in _refine_joints if n not in _JOINT_DOF_MAP]
        if _bad:
            print(f"  [head] WARNING: unknown refine joints {_bad} ignored")
        _refine_joints = [n for n in _refine_joints if n in _JOINT_DOF_MAP]
        print(f"  [head] p{person_id} free joints: {_refine_joints}")
        _free_dofs = [d for name in _refine_joints for d in _JOINT_DOF_MAP[name]]
        _free_idxs = torch.tensor(_free_dofs, device=device)
        _frozen_mask = torch.ones(63, dtype=torch.bool, device=device)
        _frozen_mask[_free_idxs] = False
        _frozen_idxs = _frozen_mask.nonzero(as_tuple=True)[0]

        upper_pose_head = refined_body_pose[0, _free_idxs].clone().detach().requires_grad_(True)
        lower_pose_frozen = refined_body_pose[0, _frozen_idxs].detach()
        # Temporal anchor for the refinement stage (only meaningful for frames > 0)
        upper_pose_anchor = refined_body_pose[0, _free_idxs].clone().detach()
        jaw_pose_anchor   = body_model.jaw_pose.detach().clone()

        for p in body_model.parameters():
            p.requires_grad_(False)
        body_model.jaw_pose.requires_grad_(True)
        # global_orient (pelvis) and transl stay FIXED here: this stage only refines
        # head/neck/torso orientation via the face landmarks. Letting them move would
        # re-fit the pelvis to upper-body-only joints and undo the hip-based
        # placement — re-introducing the levitation we just removed.
        transl_anchor = body_model.transl.detach().clone()

        _d_pose_w = torch.tensor(float(kwargs.get('head_pose_weight', 0.1)),  dtype=dtype, device=device)
        _d_data_w = torch.tensor(float(kwargs.get('head_data_weight', 15.0)), dtype=dtype, device=device)
        _d_face_w = torch.tensor(float(kwargs.get('head_face_weight', 20.0)), dtype=dtype, device=device)
        _d_jaw_w  = torch.tensor(float(kwargs.get('head_jaw_weight',  1.0)),  dtype=dtype, device=device)
        # Intra-frame: prevent head refinement from straying far from the IK result.
        _d_intra_w = torch.tensor(float(kwargs.get('head_intra_weight', 1.0)), dtype=dtype, device=device)
        # Cross-frame: anchor to previous frame's final refined upper pose.
        prev_upper_free = None
        _temp_w_val = float(kwargs.get('head_temporal_weight', 0.2))
        if frame_idx > 0 and prev_body_pose is not None:
            prev_upper_free = prev_body_pose[_free_idxs].to(device=device, dtype=dtype)
            _d_temp_w = torch.tensor(_temp_w_val, dtype=dtype, device=device)
        else:
            _d_temp_w = torch.tensor(0.0, dtype=dtype, device=device)


        head_optim = torch.optim.LBFGS(
            [upper_pose_head, body_model.jaw_pose],
            lr=kwargs.get('lr', 1.), max_iter=10,
            line_search_fn='strong_wolfe')

        # Joint-data term restricted to coco17 indices the free DOFs can move and
        # whose 3D is reliable: shoulders (5,6), elbows (7,8), wrists (9,10). The
        # spine/collar/shoulder DOFs are FREE, so anchoring these makes the torso
        # bend to follow the head/neck instead of stretching the neck while the
        # upper body stays pinned. Head landmarks (0-4) are left to floss (the dense
        # face-landmark term drives head orientation); hips/legs (11-16) are
        # excluded — unreliable 3D and their pose DOFs are frozen (zero gradient).
        _upper_body_mask = torch.zeros_like(joint_weights)
        _upper_body_mask[:, 5:11] = 1.0  # shoulders, elbows, wrists

        def _head_closure():
            head_optim.zero_grad()
            with torch.no_grad():
                upper_pose_head.data.clamp_(-torch.pi, torch.pi)
            bp = torch.zeros(1, 63, dtype=dtype, device=device)
            bp[0, _free_idxs]   = upper_pose_head
            bp[0, _frozen_idxs] = lower_pose_frozen
            out = body_model(return_verts=True, body_pose=bp,
                             return_full_pose=True)

            proj = out.joints
            # Upper-body chain only (see _upper_body_mask): reliable joints the
            # free spine/neck/arm DOFs can move. Frozen legs give zero gradient to
            # upper_pose_head anyway, and their 2D is too noisy to anchor transl.
            w    = (joint_weights * _upper_body_mask * valid_mask).unsqueeze(-1)
            jdiff = loss.robustifier(gt_joints - proj)
            jloss = (w ** 2 * jdiff).sum() * _d_data_w ** 2

            ploss = upper_pose_head.pow(2).sum() * _d_pose_w ** 2

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

            iloss = (((upper_pose_head - upper_pose_anchor).pow(2).sum()
                     + (body_model.jaw_pose - jaw_pose_anchor).pow(2).sum()) * _d_intra_w ** 2
                     + (body_model.transl - transl_anchor).pow(2).sum() * 4 ** 2)


            # Cross-frame anchor: penalise distance from previous frame's refined pose.
            tloss = torch.tensor(0.0, device=device, dtype=dtype)
            if prev_upper_free is not None:
                tloss = (upper_pose_head - prev_upper_free).pow(2).sum() * _d_temp_w ** 2

            total = jloss + ploss + floss + jploss + tloss + iloss
            total.backward()
            return total

        for step_i in range(5):
            pose_before = upper_pose_head.data.clone()
            jaw_before  = body_model.jaw_pose.data.clone()
            head_optim.step(_head_closure)
            pose_delta = (upper_pose_head.data - pose_before).norm().item()
            jaw_delta  = (body_model.jaw_pose.data - jaw_before).norm().item()
            print(f"  [head] step={step_i}  Δpose={pose_delta:.6f}  Δjaw={jaw_delta:.6f}")

        with torch.no_grad():
            refined_body_pose = torch.zeros(1, 63, dtype=dtype, device=device)
            refined_body_pose[0, _free_idxs]   = upper_pose_head.detach()
            refined_body_pose[0, _frozen_idxs] = lower_pose_frozen

            if use_vposer:
                z_refined = vposer.encode(refined_body_pose)
                pose_embedding.data.copy_(z_refined.mean)
            else:
                body_model.body_pose.data.copy_(refined_body_pose)

    if _apply_head_refinement:
        _stage_times.append(('head_refine', time.perf_counter() - _t_head))

    ################################################
    ###### Hand pose refinement         ######
    ################################################
    _t_hand = time.perf_counter()
    refine_left = refine_right = False
    if _apply_hand_refinement:
        # ---- Per-hand visibility gate ---------------------------------------
        # Refining a hand's finger pose only makes sense when that hand actually
        # has triangulated finger keypoints this frame. A fully/near-fully
        # occluded hand contributes ~0 to the data term, so the prior / WiLoR /
        # temporal anchors would be the only forces left and would drift the
        # warm-started pose with nothing to constrain it. Gate each hand on its
        # visible finger-keypoint count and freeze the one(s) that fall short.
        #
        # Hand keypoint layout in gt_joints (21 joints/hand, joint 0 = wrist
        # root, which is always present — copied from the body/mano wrist):
        #   left  hand = [17:38]  -> fingers [18:38]
        #   right hand = [38:59]  -> fingers [39:59]
        _LH_FINGERS = slice(18, 38)
        _RH_FINGERS = slice(39, 59)
        _min_kpts = 3   # min visible finger keypoints to refine a hand
        _n_lh_vis = int(valid_mask[0, _LH_FINGERS].sum().item())
        _n_rh_vis = int(valid_mask[0, _RH_FINGERS].sum().item())
        refine_left  = _n_lh_vis >= _min_kpts
        refine_right = _n_rh_vis >= _min_kpts
        print(f"  [hand_refine] p{person_id} f{frame_idx} visible fingers "
              f"L={_n_lh_vis} R={_n_rh_vis} (min={_min_kpts}) -> "
              f"refine L={refine_left} R={refine_right}")

    if _apply_hand_refinement and (refine_left or refine_right):
        for p in body_model.parameters():
            p.requires_grad_(False)
        # Only the hand(s) with enough finger keypoints get optimized; the other
        # keeps its warm-started (WiLoR / carried-over) pose untouched.
        body_model.left_hand_pose.requires_grad_(refine_left)
        body_model.right_hand_pose.requires_grad_(refine_right)

        lh_anchor = prev_left_hand_pose.to(device=device, dtype=dtype) \
                    if prev_left_hand_pose is not None else None
        rh_anchor = prev_right_hand_pose.to(device=device, dtype=dtype) \
                    if prev_right_hand_pose is not None else None

        _wilor_lh_raw = kwargs.get('init_left_hand_pose',  None)
        _wilor_rh_raw = kwargs.get('init_right_hand_pose', None)
        wilor_lh = (torch.tensor(_wilor_lh_raw, dtype=dtype, device=device).reshape(1, -1)
                    if _wilor_lh_raw is not None else None)
        wilor_rh = (torch.tensor(_wilor_rh_raw, dtype=dtype, device=device).reshape(1, -1)
                    if _wilor_rh_raw is not None else None)

        _h_data_w  = torch.tensor(float(kwargs.get('hand_data_weight',  30.0)), dtype=dtype, device=device)
        _h_prior_w = torch.tensor(float(kwargs.get('hand_refine_prior_weight', 1.5)), dtype=dtype, device=device)
        _h_wilor_w = torch.tensor(float(kwargs.get('hand_wilor_weight', 0.5)), dtype=dtype, device=device)
        _h_temp_w  = torch.tensor(float(kwargs.get('hand_temporal_weight', 0.2)) if frame_idx > 0 else 0.0, dtype=dtype, device=device)

        # Data mask: only score the refined hand(s). left=[17:38], right=[38:59].
        _hand_mask = torch.zeros_like(joint_weights)
        if refine_left:  _hand_mask[:, 17:38] = 1.0
        if refine_right: _hand_mask[:, 38:59] = 1.0

        # Free the elbow+wrist DOFs of the refined arm(s) alongside the hand pose.
        # Wrist orientation is the root of the finger kinematic chain — if it's
        # wrong after LBFGS, finger poses alone can't fix the joint positions.
        # body_pose DOFs: l_elbow=51:54, r_elbow=54:57, l_wrist=57:60, r_wrist=60:63.
        _wrist_cols = []
        if refine_left:  _wrist_cols += [57, 58, 59]
        if refine_right: _wrist_cols += [60, 61, 62]
        _wrist_idx = torch.tensor(_wrist_cols, device=device)
        _wrist_free = body_model.body_pose.data[0, _wrist_idx].clone().detach().requires_grad_(True)
        _body_pose_frozen = body_model.body_pose.data.clone()  # (1, 63), all other DOFs fixed

        _opt_params = []
        if refine_left:  _opt_params.append(body_model.left_hand_pose)
        if refine_right: _opt_params.append(body_model.right_hand_pose)
        _opt_params.append(_wrist_free)

        hand_optim = torch.optim.LBFGS(
            _opt_params,
            lr=kwargs.get('lr', 1.0), max_iter=20,
            line_search_fn='strong_wolfe')

        def _hand_closure():
            hand_optim.zero_grad()
            bp = _body_pose_frozen.clone()
            bp[0, _wrist_idx] = _wrist_free
            out = body_model(return_verts=False, body_pose=bp)
            w = (joint_weights * valid_mask * _hand_mask).unsqueeze(-1)
            # GMoF-robustified residual (matches the direct stage). Raw squared
            # error here let a single triangulation outlier blow up the loss and,
            # with stock LBFGS, drive the hand pose to NaN.
            jdiff = loss.robustifier(gt_joints - out.joints)
            hloss = (w ** 2 * jdiff).sum() * _h_data_w ** 2

            # Priors / anchors only for the hand(s) actually being optimized.
            hprior_loss = torch.tensor(0.0, device=device, dtype=dtype)
            if refine_left:
                hprior_loss = hprior_loss + torch.sum(loss.left_hand_prior(body_model.left_hand_pose))
            if refine_right:
                hprior_loss = hprior_loss + torch.sum(loss.right_hand_prior(body_model.right_hand_pose))
            hprior_loss = hprior_loss * _h_prior_w ** 2

            wilor_loss = torch.tensor(0.0, device=device, dtype=dtype)
            if _h_wilor_w.item() > 0:
                if refine_left and wilor_lh is not None:
                    wilor_loss = wilor_loss + (body_model.left_hand_pose - wilor_lh).pow(2).sum() * _h_wilor_w ** 2
                if refine_right and wilor_rh is not None:
                    wilor_loss = wilor_loss + (body_model.right_hand_pose - wilor_rh).pow(2).sum() * _h_wilor_w ** 2

            tloss_h = torch.tensor(0.0, device=device, dtype=dtype)
            if refine_left and lh_anchor is not None:
                tloss_h += (body_model.left_hand_pose - lh_anchor).pow(2).sum() * _h_temp_w ** 2
            if refine_right and rh_anchor is not None:
                tloss_h += (body_model.right_hand_pose - rh_anchor).pow(2).sum() * _h_temp_w ** 2

            total = hloss + hprior_loss + wilor_loss + tloss_h
            total.backward()
            return total

        for step_i in range(5):
            lh_before = body_model.left_hand_pose.data.clone()
            rh_before = body_model.right_hand_pose.data.clone()
            hand_optim.step(_hand_closure)
            lh_delta = (body_model.left_hand_pose.data - lh_before).norm().item()
            rh_delta = (body_model.right_hand_pose.data - rh_before).norm().item()
            print(f"  [hand_refine] step={step_i}  Δlh={lh_delta:.6f}  Δrh={rh_delta:.6f}")

        # Write the refined elbow/wrist DOFs back into body_pose
        with torch.no_grad():
            body_model.body_pose.data[0, _wrist_idx] = _wrist_free.detach()

        for p in body_model.parameters():
            p.requires_grad_(True)
        if frame_idx != 0:
            body_model.betas.requires_grad_(False)

    if _apply_hand_refinement:
        _stage_times.append(('hand_refine', time.perf_counter() - _t_hand))

    #############################################
    ###### Hard-override seated lower body  ######
    #############################################
    # Lower-body GT is unreliable — bypass optimization and hard-set the legs to the
    # seated template (tune _SEATED_POSE visually).
    _apply_seated_legs(body_model)

    #############################################
    ###### Translation jitter stabilization ######
    #############################################
    # Pull the (already optimized) root translation back toward the reference
    # captured at frame 0. delta is the vector that, added to the current transl,
    # lands the position on the reference. alpha=1 snaps fully (root perfectly
    # still); alpha<1 blends so a little real drift is allowed.
    # global _TRANSL_REF
    # with torch.no_grad():
    #     if frame_idx == 0 or person_id not in _TRANSL_REF:
    #         _TRANSL_REF[person_id] = body_model.transl.detach().clone()
    #     else:
    #         _alpha = float(kwargs.get('transl_stabilize_alpha', 1.0))
    #         if _alpha > 0.0:
    #             transl_ref = _TRANSL_REF[person_id].to(device=device, dtype=dtype)
    #             delta = transl_ref - body_model.transl.detach()
    #             body_model.transl.data.add_(_alpha * delta)

    #         # Optional: let the reference drift slowly so genuine (non-jitter)
    #         # motion is tracked while fast jitter is removed. 0.0 = fixed reference.
    #         _ema = float(kwargs.get('transl_ref_ema', 0.0))
    #         if _ema > 0.0:
    #             _TRANSL_REF[person_id] = ((1.0 - _ema) * _TRANSL_REF[person_id]
    #                                       + _ema * body_model.transl.detach())

    if use_vposer:
        body_pose = vposer.decode(pose_embedding, output_type='aa').view(1, -1)
    else:
        body_pose = body_model.body_pose.detach()

    model_type = kwargs["model_type"]  # default: 'smplx'
    if kwargs.get('save_mesh', True) == True:
      model_output = body_model(return_verts=True, body_pose=body_pose)
      vertices = model_output.vertices.detach().cpu().numpy().squeeze()
      import trimesh
      out_mesh = trimesh.Trimesh(vertices, body_model.faces, process=False)
    else:
      model_output = body_model(return_verts=False, body_pose=body_pose)
      out_mesh = None

    if device.type == 'cuda': torch.cuda.synchronize()
    _t_total = time.perf_counter() - _t_frame
    _stage_str = '  '.join(f'{n}={t:.2f}s' for n, t in _stage_times)
    print(f"  [timing/frame {frame_idx}] TOTAL={_t_total:.2f}s  [{_stage_str}]")

    body_dict ={"betas": body_model.betas.detach().cpu().numpy().tolist()[0],
                "body_pose": body_pose.detach().cpu().numpy().tolist()[0],
                "left_hand_pose": body_model.left_hand_pose.detach().cpu().numpy().tolist()[0],
                "right_hand_pose": body_model.right_hand_pose.detach().cpu().numpy().tolist()[0],
                "expression": body_model.expression.detach().cpu().numpy().tolist()[0],
                "global_orient": body_model.global_orient.detach().cpu().numpy().tolist()[0],
                "transl": body_model.transl.detach().cpu().numpy().tolist()[0]}

    return body_dict, out_mesh
