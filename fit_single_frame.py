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

from optimizers import optim_factory

import fitting
from fitting import SMPLifyLoss

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

_SEATED_POSE = {0: SEATED_HIP_X, 3: SEATED_HIP_X, 9: SEATED_KNEE_X, 12: SEATED_KNEE_X}

_TEMPORAL_HOLD_SUPPORT = {
    0: [13, 15], 3: [15], 6: [15], 9: [15],     # left  hip / knee / ankle / foot
    1: [14, 16], 4: [16], 7: [16], 10: [16],    # right hip / knee / ankle / foot
    15: [7, 9], 17: [9], 19: [9],               # left  shoulder / elbow / wrist
    16: [8, 10], 18: [10], 20: [10],            # right shoulder / elbow / wrist
}
_LH_FINGER_KPTS = list(range(18, 38))           # left-hand fingers (no wrist root)
_RH_FINGER_KPTS = list(range(39, 59))           # right-hand fingers (no wrist root)
_TEMPORAL_HOLD_MIN_MISSES = 1
_TEMPORAL_HOLD_BOOST = 8.0
_TEMPORAL_MISS_COUNT = {}


def _apply_seated_legs(body_model):
    """In-place hard-set of the seated leg template onto body_pose (no grad)."""
    with torch.no_grad():
        for dof, val in _SEATED_POSE.items():
            body_model.body_pose.data[0, dof] = val




# fit single frame
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
                    prev_global_orient=None,
                    prev_translation=None,
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

    # Prepare the weights for the different optimization stages
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

    use_vposer = False
    vposer = pose_embedding = None

    # prepare the keypoint data
    kp_data = torch.tensor(input_data, dtype=dtype).to(device=device)
    gt_joints = kp_data[:, :, :3]
    conf = kp_data[:, :, 3]
    conf_thr = float(kwargs.get('joint_conf_threshold', 0.0))
    valid_mask = (conf > 0).float()
    gt_joints = torch.nan_to_num(gt_joints, nan=0.0)
    conf = torch.clamp(conf, 0., 1.0)
    per_frame_w = valid_mask * conf

    # Face landmarks (51 inner dlib 17-67) for head refinement; occluded -> NaN so they are skipped.
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


    # Weights used for the pose prior and the shape prior
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

    # Create fitting loss
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

    # Fitting Process
    _t_frame = time.perf_counter()
    _stage_times = []
    with fitting.FittingMonitor(**kwargs) as monitor:

        _do_lbfgs = True # (frame_idx == 0) or (frame_idx % lbfgs_interval == 0)
        _apply_hand_refinement = True # bool(kwargs.get('apply_hand_refinement', True))  # (frame_idx != 0)
        _apply_head_refinement = False # bool(kwargs.get('apply_head_refinement', True))  # (frame_idx != 0)
        _go_mode = kwargs.get('global_orient_mode', 'free')
        _tr_mode = kwargs.get('translation_mode', 'free')

        # First frame: reset everything to zero.
        init_body_pose     = kwargs.get('init_body_pose',     None)
        init_global_orient = kwargs.get('init_global_orient', None)
        init_transl        = kwargs.get('init_transl',   None)

        # INIT BETAS
        if global_betas is not None:
            with torch.no_grad():
                body_model.betas.data.copy_(global_betas.to(device=device, dtype=dtype))

        # INIT BODY POSE - the same as smpler pose!
        if init_body_pose is not None:
            init_bp = torch.tensor(init_body_pose, dtype=dtype, device=device).reshape(1, 63)
            with torch.no_grad():
                body_model.body_pose.data.copy_(init_bp)

        # INIT GLOBAL ORIENT
        if init_global_orient is not None:
            init_go = torch.tensor(init_global_orient, dtype=dtype, device=device).reshape(1, 3)
            with torch.no_grad():
                body_model.global_orient.data.copy_(init_go)

        # INIT TRANSL (from the SMPLer-X fusion root triangulation)
        if init_transl is not None:
            init_tr = torch.tensor(init_transl, dtype=dtype, device=device).reshape(1, 3)
            with torch.no_grad():
                body_model.transl.data.copy_(init_tr)


        body_model.transl.requires_grad_(_tr_mode != 'frozen')
        body_model.global_orient.requires_grad_(_go_mode != 'frozen')

        if frame_idx < 25:
            body_model.betas.requires_grad_(True)
        else: 
            body_model.betas.requires_grad_(False)

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

        if _do_lbfgs:
            if frame_idx > 0:
                prev_bp = prev_body_pose.to(device=device, dtype=dtype).reshape(1, -1)
                prev_go = prev_global_orient.to(device=device, dtype=dtype).reshape(1, -1)
                prev_tr = prev_translation.to(device=device, dtype=dtype).reshape(1, -1)
            else:
                _TEMPORAL_MISS_COUNT.clear()
                prev_bp = None
                prev_go = None
                prev_tr = None 

            _ign = set(kwargs.get('joints_to_ign', []) or [])
            temporal_dof_w = torch.ones(1, 63, device=device, dtype=dtype)
            for _j, _sup in _TEMPORAL_HOLD_SUPPORT.items():
                _sup_idx = list(_sup)
                if _j == 19: _sup_idx += _LH_FINGER_KPTS
                if _j == 20: _sup_idx += _RH_FINGER_KPTS
                _obs = any((s not in _ign)
                           and bool((valid_mask[0, s] > 0).item())
                           and bool((conf[0, s] >= conf_thr).item())
                           for s in _sup_idx)
                if _obs:
                    _TEMPORAL_MISS_COUNT[_j] = 0
                else:
                    _TEMPORAL_MISS_COUNT[_j] = _TEMPORAL_MISS_COUNT.get(_j, 0) + 1
                    if _TEMPORAL_MISS_COUNT[_j] >= _TEMPORAL_HOLD_MIN_MISSES:
                        temporal_dof_w[0, 3 * _j:3 * _j + 3] = _TEMPORAL_HOLD_BOOST


            

                _held = sorted(_j for _j in _TEMPORAL_HOLD_SUPPORT
                               if temporal_dof_w[0, 3 * _j].item() > 1.0)
                print(f"  [temporal-hold] f{frame_idx} held body_pose joints={_held}")

            for opt_idx, curr_weights in enumerate(tqdm(opt_weights[:4], desc='Stage')):
                final_params = [p for p in body_model.parameters() if p.requires_grad]
                body_optimizer, body_create_graph = optim_factory.create_optimizer(final_params, **kwargs)

                curr_weights['bending_prior_weight'] = (3.17e-1 * curr_weights['body_pose_weight'])
                if use_hands:
                    joint_weights[:, 17:] = curr_weights['hand_weight']
                if use_face:
                    joint_weights[:, 59:] = curr_weights['face_weight']
                joint_weights[:, 11:13] = float(kwargs.get('hip_weight', 1.0))
                stage_weights = joint_weights * per_frame_w
                loss.reset_loss_weights(curr_weights)

                closure = monitor.create_fitting_closure(
                    body_optimizer, body_model,
                    gt_joints=gt_joints,
                    joint_weights=stage_weights,
                    loss=loss, create_graph=body_create_graph,
                    use_vposer=use_vposer, vposer=vposer,
                    pose_embedding=pose_embedding,
                    return_verts=True, return_full_pose=True,
                    prev_body_pose=prev_bp,
                    prev_global_orient=prev_go,
                    prev_translation=prev_tr,
                    smpler_body_pose=init_bp,
                    temporal_dof_weights=temporal_dof_w)

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



    # Head/neck/torso refinement: optimize upper-body DOFs directly from joints + face landmarks.
    _t_head = time.perf_counter()
    if _apply_head_refinement:  # run head refinement regardless of use_vposer
        with torch.no_grad():
            refined_body_pose = body_model.body_pose.detach().clone()  # (1, 63)

        _JOINT_DOF_MAP = {name: range(3 * k, 3 * k + 3) for k, name in enumerate([
            'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee', 'spine2',
            'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot', 'neck',
            'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist'])}

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

            body_model.body_pose.data.copy_(refined_body_pose)

    if _apply_head_refinement:
        _stage_times.append(('head_refine', time.perf_counter() - _t_head))

    # Hand pose refinement
    _t_hand = time.perf_counter()
    refine_left = refine_right = False
    if _apply_hand_refinement:
        # Per-hand visibility gate: only refine a hand with >= _min_kpts visible finger keypoints.
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

        # Also free the refined arm(s) elbow+wrist DOFs (wrist is the finger-chain root).
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

    # Hard-override seated lower body
    # Lower-body GT is unreliable — bypass optimization and hard-set the legs to the
    # seated template (tune _SEATED_POSE visually).
    _apply_seated_legs(body_model)
    if device.type == 'cuda': torch.cuda.synchronize()
    _t_total = time.perf_counter() - _t_frame
    _stage_str = '  '.join(f'{n}={t:.2f}s' for n, t in _stage_times)
    print(f"  [timing/frame {frame_idx}] TOTAL={_t_total:.2f}s  [{_stage_str}]")

    return body_model
    body_pose = body_model.body_pose.detach()

    output = body_model(return_verts=kwargs.get('save_mesh', True), body_pose=body_pose)
    

    return output
