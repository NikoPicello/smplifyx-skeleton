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
import utils
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
# Spine1/2/3 (joints 2/5/8). Also ~static for a seated subject and only weakly observed
# (just the hip→shoulder endpoints), so on a previous-frame anchor the back drifts — bends
# a little more every frame. Hold it to frame 0 too.
_SPINE_POSE_DOFS = [6, 7, 8, 15, 16, 17, 24, 25, 26]
# DOFs pinned to the FRAME-0 fitted pose on every later frame (fixed reference ⇒ no drift).
_STATIC_POSE_DOFS = _LOWER_BODY_POSE_DOFS # + _SPINE_POSE_DOFS
SEATED_HIP_X  = -1.1
SEATED_KNEE_X =  1.3
# Gentle forward spine flexion so the back reads natural instead of ramrod-straight.
# The SMPLer-X anchor pins spine1/2/3 to init_bp (see fitting.py _ANCHOR_JOINT_W) and
# nothing else observes spine curvature, so the init is what we get. Split a small bend
# across the three spine joints. Tune magnitude; flip sign if it curves backward.
SEATED_SPINE_X = 0.07

# Keys are body_pose DOF indices: hip_x (0/3), knee_x (9/12), spine1/2/3_x (6/15/24).
_SEATED_POSE = {0: SEATED_HIP_X, 3: SEATED_HIP_X, 9: SEATED_KNEE_X, 12: SEATED_KNEE_X,
                6: SEATED_SPINE_X, 15: SEATED_SPINE_X, 24: SEATED_SPINE_X}

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

# Neck (11) + collars (12,13) sit between the frozen spine3 and the moving head/shoulders, so
# they absorb that motion and jitter. They must FOLLOW the body (not freeze), so smooth them to
# the PREVIOUS frame with this boost instead of anchoring the neck to per-frame SMPLer-X (that
# anchor is removed in fitting.py _ANCHOR_JOINT_W). Higher = smoother but laggier.
_NECK_COLLAR_JOINTS = [11, 12, 13]
_NECK_COLLAR_TEMPORAL_BOOST = 8.0

_MV2D_RHO_PX      = 50.0    # GMoF scale in PIXELS (tune 30–100); now meaningful since residual is px
_MV2D_DATA_WEIGHT = 1.0
_MV2D_GO_ANCHOR_W = 0.25     # depth is observable from ≥2 opposed views → anchors optional
_MV2D_TR_ANCHOR_W = 0.25     # set small (e.g. 0.01) only as a safety net for sparse-view frames
_MV2D_CONF_FLOOR  = 0.3     # drop 2D below this score
_MV2D_STEPS       = 5
_MV2D_MAX_ITER    = 20
# Root-placement weights over COCO-17. Keep only the stable TRUNK joints strong so arm
# motion can't move the root: shoulders (5,6)=1.0 and hips (11,12)=0.5. Forearm joints
# (elbows 7,8 / wrists 9,10) swing with the arms and the head (0-4) with the neck → both
# down-weighted so they don't drag translation/orientation in this stage.
_MV2D_JOINT_W = torch.ones(17)
_MV2D_JOINT_W[[0, 1, 2, 3, 4]] = 0.1     # head/face — moves with the neck
_MV2D_JOINT_W[[7, 8, 9, 10]] = 0.05      # elbows+wrists — arm motion must NOT move the root
_MV2D_JOINT_W[[13, 14, 15, 16]] = 0.05   # knees+ankles are hard-set template → exclude from placement
_MV2D_JOINT_W[[11, 12]] = 0.5            # hips
# shoulders (5,6) stay at 1.0 — stable trunk anchors, immune to forearm motion

# Translation temporal-anchor weight (SMPLifyLoss.translation_weight). Frame 0 anchors to
# the SMPLer-X/triangulated init; later frames anchor to the previous frame and need a
# stronger pull to resist drift. Overrides the config translation_weight per frame.
_TRANSL_ANCHOR_W       = 50.0    # frame 0
_TRANSL_ANCHOR_W_LATER = 1e3   # frames > 0 (raise to fight drift)

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
    if gt_silhouettes is not None:
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
        _apply_hand_refinement = bool(kwargs.get('apply_hand_refinement', True))  # (frame_idx != 0)
        _apply_head_refinement = bool(kwargs.get('apply_head_refinement', True))  # face lands on landmarks via head/neck after body placement
        _go_mode = kwargs.get('global_orient_mode', 'free')
        _tr_mode = kwargs.get('translation_mode', 'free')

        if _go_mode != 'anchored':
          loss.reset_loss_weights({'global_orient_weight': 0.0})
        if _tr_mode != 'anchored':
          loss.reset_loss_weights({'translation_weight': 0.0})

        def _set_default_grads():
            """Single source of truth for which params optimize this frame, honoring the
            frozen go/transl modes (and betas only on frame 0). Sub-stages below
            temporarily re-mask requires_grad for their own optimization, then call this
            to restore — so 'frozen' holds everywhere and a blanket re-enable can't leak."""
            for p in body_model.parameters():
                p.requires_grad_(True)
            body_model.global_orient.requires_grad_(_go_mode != 'frozen')
            body_model.transl.requires_grad_(_tr_mode != 'frozen')
            body_model.betas.requires_grad_(frame_idx < 1)

        # First frame: reset everything to zero.
        init_body_pose     = kwargs.get('init_body_pose',     None)
        init_global_orient = kwargs.get('init_global_orient', None)
        init_transl        = kwargs.get('init_transl',   None)

        # INIT BETAS
        if global_betas is not None:
            with torch.no_grad():
                body_model.betas.data.copy_(global_betas.to(device=device, dtype=dtype))

        # INIT BODY POSE — SMPLer-X pose, but deepen the (unobserved) leg flexion to a real
        # seated pose. SMPLer-X can't see the occluded legs, so it under-bends them
        # (hip ~-0.45, knee ~0.9 ≈ a half-sit). init_bp is BOTH the warm start AND the
        # smpler anchor target, so overriding the hip/knee flexion here makes the soft anchor
        # pull the legs into a proper sit (during optimization, so pelvis/spine/collision
        # co-adapt — unlike the old end-of-frame _apply_seated_legs snap). Tune _SEATED_POSE.
        init_bp = init_go = init_tr = None
        lower_body_ref = kwargs.get('lower_body_ref', None)
        if lower_body_ref is not None:
            lower_body_ref = torch.as_tensor(lower_body_ref, dtype=dtype, device=device).reshape(-1)
        if init_body_pose is not None:
            init_bp = torch.tensor(init_body_pose, dtype=dtype, device=device).reshape(1, 63)
            if lower_body_ref is not None:
                # Frames > 0: the legs (unobserved) and spine (weakly observed) are ~static for a
                # seated subject. Hold them at the FRAME-0 fitted pose — a fixed reference —
                # instead of the per-frame SMPLer-X init. init_bp is BOTH the warm-start and the
                # smpler-anchor target (passed as smpler_body_pose below), so this points the
                # anchor at frame 0: no per-frame jitter, no drift.
                init_bp[0, _STATIC_POSE_DOFS] = lower_body_ref
            else:
                # Frame 0: deepen the occluded legs into the seated template.
                for _dof, _val in _SEATED_POSE.items():
                    init_bp[0, _dof] = _val
            with torch.no_grad():
                body_model.body_pose.data.copy_(init_bp)
        elif prev_body_pose is not None:
            with torch.no_grad():
                body_model.body_pose.data.copy_(
                    prev_body_pose.to(device=device, dtype=dtype).reshape(1, 63))


        # INIT GLOBAL ORIENT
        if init_global_orient is not None:
            init_go = torch.tensor(init_global_orient, dtype=dtype, device=device).reshape(1, 3)
            # The body sits at ‖go‖≈π, so the triangulation init can land on either side of the
            # wrap. Start on the SAME axis-angle side as the frame-0 anchor so the fit doesn't
            # begin across the π boundary from it.
            _go_ref = kwargs.get('global_orient_ref', None)
            if _go_ref is not None:
                init_go = utils.aa_nearest(
                    init_go, torch.as_tensor(_go_ref, dtype=dtype, device=device).reshape(1, 3))
            with torch.no_grad():
                body_model.global_orient.data.copy_(init_go)

        # INIT TRANSL (from the SMPLer-X fusion root triangulation)
        if init_transl is not None:
            init_tr = torch.tensor(init_transl, dtype=dtype, device=device).reshape(1, 3)
            with torch.no_grad():
                body_model.transl.data.copy_(init_tr)


        _set_default_grads()

        if use_hands:
          init_lh = kwargs.get('init_left_hand_pose',  None)
          init_rh = kwargs.get('init_right_hand_pose', None)
          with torch.no_grad():
            if init_lh is not None:
              lh_t = torch.tensor(init_lh, dtype=dtype, device=device).reshape(1, -1)
              body_model.left_hand_pose.data.copy_(lh_t)
            elif prev_left_hand_pose is not None:
              # No WiLoR for this frame — carry previous pose directly.
              body_model.left_hand_pose.data.copy_(
                prev_left_hand_pose.to(device=device, dtype=dtype))
            if init_rh is not None:
              rh_t = torch.tensor(init_rh, dtype=dtype, device=device).reshape(1, -1)
              body_model.right_hand_pose.data.copy_(rh_t)
            elif prev_right_hand_pose is not None:
              body_model.right_hand_pose.data.copy_(
                prev_right_hand_pose.to(device=device, dtype=dtype))



        if _do_lbfgs:
            if frame_idx > 0:
                prev_bp = prev_body_pose.to(device=device, dtype=dtype).reshape(1, -1)
                prev_go = prev_global_orient.to(device=device, dtype=dtype).reshape(1, -1)
                prev_tr = prev_translation.to(device=device, dtype=dtype).reshape(1, -1)
                # Root anchor target = FRAME 0, not the previous frame. global_orient/transl are
                # ~static for a seated subject; a previous-frame anchor integrates drift. These
                # refs feed BOTH the main-loop anchor (global_orient_loss/translation_loss) and
                # the MV2D stage, so the saved root stays pinned to frame 0.
                _go_ref = kwargs.get('global_orient_ref', None)
                _tr_ref = kwargs.get('translation_ref', None)
                if _go_ref is not None:
                    prev_go = torch.as_tensor(_go_ref, dtype=dtype, device=device).reshape(1, -1)
                if _tr_ref is not None:
                    prev_tr = torch.as_tensor(_tr_ref, dtype=dtype, device=device).reshape(1, -1)
                if lower_body_ref is not None:
                    # Re-target the temporal hold for the static DOFs (legs + spine) to FRAME 0
                    # rather than the previous frame — a previous-frame anchor is a reference-free
                    # random walk that drifts (the back bends a bit more every frame). Pinning all
                    # pulls (warm-start, smpler, temporal) to frame 0 keeps them fixed. Arms and
                    # hands keep their previous-frame temporal smoothing.
                    prev_bp[0, _STATIC_POSE_DOFS] = lower_body_ref
            else:
                _TEMPORAL_MISS_COUNT.clear()
                prev_bp = None
                prev_go = init_go if init_go is not None else body_model.global_orient.detach().clone()
                prev_tr = init_tr if init_tr is not None else body_model.transl.detach().clone()


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

            # Smooth the neck + collars toward the previous frame (see _NECK_COLLAR_* above):
            # the neck no longer chases per-frame SMPLer-X, so this temporal prior keeps the
            # spine3→head/shoulder transition natural and jitter-free while still following.
            for _j in _NECK_COLLAR_JOINTS:
                temporal_dof_w[0, 3 * _j:3 * _j + 3] = _NECK_COLLAR_TEMPORAL_BOOST

            # # Per-frame translation anchor strength (the buffer isn't in opt_weights, so it
            # # otherwise keeps its config init value the whole run). Set once per frame.
            # loss.reset_loss_weights({'translation_weight':
            #     _TRANSL_ANCHOR_W if frame_idx == 0 else _TRANSL_ANCHOR_W_LATER})

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
            print(body_model.transl)

    # --- Multi-view 2D reprojection: refine global_orient + transl from ALL views ---
    _t_mv = time.perf_counter()
    mv_kp2d      = kwargs.get('mv_kp2d', None)
    _mv_cams_raw = kwargs.get('silhouette_cameras', None)
    if kwargs.get('apply_mv2d_stage', True) and mv_kp2d and _mv_cams_raw:
        mv_cams = {c: fitting.build_camera_tensors(_mv_cams_raw[c], device)
                for c in mv_kp2d if c in _mv_cams_raw}
        jw = _MV2D_JOINT_W.to(device=device, dtype=dtype)        # (17,) manual mask

        for p in body_model.parameters():
            p.requires_grad_(False)
        body_model.global_orient.requires_grad_(_go_mode != 'frozen')
        body_model.transl.requires_grad_(_tr_mode != 'frozen')

        # Anchor to the PREVIOUS frame (temporal smoothing), not this frame's value — this
        # stage runs last and sets the saved root, so anchoring to the current per-frame
        # solve lets it re-jitter the root every frame. Frame 0 has no previous → use current.
        if frame_idx > 0:
            go_anchor = prev_go.detach().clone()
            tr_anchor = prev_tr.detach().clone()
        else:
            go_anchor = body_model.global_orient.detach().clone()
            tr_anchor = body_model.transl.detach().clone()

        _mv_params = [p for p in (body_model.global_orient, body_model.transl) if p.requires_grad]
        if _mv_params:
            mv_optim = torch.optim.LBFGS(_mv_params, lr=kwargs.get('lr', 1.0),
                                        max_iter=_MV2D_MAX_ITER, line_search_fn='strong_wolfe')

            def _mv_closure():
                mv_optim.zero_grad()
                # (norm-clamp removed: it hard-projected global_orient at ‖go‖=π — exactly the
                # body's resting orientation — a discontinuity that destabilised the solve.
                # aa_nearest on the anchor + the saved-output unwrap make it unnecessary.)
                out = body_model(return_verts=False)
                Jb  = out.joints[0, :17, :]                       # (17,3) COCO body, world
                dloss = Jb.new_zeros(())
                for cam_name, (kp2d, conf) in mv_kp2d.items():
                    cam  = mv_cams[cam_name]
                    proj, valid = fitting._project_to_pixels(Jb, cam)    # (17,2) px, distortion-correct
                    f    = cam['K'][0, 0].to(dtype)
                    r    = kp2d.to(dtype) - proj                  # (17,2) px
                    rob  = (_MV2D_RHO_PX ** 2) * r.pow(2) / (r.pow(2) + _MV2D_RHO_PX ** 2)
                    contrib = rob / (f ** 2)                      # focal-normalize → cross-cam comparable
                    w = (conf.to(dtype) * (conf >= _MV2D_CONF_FLOOR).to(dtype) * jw * valid.to(dtype)).unsqueeze(-1)
                    dloss = dloss + (w.pow(2) * contrib).sum()
                dloss = dloss * _MV2D_DATA_WEIGHT ** 2
                # Resolve the axis-angle pi-wrap before the orient anchor so a representation
                # flip near theta=pi doesn't spike this loss and kick global_orient/transl.
                with torch.no_grad():
                    _go_a = utils.aa_nearest(go_anchor, body_model.global_orient)
                aloss = ((body_model.global_orient - _go_a).pow(2).sum() * _MV2D_GO_ANCHOR_W ** 2
                        + (body_model.transl - tr_anchor).pow(2).sum() * _MV2D_TR_ANCHOR_W ** 2)
                total = dloss + aloss
                total.backward()
                return total

            for step_i in range(_MV2D_STEPS):
                go_b = body_model.global_orient.data.clone()
                tr_b = body_model.transl.data.clone()
                mv_optim.step(_mv_closure)
                print(f"  [mv2d] step={step_i} n_cams={len(mv_kp2d)} "
                    f"Δgo={(body_model.global_orient.data - go_b).norm().item():.6f} "
                    f"Δtr={(body_model.transl.data - tr_b).norm().item():.6f}")

        _set_default_grads()
        _stage_times.append(('mv2d', time.perf_counter() - _t_mv))


    _t_sil = time.perf_counter()
    if (loss.use_silhouette and gt_silhouettes is not None
            and kwargs.get('apply_silhouette_stage', True)):
        _sil_vis = bool(kwargs.get('sil_visualize', True))
        _sil_cam_names = sorted(kwargs.get('silhouette_cameras', {}).keys())
        if _sil_vis:
            with torch.no_grad():
                _vsil = body_model(return_verts=True).vertices
            loss.visualize_stage(_vsil, gt_silhouettes, 98, frame_idx,
                                 cam_names=_sil_cam_names)

        for p in body_model.parameters():
            p.requires_grad_(False)
        body_model.global_orient.requires_grad_(_go_mode != 'frozen')
        body_model.transl.requires_grad_(_tr_mode != 'frozen')


        go_anchor = body_model.global_orient.detach().clone()
        tr_anchor = body_model.transl.detach().clone()
        _sil_w    = torch.tensor(float(kwargs.get('sil_stage_weight',     1.0)), dtype=dtype, device=device)
        _sil_go_w = torch.tensor(float(kwargs.get('sil_go_anchor_weight', 0.1)), dtype=dtype, device=device)
        _sil_tr_w = torch.tensor(float(kwargs.get('sil_tr_anchor_weight', 0.1)), dtype=dtype, device=device)

        _sil_params = [p for p in (body_model.global_orient, body_model.transl, body_model.betas) if p.requires_grad]
        if _sil_params:
            sil_optim = torch.optim.LBFGS(_sil_params, lr=1.5, # kwargs.get('lr', 1.0),
                                          max_iter=20, line_search_fn='strong_wolfe')

            def _sil_closure():
                sil_optim.zero_grad()
                # (norm-clamp removed — see _mv_closure; it poked global_orient right at π.)
                out   = body_model(return_verts=True)
                sloss = loss.silhouette_term(out.vertices, gt_silhouettes) * _sil_w ** 2
                aloss = ((body_model.global_orient - go_anchor).pow(2).sum() * _sil_go_w ** 2
                         + (body_model.transl - tr_anchor).pow(2).sum() * _sil_tr_w ** 2)
                total = sloss + aloss
                total.backward()
                return total

            for step_i in range(3):
                go_b = body_model.global_orient.data.clone()
                tr_b = body_model.transl.data.clone()
                sil_optim.step(_sil_closure)
                print(f"  [sil] step={step_i}  "
                      f"Δgo={(body_model.global_orient.data - go_b).norm().item():.6f}  "
                      f"Δtr={(body_model.transl.data - tr_b).norm().item():.6f}")

        if _sil_vis:
            with torch.no_grad():
                _vsil = body_model(return_verts=True).vertices
            loss.visualize_stage(_vsil, gt_silhouettes, 99, frame_idx,
                                 cam_names=_sil_cam_names)

        _set_default_grads()
        _stage_times.append(('sil_align', time.perf_counter() - _t_sil))






    # Head/neck refinement: re-aim head + neck (+ jaw) so the face lands on the triangulated
    # face landmarks. Runs AFTER body placement (main fit + mv2d); placement stays fixed.
    _t_head = time.perf_counter()
    if _apply_head_refinement:  # run head refinement regardless of use_vposer
        with torch.no_grad():
            refined_body_pose = body_model.body_pose.detach().clone()  # (1, 63)

        _JOINT_DOF_MAP = {name: range(3 * k, 3 * k + 3) for k, name in enumerate([
            'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee', 'spine2',
            'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot', 'neck',
            'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist'])}

        # Free neck too: the face-landmark term then distributes the look across neck+head
        # (the quadratic pose reg below favors splitting the rotation), so the head no longer
        # cranes alone on a straight neck. Override via cfg head_refine_joints.
        _default_refine = ['neck', 'head']
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

        upper_pose_free   = refined_body_pose[0, _free_idxs].clone().detach().requires_grad_(True)
        lower_pose_frozen = refined_body_pose[0, _frozen_idxs].detach()

        for p in body_model.parameters():
            p.requires_grad_(False)
        body_model.jaw_pose.requires_grad_(True)

        _d_face_w = torch.tensor(float(kwargs.get('head_face_weight', 20.0)), dtype=dtype, device=device)
        _d_jaw_w  = torch.tensor(float(kwargs.get('head_jaw_weight',  1.0)),  dtype=dtype, device=device)
        _d_pose_w = torch.tensor(float(kwargs.get('head_pose_weight', 0.1)),  dtype=dtype, device=device)
        _d_face_rho = float(kwargs.get('head_face_rho', 0.05))

        head_optim = torch.optim.LBFGS(
            [upper_pose_free, body_model.jaw_pose],
            lr=kwargs.get('lr', 0.8), max_iter=20,
            line_search_fn='strong_wolfe')

        def _head_closure():
            head_optim.zero_grad()
            with torch.no_grad():
                upper_pose_free.data.clamp_(-torch.pi, torch.pi)
            bp = torch.zeros(1, 63, dtype=dtype, device=device)
            bp[0, _free_idxs]   = upper_pose_free
            bp[0, _frozen_idxs] = lower_pose_frozen
            out = body_model(return_verts=True, body_pose=bp,
                             return_full_pose=True)

            ploss = upper_pose_free.pow(2).sum() * _d_pose_w ** 2
            floss = torch.tensor(0.0, device=device, dtype=dtype)
            if loss.use_face_landmarks and gt_face_landmarks is not None:
                verts_d = out.vertices[0]
                tri_v   = verts_d[loss.body_faces_lmk[loss.lmk_faces_idx]]
                lmk_pos = (tri_v * loss.lmk_bary_coords.unsqueeze(-1)).sum(dim=1)
                valid_f = ~torch.isnan(gt_face_landmarks).any(dim=-1)
                gt_lmks = torch.nan_to_num(gt_face_landmarks, nan=0.0)
                d2    = (gt_lmks - lmk_pos).pow(2).sum(dim=-1)              # (51,)
                rob_f = (_d_face_rho ** 2) * d2 / (d2 + _d_face_rho ** 2)
                floss = (rob_f * valid_f.float()).sum() * _d_face_w ** 2

            jploss = torch.sum(loss.jaw_prior(out.jaw_pose.mul(_d_jaw_w)))

            total = ploss + floss + jploss
            total.backward()
            return total

        for step_i in range(5):
            pose_before = upper_pose_free.data.clone()
            jaw_before  = body_model.jaw_pose.data.clone()
            head_optim.step(_head_closure)
            pose_delta = (upper_pose_free.data - pose_before).norm().item()
            jaw_delta  = (body_model.jaw_pose.data - jaw_before).norm().item()
            print(f"  [head] step={step_i}  Δpose={pose_delta:.6f}  Δjaw={jaw_delta:.6f}")

        with torch.no_grad():
            refined_body_pose = torch.zeros(1, 63, dtype=dtype, device=device)
            refined_body_pose[0, _free_idxs]   = upper_pose_free.detach()
            refined_body_pose[0, _frozen_idxs] = lower_pose_frozen

            body_model.body_pose.data.copy_(refined_body_pose)

    if _apply_head_refinement:
        _stage_times.append(('head_refine', time.perf_counter() - _t_head))


    # Hand pose refinement
    _t_hand = time.perf_counter()
    refine_left  = _apply_hand_refinement and (
        bool((valid_mask[0, 17:38] > 0).any().item())
        or kwargs.get('init_left_hand_pose') is not None)
    refine_right = _apply_hand_refinement and (
        bool((valid_mask[0, 38:59] > 0).any().item())
        or kwargs.get('init_right_hand_pose') is not None)

    if _apply_hand_refinement and (refine_left or refine_right):
        # ── Two-phase hand refinement ────────────────────────────────────────
        # Split by kinematics so the two jobs don't fight:
        #   Phase 1 PLACE     : free shoulder+elbow, data = ARM keypoints (elbow/wrist) →
        #                       snap the wrist JOINT. (A joint's own rotation can't move it,
        #                       so the wrist DOF is useless here; the shoulder is essential.)
        #   Phase 2 ARTICULATE: lock shoulder+elbow, free WRIST rotation + hand pose,
        #                       data = fingers → re-aim the hand and curl the fingers.
        # Both anneal rho coarse→fine for a clean snap.
        _h_data_w  = torch.tensor(float(kwargs.get('hand_data_weight',  30.0)), dtype=dtype, device=device)
        _h_prior_w = torch.tensor(float(kwargs.get('hand_refine_prior_weight', 1.5)), dtype=dtype, device=device)
        _h_wilor_w = torch.tensor(float(kwargs.get('hand_wilor_weight', 0.5)), dtype=dtype, device=device)
        _h_temp_w  = torch.tensor(float(kwargs.get('hand_temporal_weight', 0.2)) if frame_idx > 0 else 0.0, dtype=dtype, device=device)
        _h_arm_w   = torch.tensor(float(kwargs.get('hand_wrist_anchor_weight', 3.0)), dtype=dtype, device=device)
        _place_rho  = float(kwargs.get('hand_place_rho',  0.15))   # loose: gross hand placement
        _finger_rho = float(kwargs.get('hand_finger_rho', 0.05))  # tight: discriminate finger detail

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

        # Per-hand keypoint masks (51-joint skeleton): body L shoulder=5 elbow=7 wrist=9,
        # R shoulder=6 elbow=8 wrist=10; hands L root=17 fingers 18:38, R root=38 fingers 39:59.
        # Phase 1 fits ONLY the arm keypoints (elbow + wrist + hand-root). Feeding it the
        # fingers would make the free arm contort to best-fit the rigid uncurled hand and
        # misplace the wrist; the fingers are Phase 2's job, with the arm locked.
        _arm_kp_mask = torch.zeros_like(joint_weights)
        if refine_left:  _arm_kp_mask[:, [7, 9, 17]]  = 1.0
        if refine_right: _arm_kp_mask[:, [8, 10, 38]] = 1.0
        _arm_kp_w = (joint_weights * valid_mask * _arm_kp_mask)   # (1, J)

        _finger_mask = torch.zeros_like(joint_weights)
        if refine_left:  _finger_mask[:, 18:38] = 1.0
        if refine_right: _finger_mask[:, 39:59] = 1.0
        _finger_w = (joint_weights * valid_mask * _finger_mask)   # (1, J)

        # ---- Phase 1: PLACE the arm — free shoulder+elbow (the DOFs that move the wrist
        # JOINT; its own rotation does not). Data = arm keypoints. Anneal rho. ----
        _arm_cols = []
        if refine_left:  _arm_cols += [45, 46, 47, 51, 52, 53]   # L shoulder + elbow
        if refine_right: _arm_cols += [48, 49, 50, 54, 55, 56]   # R shoulder + elbow
        _arm_idx          = torch.tensor(_arm_cols, device=device)
        _body_pose_frozen = body_model.body_pose.data.clone()    # (1, 63), non-arm DOFs fixed
        _arm_anchor       = _body_pose_frozen[0, _arm_idx].clone()
        _arm_free         = _body_pose_frozen[0, _arm_idx].clone().detach().requires_grad_(True)

        for p in body_model.parameters():
            p.requires_grad_(False)

        place_optim = torch.optim.LBFGS([_arm_free], lr=kwargs.get('lr', 1.2),
                                        max_iter=20, line_search_fn='strong_wolfe')
        _place_rho_now = _place_rho * 3.0     # annealed coarse→fine in the loop below

        def _place_closure():
            place_optim.zero_grad()
            bp = _body_pose_frozen.clone()
            bp[0, _arm_idx] = _arm_free
            out = body_model(return_verts=False, body_pose=bp)
            d2  = (gt_joints - out.joints).pow(2).sum(dim=-1)                # (1, J) per-joint dist²
            rob = (_place_rho_now ** 2) * d2 / (d2 + _place_rho_now ** 2)
            dloss = (_arm_kp_w ** 2 * rob).sum() * _h_data_w ** 2
            aloss = (_arm_free - _arm_anchor).pow(2).sum() * _h_arm_w ** 2   # gentle stabilizer
            total = dloss + aloss
            total.backward()
            return total

        for step_i in range(5):
            _place_rho_now = _place_rho * (3.0 ** (1.0 - step_i / 4.0))      # 3x → 1x
            arm_before = _arm_free.data.clone()
            place_optim.step(_place_closure)
            print(f"  [hand_place] step={step_i}  rho={_place_rho_now:.3f}  "
                  f"Δarm={(_arm_free.data - arm_before).norm().item():.6f}")

        with torch.no_grad():
            body_model.body_pose.data[0, _arm_idx] = _arm_free.detach()

        # ---- Phase 2: ARTICULATE — arm (shoulder+elbow) LOCKED. Free the WRIST rotation
        # (so the hand can re-aim — fingers can't reach a mis-oriented hand) + hand pose.
        # Data = fingers; anneal rho coarse→fine. ----
        _wrist_cols = []
        if refine_left:  _wrist_cols += [57, 58, 59]   # L wrist
        if refine_right: _wrist_cols += [60, 61, 62]   # R wrist
        _wrist_idx     = torch.tensor(_wrist_cols, device=device)
        _body_pose_art = body_model.body_pose.data.clone()
        _wrist_anchor  = _body_pose_art[0, _wrist_idx].clone()
        _wrist_free    = _body_pose_art[0, _wrist_idx].clone().detach().requires_grad_(True)

        for p in body_model.parameters():
            p.requires_grad_(False)
        body_model.left_hand_pose.requires_grad_(refine_left)
        body_model.right_hand_pose.requires_grad_(refine_right)

        _art_params = [_wrist_free]
        if refine_left:  _art_params.append(body_model.left_hand_pose)
        if refine_right: _art_params.append(body_model.right_hand_pose)
        art_optim = torch.optim.LBFGS(_art_params, lr=kwargs.get('lr', 0.8),
                                      max_iter=20, line_search_fn='strong_wolfe')
        _finger_rho_now = _finger_rho * 3.0   # annealed coarse→fine in the loop below

        def _art_closure():
            art_optim.zero_grad()
            bp = _body_pose_art.clone()
            bp[0, _wrist_idx] = _wrist_free
            out = body_model(return_verts=False, body_pose=bp)
            d2  = (gt_joints - out.joints).pow(2).sum(dim=-1)                # (1, J)
            rob = (_finger_rho_now ** 2) * d2 / (d2 + _finger_rho_now ** 2)
            hloss = (_finger_w ** 2 * rob).sum() * _h_data_w ** 2

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
                tloss_h = tloss_h + (body_model.left_hand_pose - lh_anchor).pow(2).sum() * _h_temp_w ** 2
            if refine_right and rh_anchor is not None:
                tloss_h = tloss_h + (body_model.right_hand_pose - rh_anchor).pow(2).sum() * _h_temp_w ** 2

            # gentle: keep the wrist from spinning to chase noisy finger keypoints
            wloss = (_wrist_free - _wrist_anchor).pow(2).sum() * _h_arm_w ** 2

            total = hloss + hprior_loss + wilor_loss + tloss_h + wloss
            total.backward()
            return total

        for step_i in range(5):
            _finger_rho_now = _finger_rho * (3.0 ** (1.0 - step_i / 4.0))    # 3x → 1x
            lh_before = body_model.left_hand_pose.data.clone()
            rh_before = body_model.right_hand_pose.data.clone()
            wr_before = _wrist_free.data.clone()
            art_optim.step(_art_closure)
            lh_delta = (body_model.left_hand_pose.data - lh_before).norm().item()
            rh_delta = (body_model.right_hand_pose.data - rh_before).norm().item()
            wr_delta = (_wrist_free.data - wr_before).norm().item()
            print(f"  [hand_art] step={step_i}  rho={_finger_rho_now:.3f}  "
                  f"Δwrist={wr_delta:.6f}  Δlh={lh_delta:.6f}  Δrh={rh_delta:.6f}")

        with torch.no_grad():
            body_model.body_pose.data[0, _wrist_idx] = _wrist_free.detach()

        # ---- Phase 3: SNAP — re-place the now-curled hand onto the FINGER cloud. Phase 1
        # placed the wrist from arm keypoints with a flat hand; Phase 2 curled the fingers but
        # could not translate the hand (shoulder+elbow locked). So the fingers never moved the
        # hand into position. With the curl now FROZEN, free shoulder+elbow+wrist and fit the
        # arm + finger keypoints together: the rigid curled hand slides onto the keypoints
        # (ICP-style) instead of contorting the arm. Fingers (~20 kp) outvote the arm kp (~3),
        # so it snaps to the fingers; the arm kp + gentle pose anchor keep it body-consistent. ----
        if kwargs.get('hand_snap_stage', False):
            _snap_idx    = torch.cat([_arm_idx, _wrist_idx])           # shoulder+elbow+wrist
            _bp_snap     = body_model.body_pose.data.clone()
            _snap_anchor = _bp_snap[0, _snap_idx].clone()
            _snap_free   = _bp_snap[0, _snap_idx].clone().detach().requires_grad_(True)
            _snap_w      = _arm_kp_w + _finger_w                       # arm + fingers, conf/valid-masked

            for p in body_model.parameters():
                p.requires_grad_(False)                                # hand pose stays frozen (keep the curl)
            snap_optim = torch.optim.LBFGS([_snap_free], lr=kwargs.get('lr', 1.0),
                                           max_iter=20, line_search_fn='strong_wolfe')
            _snap_rho_now = _finger_rho * 3.0

            def _snap_closure():
                snap_optim.zero_grad()
                bp = _bp_snap.clone()
                bp[0, _snap_idx] = _snap_free
                out = body_model(return_verts=False, body_pose=bp)     # hand_pose frozen → rigid curled hand
                d2  = (gt_joints - out.joints).pow(2).sum(dim=-1)
                rob = (_snap_rho_now ** 2) * d2 / (d2 + _snap_rho_now ** 2)
                dloss = (_snap_w ** 2 * rob).sum() * _h_data_w ** 2
                aloss = (_snap_free - _snap_anchor).pow(2).sum() * _h_arm_w ** 2   # gentle stabilizer
                total = dloss + aloss
                total.backward()
                return total

            for step_i in range(5):
                _snap_rho_now = _finger_rho * (3.0 ** (1.0 - step_i / 4.0))        # 3x → 1x
                snap_before = _snap_free.data.clone()
                snap_optim.step(_snap_closure)
                print(f"  [hand_snap] step={step_i}  rho={_snap_rho_now:.3f}  "
                      f"Δsnap={(_snap_free.data - snap_before).norm().item():.6f}")

            with torch.no_grad():
                body_model.body_pose.data[0, _snap_idx] = _snap_free.detach()

        _set_default_grads()



    # Hard-override seated lower body
    # Lower-body GT is unreliable — bypass optimization and hard-set the legs to the
    # seated template (tune _SEATED_POSE visually).
    # _apply_seated_legs(body_model)





    if device.type == 'cuda': torch.cuda.synchronize()
    _t_total = time.perf_counter() - _t_frame
    _stage_str = '  '.join(f'{n}={t:.2f}s' for n, t in _stage_times)
    print(f"  [timing/frame {frame_idx}] TOTAL={_t_total:.2f}s  [{_stage_str}]")

    return body_model
