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
import time
import smplx
import traceback
import pickle


from cmd_parser import parse_config
from utils import JointMapper, aa_nearest
from prior import create_prior
from cvars import *

torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
import random as _random
np.random.seed(42)
_random.seed(42)
torch.use_deterministic_algorithms(True, warn_only=True)

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

    from temporal_window import WIN_SIZE, run_windowed
    window_body_params = {**model_params, 'batch_size': WIN_SIZE}
    window_body_model  = smplx.create(**window_body_params).to(device=device)


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

    mamma_folder = args.get('mamma_folder', None)
    mamma_data = None
    if mamma_folder is not None:
        from mamma_loader import load_mamma
        _nose_traj  = dataset_obj.body_data[:, 0, :3]
        _nose_valid = dataset_obj.body_data[:, 0, 3] > 0
        mamma_data = load_mamma(mamma_folder, _nose_traj, _nose_valid, device, dtype)
        if mamma_data is not None:
            print(f"[mamma] loaded body_id-{mamma_data['body_id']} "
                  f"({mamma_data['global_orient'].shape[0]} frames)")


    # Per-frame SMPLer-X init: body_pose / global_orient / transl / betas (world frame).
    init_bps, init_gos, init_trs, init_betas = dataset_obj.get_init_body(init_poses=True)
    init_left_hand_poses  = dataset_obj.get_init_hand_poses('left')
    init_right_hand_poses = dataset_obj.get_init_hand_poses('right')

    # mamma_loader now sources betas from the SAME file as the pose/root it returns (verified
    # ~2.5cm median vs our own triangulation) — safe to trust directly again.
    _betas_src = 'SMPLer-X'
    # if mamma_data is not None:
    #     init_betas = mamma_data['betas'].detach().cpu().numpy()
    #     _betas_src = f"mamma body_id-{mamma_data['body_id']}"

    global_betas = None
    if init_betas is not None:
        init_arr = np.asarray(init_betas, dtype=np.float64 if float_dtype == 'float64' else np.float32).flatten()
        nb = body_model.num_betas
        if len(init_arr) > nb:
            init_arr = init_arr[:nb]
        elif len(init_arr) < nb:
            init_arr = np.pad(init_arr, (0, nb - len(init_arr)))
        global_betas = torch.as_tensor(init_arr, dtype=dtype, device=device).reshape(1, -1)
        preview = global_betas.detach().cpu().numpy().flatten()[:5].round(3).tolist()
        print(f"Using injected betas from {_betas_src} (shape={list(global_betas.shape)}, original={len(np.asarray(init_betas).flatten())}): {preview} ...")
    if not os.path.exists(os.path.join(args['output_folder'], sequence_name, 'meshes')):
        os.makedirs(os.path.join(args['output_folder'], sequence_name, 'meshes'))
    smplx_stored_path = os.path.join(args['output_folder'], sequence_name, 'body_smplx.json')
    failed_frames = []
    _timing_frames = []   # list of {frame, fit_s, mesh_s}

    # ===================================================================
    # Stage A: one batched, windowed body+root fit for the WHOLE sequence
    # (replaces the per-frame fitting loop). Hands/face stay at their init
    # warm-start here; Stage B (per-frame head/hand refinement) is TODO.
    # ===================================================================
    max_frames = args.get('max_frames', -1)   # --max-frames from fitter_pipeline.py; -1 = all available
    N = len(dataset_obj) if max_frames < 0 else min(len(dataset_obj), max_frames)
    if init_bps is not None and len(init_bps) < N:
        print(f"[init] SMPLer-X init covers only {len(init_bps)}/{N} frames → fitting that range "
              f"(re-run smpler_fusion over the full video for longer fits)")
        N = len(init_bps)

    def _seq(arr, d, n=None):
        return torch.stack([torch.as_tensor(np.asarray(arr[i], dtype=np.float32),
                            dtype=dtype, device=device).reshape(d) for i in range(N if n is None else n)])

    def _gap_fill_keypoints(a, conf_thr, max_gap, fill_conf):
        """a: (N,J,4). Linearly interpolate the 3D position across missing runs (conf<=conf_thr)
        of length <= max_gap that are BRACKETED by observed frames, stamping filled points with
        `fill_conf`. Removes the observed<->unobserved toggle that yanks flickering limbs; a fully
        unseen run (no bracket, or too long) is left untouched for the prior/smoothing to hold.
        Returns (filled_copy, n_filled)."""
        a = a.copy()
        Nn, Jn = a.shape[0], a.shape[1]
        n_filled = 0
        for j in range(Jn):
            obs = (a[:, j, 3] > conf_thr) & np.isfinite(a[:, j, :3]).all(1)
            t = 0
            while t < Nn:
                if obs[t]:
                    t += 1; continue
                lo = t
                while t < Nn and not obs[t]:
                    t += 1
                hi, run = t, t - lo                        # hi = first observed after the run (or Nn)
                if lo > 0 and hi < Nn and 0 < run <= max_gap:
                    p0, p1 = a[lo - 1, j, :3], a[hi, j, :3]
                    for k in range(run):
                        w = (k + 1) / (run + 1)
                        a[lo + k, j, :3] = (1 - w) * p0 + w * p1
                        a[lo + k, j, 3]  = fill_conf
                    n_filled += run
        return a, n_filled

    # 3D keypoints (numpy first so we can temporally gap-fill flickering detections) + body-only
    # data weights (hands/face -> Stage B). conf is sharpened (conf**KP_CONF_POWER) so a low-conf
    # flickering keypoint yanks its limb less and the temporal smoothing carries it instead.
    kp_np = np.stack([np.asarray(dataset_obj[i], dtype=np.float32) for i in range(N)]).squeeze(1)  # (N,J,4)
    _conf_thr = float(args.get('joint_conf_threshold', 0.0))
    kp_np, _n_fill = _gap_fill_keypoints(kp_np, _conf_thr, KP_FILL_MAX_GAP, KP_FILL_CONF)
    if _n_fill:
        print(f"[kp gapfill] interpolated {_n_fill} keypoint-frame(s) across gaps <= {KP_FILL_MAX_GAP}")
    kp = torch.as_tensor(kp_np, dtype=dtype, device=device)                        # (N, J, 4)
    gt_joints_all = torch.nan_to_num(kp[..., :3], nan=0.0)                          # (N, J, 3)
    conf  = kp[..., 3].clamp(0.0, 1.0)
    valid = (kp[..., 3] > 0).float()
    jw = joint_weights.clone()                                                     # (1, J)
    jw[:, 17:] = 0.0                                                               # hands + face -> Stage B
    for _j in (args.get('joints_to_ign', []) or []):
        jw[:, _j] = 0.0                                                            # ignored (knees/ankles)
    jw[:, 11:13] = float(args.get('hip_weight', 1.0))
    weights_all = jw * valid * conf ** KP_CONF_POWER                               # (N, J); conf-sharpened

    # Warm start from SMPLer-X + seated-leg override; anchor targets for the root term.
    # Legs: SEATED_LEGS is the fallback template; when the per-camera SMPLer-X export exists,
    # the GB-view median seated legs override it (FREEZE_LEGS then holds them through Stage A).
    # NOTE these are GB-pelvis-relative until align_hips_to_root runs after the static root.
    # Spine: deliberately NOT templated — the SMPLer-X per-frame spine (~50° lumbar slouch here)
    # is both the static-root template and Stage A's warm start, so root pitch and posture stay
    # consistent (stamping +4° mis-pitched the frozen root → S-kink / neck-dump compensation).
    from temporal_window import load_static_leg_pose, align_hips_to_root, LEG_POSE_CAM, _LEG_COLS
    leg_pose, gb_go = load_static_leg_pose(args.get('smpler_folder'), args.get('person_id', 0),
                                           device, dtype)                    # (18,)/(3,) or None
    # use mamma's own body_pose as the init, not just SMPLer-X's — it must match mamma's
    # (fixed) root, which SMPLer-X's body_pose was never calibrated against. .clone() so the
    # leg overrides below don't mutate mamma_data['body_pose'] itself (aliased slice).
    bp_init = mamma_data['body_pose'][:N].clone() if mamma_data is not None else _seq(init_bps, (63,))
    # Legs always come from the SMPLer-X GB seated template, mamma included — mamma's own legs
    # are occluded/unreliable even in clips where its arms/head are good.
    for _dof, _val in SEATED_LEGS.items():
        bp_init[:, _dof] = _val
    if leg_pose is not None:
        bp_init[:, _LEG_COLS] = leg_pose
    go_init = _seq(init_gos, (3,)) if init_gos is not None else torch.zeros(N, 3, dtype=dtype, device=device)
    tr_init = _seq(init_trs, (3,)) if init_trs is not None else torch.zeros(N, 3, dtype=dtype, device=device)
    go_ref_all, tr_ref_all = go_init.clone(), tr_init.clone()
    betas = global_betas if global_betas is not None else body_model.betas.detach()[:1].clone()

    _kp_full = torch.as_tensor(dataset_obj.body_data, dtype=dtype, device=device)   # (N_full,17,4)
    _cf_full = (_kp_full[..., 3] > 0).to(dtype) * _kp_full[..., 3].clamp(0.0, 1.0)

    # Cross-check/correct against observed bone lengths even when betas came from mamma — a
    # wrong/borderline segment assignment can silently poison mamma's stored betas the same way
    # it did pose; the anchor is loose so an already-good mamma shape barely moves.
    from temporal_window import refine_betas_bone_lengths
    betas = refine_betas_bone_lengths(body_model, betas, _kp_full[..., :3].contiguous(), _cf_full)

    from temporal_window import (FREEZE_ROOT, solve_static_root, build_root_2d_inputs)
    _N_2d = len(dataset_obj) if FREEZE_ROOT else N
    _cams, _gt2d, _conf2d = build_root_2d_inputs(
        args.get('silhouette_cameras'), mv_rtmo, args.get('person_id', 0), _N_2d, device, dtype)
    if FREEZE_ROOT:
        _t_rt = time.time()
        # The root solve can only use frames that HAVE an init body: a benchmark-sized
        # smpler_fusion export (e.g. 50 frames) truncates the solve to that range — the other
        # arrays (keypoints/2D) stay full-video sized, so slice them consistently.
        _N_rt = min(len(dataset_obj), len(init_bps))
        if mamma_data is not None:
            _N_rt = min(_N_rt, mamma_data['body_pose'].shape[0])
        if _N_rt < len(dataset_obj):
            print(f"[static root] init covers {_N_rt}/{len(dataset_obj)} frames → solving on that range "
                  f"(re-run smpler_fusion over the full video to restore whole-video hip evidence)")
        # Trunk/spine template + root warm start: mamma's own (matches bp_init above, so root
        # pitch stays consistent with Stage A's warm start) when available, else SMPLer-X's.
        # Legs always come from the SMPLer-X GB seated template — mamma's own legs are unreliable.
        if mamma_data is not None:
            print(f"[static root] trunk template + warm start from mamma body_id-{mamma_data['body_id']}; "
                  f"legs from the SMPLer-X GB seated template")
            _bp_full = mamma_data['body_pose'][:_N_rt].clone()
            _go_full = mamma_data['global_orient'][:_N_rt].clone()
            _tr_full = mamma_data['transl'][:_N_rt].clone()
        else:
            _bp_full = _seq(init_bps, (63,), _N_rt)
            _go_full = _seq(init_gos, (3,), _N_rt) if init_gos is not None else torch.zeros(_N_rt, 3, dtype=dtype, device=device)
            _tr_full = _seq(init_trs, (3,), _N_rt) if init_trs is not None else torch.zeros(_N_rt, 3, dtype=dtype, device=device)
        for _dof, _val in SEATED_LEGS.items():   # legs only — spine stays as set above
            _bp_full[:, _dof] = _val
        if leg_pose is not None:   # same legs Stage A will hold → unbiased 2D knee/ankle votes
            _bp_full[:, _LEG_COLS] = leg_pose
        _kp_rt   = _kp_full[:_N_rt, :, :3].contiguous()
        _w_full  = _cf_full[:_N_rt] ** KP_CONF_POWER
        _go_s, _tr_s = solve_static_root(
            window_body_model, betas, _bp_full, _go_full, _tr_full,
            _kp_rt, _w_full,
            cams=_cams, gt2d_all=_gt2d, conf2d_all=_conf2d)
        # GB's hip angles are relative to ITS pelvis; the solved root resolves the seated
        # pelvis-pitch vs hip-flexion ambiguity differently (~40-47° here). Transport the hips
        # under the solved root, re-solve the root ONCE with the corrected legs (they vote in
        # its 2D knee/ankle term), then align to the final root (the residual is second-order).
        _sc = (args.get('silhouette_cameras') or {}).get(LEG_POSE_CAM)
        if leg_pose is not None and _sc is not None:
            _bp_full[:, _LEG_COLS] = align_hips_to_root(leg_pose, gb_go, _sc['R'], _go_s)
            _go_s, _tr_s = solve_static_root(
                window_body_model, betas, _bp_full, _go_full, _tr_full,
                _kp_rt, _w_full,
                cams=_cams, gt2d_all=_gt2d, conf2d_all=_conf2d)
            # keep leg_pose RAW (GB-pelvis-relative) — the root refit re-aligns from it again
            _leg_aligned = align_hips_to_root(leg_pose, gb_go, _sc['R'], _go_s)
            bp_init[:, _LEG_COLS] = _leg_aligned
            _bp_full[:, _LEG_COLS] = _leg_aligned
        go_init = _go_s.expand(N, -1).contiguous()
        tr_init = _tr_s.expand(N, -1).contiguous()
        go_ref_all, tr_ref_all = go_init.clone(), tr_init.clone()
        print(f"[timing] static root solve: {time.time() - _t_rt:.2f}s")

    # Stillness-anchor reference (temporal_window.refine_window_body's L_still): mamma's own
    # occlusion-gated body_pose when available, else None (falls back to each window's own mean).
    bp_ref_all = mamma_data['body_pose'][:N] if mamma_data is not None else None

    _t_stageA = time.time()
    bp_all, go_all, tr_all = run_windowed(
        window_body_model, body_pose_prior, angle_prior,
        gt_joints_all, weights_all, betas,
        bp_init, go_init, tr_init, go_ref_all, tr_ref_all,
        bp_ref_all=bp_ref_all)
    _dt_stageA = time.time() - _t_stageA
    print(f"[timing] Stage A windowed fit ({N} frames): {_dt_stageA:.2f}s  ({_dt_stageA/max(N,1):.2f}s/frame)")

    # ===== Root refit (the mv2d check): re-solve the static root on the FITTED pose. The first
    # solve used the init template trunk; with the Stage-A trunk the 3D + multi-view 2D evidence
    # either CONFIRMS the root or corrects the residual template bias — once, jitter-free. Legs
    # are re-aligned and Stage A re-runs (warm start) only if the correction matters. =====
    from temporal_window import (ROOT_REFIT, ROOT_REFIT_THR_MM, ROOT_REFIT_THR_DEG, aa_angle_deg)
    if FREEZE_ROOT and ROOT_REFIT:
        _t_rf = time.time()
        _w_rf = valid * conf ** KP_CONF_POWER          # conf-only weights, same as the 1st solve
        _gt2d_N   = None if not _cams else {c: _gt2d[c][:N] for c in _cams}
        _conf2d_N = None if not _cams else {c: _conf2d[c][:N] for c in _cams}
        _go_rf, _tr_rf = solve_static_root(
            window_body_model, betas, bp_all, go_all, tr_all,
            gt_joints_all, _w_rf, cams=_cams, gt2d_all=_gt2d_N, conf2d_all=_conf2d_N)
        _ddeg = float(aa_angle_deg(_go_rf, go_all[:1]))
        _dmm  = float((_tr_rf - tr_all[:1]).norm()) * 1000.0
        if _ddeg > ROOT_REFIT_THR_DEG or _dmm > ROOT_REFIT_THR_MM:
            print(f"[root refit] Δ={_ddeg:.2f}° / {_dmm:.1f}mm → applied; legs re-aligned; Stage A re-run")
            go_init = _go_rf.expand(N, -1).contiguous()
            tr_init = _tr_rf.expand(N, -1).contiguous()
            go_ref_all, tr_ref_all = go_init.clone(), tr_init.clone()
            if leg_pose is not None and _sc is not None:
                bp_all[:, _LEG_COLS] = align_hips_to_root(leg_pose, gb_go, _sc['R'], _go_rf)
            bp_all, go_all, tr_all = run_windowed(
                window_body_model, body_pose_prior, angle_prior,
                gt_joints_all, weights_all, betas,
                bp_all, go_init, tr_init, go_ref_all, tr_ref_all,
                bp_ref_all=bp_ref_all)
        else:
            print(f"[root refit] root CONFIRMED (Δ={_ddeg:.2f}° / {_dmm:.1f}mm ≤ "
                  f"{ROOT_REFIT_THR_DEG}°/{ROOT_REFIT_THR_MM}mm) — kept")
        print(f"[timing] root refit: {time.time() - _t_rf:.2f}s")

    def _arm_kp_resid(bp_a, go_a, tr_a, lh_a=None, rh_a=None):
        """DIAGNOSTIC: mean 3D residual (mm) of the elbow/wrist keypoints over observed frames,
        plus the MODEL's R-arm bone lengths vs the gt triangulation (kinematic-consistency check)."""
        idxs = {'Lelb': 7, 'Relb': 8, 'Lwri': 9, 'Rwri': 10}
        acc = {k: [] for k in idxs}
        ua, fa, ua_gt, fa_gt = [], [], [], []
        z = torch.zeros(1, 45, dtype=dtype, device=device)
        with torch.no_grad():
            for i in range(N):
                body_model.body_pose.data.copy_(bp_a[i:i + 1])
                body_model.global_orient.data.copy_(go_a[i:i + 1])
                body_model.transl.data.copy_(tr_a[i:i + 1])
                body_model.betas.data.copy_(betas)
                body_model.left_hand_pose.data.copy_(z if lh_a is None else lh_a[i:i + 1])
                body_model.right_hand_pose.data.copy_(z if rh_a is None else rh_a[i:i + 1])
                J = body_model(return_verts=False).joints[0]
                for k, j in idxs.items():
                    if float(valid[i, j]) > 0:
                        acc[k].append(float((gt_joints_all[i, j] - J[j]).norm()) * 1000)
                ua.append(float((J[6] - J[8]).norm()) * 100); fa.append(float((J[8] - J[10]).norm()) * 100)
                if float(valid[i, 6]) > 0 and float(valid[i, 8]) > 0:
                    ua_gt.append(float((gt_joints_all[i, 6] - gt_joints_all[i, 8]).norm()) * 100)
                if float(valid[i, 8]) > 0 and float(valid[i, 10]) > 0:
                    fa_gt.append(float((gt_joints_all[i, 8] - gt_joints_all[i, 10]).norm()) * 100)
        _m = lambda v: (round(sum(v) / len(v), 1) if v else None)
        print(f"    [R-arm bones] model upperarm={_m(ua)}cm forearm={_m(fa)}cm  |  "
              f"gt upperarm={_m(ua_gt)}cm forearm={_m(fa_gt)}cm")
        return {k: (round(sum(v) / len(v), 1) if v else None) for k, v in acc.items()}

    print(f"[arm-resid] after Stage A:            {_arm_kp_resid(bp_all, go_all, tr_all)}")

    # ===== Stage B — hands: refine hand pose + arm reach (go/tr + non-arm body_pose FIXED) =====
    from temporal_window import run_windowed_hands, build_hand_inputs
    hand_w_all = torch.zeros_like(weights_all)
    hand_w_all[:, 17:59] = (valid * conf)[:, 17:59]                     # 3D hand keypoint weights
    # Also fit the ARM body keypoints (elbow 7/8, wrist 9/10): the hand stage moves the arm reach,
    # so without these the elbow drifts off its keypoint while the hand is placed. Boosted so the
    # ~4 arm points aren't outvoted by the ~40 finger points (mirrors the per-frame 'place' phase).
    hand_w_all[:, [7, 8, 9, 10]] = 2.0 * (valid * conf)[:, [7, 8, 9, 10]]
    lh_all, rh_all, wilor_lh_all, wilor_rh_all = build_hand_inputs(
        init_left_hand_poses, init_right_hand_poses, N, device, dtype)
    if left_hand_prior is not None and right_hand_prior is not None and (
            bool((hand_w_all > 0).any()) or float(wilor_lh_all.abs().sum() + wilor_rh_all.abs().sum()) > 0):
        _t_h = time.time()
        lh_all, rh_all, bp_all = run_windowed_hands(
            window_body_model, left_hand_prior, right_hand_prior,
            gt_joints_all, hand_w_all, betas, bp_all, go_all, tr_all,
            lh_all, rh_all, wilor_lh_all, wilor_rh_all)
        print(f"[timing] Stage B hands ({N} frames): {time.time() - _t_h:.2f}s")
    else:
        print("[stageB] no hand keypoints / WiLoR / prior → hand refinement skipped")

    print(f"[arm-resid] after hand stage:         {_arm_kp_resid(bp_all, go_all, tr_all, lh_all, rh_all)}")

    # ===== Stage B — head: refine neck+head + jaw onto the face landmarks (go/tr + rest fixed) =====
    from temporal_window import run_windowed_head, build_face_landmark_embedding
    face_w_all = torch.zeros_like(weights_all)
    face_w_all[:, 76:127] = (valid * conf)[:, 76:127]                   # inner face landmark weights
    face_w_all[:, 0:5]    = (valid * conf)[:, 0:5]                      # nose/eyes/EARS: skull orientation
    _E = int(getattr(window_body_model, 'num_expression_coeffs', 10))
    jaw_all  = torch.zeros(N, 3, dtype=dtype, device=device)
    expr_all = torch.zeros(N, _E, dtype=dtype, device=device)
    leye_all = torch.zeros(N, 3, dtype=dtype, device=device)
    reye_all = torch.zeros(N, 3, dtype=dtype, device=device)
    # True dlib landmarks live on the mesh surface (deform with expression) — the static model face
    # joints don't, so fitting those bottomed out ~5cm. Load the barycentric embedding for them.
    _lmk_emb = build_face_landmark_embedding(
        args['model_folder'], args.get('gender', 'neutral'),
        window_body_model.faces_tensor, device, dtype)
    if jaw_prior is not None and bool((face_w_all > 0).any()):
        _t_hd = time.time()
        jaw_all, expr_all, leye_all, reye_all, bp_all = run_windowed_head(
            window_body_model, jaw_prior, gt_joints_all, face_w_all, betas,
            bp_all, go_all, tr_all, jaw_all, expr_all, leye_all, reye_all, lmk_emb=_lmk_emb)
        print(f"[timing] Stage B head ({N} frames): {time.time() - _t_hd:.2f}s")
    else:
        print("[stageB] no face landmarks / jaw prior → head refinement skipped")

    # ===== Stage C: offline whole-sequence smoothing of ALL trajectories (global, no seams) =====
    from temporal_window import smooth_all_outputs
    _t_sm = time.time()
    (bp_all, go_all, tr_all, lh_all, rh_all,
     jaw_all, expr_all, leye_all, reye_all) = smooth_all_outputs(
        bp_all, go_all, tr_all, lh_all, rh_all, jaw_all, expr_all, leye_all, reye_all)
    print(f"[timing] Stage C smoothing ({N} frames): {time.time() - _t_sm:.2f}s")
    print(f"[arm-resid] after Stage C smoothing:  {_arm_kp_resid(bp_all, go_all, tr_all, lh_all, rh_all)}")

    # ===== write the fully-refined bodies (Stage A + hands + head + smoothing) =====
    with open(smplx_stored_path, 'w') as f:
        prev_saved_go = None   # for axis-angle unwrap of the saved global_orient trajectory
        for idx in range(N):
            try:
                # Load the Stage-A body+root + Stage-B refined hands into the (batch-1) model.
                with torch.no_grad():
                    body_model.body_pose.data.copy_(bp_all[idx:idx + 1])
                    body_model.global_orient.data.copy_(go_all[idx:idx + 1])
                    body_model.transl.data.copy_(tr_all[idx:idx + 1])
                    body_model.betas.data.copy_(betas)
                    body_model.left_hand_pose.data.copy_(lh_all[idx:idx + 1])
                    body_model.right_hand_pose.data.copy_(rh_all[idx:idx + 1])
                    body_model.jaw_pose.data.copy_(jaw_all[idx:idx + 1])
                    body_model.expression.data.copy_(expr_all[idx:idx + 1])
                    body_model.leye_pose.data.copy_(leye_all[idx:idx + 1])
                    body_model.reye_pose.data.copy_(reye_all[idx:idx + 1])

                _t_mesh = time.time()
                output = body_model(return_verts=args.get('save_mesh', True))

                # Unwrap the saved global_orient so the trajectory stays continuous across the
                # axis-angle pi-boundary (same rotation, no spurious ~2*pi sign flip).
                _go_save = output.global_orient.detach().reshape(1, 3)
                if prev_saved_go is not None:
                    _go_save = aa_nearest(_go_save, prev_saved_go)
                prev_saved_go = _go_save.clone()

                body_dict = {"frame_idx": idx,
                             "betas": output.betas.detach().cpu().numpy().tolist()[0],
                             "body_pose": output.body_pose.detach().cpu().numpy().tolist()[0],
                             "left_hand_pose": output.left_hand_pose.detach().cpu().numpy().tolist()[0],
                             "right_hand_pose": output.right_hand_pose.detach().cpu().numpy().tolist()[0],
                             "expression": output.expression.detach().cpu().numpy().tolist()[0],
                             "jaw_pose": body_model.jaw_pose.detach().cpu().numpy().tolist()[0],
                             "leye_pose": body_model.leye_pose.detach().cpu().numpy().tolist()[0],
                             "reye_pose": body_model.reye_pose.detach().cpu().numpy().tolist()[0],
                             "global_orient": _go_save.cpu().numpy().tolist()[0],
                             "transl": output.transl.detach().cpu().numpy().tolist()[0]}
                f.write(json.dumps(body_dict) + '\n')
                f.flush()

                _dt_mesh = 0.0
                if args.get('save_mesh', True):
                    import trimesh
                    vertices = output.vertices.detach().cpu().numpy().squeeze()
                    body_mesh = trimesh.Trimesh(vertices, body_model.faces, process=False)
                    mesh_path = os.path.join(args["output_folder"], sequence_name, "meshes", f"{idx:06d}_fit.obj")
                    body_mesh.export(mesh_path)
                    _dt_mesh = time.time() - _t_mesh

                _timing_frames.append({'frame': idx, 'fit_s': 0.0, 'mesh_s': round(_dt_mesh, 3)})
                print(f"[write/frame {idx}] mesh_export={_dt_mesh:.3f}s")
            except Exception as e:
                print('Writing sequence {} failed at frame {} with error: {}'.format(
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
            'stage_a_total_s':  round(_dt_stageA, 3),
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
        print(f"[timing summary]  stage_a={_timing_summary['stage_a_total_s']}s"
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
