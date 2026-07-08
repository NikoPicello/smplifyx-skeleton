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
from utils import JointMapper, aa_nearest, load_gt_silhouettes
from prior import create_prior
from fit_single_frame import fit_single_frame
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
    failed_frames = []
    _timing_frames = []   # list of {frame, fit_s, mesh_s}

    # ===================================================================
    # Stage A: one batched, windowed body+root fit for the WHOLE sequence
    # (replaces the per-frame fitting loop). Hands/face stay at their init
    # warm-start here; Stage B (per-frame head/hand refinement) is TODO.
    # ===================================================================
    N = min(len(dataset_obj), 25)   # benchmark cap — raise (or lower WIN_SIZE) to exercise seams

    def _seq(arr, d):
        return torch.stack([torch.as_tensor(np.asarray(arr[i], dtype=np.float32),
                            dtype=dtype, device=device).reshape(d) for i in range(N)])

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

    # Warm start from SMPLer-X + seated-leg override; anchor targets for the leg/root terms.
    bp_init = _seq(init_bps, (63,))
    for _dof, _val in SEATED_POSE.items():
        bp_init[:, _dof] = _val
    go_init = _seq(init_gos, (3,)) if init_gos is not None else torch.zeros(N, 3, dtype=dtype, device=device)
    tr_init = _seq(init_trs, (3,)) if init_trs is not None else torch.zeros(N, 3, dtype=dtype, device=device)
    leg_ref_all = bp_init[:, LOWER_BODY_POSE_DOFS].clone()
    go_ref_all, tr_ref_all = go_init.clone(), tr_init.clone()
    betas1 = global_betas if global_betas is not None else body_model.betas.detach()[:1].clone()

    # ===== Stage 0: refine the shared betas to the OBSERVED bone lengths (pose-invariant). =====
    # SMPLer-X shape is the init/anchor, but its limb lengths can be cm-level wrong (p0's arm was
    # ~8cm short → elbow unfittable). One solve per person, then frozen for the whole sequence.
    from temporal_window import refine_betas_bone_lengths
    betas1 = refine_betas_bone_lengths(body_model, betas1, gt_joints_all, valid * conf)

    _t_stageA = time.time()
    bp_all, go_all, tr_all = run_windowed(
        window_body_model, body_pose_prior, angle_prior,
        gt_joints_all, weights_all, betas1,
        bp_init, go_init, tr_init, leg_ref_all, go_ref_all, tr_ref_all)
    _dt_stageA = time.time() - _t_stageA
    print(f"[timing] Stage A windowed fit ({N} frames): {_dt_stageA:.2f}s  ({_dt_stageA/max(N,1):.2f}s/frame)")

    # ===== Stage B — placement: refine go/tr via batched multi-view 2D reprojection (body fixed) =====
    from temporal_window import run_windowed_placement, build_placement_inputs
    _cams, _gt2d, _conf2d = build_placement_inputs(
        args.get('silhouette_cameras'), mv_rtmo, args.get('person_id', 0), N, device, dtype)
    if _cams:
        _t_pl = time.time()
        go_all, tr_all = run_windowed_placement(
            window_body_model, _cams, _gt2d, _conf2d, betas1, bp_all, go_all, tr_all)
        print(f"[timing] Stage B placement ({N} frames, {len(_cams)} cams): {time.time() - _t_pl:.2f}s")
    else:
        print("[stageB] no 2D keypoints / cameras available → placement skipped")

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
                body_model.betas.data.copy_(betas1)
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

    print(f"[arm-resid] after Stage A + placement: {_arm_kp_resid(bp_all, go_all, tr_all)}")

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
            gt_joints_all, hand_w_all, betas1, bp_all, go_all, tr_all,
            lh_all, rh_all, wilor_lh_all, wilor_rh_all)
        print(f"[timing] Stage B hands ({N} frames): {time.time() - _t_h:.2f}s")
    else:
        print("[stageB] no hand keypoints / WiLoR / prior → hand refinement skipped")

    print(f"[arm-resid] after hand stage:         {_arm_kp_resid(bp_all, go_all, tr_all, lh_all, rh_all)}")

    # ===== Stage B — head: refine neck+head + jaw onto the face landmarks (go/tr + rest fixed) =====
    from temporal_window import run_windowed_head, build_face_landmark_embedding
    face_w_all = torch.zeros_like(weights_all)
    face_w_all[:, 76:127] = (valid * conf)[:, 76:127]                   # inner face landmark weights
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
            window_body_model, jaw_prior, gt_joints_all, face_w_all, betas1,
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

    # ===== write the fully-refined bodies (Stage A + placement + hands + head + smoothing) =====
    with open(smplx_stored_path, 'w') as f:
        prev_saved_go = None   # for axis-angle unwrap of the saved global_orient trajectory
        for idx in range(N):
            try:
                # Load the Stage-A body+root + Stage-B refined hands into the (batch-1) model.
                with torch.no_grad():
                    body_model.body_pose.data.copy_(bp_all[idx:idx + 1])
                    body_model.global_orient.data.copy_(go_all[idx:idx + 1])
                    body_model.transl.data.copy_(tr_all[idx:idx + 1])
                    body_model.betas.data.copy_(betas1)
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
