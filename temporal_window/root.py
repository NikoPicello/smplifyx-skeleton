# -*- coding: utf-8 -*-
"""
temporal_window.root — static root: solve ONE (go, tr) for the whole sequence, then FREEZE it.

The subjects are SEATED and their hips are table-occluded in almost every frame (005013/lego:
p0 Rhip observed 4774/9345 frames, p1's hips ~100/9345), so the per-frame root is placed by the
SHOULDERS — which genuinely lean (p95 ~8-13cm reaching over the table). Root-vs-spine is then
underdetermined and the solved root wanders/jitters per frame (p0 even excursed 4.6m during a
detection dropout). When the hips ARE observed they sit still (p0: 8.7mm median drift over the
whole 5min video) → the pelvis is truly static. So: fit ONE (go, tr) against the trunk keypoints
of a strided frame subsample (GMoF → transient leans saturate as outliers → a robust "resting"
root), freeze it, and let Stage A express all motion through body_pose — leans go to the spine,
where they anatomically belong. Kills root jitter identically to zero and the excursion failure
class by construction.

The solve and the freeze are now SEPARATE decisions:
  SOLVE_STATIC_ROOT — run the robust static solve above. Keep it True: it is the best global
                      placement available (annealed GMoF, multi-view 2D hips/knees) and, when
                      the root is free, it is also the warm start AND the L_root anchor target.
                      With it False the root falls back to the raw per-frame SMPLer-X init,
                      which misses the triangulated shoulders by 8-18cm (see ROOT_*_ANCHOR_W).
  FREEZE_ROOT       — hold that solve fixed for every frame (True), or let Stage A refine it
                      per frame around the anchor (False). Free lets the pelvis answer to the
                      rest of the body instead of forcing every lean into the spine (the arched
                      back / unnatural trunk), at the cost of re-opening the wander/jitter class
                      above — which is what LAMBDA_ROOT (the anchor, in body.py) plus the
                      already-stiffer root smoothness (LAMBDA_*_TR/GO) and the spine pin exist
                      to contain.

Also holds the hip-lift seed (lift a single 2D hip detection into 3D for the opening window
only) that feeds this solve real pelvis ORIENTATION evidence — see its own section below.
"""
from __future__ import absolute_import, print_function, division

import math

import cv2 as cv
import numpy as np
import torch

from smplx.lbs import batch_rodrigues   # axis-angle -> rotation matrix (exp map, smooth for all theta)
from smplx.lbs import blend_shapes, vertices2joints   # pelvis(betas) for the static-root reduction

from utils import build_camera_tensors, _project_to_pixels   # multi-view 2D term of the static root

from . import _common
from ._common import _aa_unwrap, _aa_to_6d, _cap, _f

SOLVE_STATIC_ROOT = True
FREEZE_ROOT   = False
ROOT_STRIDE   = 30            # fit every k-th frame (auto-lowered so short clips keep >=WIN_SIZE)
ROOT_FRAME_START = 0          # first frame (of the full sequence) eligible for the static root solve
ROOT_FRAME_COUNT = None       # if set, only the ROOT_FRAME_COUNT frames starting at ROOT_FRAME_START
                              # are eligible (stride auto-drops to 1 for a pool this small -- same
                              # "short clips: use every frame" logic as ROOT_STRIDE below). None
                              # (default) uses the whole sequence, unchanged from today. Lets the
                              # static root be seeded from a short early window instead of a
                              # whole-sequence median that a later genuine reorientation would
                              # otherwise dominate.
ROOT_TRUNK_KP = [5, 6, 11, 12]   # shoulders + hips (COCO ids in the mapped layout). Hips were
                              # excluded here for years because real triangulated hips are almost
                              # never confident in these seated/table sessions (see BETAS_SEGMENTS'
                              # dropped 'trunk' pair) -- elsewhere in the clip they stay near-zero
                              # confidence and contribute ~nothing to L3d below. lift_hip_seed()
                              # (this file) now fills them for the OPENING window only, from a
                              # single camera's 2D detection + both shoulders -- see its docstring.
_TRUNK_KP_NAMES = {5: 'Lsho', 6: 'Rsho', 11: 'Lhip', 12: 'Rhip'}   # for the per-joint resid print
                                                                    # below -- keyed by COCO id, not
                                                                    # position, so it stays correct
                                                                    # for any subset/order of ROOT_TRUNK_KP
# GMoF scales, ANNEALED coarse→fine over ROOT_STEPS (the recurring rho-saturation trap, third
# time: the init pelvis starts ~10cm / 60-96px off the RTMO hips — beyond 2ρ at a FIXED fine ρ
# both hip terms are saturated with near-zero gradient, so the solve polished the shoulders and
# LEFT THE PELVIS WHERE THE INIT PUT IT → the fold at the waist / "arching back". Start wide so
# the pelvis is actually pulled in, finish fine so transient leans go back to being outliers.)
ROOT_RHO0,    ROOT_RHO1    = 0.20, 0.05    # 3D (m)
ROOT_RHO_PX0, ROOT_RHO_PX1 = 200.0, 50.0   # 2D (px)
ROOT_DATA_W   = 100.0         # 3D trunk data weight (Stage-A scale)
ROOT_DATA_W_2D = 100.0        # 2D reprojection weight (focal-normalised). RAISED so the 2D
                              # hip/knee evidence OWNS the pelvis placement — the 3D shoulders
                              # must not drag the pelvis through an uncertain trunk template.
ROOT_CONF_FLOOR = 0.3         # ignore 2D detections below this score
ROOT_STEPS    = 3             # LBFGS steps (max_iter 20 each; also the rho annealing schedule)
# Light 6D/L2 anchor to the SMPLer-X init median — tie-break for the last soft DOFs only.
# LOOSENED from 10: the init root is measurably inconsistent with the triangulated shoulders
# (its own root+spine pair misses them by 8-18cm), so it must not out-vote the 2D pelvis data.
ROOT_GO_ANCHOR_W = 3.0
ROOT_TR_ANCHOR_W = 3.0
# ── root refit (the mv2d check): re-solve the static root on the FITTED pose after Stage A ────
# The initial static root is solved with the INIT template trunk (SMPLer-X spine + GB legs), so
# any template error biases it. After Stage A the trunk is data-fit; one more static solve with
# the same 3D + multi-view 2D evidence either CONFIRMS the root or corrects the residual bias —
# ONCE for the whole sequence. (A per-frame mv2d placement under FREEZE_ROOT would re-introduce
# the root-wander failure class and swing the world-aligned frozen legs off the GB image.)
# If the correction exceeds the thresholds the legs are re-aligned and Stage A re-runs (warm).
ROOT_REFIT         = True
ROOT_REFIT_THR_MM  = 5.0
ROOT_REFIT_THR_DEG = 0.5
# 2D mask for the STATIC solve — the PELVIS is OBSERVED here, not inferred: two-camera hip
# rays pin its position, and the knees (ends of the world-aligned frozen GB legs) pin its
# pitch. The trunk template's curl (SMPLer-X spine) is uncertain and pelvis-relative, so with
# weak lower-body weights the 3D-shoulder term drags the pelvis through the template instead
# (measured after the spine-template change: 2.5-6cm pelvis drift, knees up to 85px off).
# Weights swept offline (scratchpad root_experiment.py): hips 2.0 / knees 1.0 / DATA_W_2D 200
# / anchors 3.0 → hips+knees land at 13-57px (was 68-141px), beating even the old
# straight-template root; the solved pelvis independently converges toward GB-SMPLer-X's
# pelvis (delta 30°→10-14° for p1).
_ROOT_2D_W = torch.ones(17)
_ROOT_2D_W[[0, 1, 2, 3, 4]] = 0.1    # nose/eyes/ears — move with the neck
_ROOT_2D_W[[7, 8, 9, 10]]   = 0.05   # elbows/wrists — arm motion must not tilt the root
_ROOT_2D_W[[11, 12]]         = 2.0   # hips — pelvis POSITION (two-camera rays)
_ROOT_2D_W[[13, 14, 15, 16]] = 1.0   # knees/ankles — pelvis PITCH via the frozen aligned legs

# ── hip seed: lift a single 2D hip detection into 3D, OPENING WINDOW only ──────────────────────
# Hips are essentially never triangulatable in these seated/table sessions: checked across 7
# sessions, they're occluded in every camera but one (occasionally two, never confidently) -- 2-
# camera agreement basically doesn't happen, which is why BETAS_SEGMENTS dropped 'trunk' and
# ROOT_TRUNK_KP excluded hips for years. But a SINGLE camera's 2D RTMO hip detection is available
# almost everywhere, and its own line of sight passes within a few mm of the true hip (checked
# against the rare frames that DO triangulate confidently) -- the only missing piece is DEPTH
# along that ray, which the two reliably-triangulated shoulders resolve: anchor the hip to BOTH
# shoulders (same-side + cross-side bone length, each an independent distance constraint on the
# one unknown depth) and solve. Using both shoulders instead of just the same-side one also avoids
# the near/far root ambiguity a single sphere has -- the cross-side anchor only agrees with one of
# the two roots, so this self-disambiguates instead of needing a rig-specific rule.
#
# Restricted to the OPENING window (ROOT_FRAME_START/ROOT_FRAME_COUNT, the same range the static
# root solve itself uses) because that's the one place a confident hip fix earns its keep: seeding
# (go, tr) with real pelvis ORIENTATION evidence, not just the two-shoulder position solve_static_
# root was limited to before. It is NOT a general per-frame Stage-A signal -- feeding it every
# frame of a multi-thousand-frame clip was the original idea here, dropped once it was clear real
# triangulated hips can't be trusted "period", not just occasionally.
HIP_LIFT_CONF        = 0.5   # confidence stamped on the injected pseudo-observation. Kept below a
                              # well-observed real joint (shoulders routinely score 0.8-0.99) so
                              # genuine data always outweighs it; a frame's hip slot is only
                              # overwritten when that RAISES its confidence. Gets squared twice
                              # downstream (KP_CONF_POWER, then again inside solve_static_root's
                              # L3d) -- retune against the per-joint mm residual the static root
                              # already prints (_TRUNK_KP_NAMES has carried Lhip/Rhip entries since
                              # before this existed) rather than trusting this value blindly.
HIP_LIFT_MIN_SAMPLES  = 20    # below this many in-clip confident triangulated pairs, don't trust
                              # this clip's own median bone length -- fall back to the CURRENT
                              # betas' rest-pose distance instead (vertices2joints, the same call
                              # solve_static_root uses for pelvis0).
HIP_LIFT_BAD_CAM_PX   = 15.0  # shoulder-reprojection sanity gate: exclude a camera from the lift if
                              # body.npy's own (reliable) shoulder reprojects >this many px off that
                              # camera's own RTMO shoulder detection. Found empirically: GB fails
                              # this on every one of 7 sessions checked (20-40px, 3 independent
                              # calibration dates) while GF and the others clear it -- a real,
                              # camera-specific issue, not an artifact of this method.
_LSHO, _RSHO, _LHIP, _RHIP = 5, 6, 11, 12   # COCO ids, same layout as ROOT_TRUNK_KP/_TRUNK_KP_NAMES

_CFG_OVERRIDABLE = [
    'SOLVE_STATIC_ROOT', 'FREEZE_ROOT', 'ROOT_STRIDE', 'ROOT_FRAME_START', 'ROOT_FRAME_COUNT',
    'ROOT_DATA_W', 'ROOT_CONF_FLOOR',
    'ROOT_STEPS', 'ROOT_GO_ANCHOR_W', 'ROOT_TR_ANCHOR_W',
    'ROOT_REFIT', 'ROOT_REFIT_THR_MM', 'ROOT_REFIT_THR_DEG',
]


def configure(args):
    """Overwrite the tuning constants named in _CFG_OVERRIDABLE from a parsed cfg/CLI
    args dict (see cmd_parser.py). Called once by temporal_window.configure(), before any
    fitting stage runs. Keys absent from `args` (or None) leave the module default
    untouched."""
    g = globals()
    for name in _CFG_OVERRIDABLE:
        key = name.lower()
        if args.get(key) is not None:
            g[name] = args[key]


def _weighted_median(values, weights):
    values, weights = np.asarray(values, dtype=np.float64), np.asarray(weights, dtype=np.float64)
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cw = np.cumsum(weights)
    return float(values[np.searchsorted(cw, cw[-1] / 2.0)])


def _hip_bone_lengths(kp_np, cf_np, betas1, model_ref):
    """Per-side (same-side, cross-side) shoulder->hip length: confidence-weighted median over the
    WHOLE clip's own triangulation when there are enough samples (HIP_LIFT_MIN_SAMPLES), else the
    current betas' rest-pose distance. kp_np/cf_np: (N,17,3)/(N,17) numpy, the full clip."""
    pairs = {'L': (_LSHO, _LHIP, _RSHO), 'R': (_RSHO, _RHIP, _LSHO)}   # side -> (sho, hip, cross_sho)
    out_same, out_cross, n_used = {}, {}, {}
    for side, (sho, hip, cross_sho) in pairs.items():
        cs, ch, cc = cf_np[:, sho], cf_np[:, hip], cf_np[:, cross_sho]
        ok = (cs > 0) & (ch > 0) & (cc > 0) & np.isfinite(kp_np[:, [sho, hip, cross_sho]]).all((1, 2))
        n_used[side] = int(ok.sum())
        if n_used[side] >= HIP_LIFT_MIN_SAMPLES:
            out_same[side]  = _weighted_median(np.linalg.norm(kp_np[ok, sho] - kp_np[ok, hip], axis=1),
                                                np.minimum(cs[ok], ch[ok]))
            out_cross[side] = _weighted_median(np.linalg.norm(kp_np[ok, cross_sho] - kp_np[ok, hip], axis=1),
                                                np.minimum(cc[ok], ch[ok]))
    missing = [s for s in ('L', 'R') if s not in out_same or s not in out_cross]
    if missing:
        with torch.no_grad():
            v_shaped = model_ref.v_template + blend_shapes(betas1, model_ref.shapedirs)
            J = vertices2joints(model_ref.J_regressor, v_shaped)[0].cpu().numpy()   # rest-pose (J,3)
        # SMPL-X body joint ids (see visualization/vis_joint_mapping.py): 0 pelvis, 1 l_hip,
        # 2 r_hip, 16 l_shoulder, 17 r_shoulder -- direct pelvis children, unaffected by body_pose.
        fb_same  = {'L': float(np.linalg.norm(J[1] - J[16])), 'R': float(np.linalg.norm(J[2] - J[17]))}
        fb_cross = {'L': float(np.linalg.norm(J[1] - J[17])), 'R': float(np.linalg.norm(J[2] - J[16]))}
        for s in missing:
            out_same.setdefault(s, fb_same[s])
            out_cross.setdefault(s, fb_cross[s])
    return out_same, out_cross, n_used


def _shoulder_sanity_gate(silhouette_cameras, mv_rtmo, kp_np, cf_np, person_id, bad_px=HIP_LIFT_BAD_CAM_PX):
    """Reproject body.npy's own shoulders into each camera and compare to that camera's own RTMO
    shoulder detection. A camera whose median error exceeds bad_px is excluded below -- catches a
    mis-calibrated camera (GB, on every session checked so far) before it corrupts the seed."""
    good = []
    for cam_name, cam in silhouette_cameras.items():
        arr = mv_rtmo.get(cam_name)
        if arr is None:
            continue
        K, D, R = (np.asarray(cam[k], dtype=np.float64) for k in ('K', 'D', 'R'))
        T = np.asarray(cam['T'], dtype=np.float64).reshape(3)
        rvec, _ = cv.Rodrigues(R)
        errs = []
        for fidx in range(min(len(arr), kp_np.shape[0])):
            det = arr[fidx].get(person_id) if isinstance(arr[fidx], dict) else None
            if det is None:
                continue
            for sho in (_LSHO, _RSHO):
                if cf_np[fidx, sho] <= 0 or det['keypoint_scores'][sho] < 0.3:
                    continue
                S = kp_np[fidx, sho]
                if not np.isfinite(S).all():
                    continue
                proj, _ = cv.projectPoints(S.reshape(1, 3), rvec, T.reshape(3, 1), K, D)
                errs.append(np.linalg.norm(proj.reshape(2) - det['keypoints'][sho]))
        if errs and np.median(errs) <= bad_px:
            good.append(cam_name)
    return good


def _camera_ray_batch(pts, K, D, R, T):
    """World-space (camera_center (3,), unit_ray_directions (N,3)) through pixels pts (N,2)."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    norm = cv.undistortPoints(pts, K, D).reshape(-1, 2)
    d_cam = np.concatenate([norm, np.ones((norm.shape[0], 1))], axis=1)
    d_cam /= np.linalg.norm(d_cam, axis=1, keepdims=True)
    return -R.T @ T, d_cam @ R


def _ray_sphere_lift_batch(C, U, S, L):
    """Larger positive t with |C+t*U-S|==L per row (see the section comment for why the far root;
    verified against real triangulated hips on 7 sessions). NaN where no positive root exists."""
    v = C[None, :] - S
    b = np.einsum('ij,ij->i', v, U)
    c = np.einsum('ij,ij->i', v, v) - L * L
    disc = b * b - c
    sq = np.sqrt(np.clip(disc, 0, None))
    t_far, t_near = -b + sq, -b - sq
    t = np.where(t_far > 0, t_far, np.where(t_near > 0, t_near, np.nan))
    return np.where(disc >= 0, t, np.nan)


def _dual_anchor_lift_batch(C, U, S_same, len_same, S_cross, len_cross, t0, n_iter=8):
    """Vectorized Gauss-Newton: resolve the single unknown depth t against BOTH shoulder anchors
    at once (self-disambiguates the near/far root -- see section comment) for many rays at once."""
    v1, v2 = C[None, :] - S_same, C[None, :] - S_cross
    b1, c1 = np.einsum('ij,ij->i', v1, U), np.einsum('ij,ij->i', v1, v1)
    b2, c2 = np.einsum('ij,ij->i', v2, U), np.einsum('ij,ij->i', v2, v2)
    t = t0.copy()
    for _ in range(n_iter):
        s1 = np.sqrt(np.clip(t * t + 2 * b1 * t + c1, 1e-9, None))
        s2 = np.sqrt(np.clip(t * t + 2 * b2 * t + c2, 1e-9, None))
        g, h = s1 - len_same, s2 - len_cross
        gp, hp = (t + b1) / s1, (t + b2) / s2
        t = t - (g * gp + h * hp) / np.clip(gp * gp + hp * hp, 1e-9, None)
    return t


def lift_hip_seed(betas1, model_ref, kp_full, cf_full, silhouette_cameras, mv_rtmo, person_id):
    """Fill kp_full/cf_full's hip slots (COCO ids 11/12) for the OPENING window
    (ROOT_FRAME_START..+ROOT_FRAME_COUNT, whole clip if ROOT_FRAME_COUNT is None) with a single
    pooled 3D estimate per side, lifted from whichever single camera has a confident 2D RTMO hip
    detection each frame, anchored to both reliably-triangulated shoulders (see the section
    comment above). ONE pooled point per side, not a per-frame value: solve_static_root already
    assumes the trunk is ~static over this window, so a single robust (median) estimate matches
    that assumption better than injecting per-frame noise would. Only overwrites a frame's slot
    where doing so RAISES its confidence (HIP_LIFT_CONF) -- never weakens a real observation.
    Returns (kp_full, cf_full): new tensors if anything was injected, the originals otherwise."""
    if not silhouette_cameras or not mv_rtmo:
        return kp_full, cf_full
    device, dtype = kp_full.device, kp_full.dtype
    kp_np = kp_full[..., :3].detach().cpu().numpy().astype(np.float64)
    cf_np = cf_full.detach().cpu().numpy().astype(np.float64)
    N = kp_np.shape[0]
    lo = min(ROOT_FRAME_START, N)
    hi = N if ROOT_FRAME_COUNT is None else min(lo + ROOT_FRAME_COUNT, N)

    trunk, cross, n_used = _hip_bone_lengths(kp_np, cf_np, betas1, model_ref)
    good_cams = _shoulder_sanity_gate(silhouette_cameras, mv_rtmo, kp_np, cf_np, person_id)
    if not good_cams:
        print("[hip seed] no camera passed the shoulder sanity check -- skipping")
        return kp_full, cf_full
    print(f"[hip seed] trunk(cm) L={trunk['L']*100:.1f}(n={n_used['L']}) R={trunk['R']*100:.1f}(n={n_used['R']})"
          f"  good_cams={good_cams}  window=[{lo},{hi})")

    pairs = {'L': (_LSHO, _LHIP, _RSHO), 'R': (_RSHO, _RHIP, _LSHO)}
    pooled = {}
    for side, (sho, hip, cross_sho) in pairs.items():
        best = {}
        for cam_name in good_cams:
            cam, arr = silhouette_cameras[cam_name], mv_rtmo[cam_name]
            K, D, R = (np.asarray(cam[k], dtype=np.float64) for k in ('K', 'D', 'R'))
            T = np.asarray(cam['T'], dtype=np.float64).reshape(3)
            fidxs, pxpy, S_same, S_cross, scores = [], [], [], [], []
            for fidx in range(lo, min(hi, len(arr))):
                det = arr[fidx].get(person_id) if isinstance(arr[fidx], dict) else None
                if det is None or det['keypoint_scores'][hip] < 0.3:
                    continue
                cs, ss = cf_np[fidx, sho], kp_np[fidx, sho]
                cc, sc = cf_np[fidx, cross_sho], kp_np[fidx, cross_sho]
                if cs <= 0 or cc <= 0 or not (np.isfinite(ss).all() and np.isfinite(sc).all()):
                    continue
                fidxs.append(fidx); pxpy.append(det['keypoints'][hip])
                S_same.append(ss); S_cross.append(sc); scores.append(float(det['keypoint_scores'][hip]))
            if not fidxs:
                continue
            pxpy, S_same, S_cross = np.asarray(pxpy), np.asarray(S_same), np.asarray(S_cross)
            C, U = _camera_ray_batch(pxpy, K, D, R, T)
            t0 = _ray_sphere_lift_batch(C, U, S_same, trunk[side])
            t0 = np.where(np.isfinite(t0), t0, np.einsum('ij,ij->i', S_same - C[None, :], U))
            t = _dual_anchor_lift_batch(C, U, S_same, trunk[side], S_cross, cross[side], t0)
            P = C[None, :] + t[:, None] * U
            for i, fidx in enumerate(fidxs):
                if t[i] > 0 and (fidx not in best or scores[i] > best[fidx][0]):
                    best[fidx] = (scores[i], P[i])
        if best:
            pooled[side] = np.median(np.array([p for _, p in best.values()]), axis=0)
            print(f"[hip seed] {side}hip pooled from {len(best)}/{hi - lo} window frames")
        else:
            print(f"[hip seed] {side}hip: no confident 2D detection from any good camera in the window")

    if not pooled:
        return kp_full, cf_full
    kp_full, cf_full = kp_full.clone(), cf_full.clone()
    for side, (sho, hip, cross_sho) in pairs.items():
        if side not in pooled:
            continue
        P = torch.as_tensor(pooled[side], dtype=dtype, device=device)
        for fidx in range(lo, hi):
            if float(cf_full[fidx, hip]) < HIP_LIFT_CONF:
                kp_full[fidx, hip, :3] = P
                cf_full[fidx, hip] = HIP_LIFT_CONF
    return kp_full, cf_full


# ── static root: ONE (go, tr) for the whole sequence (FREEZE_ROOT) ─────────────────────────────
def solve_static_root(model_W, betas1, bp_all, go_all, tr_all, gt_joints_all, weights_all,
                      cams=None, gt2d_all=None, conf2d_all=None):
    """Solve the single frozen root. Rigid-root reduction: with body_pose fixed, changing only
    (go, tr) moves every output joint by  j → R(go)·(j₀ − pelvis₀) + pelvis₀ + tr,  where j₀ are
    the joints at go=0/tr=0 (precomputed once, chunked through the batch-W model) and pelvis₀
    depends on betas alone — so the optimisation loop never runs the body model.
        bp_all/go_all/tr_all: (N,·) SMPLer-X init (bp gives each frame's articulation; go/tr give
        the warm start via their strided component-wise median, go unwrapped first).
    Data: 3D trunk keypoints (GMoF, rho ANNEALED ROOT_RHO0→1) on every stride-th frame within
    [ROOT_FRAME_START, ROOT_FRAME_START+ROOT_FRAME_COUNT) (the whole sequence if ROOT_FRAME_COUNT
    is None) + the multi-view 2D trunk reprojection (_ROOT_2D_W mask, GMoF). Returns (go, tr),
    each (1,3) detached."""
    device, dt = go_all.device, go_all.dtype
    N, W = gt_joints_all.shape[0], _common.WIN_SIZE
    tkp = torch.as_tensor(ROOT_TRUNK_KP, device=device, dtype=torch.long)

    pool_lo = min(ROOT_FRAME_START, N)
    pool_hi = N if ROOT_FRAME_COUNT is None else min(pool_lo + ROOT_FRAME_COUNT, N)
    stride = max(1, min(ROOT_STRIDE, (pool_hi - pool_lo) // W))   # short clips/pools: use every frame
    idx = torch.arange(pool_lo, pool_hi, stride, device=device)
    idx = idx[(weights_all[idx][:, tkp] > 0).any(1)]   # keep frames with >=1 trunk observation
    n = int(len(idx))
    go0 = _aa_unwrap(go_all[idx if n else slice(None)]).median(0, keepdim=True).values
    tr0 = (tr_all[idx] if n else tr_all).median(0, keepdim=True).values
    if n == 0:
        print("[static root] no trunk observations → freezing at the init median")
        return go0.detach(), tr0.detach()

    # joints at go=0/tr=0 for the selected frames (chunked, no grad) + pelvis (betas only)
    betasW = betas1.expand(W, -1).contiguous()
    zero3 = torch.zeros(W, 3, dtype=dt, device=device)
    with torch.no_grad():
        j0 = []
        for s in range(0, n, W):
            ii = idx[s:s + W]
            bpc = bp_all[ii]
            if len(ii) < W:
                bpc = torch.cat([bpc, bpc[-1:].expand(W - len(ii), -1)], dim=0)
            j0.append(model_W(betas=betasW, body_pose=bpc, global_orient=zero3, transl=zero3,
                              return_verts=False).joints[:len(ii), :17])
        j0 = torch.cat(j0, dim=0)                                                    # (n,17,3)
        v_shaped = model_W.v_template + blend_shapes(betas1, model_W.shapedirs)
        pelvis0  = vertices2joints(model_W.J_regressor, v_shaped)[0, 0]              # (3,)
    j0c = j0 - pelvis0                       # pelvis-centred: rotate these, then add pelvis0 + tr
    gt, wkp = gt_joints_all[idx][:, tkp], weights_all[idx][:, tkp]                   # (n,4,·)

    have2d = bool(cams) and gt2d_all is not None
    if have2d:
        jw2  = _ROOT_2D_W.to(device=device, dtype=dt)
        gt2d = {c: gt2d_all[c][idx] for c in cams}
        cf2d = {c: conf2d_all[c][idx] for c in cams}

    go = go0.clone().requires_grad_(True)
    tr = tr0.clone().requires_grad_(True)
    go6_0 = _aa_to_6d(go0)                                             # anchor target (const)
    opt = torch.optim.LBFGS([go, tr], lr=1.0, max_iter=20, line_search_fn='strong_wolfe')
    _call_i = 0

    def closure(backward=True, rho3=ROOT_RHO1, rho2=ROOT_RHO_PX1):
        nonlocal _call_i
        if backward:
            opt.zero_grad()
        R  = batch_rodrigues(go)[0]                                                  # (3,3)
        Jb = j0c @ R.t() + pelvis0 + tr                                              # (n,17,3)
        d2  = (gt - Jb[:, tkp]).pow(2).sum(-1)                                       # (n,4)
        rob = rho3 ** 2 * d2 / (d2 + rho3 ** 2)                                      # GMoF
        L3d = (wkp ** 2 * rob).sum(1).mean() * ROOT_DATA_W ** 2
        L2d = go.new_zeros(())
        if have2d:
            for cname, cam in cams.items():
                proj, vld = _project_to_pixels(Jb, cam)
                proj  = proj.reshape(n, 17, 2)
                vld   = vld.reshape(n, 17).to(dt)
                f     = cam['K'][0, 0].to(dt)
                r     = gt2d[cname] - proj
                rob2  = rho2 ** 2 * r.pow(2) / (r.pow(2) + rho2 ** 2)
                c     = cf2d[cname]
                w2    = (c * (c >= ROOT_CONF_FLOOR).to(dt) * jw2 * vld).unsqueeze(-1)
                L2d   = L2d + (w2.pow(2) * rob2 / (f ** 2)).sum()
            L2d = L2d * ROOT_DATA_W_2D ** 2 / n                                      # per-frame mean
        L_anc = (ROOT_GO_ANCHOR_W * (_aa_to_6d(go) - go6_0).pow(2).sum()
                 + ROOT_TR_ANCHOR_W * (tr - tr0).pow(2).sum())
        total = _cap(L3d) + _cap(L2d) + _cap(L_anc)
        if backward:
            total.backward()
            _call_i += 1
            if _call_i % _common.LOG_EVERY == 0:
                print(f"  [root] 3d={_f(L3d):8.3f} 2d={_f(L2d):8.3f} anc={_f(L_anc):7.3f} tot={_f(total):8.3f}")
        return total

    best_loss = float('inf')
    best_state = [p.detach().clone() for p in (go, tr)]
    for si in range(ROOT_STEPS):
        t = si / max(ROOT_STEPS - 1, 1)                       # anneal coarse → fine
        rho3 = ROOT_RHO0 * (ROOT_RHO1 / ROOT_RHO0) ** t
        rho2 = ROOT_RHO_PX0 * (ROOT_RHO_PX1 / ROOT_RHO_PX0) ** t
        snapshot = [p.detach().clone() for p in (go, tr)]
        loss = float(opt.step(lambda: closure(rho3=rho3, rho2=rho2)))
        if not (math.isfinite(loss) and all(bool(torch.isfinite(p).all()) for p in (go, tr))):
            print("  [root] non-finite step → restoring best, stop")
            break
        if loss < best_loss:
            best_loss = loss; best_state = snapshot
    final = float(closure(backward=False))
    if not (math.isfinite(final) and final <= best_loss):
        with torch.no_grad():
            for p, s in zip((go, tr), best_state):
                p.data.copy_(s)

    with torch.no_grad():   # diagnostics + exact-forward probe of the rigid reduction
        R = batch_rodrigues(go)[0]
        dj = ((j0c[:, tkp] @ R.t() + pelvis0 + tr) - gt).norm(dim=-1) * 1000          # (n,4)
        per = '  '.join(f"{_TRUNK_KP_NAMES.get(k, str(k))}={float(dj[wkp[:, i] > 0, i].median()):.0f}"
                        for i, k in enumerate(ROOT_TRUNK_KP)
                        if bool((wkp[:, i] > 0).any()))
        print(f"[static root] per-joint 3D resid (median mm): {per}")
        d = dj[wkp > 0]
        ang = torch.rad2deg(torch.arccos(
            ((batch_rodrigues(go0)[0] * R).sum() - 1).mul(0.5).clamp(-1, 1)))
        m  = min(n, W)
        bpc = bp_all[idx[:m]]
        if m < W:
            bpc = torch.cat([bpc, bpc[-1:].expand(W - m, -1)], dim=0)
        jf = model_W(betas=betasW, body_pose=bpc, global_orient=go.expand(W, -1).contiguous(),
                     transl=tr.expand(W, -1).contiguous(), return_verts=False).joints[:m, :17]
        probe = (jf - (j0c[:m] @ R.t() + pelvis0 + tr)).norm(dim=-1).max() * 1000
        print(f"[static root] fit {n} frames from [{pool_lo},{pool_hi}) (stride {stride})  "
              f"trunk resid p50={float(d.median()):5.1f} "
              f"p95={float(d.quantile(0.95)):5.1f} mm  Δinit-median {float(ang):.1f}° / "
              f"{float((tr - tr0).norm()) * 1000:.1f}mm  (rigid-model check {float(probe):.2f}mm)")
    return go.detach(), tr.detach()


# ── multi-view 2D inputs for the static-root solve ────────────────────────────
def build_root_2d_inputs(silhouette_cameras, mv_rtmo, person_id, N, device, dtype):
    """Assemble the multi-view 2D inputs for solve_static_root. Returns (cams, gt2d_all,
    conf2d_all) — or (None, None, None) if 2D detections / cameras are unavailable (the root
    is then solved from 3D only). silhouette_cameras: {cam:{K,D,R,T,image_size}}; mv_rtmo:
    {cam: per-frame array}, arr[idx][person_id] = {'keypoints'(17,2), 'keypoint_scores'(17,)}."""
    if not silhouette_cameras or not mv_rtmo:
        return None, None, None
    cams, gt2d_all, conf2d_all = {}, {}, {}
    for cam_name, arr in mv_rtmo.items():
        if cam_name not in silhouette_cameras:
            continue
        cams[cam_name] = build_camera_tensors(silhouette_cameras[cam_name], device)
        kp2 = np.zeros((N, 17, 2), dtype=np.float32)   # accumulate on CPU: one upload per cam,
        cf2 = np.zeros((N, 17),    dtype=np.float32)   # not N (matters for the full-video solve)
        for idx in range(N):
            if idx >= len(arr):
                continue
            det = arr[idx].get(person_id) if isinstance(arr[idx], dict) else None
            if isinstance(det, dict) and 'keypoints' in det:
                kp2[idx] = np.asarray(det['keypoints'],       dtype=np.float32)
                cf2[idx] = np.asarray(det['keypoint_scores'], dtype=np.float32)
        gt2d_all[cam_name]   = torch.as_tensor(kp2, dtype=dtype, device=device)
        conf2d_all[cam_name] = torch.as_tensor(cf2, dtype=dtype, device=device)
    return (cams, gt2d_all, conf2d_all) if cams else (None, None, None)
