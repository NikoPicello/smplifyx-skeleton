#!/usr/bin/env python3
"""Prototype: lift a SINGLE-CAMERA 2D hip detection (RTMO) into a 3D point, instead of
requiring the 2-camera agreement that plain triangulation needs and that hips almost
never get in seated/table sessions (body.npy IS RTMO, triangulated -- a hip with <2
confident views just comes out as confidence=0/NaN; see BETAS_SEGMENTS's dropped
'trunk' pair and ROOT_TRUNK_KP in temporal_window.py for the existing workarounds).

Geometry: ray-sphere intersection, i.e. trilateration with one leg degenerated to a
camera ray instead of a second sphere.
    - C, u        : camera center + unit ray direction through the undistorted 2D hip
                    pixel (world space).
    - S           : the SAME-SIDE shoulder position (5=Lsho/11=Lhip, 6=Rsho/12=Rhip --
                    same pairing as the commented-out 'trunk' entry in BETAS_SEGMENTS),
                    taken straight from body.npy -- shoulders triangulate fine, they're
                    essentially never table-occluded.
    - L           : the same-side shoulder->hip length, robust-median from whatever
                    triangulated (shoulder,hip) pairs exist in this clip (usually few --
                    that's the whole problem this script is working around).
Solve |C + t*u - S| = L for t: a quadratic, two roots. Checked empirically against
session 001080/lego_task (both people, GB+GF): the perpendicular distance from the ray
to the TRUE triangulated hip is 4-9mm median (i.e. the ray itself, and the calibration/
frame-alignment behind it, is essentially exact) -- but the true point sits at t_far
(median 229.0/235.7cm vs. a true depth of 228.6/237.2cm), NOT t_near (162.8/189.3cm,
off by 40-65cm). So this picks the LARGER positive root. My first instinct here was
"the camera sees the near surface, take the near root" -- that reasoning was simply
wrong for this rig's geometry (the ray's closest approach to the shoulder falls short
of the body's true depth, so the near root is a phantom point in front of the torso,
not an anatomical one); trust the data over the intuition. This can still flip on a
near-tangent ray (both roots close together, or on a rig with a very different
camera-subject arrangement than this one); that's what the two-hip refinement below is
for, and why every call also returns BOTH roots for the caller to sanity-check against.

If BOTH hips are visible in the SAME camera frame, don't lift them independently --
jointly refine (t_L, t_R) against trunk_L, trunk_R AND the (also robust-median)
hip-width. Hip-width is what actually disambiguates a near/far root mix-up: pairing the
wrong root on either side blows up the hip-width residual almost every time.

Validation: for the rare frames where body.npy's hip triangulation IS confident, the
lift is computed WITHOUT ever looking at that value, so comparing the two is a true
holdout check, not a fit-then-compare.

Usage:
    python hip_lift_prototype.py --sid 001080 --activity lego_task --person-id 0
"""
import argparse
import os.path as osp

import cv2 as cv
import numpy as np
from scipy.optimize import least_squares

_SCRIPT_DIR = osp.dirname(osp.abspath(__file__))
_RESOURCES  = osp.normpath(osp.join(_SCRIPT_DIR, '..', '..', 'resources'))
CALIBS_ROOT = osp.join(_RESOURCES, 'calibs')
SESS_ROOT   = osp.join(_RESOURCES, 'all_sessions')
TRIG_ROOT   = osp.join(_RESOURCES, 'triangulation_results')
RTMO_ROOT   = osp.join(_RESOURCES, 'rtmo_results')

# raw calib filename -> canonical camera name (matches visualization/vis_fit_on_video.py)
CAM_MAP = {'GC': 'GB', 'HC': 'GF', 'Z1': 'FC1', 'Z2': 'FC2', 'N1': 'HA1', 'N2': 'HA2'}

# COCO-17 indices (data_parser.py get_model2data / vis_fit_on_video.py MV2D_DRAW_JOINTS)
LSHO, RSHO, LHIP, RHIP = 5, 6, 11, 12
HIP_PAIRS = {'L': (LSHO, LHIP), 'R': (RSHO, RHIP)}

CONF_FLOOR_2D = 0.3   # mirrors MV2D_CONF_FLOOR / ROOT_CONF_FLOOR in temporal_window.py
CONF_FLOOR_3D = 0.0   # any observed (>0) 3D sample counts as INPUT to the lift (shoulders:
                      # empirically fine even at this floor, see the GF reprojection check --
                      # shoulders are rarely near the noise floor the way hips are)
VALID_CONF_3D = 0.1   # mirrors BETAS_CONF_THR in temporal_window.py -- the bar for a hip
                      # triangulation to count as trustworthy GROUND TRUTH for validation.
                      # Keep this separate from CONF_FLOOR_3D: a hip sample at conf=0.02
                      # (2-view, cm-scale sigma_m) is not a fair thing to validate the lift
                      # against, even though it's ">0".


# ── loading ────────────────────────────────────────────────────────────────────────
def load_calib(session_id):
    with open(osp.join(SESS_ROOT, session_id, 'session_data.txt')) as f:
        calib_date = f.readlines()[1][11:].strip()
    calib_dir = osp.join(CALIBS_ROOT, calib_date)
    cams = {}
    for raw_name, cam_name in CAM_MAP.items():
        p = osp.join(calib_dir, f'{raw_name}.yml')
        if not osp.isfile(p):
            continue
        fs = cv.FileStorage(p, cv.FILE_STORAGE_READ)
        cams[cam_name] = dict(K=fs.getNode('K').mat(), D=fs.getNode('D').mat(),
                               R=fs.getNode('R').mat(), T=fs.getNode('T').mat().reshape(3))
        fs.release()
    return cams


def load_body_npy(session_id, activity, person_id):
    p = osp.join(TRIG_ROOT, session_id, activity, f'p{person_id}', 'body.npy')
    raw = np.load(p, allow_pickle=True).item()
    return {k: v for k, v in raw.items() if isinstance(k, int)}   # {fidx: {kpts_3d, confidence}}


def load_rtmo(session_id, activity, cam_name):
    p = osp.join(RTMO_ROOT, session_id, activity, f'{cam_name}_rtmo.npy')
    return np.load(p, allow_pickle=True) if osp.isfile(p) else None


# ── bone-length priors from whatever triangulation this clip has ───────────────────
def weighted_median(values, weights):
    values, weights = np.asarray(values, dtype=np.float64), np.asarray(weights, dtype=np.float64)
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cw = np.cumsum(weights)
    return float(values[np.searchsorted(cw, cw[-1] / 2.0)])


def bone_length_stats(body_frames):
    """Confidence-weighted median same-side shoulder->hip length + hip width."""
    out = {}
    for side, (sho, hip) in HIP_PAIRS.items():
        lens, ws = [], []
        for fr in body_frames.values():
            cs, ch = fr['confidence'][sho], fr['confidence'][hip]
            if cs <= CONF_FLOOR_3D or ch <= CONF_FLOOR_3D:
                continue
            ps, ph = fr['kpts_3d'][sho], fr['kpts_3d'][hip]
            if not (np.isfinite(ps).all() and np.isfinite(ph).all()):
                continue
            lens.append(np.linalg.norm(ps - ph)); ws.append(min(cs, ch))
        out[side] = (weighted_median(lens, ws), len(lens)) if lens else (None, 0)
    widths, ws = [], []
    for fr in body_frames.values():
        cl, cr = fr['confidence'][LHIP], fr['confidence'][RHIP]
        if cl <= CONF_FLOOR_3D or cr <= CONF_FLOOR_3D:
            continue
        pl, pr = fr['kpts_3d'][LHIP], fr['kpts_3d'][RHIP]
        if not (np.isfinite(pl).all() and np.isfinite(pr).all()):
            continue
        widths.append(np.linalg.norm(pl - pr)); ws.append(min(cl, cr))
    out['hip_width'] = (weighted_median(widths, ws), len(widths)) if widths else (None, 0)
    return out


CROSS_PAIRS = {'L': (RSHO, LHIP), 'R': (LSHO, RHIP)}   # hip side -> (OPPOSITE shoulder, this hip)


def cross_length_stats(body_frames):
    """Confidence-weighted median CROSS shoulder->hip length (opposite side). The second,
    independent anchor for dual_anchor_lift below -- same provenance/threshold logic as
    bone_length_stats, just paired with the other shoulder."""
    out = {}
    for side, (sho, hip) in CROSS_PAIRS.items():
        lens, ws = [], []
        for fr in body_frames.values():
            cs, ch = fr['confidence'][sho], fr['confidence'][hip]
            if cs <= CONF_FLOOR_3D or ch <= CONF_FLOOR_3D:
                continue
            ps, ph = fr['kpts_3d'][sho], fr['kpts_3d'][hip]
            if not (np.isfinite(ps).all() and np.isfinite(ph).all()):
                continue
            lens.append(np.linalg.norm(ps - ph)); ws.append(min(cs, ch))
        out[side] = (weighted_median(lens, ws), len(lens)) if lens else (None, 0)
    return out


# ── geometry ─────────────────────────────────────────────────────────────────────────
def camera_ray(px, py, K, D, R, T):
    """World-space (camera_center, unit_ray_direction) through pixel (px,py)."""
    pts = np.array([[[px, py]]], dtype=np.float64)
    norm = cv.undistortPoints(pts, K, D).reshape(2)      # normalized ideal-camera coords
    d_cam = np.array([norm[0], norm[1], 1.0])
    d_cam /= np.linalg.norm(d_cam)
    d_world = R.T @ d_cam
    C_world = -R.T @ T
    return C_world, d_world


def camera_ray_batch(pts, K, D, R, T):
    """Vectorized camera_ray: pts (N,2) pixel coords -> C (3,) shared camera center,
    U (N,3) unit ray directions -- one cv.undistortPoints call instead of N."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    norm = cv.undistortPoints(pts, K, D).reshape(-1, 2)
    d_cam = np.concatenate([norm, np.ones((norm.shape[0], 1))], axis=1)
    d_cam /= np.linalg.norm(d_cam, axis=1, keepdims=True)
    d_world = d_cam @ R              # row-vector form of R.T @ d_cam per row
    C_world = -R.T @ T
    return C_world, d_world


def ray_sphere_lift(C, u, S, L):
    """LARGER positive t with |C + t*u - S| == L (see module docstring for why: verified
    empirically against 001080/lego_task, the far root matches the true hip depth to
    within ~1-2cm while the near root misses by 40-65cm on this rig). Returns
    (point, t_chosen, disc, t_near, t_far) so callers can sanity-check both roots."""
    v = C - S
    b = v @ u
    c = v @ v - L * L
    disc = b * b - c
    if disc < 0:
        return None, None, disc, None, None
    sq = np.sqrt(disc)
    t_near, t_far = -b - sq, -b + sq
    for t in (t_far, t_near):     # prefer far; fall back to near only if far is behind the camera
        if t > 0:
            return C + t * u, t, disc, t_near, t_far
    return None, None, disc, t_near, t_far


def ray_sphere_lift_batch(C, U, S, L):
    """Vectorized ray_sphere_lift: C (3,), U (N,3), S (N,3), L scalar -> t (N,) far root,
    NaN where disc<0 or neither root is positive."""
    v = C[None, :] - S
    b = np.einsum('ij,ij->i', v, U)
    c = np.einsum('ij,ij->i', v, v) - L * L
    disc = b * b - c
    sq = np.sqrt(np.clip(disc, 0, None))
    t_far, t_near = -b + sq, -b - sq
    t = np.where(t_far > 0, t_far, np.where(t_near > 0, t_near, np.nan))
    return np.where(disc >= 0, t, np.nan)


def dual_anchor_lift(C, u, S_same, len_same, S_cross, len_cross):
    """Resolve the single-ray depth ambiguity with BOTH shoulders as independent distance
    anchors instead of one sphere + an empirically-found near/far preference: the
    cross-side shoulder sits off to the side, so its own sphere generically crosses the
    ray near only ONE of the same-side sphere's two roots -- this should self-disambiguate
    rather than leaning on a rig-specific rule. Single unknown (t along the ray), two
    residuals -> a well-conditioned 1D least-squares, initialized from the same-side
    closed-form far root (falls back to closest-approach if that sphere doesn't intersect
    the ray at all)."""
    _, t0, _, _, _ = ray_sphere_lift(C, u, S_same, len_same)
    if t0 is None:
        t0 = float((S_same - C) @ u)

    def resid(x):
        P = C + x[0] * u
        return [np.linalg.norm(P - S_same) - len_same,
                np.linalg.norm(P - S_cross) - len_cross]

    sol = least_squares(resid, x0=[t0])
    t = float(sol.x[0])
    return C + t * u, t, sol


def dual_anchor_lift_batch(C, U, S_same, len_same, S_cross, len_cross, t0, n_iter=8):
    """Vectorized Gauss-Newton version of dual_anchor_lift for many (frame, ray) pairs
    against ONE camera at once -- a per-frame scipy.optimize.least_squares call is far too
    slow to run over a real video (tried it: didn't finish one session in several minutes),
    and this needs to be fast regardless since a from-scratch fitting pipeline this size
    would need it batched anyway. Closed form per Newton step (u is unit, so
    |C+t*u-S|^2 = t^2 + 2*b*t + c with b=(C-S)@u, c=|C-S|^2 -- same algebra as
    ray_sphere_lift, just kept symbolic in t instead of solved for the zero-residual case).
        C: (3,) one camera center. U: (N,3) unit ray directions, one per frame.
        S_same, S_cross: (N,3) per-frame shoulder positions. len_same/len_cross: scalars.
        t0: (N,) initial guess (e.g. from the closed-form same-side far root).
    Returns t: (N,) refined depths (NOT clipped to positive -- caller should check t>0)."""
    v1, v2 = C[None, :] - S_same, C[None, :] - S_cross
    b1, c1 = np.einsum('ij,ij->i', v1, U), np.einsum('ij,ij->i', v1, v1)
    b2, c2 = np.einsum('ij,ij->i', v2, U), np.einsum('ij,ij->i', v2, v2)
    t = t0.copy()
    for _ in range(n_iter):
        s1 = np.sqrt(np.clip(t * t + 2 * b1 * t + c1, 1e-9, None))
        s2 = np.sqrt(np.clip(t * t + 2 * b2 * t + c2, 1e-9, None))
        g, h = s1 - len_same, s2 - len_cross
        gp, hp = (t + b1) / s1, (t + b2) / s2           # dg/dt, dh/dt
        grad = g * gp + h * hp                           # (1/2) dF/dt
        gn   = gp * gp + hp * hp                          # Gauss-Newton curvature approx
        t = t - grad / np.clip(gn, 1e-9, None)
    return t


def joint_two_hip_lift(ray_L, ray_R, S_L, S_R, trunk_L, trunk_R, hip_width):
    (C_L, u_L), (C_R, u_R) = ray_L, ray_R

    def t0_guess(C, u, S, L):
        _, t, disc, _, _ = ray_sphere_lift(C, u, S, L)
        return t if t is not None else float((S - C) @ u)   # closest-approach fallback

    x0 = [t0_guess(C_L, u_L, S_L, trunk_L), t0_guess(C_R, u_R, S_R, trunk_R)]

    def resid(x):
        tL, tR = x
        PL, PR = C_L + tL * u_L, C_R + tR * u_R
        return [np.linalg.norm(PL - S_L) - trunk_L,
                np.linalg.norm(PR - S_R) - trunk_R,
                np.linalg.norm(PL - PR) - hip_width]

    sol = least_squares(resid, x0=x0)
    tL, tR = sol.x
    return C_L + tL * u_L, C_R + tR * u_R, sol


def shoulder_sanity_check(cams, rtmo_by_cam, body_frames, person_id):
    """Reproject body.npy's (reliably-triangulated) shoulders into each camera and compare
    to that camera's own RTMO shoulder detection for the SAME frame index. Shoulders are
    rarely occluded on either side, so a large error here means the camera's calib and/or
    the body.npy<->RTMO frame alignment is off for THIS session -- a lift built on that
    camera's rays would inherit the same error for free, well before the hip geometry is
    even in play. Cheap to run before trusting any lift numbers below."""
    print("\n  shoulder reprojection sanity check (body.npy -> camera vs. that camera's own RTMO):")
    for cam_name, arr in sorted(rtmo_by_cam.items()):
        cam = cams[cam_name]
        rvec, _ = cv.Rodrigues(cam['R'])
        errs = []
        for fidx, fr in body_frames.items():
            det = arr[fidx].get(person_id) if fidx < len(arr) and isinstance(arr[fidx], dict) else None
            if det is None:
                continue
            for sho in (LSHO, RSHO):
                if fr['confidence'][sho] <= CONF_FLOOR_3D or det['keypoint_scores'][sho] < CONF_FLOOR_2D:
                    continue
                S = fr['kpts_3d'][sho]
                if not np.isfinite(S).all():
                    continue
                proj, _ = cv.projectPoints(S.reshape(1, 3), rvec, cam['T'].reshape(3, 1),
                                            cam['K'], cam['D'])
                errs.append(np.linalg.norm(proj.reshape(2) - det['keypoints'][sho]))
        if errs:
            errs = np.array(errs)
            flag = "  <-- suspect calib/alignment for this camera" if np.median(errs) > 15 else ""
            print(f"    {cam_name:5s}: n={len(errs):4d}  median={np.median(errs):5.1f}px  "
                  f"p90={np.percentile(errs, 90):5.1f}px{flag}")
        else:
            print(f"    {cam_name:5s}: no overlapping confident shoulder samples to check")


# ── driver ───────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sid', default='001080')
    ap.add_argument('--activity', default='lego_task')
    ap.add_argument('--person-id', type=int, default=0)
    ap.add_argument('--conf-floor-2d', type=float, default=CONF_FLOOR_2D)
    args = ap.parse_args()

    cams = load_calib(args.sid)
    body_frames = load_body_npy(args.sid, args.activity, args.person_id)
    bone = bone_length_stats(body_frames)
    print(f"[{args.sid}/{args.activity}/p{args.person_id}]  triangulated frames available: {len(body_frames)}")
    for k, (val, n) in bone.items():
        print(f"  {k:10s}: {val*100:.1f}cm  (n={n})" if val is not None else f"  {k:10s}: NO SAMPLES")
    if bone['L'][0] is None or bone['R'][0] is None:
        print("  -> can't derive both trunk lengths from this clip's own triangulation, aborting.")
        return
    trunk = {'L': bone['L'][0], 'R': bone['R'][0]}
    hip_width = bone['hip_width'][0]   # may be None -> single-hip lifts only, no joint refine

    rtmo_by_cam = {c: load_rtmo(args.sid, args.activity, c) for c in cams}
    rtmo_by_cam = {c: v for c, v in rtmo_by_cam.items() if v is not None}
    n_frames = max((len(v) for v in rtmo_by_cam.values()), default=0)
    print(f"\n  cameras with RTMO: {sorted(rtmo_by_cam)}   (video length ~{n_frames} frames)")

    shoulder_sanity_check(cams, rtmo_by_cam, body_frames, args.person_id)

    # per-camera visibility diagnostic: how often is the person even detected vs. how
    # often does a given hip clear the confidence floor -- this is the "which camera
    # actually sees the hips" question, made concrete.
    print("\n  per-camera hip visibility (person present / Lhip>=floor / Rhip>=floor):")
    for cam_name, arr in sorted(rtmo_by_cam.items()):
        present = lhip_ok = rhip_ok = 0
        for fr in arr:
            det = fr.get(args.person_id) if isinstance(fr, dict) else None
            if det is None:
                continue
            present += 1
            if det['keypoint_scores'][LHIP] >= args.conf_floor_2d:
                lhip_ok += 1
            if det['keypoint_scores'][RHIP] >= args.conf_floor_2d:
                rhip_ok += 1
        print(f"    {cam_name:5s}: present={present:5d}  Lhip_ok={lhip_ok:5d}  Rhip_ok={rhip_ok:5d}")

    # lift every (frame, camera, side) that clears the 2D floor AND has a confident
    # same-side 3D shoulder in body.npy for that frame.
    lifts = {}   # (fidx, side) -> list of (cam_name, score, point, t, disc)
    for cam_name, arr in rtmo_by_cam.items():
        cam = cams[cam_name]
        for fidx, fr in enumerate(arr):
            det = fr.get(args.person_id) if isinstance(fr, dict) else None
            if det is None:
                continue
            body_fr = body_frames.get(fidx)
            if body_fr is None:
                continue
            for side, (sho, hip) in HIP_PAIRS.items():
                score = det['keypoint_scores'][hip]
                if score < args.conf_floor_2d:
                    continue
                cs = body_fr['confidence'][sho]
                S = body_fr['kpts_3d'][sho]
                if cs <= CONF_FLOOR_3D or not np.isfinite(S).all():
                    continue
                px, py = det['keypoints'][hip]
                C, u = camera_ray(px, py, cam['K'], cam['D'], cam['R'], cam['T'])
                P, t, disc, _, _ = ray_sphere_lift(C, u, S, trunk[side])
                if P is None:
                    continue
                lifts.setdefault((fidx, side), []).append((cam_name, float(score), P, t, disc))

    # collapse multi-camera duplicates to the highest-confidence camera (a full
    # multi-view bundle is future work, not needed to validate the single-view idea)
    best = {}
    for key, cands in lifts.items():
        best[key] = max(cands, key=lambda c: c[1])

    frames_with_lift = {f for (f, _s) in best}
    frames_with_gt_hip = {f for f, fr in body_frames.items()
                          if fr['confidence'][LHIP] > CONF_FLOOR_3D or fr['confidence'][RHIP] > CONF_FLOOR_3D}
    print(f"\n  frames with >=1 lifted hip:            {len(frames_with_lift)} / {len(body_frames)} triangulated frames in range")
    print(f"  frames with a genuinely observed 3D hip: {len(frames_with_gt_hip)} (confidence>0 either side)")

    # joint two-hip refine wherever both sides came from the SAME camera this frame
    refined = {}
    if hip_width is not None:
        for fidx in frames_with_lift:
            if (fidx, 'L') in best and (fidx, 'R') in best:
                cam_l, _, _, _, _ = best[(fidx, 'L')]
                cam_r, _, _, _, _ = best[(fidx, 'R')]
                if cam_l != cam_r:
                    continue
                cam = cams[cam_l]
                arr = rtmo_by_cam[cam_l]
                det = arr[fidx][args.person_id]
                pxL, pyL = det['keypoints'][LHIP]
                pxR, pyR = det['keypoints'][RHIP]
                rayL = camera_ray(pxL, pyL, cam['K'], cam['D'], cam['R'], cam['T'])
                rayR = camera_ray(pxR, pyR, cam['K'], cam['D'], cam['R'], cam['T'])
                S_L, S_R = body_frames[fidx]['kpts_3d'][LSHO], body_frames[fidx]['kpts_3d'][RSHO]
                if not (np.isfinite(S_L).all() and np.isfinite(S_R).all()):
                    continue
                PL, PR, sol = joint_two_hip_lift(rayL, rayR, S_L, S_R, trunk['L'], trunk['R'], hip_width)
                refined[fidx] = (PL, PR, sol)
        print(f"  frames refined jointly (both hips, same camera): {len(refined)}")

    # ── validation: compare against the RARE frames where triangulation itself clears a
    # meaningful confidence bar (VALID_CONF_3D, not just >0). The lift never sees this
    # value -- true holdout, not a fit check.
    print(f"\n  validation vs. body.npy's own confident (>{VALID_CONF_3D}) triangulated hip:")
    for side, (_, hip) in HIP_PAIRS.items():
        all_conf = [fr['confidence'][hip] for fr in body_frames.values() if fr['confidence'][hip] > 0]
        errs = []
        for fidx, fr in body_frames.items():
            if fr['confidence'][hip] < VALID_CONF_3D or not np.isfinite(fr['kpts_3d'][hip]).all():
                continue
            key = (fidx, side)
            if key not in best:
                continue
            gt = fr['kpts_3d'][hip]
            P = best[key][2]
            errs.append(np.linalg.norm(P - gt) * 1000.0)
        if errs:
            errs = np.array(errs)
            print(f"    {side}hip: n={len(errs)}  median={np.median(errs):.0f}mm  "
                  f"p90={np.percentile(errs, 90):.0f}mm  max={errs.max():.0f}mm")
        else:
            rng = f"[{min(all_conf):.3f},{max(all_conf):.3f}]" if all_conf else "no nonzero samples"
            print(f"    {side}hip: no sample in this clip clears conf>{VALID_CONF_3D} "
                  f"(observed nonzero-confidence range: {rng}) -- this clip can't validate "
                  f"accuracy for this side, only coverage")

    # a few concrete examples, single-ray and (if any) jointly-refined
    print("\n  example single-ray lifts:")
    for i, (key, (cam_name, score, P, t, disc)) in enumerate(sorted(best.items())[:5]):
        fidx, side = key
        print(f"    frame {fidx:4d} {side}hip  cam={cam_name}  score={score:.2f}  t={t*100:.1f}cm  "
              f"P={np.round(P, 3).tolist()}")
    if refined:
        print("\n  example jointly-refined (both hips) lifts:")
        for i, (fidx, (PL, PR, sol)) in enumerate(sorted(refined.items())[:5]):
            print(f"    frame {fidx:4d}  PL={np.round(PL, 3).tolist()}  PR={np.round(PR, 3).tolist()}  "
                  f"width_now={np.linalg.norm(PL - PR)*100:.1f}cm (target {hip_width*100:.1f}cm)  "
                  f"resid_norm={np.linalg.norm(sol.fun):.4f}")


if __name__ == '__main__':
    main()
