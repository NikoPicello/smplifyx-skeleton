# -*- coding: utf-8 -*-
"""
Windowed temporal SMPLX fitting — Stage A: batched body + root.

Replaces the per-frame main body solve (fit_single_frame's LBFGS body loop) for the
bulk of a sequence. A window of W frames is optimised JOINTLY from the SMPLer-X warm
start, with a 3D keypoint data term + GMM/angle priors per frame and velocity +
acceleration smoothness COUPLING neighbouring frames.

Why this subsumes the per-frame temporal band-aids: today's temporal term pulls frame
t toward a *frozen* frame t-1 (a constant). Here the coupling is two-sided between
*free* variables, so:
  - an unobserved/low-conf frame is INTERPOLATED from BOTH neighbours (no occlusion
    hold / neck-collar boost needed),
  - drift is removed globally (no frame-0 root/leg anchor needed; the 3D data already
    observes the root absolutely),
  - window seams are C1 by construction — the OVERLAP frames are pinned to the
    previous window's committed solve, and acceleration couples across the seam.

Speed: batching amortises the GPU launch + LBFGS line-search overhead that dominates
batch=1, and we drop the collision BVH + nvdiffrast silhouette here. Windowing is also
what makes a 9000-frame clip fit in memory (a single batch would not).

All tuning lives here as module constants (no cross-module arg plumbing). main.py reads
WIN_SIZE to build the batched model.
"""
from __future__ import absolute_import, print_function, division

import math
import os.path as osp

import numpy as np
import torch

from smplx.lbs import batch_rodrigues   # axis-angle -> rotation matrix (exp map, smooth for all theta)

from utils import aa_nearest
from cvars import LOWER_BODY_POSE_DOFS   # legs + spine DOFs (the seated anchor set)
from fitting import build_camera_tensors, _project_to_pixels   # multi-view 2D reprojection (Stage B)

# ── window geometry ──────────────────────────────────────────────────────────
WIN_SIZE    = 16      # frames optimised jointly (== batched model batch_size). >= N ⇒ full-seq.
WIN_OVERLAP = 8       # boundary frames pinned to the previous window's solve (>=2 ⇒ C1 seam)

# ── temporal smoothness (the new core) ───────────────────────────────────────
# Acceleration > velocity: penalise JERK, not motion, so fast-but-smooth moves aren't damped.
LAMBDA_VEL_BP, LAMBDA_ACC_BP = 20.0,  70.0
LAMBDA_VEL_TR, LAMBDA_ACC_TR = 60.0, 120.0
LAMBDA_VEL_GO, LAMBDA_ACC_GO = 60.0, 350.0

# ── anchors ──────────────────────────────────────────────────────────────────
LAMBDA_LEG  = 5.0     # under-observed seated legs/spine → hold near the SMPLer-X seated ref
LAMBDA_ROOT = 0.0     # 3D data observes the root; raise only if the trajectory drifts
LAMBDA_BND  = 1e3     # pin overlap frames to the previous window's committed solve
LAMBDA_GO_ANCHOR = 20.0  # observability-gated pull of go toward the window's well-observed
                        # orientation; self-gates to 0 on fully-observed frames (p0 untouched).
                        # Stops an underdetermined global_orient (p1: dropped-out arm) drifting.
LAMBDA_BP_STILL  = 60.0  # ABSOLUTE "keep in place" anchor: pull each frame's body_pose toward the
                        # window-mean pose. Smoothness only penalises jerk, so a small steady wobble
                        # survives; this pins the pose to a value and kills it. Raise until the
                        # vibration stops. NOTE it also resists genuine motion, so if the arms lag,
                        # switch its DOF set to the trunk only (exclude shoulders/elbows/wrists).

# ── data / priors ────────────────────────────────────────────────────────────
DATA_RHO     = 0.25   # GMoF scale (metres), matches the per-frame fit
LAMBDA_POSE  = 1.0    # GMM body-pose prior (light: Stage A refines an already-plausible init)
LAMBDA_ANGLE = 0.3    # knee/elbow hyper-extension prior

# Coarse→fine: smooth + regularise first, then let the data sharpen.
STAGE_SCHEDULE = [
    dict(data=100.0, temporal=2.0, lbfgs_steps=2),
    dict(data=150.0, temporal=1.0, lbfgs_steps=2),
    dict(data=200.0, temporal=0.5, lbfgs_steps=2),
]

# ── Stage 0 — betas refinement: fit the SHAPE to the observed bone lengths ────────────────────
# SMPLer-X betas are a good INIT but their limb lengths can be off by several cm (p0's model arm
# was ~8cm shorter than the triangulated one → the elbow could NEVER fit, whatever the pose).
# Bone lengths are pose-INVARIANT (skeleton segments move rigidly), so betas can be fit directly
# to the median observed segment lengths — no pose estimate needed, no chicken-and-egg with
# Stage A. Solved ONCE, anchored to the SMPLer-X init, then shared+frozen for the whole sequence
# (cross-frame consistency preserved). Segments = COCO-17 mapped-joint pairs; each is weighted by
# how often both endpoints are observed, so an unseen limb (p1's Rwrist) just drops out.
# Symmetric groups: L/R pooled into ONE target. SMPL-X's shape space is bilaterally symmetric —
# it CANNOT give the two arms different lengths — so per-side targets over-specify and force the
# fit to chase the impossible. Both people's LEFT arm also triangulates ~3cm shorter than the
# right (systematic bias), so within a pool the samples are conf-weighted: the better-observed
# side arbitrates.
BETAS_SEGMENTS = {
    'upperarm':  [(5, 7), (6, 8)],
    'forearm':   [(7, 9), (8, 10)],
    'shoulders': [(5, 6)],
}
BETAS_LEN_W    = 1e3    # bone-length data term (m² residuals → O(1) loss)
BETAS_RHO      = 0.03   # GMoF (m) on segment residuals: a corrupt target (p1's 21cm L forearm,
                        # its only observed side) SATURATES instead of hijacking the coupled shape
                        # directions and dragging the whole arm short.
BETAS_ANCHOR_W = 0.05    # stay near the SMPLer-X init (good overall shape estimate)
BETAS_STEPS    = 3
BETAS_CONF_THR = 0.1    # a segment endpoint below this conf doesn't count as observed

# ── Stage B — placement: batched multi-view 2D reprojection (refine go + tr only) ─────────────
# Runs AFTER Stage A with body_pose FIXED. Refines global_orient + transl so the body reprojects
# onto the 2D detections in every view, but batched over the window with vel/accel coupling — so
# it stays smooth (the old per-frame mv2d re-jittered the root each frame). Softly anchored to the
# Stage-A go/tr so it refines rather than replaces.
MV2D_RHO_PX     = 50.0     # GMoF scale in PIXELS
MV2D_CONF_FLOOR = 0.3      # ignore 2D detections below this score
MV2D_DATA_W     = 100.0    # brings the focal-normalised reprojection up to O(1) vs the reg terms
MV2D_MAX_ITER   = 20
MV2D_STEPS      = 3        # LBFGS steps per window
LAMBDA_MV_VEL_GO, LAMBDA_MV_ACC_GO = 10.0, 20.0   # keep the refined go smooth
LAMBDA_MV_VEL_TR, LAMBDA_MV_ACC_TR = 10.0, 20.0
# Asymmetric anchor to the Stage-A solve: HOLD go hard (it's already smooth+correct, and per-frame
# 2D noise would only re-jitter it), but anchor tr only weakly so the multi-view 2D can refine the
# depth/position it observes well. (Placement's win is on tr; go should stay = Stage A.)
LAMBDA_MV_GO_ANCHOR = 200.0
LAMBDA_MV_TR_ANCHOR = 100.0
# Trunk-only COCO-17 mask: arm/head detections must NOT drag the root placement.
_MV2D_TRUNK_W = torch.ones(17)
_MV2D_TRUNK_W[[0, 1, 2, 3, 4]]   = 0.1     # nose/eyes/ears — move with the neck
_MV2D_TRUNK_W[[7, 8, 9, 10]]     = 0.05    # elbows/wrists — arm motion must not move the root
_MV2D_TRUNK_W[[13, 14, 15, 16]]  = 0.05    # knees/ankles — template-set, exclude from placement
_MV2D_TRUNK_W[[11, 12]]          = 0.5     # hips

# ── Stage B — hand refinement: batched, refine hand pose + arm REACH only (go/tr fixed) ────────
# Fits the 3D hand keypoints by moving the hand poses + the arm cols (shoulder/elbow/wrist) of
# body_pose that carry the hand into place; go/tr and the rest of body_pose stay FIXED. WiLoR init
# is the anchor + a hand prior regularises, and vel/accel coupling keeps hands smooth across the
# window. (Single coherent batched solve — the per-frame version splits this into place/articulate
# /snap phases; batching + coupling lets one annealed solve do the job.)
HAND_L_KP = list(range(17, 38))            # left  hand keypoints (root + 20 fingers) in the mapping
HAND_R_KP = list(range(38, 59))            # right hand keypoints
_ARM_COLS = [45, 46, 47, 51, 52, 53, 57, 58, 59,     # L shoulder / elbow / wrist body_pose cols
             48, 49, 50, 54, 55, 56, 60, 61, 62]     # R shoulder / elbow / wrist
HAND_RHO0, HAND_RHO1 = 0.15, 0.05          # GMoF scale (m): anneal coarse→fine over the steps
HAND_DATA_W   = 50.0                        # 3D hand-keypoint data weight
HAND_WILOR_W  = 0.8                         # pull hand pose toward the WiLoR init
HAND_PRIOR_W  = 0.1                         # L2 hand-pose prior (plausible fingers)
HAND_ARM_ANCHOR = 0.5                       # keep the arm cols near the Stage-A reach (don't wander)
LAMBDA_HAND_VEL, LAMBDA_HAND_ACC = 5.0, 15.0   # temporal coupling on hand pose + arm cols
HAND_STEPS    = 5
# PLACE sub-phase (DISABLED, kept behind the flag): fit ONLY the arm keypoints with the arm cols
# first, no fingers — was added to escape a local min that turned out to be a SHAPE problem (arm
# too short from the SMPLer-X betas; see Stage 0). With betas corrected, the single mixed solve
# reaches the same arm residuals without it (A/B tested), at half the hand-stage time.
_ARM_KP = [7, 8, 9, 10, 17, 38]             # elbow(7,8) + wrist(9,10) + hand-root(17,38) keypoints
HAND_PLACE_RHO0, HAND_PLACE_RHO1 = 0.20, 0.05
HAND_PLACE_STEPS = 0                        # 0 disables the place phase

# ── Stage B — head refinement: batched, re-aim neck+head + jaw onto the face landmarks ─────────
# Refines the neck+head body_pose cols + jaw_pose to land the 51 inner face landmarks; go/tr and
# the rest of body_pose stay FIXED. Uses the model's mapped face joints (indices 76:127) directly
# as the landmark set. vel/accel coupling + a light anchor to Stage A keep the head smooth.
_HEAD_COLS   = [33, 34, 35, 42, 43, 44]     # neck (joint 11) + head (joint 14) body_pose cols
FACE_KP      = list(range(76, 127))         # 51 inner face landmarks in the mapped-joint layout
# GMoF scale (m), ANNEALED coarse→fine: the face starts ~7cm off, so a fixed tight rho saturates
# the robustifier and starves the gradient; start wide to snap the head in, then tighten to refine.
HEAD_RHO0, HEAD_RHO1 = 0.20, 0.05
HEAD_FACE_W   = 20.0                         # face-landmark data weight
HEAD_JAW_W    = 1.0                          # jaw L2 prior
HEAD_POSE_W   = 0.1                          # keep neck/head near neutral
HEAD_ANCHOR   = 0.1                          # keep neck/head near the Stage-A value (don't wander)
HEAD_EXPR_W   = 1.0                          # L2 expression prior (keep blendshapes plausible)
HEAD_EYE_W    = 1.0                          # L2 eye-pose reg toward neutral (eyes barely rotate)
LAMBDA_HEAD_VEL, LAMBDA_HEAD_ACC = 5.0, 15.0   # temporal coupling on neck/head + jaw + expr + eyes
HEAD_STEPS    = 5

# ── Stage C — offline whole-sequence smoothing (runs AFTER all fitting, before the writer) ─────
# Whittaker–Eilers: x* = argmin Σ|x_t − y_t|² + λ Σ|x_{t+1} − 2 x_t + x_{t-1}|² per channel — the
# same acceleration prior as Stage A but GLOBAL (whole sequence coupled at once: no window seams)
# and with a closed-form banded solve (O(N), milliseconds for 9000 frames). Pure signal processing
# on the finished trajectories: no data-term tension, zero phase lag. The mesh is a deterministic
# function of the params, so smoothing the params smooths the mesh exactly.
# λ dials: higher = smoother but starts damping real motion. λ=0 disables a group.
# Rule of thumb: perceived cutoff ~ (1/λ)^(1/4) of Nyquist — λ 10→gentle, 100→strong, 1000→heavy.
SMOOTH_LAM_BP   = 50.0    # body_pose (63) — the visible body vibration
SMOOTH_LAM_GO   = 200.0   # global_orient — smoothed on the UNWRAPPED (aa_nearest) trajectory
SMOOTH_LAM_TR   = 200.0   # translation
SMOOTH_LAM_HAND = 20.0    # hand poses (fingers move fast; keep light)
SMOOTH_LAM_HEAD = 20.0    # jaw + expression + eyes


def smooth_sequence(y, lam):
    """Whittaker–Eilers smoother over the whole sequence. y: (N, D) tensor; returns (N, D).
    Solves the pentadiagonal SPD system (I + λ D2ᵀD2) x = y once for all D channels."""
    N = y.shape[0]
    if lam <= 0 or N < 3:
        return y
    from scipy.linalg import solveh_banded
    yn = y.detach().cpu().numpy().astype(np.float64)
    d0 = np.ones(N)                       # I  +  λ·(D2ᵀD2) diagonals
    d0[0:N - 2] += lam; d0[1:N - 1] += 4.0 * lam; d0[2:N] += lam
    d1 = np.zeros(N - 1)
    d1[0:N - 2] += -2.0 * lam; d1[1:N - 1] += -2.0 * lam
    d2 = np.full(N - 2, lam)
    ab = np.zeros((3, N))                 # upper banded form for solveh_banded
    ab[0, 2:] = d2
    ab[1, 1:] = d1
    ab[2, :]  = d0
    x = solveh_banded(ab, yn, lower=False)
    return torch.as_tensor(x, dtype=y.dtype, device=y.device)


def smooth_all_outputs(bp, go, tr, lh, rh, jaw, expr, leye, reye):
    """Stage C over every saved trajectory. go is unwrapped (aa_nearest chain) BEFORE smoothing so
    the filter never sees a 2π rep jump; the returned go stays on that continuous branch (the
    writer's saved-output unwrap is a no-op on it). Prints accel before → after per group."""
    gou = _aa_unwrap(go)
    groups = [('body_pose', bp,  SMOOTH_LAM_BP), ('global_orient', gou, SMOOTH_LAM_GO),
              ('transl',    tr,  SMOOTH_LAM_TR), ('left_hand',     lh,  SMOOTH_LAM_HAND),
              ('right_hand', rh, SMOOTH_LAM_HAND), ('jaw',         jaw, SMOOTH_LAM_HEAD),
              ('expr',      expr, SMOOTH_LAM_HEAD), ('leye',       leye, SMOOTH_LAM_HEAD),
              ('reye',      reye, SMOOTH_LAM_HEAD)]
    out = []
    for name, x, lam in groups:
        xs = smooth_sequence(x, lam)
        if name in ('body_pose', 'global_orient', 'transl'):
            print(f"[stageC smooth] {name:13s} λ={lam:6.0f}  accel {_d2(x).item():.2e} → {_d2(xs).item():.2e}")
        out.append(xs)
    return out


# ── helpers ──────────────────────────────────────────────────────────────────
def _d1(x):
    """Mean squared velocity over the window (sum over the last/param dim, mean over time)."""
    return (x[1:] - x[:-1]).pow(2).sum(-1).mean()


def _d2(x):
    """Mean squared acceleration (second difference) over the window."""
    return (x[2:] - 2.0 * x[1:-1] + x[:-2]).pow(2).sum(-1).mean()


def _aa_unwrap(go):
    """Make a (W,3) axis-angle trajectory continuous across the |theta|=pi boundary so the
    vel/accel terms measure the true rotation change, not a spurious ~2*pi vector jump.
    Reuses utils.aa_nearest; the chosen 2*pi*m offset is constant w.r.t. the variable, so
    gradients pass straight through (same trick the per-frame global_orient anchor uses)."""
    rows = [go[:1]]
    for w in range(1, go.shape[0]):
        rows.append(aa_nearest(go[w:w + 1], rows[-1].detach()))
    return torch.cat(rows, dim=0)


def _aa_to_6d(aa):
    """(W,3) axis-angle -> (W,6) 6D rotation rep (first two columns of R), for the smoothness
    term ONLY. AA->matrix is the exponential map (batch_rodrigues), analytic for ALL theta, so
    gradients flow cleanly; the |theta|=pi pathology lives in the INVERSE (matrix->AA) map, which
    we never take. Measuring vel/accel here penalises the TRUE orientation change, free of the
    axis-angle metric distortion + 2*pi wrap near theta=pi. `go` itself stays axis-angle (this is
    a read-only metric), so the saved output needs no conversion and hits no singularity."""
    R = batch_rodrigues(aa)                       # (W,3,3)
    return R[:, :, :2].reshape(aa.shape[0], 6)


TERM_CAP = 1e5   # per-term clamp: keeps one spike finite so the line search can reject it


def _f(x):
    """Safe scalar for logging (avoids the autograd warning from float() on a grad tensor)."""
    return x.item() if torch.is_tensor(x) else float(x)


def _cap(x, cap=TERM_CAP):
    """Scale a loss term down if it exceeds `cap`, preserving gradient direction. Mirrors
    SMPLifyLoss._clamp_term — stops a finite-but-huge term (e.g. the GMM prior when a frame
    wanders out of support) from compounding to inf/nan and breaking the line search."""
    if torch.is_tensor(x):
        v = x.detach()
        if bool(torch.isfinite(v)) and float(v) > cap:
            return x * (cap / float(v))
    return x


# ── Stage 0: fit betas to the observed bone lengths (pose-invariant, once per person) ─────────
def refine_betas_bone_lengths(model_1, betas0, gt_joints_all, conf_all):
    """Refine the shared betas so the model's skeleton matches the OBSERVED bone lengths.
        model_1:      batch-1 SMPLX model (used at zero pose; lengths are pose-invariant)
        betas0:       (1, B) SMPLer-X init — warm start AND anchor
        gt_joints_all (N, J, 3), conf_all (N, J): triangulated keypoints + validity*conf
    Target per symmetric group = conf-WEIGHTED MEDIAN over all (frame, side) samples; residuals
    are GMoF-robustified so one corrupt segment can't hijack the coupled shape directions.
    Returns the refined betas (1, B), detached. Prints per-group model-vs-gt lengths."""
    device, dt = betas0.device, betas0.dtype

    def _wmedian(v, w):
        i = torch.argsort(v)
        v, w = v[i], w[i]
        c = torch.cumsum(w, 0)
        return float(v[min(int(torch.searchsorted(c, c[-1] * 0.5)), len(v) - 1)])

    # Pooled conf-weighted target per group + its share of possible observations (fit weight).
    tgt, wseg, nobs = {}, {}, {}
    for name, pairs in BETAS_SEGMENTS.items():
        vs, ws = [], []
        for a, b in pairs:
            ok = (conf_all[:, a] > BETAS_CONF_THR) & (conf_all[:, b] > BETAS_CONF_THR)
            if int(ok.sum()) == 0:
                continue
            vs.append((gt_joints_all[ok, a] - gt_joints_all[ok, b]).norm(dim=-1))
            ws.append(torch.minimum(conf_all[ok, a], conf_all[ok, b]))
        if not vs:
            continue
        v, w = torch.cat(vs), torch.cat(ws)
        tgt[name]  = _wmedian(v, w)
        wseg[name] = float(len(v)) / (conf_all.shape[0] * len(pairs))
        nobs[name] = len(v)
    if not tgt:
        print("[stage0 betas] no observed segments → keeping SMPLer-X betas")
        return betas0

    zero_bp = torch.zeros(1, 63, dtype=dt, device=device)
    zero_3  = torch.zeros(1, 3,  dtype=dt, device=device)

    def _seg_lengths(b):
        J = model_1(betas=b, body_pose=zero_bp, global_orient=zero_3, transl=zero_3,
                    return_verts=False).joints[0]
        # model is L/R-symmetric; average the pair's sides anyway for numerical safety
        return {n: torch.stack([(J[a] - J[bb]).norm() for a, bb in pairs]).mean()
                for n, pairs in BETAS_SEGMENTS.items() if n in tgt}

    betas = betas0.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([betas], lr=1.0, max_iter=20, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        lens = _seg_lengths(betas)
        L_len = betas.new_zeros(())
        for n in lens:
            d2 = (lens[n] - tgt[n]).pow(2)
            L_len = L_len + wseg[n] * BETAS_RHO ** 2 * d2 / (d2 + BETAS_RHO ** 2)   # GMoF
        L = BETAS_LEN_W * L_len + BETAS_ANCHOR_W * (betas - betas0).pow(2).sum()
        L.backward()
        return L

    with torch.no_grad():
        before = {n: float(v) for n, v in _seg_lengths(betas0).items()}
    for _ in range(BETAS_STEPS):
        opt.step(closure)
    if not bool(torch.isfinite(betas).all()):
        print("[stage0 betas] non-finite result → keeping SMPLer-X betas")
        return betas0
    with torch.no_grad():
        after = {n: float(v) for n, v in _seg_lengths(betas).items()}
    for n in sorted(tgt):
        sat = "  [saturated: target >2*rho from model — likely corrupt]" \
            if abs(after[n] - tgt[n]) > 2 * BETAS_RHO else ""
        print(f"[stage0 betas] {n:10s} tgt={100*tgt[n]:5.1f}cm (n={nobs[n]:2d})  "
              f"model {100*before[n]:5.1f} → {100*after[n]:5.1f}cm{sat}")
    print(f"[stage0 betas] |Δbetas|={float((betas.detach() - betas0).norm()):.3f}")
    return betas.detach()


# ── one window ───────────────────────────────────────────────────────────────
def refine_window_body(model_W, body_pose_prior, angle_prior,
                       gt_joints, weights, betas,
                       bp0, go0, tr0, bp_leg_ref, go_ref, tr_ref,
                       carry=None, frame_lo=0):
    """Jointly fit one window of W frames. Shapes (W == WIN_SIZE, J mapped joints, B betas):
        gt_joints  (W, J, 3)   weights (W, J)    betas (W, B)   [betas frozen, shared]
        bp0        (W, 63)     go0/tr0 (W, 3)                   [SMPLer-X warm start]
        bp_leg_ref (W, |static|)   go_ref/tr_ref (W, 3)         [anchor targets]
        carry: dict(k=LongTensor[O], bp=(O,63), go=(O,3), tr=(O,3)) or None (first window)
    Returns bp, go, tr each (W, ·), detached.
    """
    device = bp0.device
    bp = bp0.clone().requires_grad_(True)
    go = go0.clone().requires_grad_(True)
    tr = tr0.clone().requires_grad_(True)
    static = torch.as_tensor(LOWER_BODY_POSE_DOFS, device=device, dtype=torch.long)

    for si, st in enumerate(STAGE_SCHEDULE):
        opt = torch.optim.LBFGS([bp, go, tr], lr=1.0, max_iter=20,
                                line_search_fn='strong_wolfe')

        def closure(backward=True):
            if backward:
                opt.zero_grad()
            out = model_W(betas=betas, body_pose=bp, global_orient=go, transl=tr,
                          return_verts=False)
            r   = gt_joints - out.joints                                   # (W, J, 3)
            rob = DATA_RHO ** 2 * r.pow(2) / (r.pow(2) + DATA_RHO ** 2)     # GMoF
            L_data = (weights.unsqueeze(-1) ** 2 * rob).sum(dim=(1, 2)).mean() * st['data'] ** 2

            L_pri = (body_pose_prior(bp, betas).mean() * LAMBDA_POSE
                     + angle_prior(bp).sum(-1).mean() * LAMBDA_ANGLE)

            gou = _aa_unwrap(go)    # AA-continuous: still used by the root/seam anchors below
            go6 = _aa_to_6d(go)     # rotation-faithful 6D: drives the smoothness term (no unwrap)
            tw  = st['temporal']
            L_vel = tw * (LAMBDA_VEL_BP * _d1(bp) + LAMBDA_VEL_TR * _d1(tr) + LAMBDA_VEL_GO * _d1(go6))
            L_acc = tw * (LAMBDA_ACC_BP * _d2(bp) + LAMBDA_ACC_TR * _d2(tr) + LAMBDA_ACC_GO * _d2(go6))

            L_leg  = LAMBDA_LEG * (bp[:, static] - bp_leg_ref).pow(2).sum(-1).mean()
            L_root = LAMBDA_ROOT * ((gou - go_ref).pow(2).sum(-1).mean()
                                    + (tr - tr_ref).pow(2).sum(-1).mean())

            # Observability-gated go anchor. Gate = how many body joints THIS frame is missing
            # vs. the joints seen SOMEWHERE in the window: 0 when the frame is as observed as the
            # best (all of p0, and p1's good frames), rising as joints drop out (p1's arm flicker).
            # Target = the observability-weighted mean 6D orientation of the window (DETACHED, so
            # the pull is one-way: sparse frames -> the well-observed consensus, never the reverse).
            n_obs   = (weights > 0).float().sum(1)                            # (W,) observed joints/frame
            n_used  = (weights.max(0).values > 0).float().sum().clamp(min=1)  # joints seen anywhere in win
            gate    = (1.0 - n_obs / n_used).clamp(min=0.0)                   # (W,)
            cw      = (n_obs / n_obs.sum().clamp(min=1)).unsqueeze(1)         # (W,1) trust well-observed
            go6_ref = (go6.detach() * cw).sum(0, keepdim=True)               # (1,6) consensus orientation
            L_goanc = LAMBDA_GO_ANCHOR * (gate.unsqueeze(1) * (go6 - go6_ref).pow(2)).sum(-1).mean()

            # Stillness anchor: pull body_pose toward its OWN window-mean pose (DETACHED reference),
            # an absolute pin that kills the residual per-frame wobble smoothing leaves behind.
            L_still = LAMBDA_BP_STILL * (bp - bp.detach().mean(0, keepdim=True)).pow(2).sum(-1).mean()

            L_bnd = bp.new_zeros(())
            if carry is not None:
                k = carry['k']
                L_bnd = LAMBDA_BND * ((bp[k] - carry['bp']).pow(2).sum()
                                      + (gou[k] - carry['go']).pow(2).sum()
                                      + (tr[k] - carry['tr']).pow(2).sum())

            # Cap each term so a single spike stays finite — strong-Wolfe can then reject the
            # step and back off (as it does for finite spikes) instead of hitting inf/nan.
            L_data, L_pri = _cap(L_data), _cap(L_pri)
            L_vel,  L_acc = _cap(L_vel),  _cap(L_acc)
            L_leg,  L_root, L_bnd, L_goanc = _cap(L_leg), _cap(L_root), _cap(L_bnd), _cap(L_goanc)
            L_still = _cap(L_still)

            total = L_data + L_pri + L_vel + L_acc + L_leg + L_root + L_bnd + L_goanc + L_still
            if backward:
                total.backward()
                torch.nn.utils.clip_grad_norm_([bp, go, tr], 10.0)
                # compact one-line log (fixed columns)
                print(f"  [win f{frame_lo:05d} s{si}] data={_f(L_data):7.3f} pri={_f(L_pri):6.3f} "
                      f"vel={_f(L_vel):6.3f} acc={_f(L_acc):6.3f} leg={_f(L_leg):6.3f} "
                      f"stl={_f(L_still):6.3f} bnd={_f(L_bnd):7.3f} tot={_f(total):7.3f}")
            return total

        # Keep-best (consistent pairing). L-BFGS.step() returns the PRE-step loss, so snapshot
        # the PRE-step params to pair with it (matches fitting.run_fitting). A non-finite step
        # stops the stage. After the loop, score the final post-step state once (no backward)
        # and keep it only if it is finite and actually <= the best snapshot; else restore best.
        best_loss = float('inf')
        best_state = [p.detach().clone() for p in (bp, go, tr)]   # stage-start fallback (finite)
        for _ in range(st['lbfgs_steps']):
            snapshot = [p.detach().clone() for p in (bp, go, tr)]   # pre-step: matches `loss`
            loss = float(opt.step(closure))
            if not (math.isfinite(loss) and all(bool(torch.isfinite(p).all()) for p in (bp, go, tr))):
                print(f"  [win f{frame_lo:05d} s{si}] non-finite step → restoring best, stop stage")
                break
            if loss < best_loss:
                best_loss = loss
                best_state = snapshot
        final_loss = float(closure(backward=False))
        if not (math.isfinite(final_loss) and final_loss <= best_loss):
            with torch.no_grad():
                for p, s in zip((bp, go, tr), best_state):
                    p.data.copy_(s)

    return bp.detach(), go.detach(), tr.detach()


# ── slide windows across the whole sequence ──────────────────────────────────
def run_windowed(model_W, body_pose_prior, angle_prior,
                 gt_joints_all, weights_all, betas1,
                 bp_init, go_init, tr_init, leg_ref_all, go_ref_all, tr_ref_all):
    """Sliding-window Stage A over the full sequence. All *_all tensors are (N, ·) on device;
    betas1 is (1, B) shared+frozen. The final short window is padded to WIN_SIZE (replicated
    last frame, zero data weight, not committed). Returns bp, go, tr each (N, ·).
    """
    N = gt_joints_all.shape[0]
    W, O = WIN_SIZE, WIN_OVERLAP
    bp_out, go_out, tr_out = bp_init.clone(), go_init.clone(), tr_init.clone()
    betasW = betas1.expand(W, -1).contiguous()

    def _pad(x, n):
        return x if n == W else torch.cat([x, x[-1:].expand(W - n, *x.shape[1:])], dim=0)

    carry, start = None, 0
    resid_sum, resid_cnt = 0.0, 0   # guardrail: mean per-joint 3D error over COMMITTED frames
    while start < N:
        end = min(start + W, N)
        n   = end - start
        sl  = slice(start, end)

        w_pad  = weights_all[sl]
        if n < W:   # padded frames carry no data weight
            w_pad = torch.cat([w_pad, w_pad.new_zeros(W - n, w_pad.shape[1])], dim=0)
        gt_pad = _pad(gt_joints_all[sl], n)

        bp_s, go_s, tr_s = refine_window_body(
            model_W, body_pose_prior, angle_prior,
            gt_pad, w_pad, betasW,
            _pad(bp_out[sl], n), _pad(go_out[sl], n), _pad(tr_out[sl], n),
            _pad(leg_ref_all[sl], n), _pad(go_ref_all[sl], n), _pad(tr_ref_all[sl], n),
            carry=carry, frame_lo=start)

        commit_lo = start if carry is None else start + O   # never recommit the overlap

        # Guardrail: smoothness alone can't tell over-damping from good tracking. Report the mean
        # per-joint 3D error (mm) on the frames we actually commit, over OBSERVED joints only.
        with torch.no_grad():
            outj  = model_W(betas=betasW, body_pose=bp_s, global_orient=go_s,
                            transl=tr_s, return_verts=False).joints
            obs   = (w_pad > 0)
            pf_mm = 1000.0 * ((gt_pad - outj).norm(dim=-1) * obs).sum(1) / obs.sum(1).clamp(min=1)
        cw = torch.tensor([w for w in range(n) if start + w >= commit_lo],
                          device=pf_mm.device, dtype=torch.long)
        if len(cw):
            print(f"  [win f{start:05d}] committed {len(cw):2d}  resid={float(pf_mm[cw].mean()):6.1f} mm")
            resid_sum += float(pf_mm[cw].sum()); resid_cnt += int(len(cw))

        for w in range(n):
            f = start + w
            if f >= commit_lo:
                bp_out[f], go_out[f], tr_out[f] = bp_s[w], go_s[w], tr_s[w]

        if end == N:
            break
        start = end - O
        carry = dict(k=torch.arange(O, device=bp_out.device),
                     bp=bp_out[start:start + O].clone(),
                     go=_aa_unwrap(go_out[start:start + O]).clone(),
                     tr=tr_out[start:start + O].clone())

    if resid_cnt:
        # Companion to the residual: output jerk on the full committed trajectory (across seams).
        # global_orient in the SAME 6D metric the loss damps, so it's the knob feedback for
        # LAMBDA_ACC_GO. Resid steady + this dropping = jitter fixed, not over-damped.
        with torch.no_grad():
            bp_acc = _d2(bp_out).item()
            go_acc = _d2(_aa_to_6d(go_out)).item()
            tr_acc = _d2(tr_out).item()
        print(f"[stageA] mean committed residual: {resid_sum / resid_cnt:6.1f} mm  ({resid_cnt} frames)")
        print(f"[stageA] output accel  body_pose={bp_acc:.2e}  global_orient(6D)={go_acc:.2e}  transl={tr_acc:.2e}")
    return bp_out, go_out, tr_out


# ── Stage B: one window of 2D-reprojection placement (refine go + tr, body fixed) ─────────────
def refine_window_placement(model_W, cams, gt2d, conf2d, betas, bp, go0, tr0,
                            go_ref, tr_ref, carry=None, frame_lo=0):
    """Batched multi-view 2D reprojection over one window. body_pose (bp) is FIXED at the Stage-A
    solve; only go, tr move — with vel/accel coupling so the placement stays smooth across the
    window, and a soft anchor to the Stage-A go/tr so it refines rather than replaces.
        cams:   {cam: camera-tensor dict}
        gt2d:   {cam: (W,17,2) px}   conf2d: {cam: (W,17)}
    Returns go, tr each (W,3) detached.
    """
    device = go0.device
    go = go0.clone().requires_grad_(True)
    tr = tr0.clone().requires_grad_(True)
    jw = _MV2D_TRUNK_W.to(device=device, dtype=go.dtype)              # (17,)
    opt = torch.optim.LBFGS([go, tr], lr=1.0, max_iter=MV2D_MAX_ITER, line_search_fn='strong_wolfe')

    def closure(backward=True):
        if backward:
            opt.zero_grad()
        out = model_W(betas=betas, body_pose=bp, global_orient=go, transl=tr, return_verts=False)
        Jb  = out.joints[:, :17, :]                                  # (W,17,3) COCO body, world
        Wn  = Jb.shape[0]
        L_rep = go.new_zeros(())
        for cam_name, cam in cams.items():
            proj, valid = _project_to_pixels(Jb, cam)                # (W*17,2),(W*17,)
            proj  = proj.reshape(Wn, 17, 2)
            valid = valid.reshape(Wn, 17).to(go.dtype)
            f     = cam['K'][0, 0].to(go.dtype)
            r     = gt2d[cam_name] - proj                            # (W,17,2) px
            rob   = MV2D_RHO_PX ** 2 * r.pow(2) / (r.pow(2) + MV2D_RHO_PX ** 2)
            c     = conf2d[cam_name]
            w     = (c * (c >= MV2D_CONF_FLOOR).to(go.dtype) * jw * valid).unsqueeze(-1)   # (W,17,1)
            L_rep = L_rep + (w.pow(2) * rob / (f ** 2)).sum()        # focal-normalised → cross-cam
        L_rep = L_rep * MV2D_DATA_W ** 2

        go6 = _aa_to_6d(go)
        L_smooth = (LAMBDA_MV_VEL_GO * _d1(go6) + LAMBDA_MV_ACC_GO * _d2(go6)
                    + LAMBDA_MV_VEL_TR * _d1(tr) + LAMBDA_MV_ACC_TR * _d2(tr))
        gou   = _aa_unwrap(go)   # only for the seam pin (L_bnd), consistent with carry['go']
        # Anchor go in 6D — rotation-faithful, NO 2*pi branch. Comparing unwrapped gou to the RAW
        # Stage-A go_ref mismatches by ~2*pi near theta=pi (p0 sits there) and explodes the anchor.
        L_anc = (LAMBDA_MV_GO_ANCHOR * (go6 - _aa_to_6d(go_ref)).pow(2).sum(-1).mean()
                 + LAMBDA_MV_TR_ANCHOR * (tr - tr_ref).pow(2).sum(-1).mean())
        L_bnd = go.new_zeros(())
        if carry is not None:
            k = carry['k']
            L_bnd = LAMBDA_BND * ((gou[k] - carry['go']).pow(2).sum() + (tr[k] - carry['tr']).pow(2).sum())

        total = _cap(L_rep) + _cap(L_smooth) + _cap(L_anc) + _cap(L_bnd)
        if backward:
            total.backward()
            torch.nn.utils.clip_grad_norm_([go, tr], 10.0)
            print(f"  [mv2d f{frame_lo:05d}] rep={_f(L_rep):8.4f} smo={_f(L_smooth):7.4f} "
                  f"anc={_f(L_anc):7.4f} bnd={_f(L_bnd):7.4f} tot={_f(total):8.4f}")
        return total

    best_loss = float('inf')
    best_state = [p.detach().clone() for p in (go, tr)]
    for _ in range(MV2D_STEPS):
        snapshot = [p.detach().clone() for p in (go, tr)]
        loss = float(opt.step(closure))
        if not (math.isfinite(loss) and all(bool(torch.isfinite(p).all()) for p in (go, tr))):
            print(f"  [mv2d f{frame_lo:05d}] non-finite step → restoring best, stop")
            break
        if loss < best_loss:
            best_loss = loss; best_state = snapshot
    final = float(closure(backward=False))
    if not (math.isfinite(final) and final <= best_loss):
        with torch.no_grad():
            for p, s in zip((go, tr), best_state):
                p.data.copy_(s)
    return go.detach(), tr.detach()


# ── slide the placement refine across the whole sequence ──────────────────────
def run_windowed_placement(model_W, cams, gt2d_all, conf2d_all, betas1,
                           bp_all, go_all, tr_all):
    """Sliding-window Stage-B placement (2D reprojection) over the sequence. Refines go+tr with
    the Stage-A body_pose FIXED, anchored to the Stage-A go/tr. cams: {cam: tensor dict};
    gt2d_all: {cam:(N,17,2)}, conf2d_all: {cam:(N,17)}. Returns refined go, tr each (N,3)."""
    N = bp_all.shape[0]
    W, O = WIN_SIZE, WIN_OVERLAP
    go_out, tr_out = go_all.clone(), tr_all.clone()
    go_ref_all, tr_ref_all = go_all.clone(), tr_all.clone()          # anchor targets = Stage-A solve
    betasW = betas1.expand(W, -1).contiguous()

    def _pad(x, n):
        return x if n == W else torch.cat([x, x[-1:].expand(W - n, *x.shape[1:])], dim=0)

    carry, start = None, 0
    while start < N:
        end = min(start + W, N); n = end - start; sl = slice(start, end)
        gt2d   = {c: _pad(gt2d_all[c][sl], n) for c in cams}
        conf2d = {c: (conf2d_all[c][sl] if n == W else
                      torch.cat([conf2d_all[c][sl], conf2d_all[c].new_zeros(W - n, 17)], 0)) for c in cams}

        go_s, tr_s = refine_window_placement(
            model_W, cams, gt2d, conf2d, betasW,
            _pad(bp_all[sl], n), _pad(go_out[sl], n), _pad(tr_out[sl], n),
            _pad(go_ref_all[sl], n), _pad(tr_ref_all[sl], n),
            carry=carry, frame_lo=start)

        commit_lo = start if carry is None else start + O
        for w in range(n):
            f = start + w
            if f >= commit_lo:
                go_out[f], tr_out[f] = go_s[w], tr_s[w]
        if end == N:
            break
        start = end - O
        carry = dict(k=torch.arange(O, device=go_out.device),
                     go=_aa_unwrap(go_out[start:start + O]).clone(),
                     tr=tr_out[start:start + O].clone())

    with torch.no_grad():
        print(f"[stageB placement] output accel  global_orient(6D)={_d2(_aa_to_6d(go_out)).item():.2e}  "
              f"transl={_d2(tr_out).item():.2e}")
    return go_out, tr_out


def build_placement_inputs(silhouette_cameras, mv_rtmo, person_id, N, device, dtype):
    """Assemble Stage-B placement inputs from raw pipeline data. Returns (cams, gt2d_all,
    conf2d_all) — or (None, None, None) if 2D detections / cameras are unavailable (placement
    is then skipped). silhouette_cameras: {cam:{K,D,R,T,image_size}}; mv_rtmo: {cam: per-frame
    array}, arr[idx][person_id] = {'keypoints'(17,2), 'keypoint_scores'(17,)}."""
    if not silhouette_cameras or not mv_rtmo:
        return None, None, None
    cams, gt2d_all, conf2d_all = {}, {}, {}
    for cam_name, arr in mv_rtmo.items():
        if cam_name not in silhouette_cameras:
            continue
        cams[cam_name] = build_camera_tensors(silhouette_cameras[cam_name], device)
        kp2 = torch.zeros(N, 17, 2, dtype=dtype, device=device)
        cf2 = torch.zeros(N, 17, dtype=dtype, device=device)
        for idx in range(N):
            if idx >= len(arr):
                continue
            det = arr[idx].get(person_id) if isinstance(arr[idx], dict) else None
            if isinstance(det, dict) and 'keypoints' in det:
                kp2[idx] = torch.as_tensor(np.asarray(det['keypoints'],       dtype=np.float32), device=device)
                cf2[idx] = torch.as_tensor(np.asarray(det['keypoint_scores'], dtype=np.float32), device=device)
        gt2d_all[cam_name], conf2d_all[cam_name] = kp2, cf2
    return (cams, gt2d_all, conf2d_all) if cams else (None, None, None)


# ── Stage B: one window of hand refinement (hand pose + arm reach; go/tr + non-arm body fixed) ─
def refine_window_hands(model_W, left_hand_prior, right_hand_prior,
                        gt_joints, hand_w, betas, bp, go, tr,
                        lh0, rh0, wilor_lh, wilor_rh, carry=None, frame_lo=0):
    """Batched hand refinement over one window. Optimises left/right hand pose (W,45 each) + the
    arm cols of body_pose (shoulder/elbow/wrist) that reach the hand into place; go/tr and the
    non-arm body_pose stay FIXED. Fits the 3D hand keypoints (GMoF, rho annealed) with a WiLoR
    anchor, an L2 hand prior, and vel/accel temporal coupling. Returns lh, rh, bp (W,·) detached."""
    device = bp.device
    arm_idx = torch.as_tensor(_ARM_COLS, device=device, dtype=torch.long)
    bp_fixed = bp.clone()                                    # non-arm cols stay at Stage-A value
    arm      = bp[:, arm_idx].clone().requires_grad_(True)   # (W, 18) arm reach
    arm_ref  = bp[:, arm_idx].detach().clone()
    lh = lh0.clone().requires_grad_(True)                    # (W, 45)
    rh = rh0.clone().requires_grad_(True)

    # ── PLACE phase: position the whole arm from the arm keypoints ONLY (no fingers), so the
    # elbow lands on its keypoint before the finger fit can tug the wrist and stall it. ──
    arm_kp = torch.as_tensor(_ARM_KP, device=device, dtype=torch.long)
    place_opt = torch.optim.LBFGS([arm], lr=1.0, max_iter=20, line_search_fn='strong_wolfe')

    def _place(rho):
        place_opt.zero_grad()
        bpf = bp_fixed.clone(); bpf[:, arm_idx] = arm
        out = model_W(betas=betas, body_pose=bpf, global_orient=go, transl=tr,
                      left_hand_pose=lh, right_hand_pose=rh, return_verts=False)
        d2  = (gt_joints[:, arm_kp] - out.joints[:, arm_kp]).pow(2).sum(-1)
        rob = rho ** 2 * d2 / (d2 + rho ** 2)
        L = ((hand_w[:, arm_kp] ** 2 * rob).sum(1).mean() * HAND_DATA_W ** 2
             + HAND_ARM_ANCHOR * (arm - arm_ref).pow(2).sum(-1).mean()
             + LAMBDA_HAND_VEL * _d1(arm) + LAMBDA_HAND_ACC * _d2(arm))
        L.backward()
        return L

    for si in range(HAND_PLACE_STEPS):
        rho_p = HAND_PLACE_RHO0 * (HAND_PLACE_RHO1 / HAND_PLACE_RHO0) ** (si / max(HAND_PLACE_STEPS - 1, 1))
        place_opt.step(lambda: _place(rho_p))

    opt = torch.optim.LBFGS([arm, lh, rh], lr=1.0, max_iter=20, line_search_fn='strong_wolfe')

    def closure(backward=True, rho=HAND_RHO1):
        if backward:
            opt.zero_grad()
        bpf = bp_fixed.clone()
        bpf[:, arm_idx] = arm
        out = model_W(betas=betas, body_pose=bpf, global_orient=go, transl=tr,
                      left_hand_pose=lh, right_hand_pose=rh, return_verts=False)
        d2  = (gt_joints - out.joints).pow(2).sum(-1)                       # (W, J)
        rob = rho ** 2 * d2 / (d2 + rho ** 2)
        L_data = (hand_w ** 2 * rob).sum(1).mean() * HAND_DATA_W ** 2
        L_wilor = go.new_zeros(())
        if wilor_lh is not None:
            L_wilor = L_wilor + (lh - wilor_lh).pow(2).sum(-1).mean() * HAND_WILOR_W ** 2
        if wilor_rh is not None:
            L_wilor = L_wilor + (rh - wilor_rh).pow(2).sum(-1).mean() * HAND_WILOR_W ** 2
        L_prior = (left_hand_prior(lh).mean() + right_hand_prior(rh).mean()) * HAND_PRIOR_W ** 2
        L_temp  = (LAMBDA_HAND_VEL * (_d1(lh) + _d1(rh) + _d1(arm))
                   + LAMBDA_HAND_ACC * (_d2(lh) + _d2(rh) + _d2(arm)))
        L_arm   = HAND_ARM_ANCHOR * (arm - arm_ref).pow(2).sum(-1).mean()
        L_bnd = go.new_zeros(())
        if carry is not None:
            k = carry['k']
            L_bnd = LAMBDA_BND * ((lh[k] - carry['lh']).pow(2).sum() + (rh[k] - carry['rh']).pow(2).sum()
                                  + (arm[k] - carry['arm']).pow(2).sum())
        total = (_cap(L_data) + _cap(L_wilor) + _cap(L_prior) + _cap(L_temp)
                 + _cap(L_arm) + _cap(L_bnd))
        if backward:
            total.backward()
            torch.nn.utils.clip_grad_norm_([arm, lh, rh], 10.0)
            print(f"  [hand f{frame_lo:05d}] data={_f(L_data):8.3f} wil={_f(L_wilor):6.3f} "
                  f"pri={_f(L_prior):6.3f} tmp={_f(L_temp):6.3f} arm={_f(L_arm):6.3f} tot={_f(total):8.3f}")
        return total

    best_loss = float('inf')
    best_state = [p.detach().clone() for p in (arm, lh, rh)]
    for si in range(HAND_STEPS):
        rho = HAND_RHO0 * (HAND_RHO1 / HAND_RHO0) ** (si / max(HAND_STEPS - 1, 1))   # anneal coarse→fine
        snapshot = [p.detach().clone() for p in (arm, lh, rh)]
        loss = float(opt.step(lambda: closure(rho=rho)))
        if not (math.isfinite(loss) and all(bool(torch.isfinite(p).all()) for p in (arm, lh, rh))):
            print(f"  [hand f{frame_lo:05d}] non-finite step → restoring best, stop")
            break
        if loss < best_loss:
            best_loss = loss; best_state = snapshot
    final = float(closure(backward=False))
    if not (math.isfinite(final) and final <= best_loss):
        with torch.no_grad():
            for p, s in zip((arm, lh, rh), best_state):
                p.data.copy_(s)

    bp_out = bp.clone()
    bp_out[:, arm_idx] = arm.detach()
    return lh.detach(), rh.detach(), bp_out


def run_windowed_hands(model_W, left_hand_prior, right_hand_prior,
                       gt_joints_all, hand_w_all, betas1, bp_all, go_all, tr_all,
                       lh_all, rh_all, wilor_lh_all, wilor_rh_all):
    """Sliding-window Stage-B hand refinement. Refines hand poses + arm reach with go/tr and the
    non-arm body_pose FIXED. Returns lh, rh each (N,45) and the updated bp (N,63)."""
    N = bp_all.shape[0]
    W, O = WIN_SIZE, WIN_OVERLAP
    bp_out, lh_out, rh_out = bp_all.clone(), lh_all.clone(), rh_all.clone()
    betasW = betas1.expand(W, -1).contiguous()

    def _pad(x, n):
        return x if n == W else torch.cat([x, x[-1:].expand(W - n, *x.shape[1:])], dim=0)

    carry, start = None, 0
    while start < N:
        end = min(start + W, N); n = end - start; sl = slice(start, end)
        hw = hand_w_all[sl]
        if n < W:
            hw = torch.cat([hw, hw.new_zeros(W - n, hw.shape[1])], dim=0)
        lh_s, rh_s, bp_s = refine_window_hands(
            model_W, left_hand_prior, right_hand_prior,
            _pad(gt_joints_all[sl], n), hw, betasW,
            _pad(bp_out[sl], n), _pad(go_all[sl], n), _pad(tr_all[sl], n),
            _pad(lh_out[sl], n), _pad(rh_out[sl], n),
            _pad(wilor_lh_all[sl], n), _pad(wilor_rh_all[sl], n),
            carry=carry, frame_lo=start)

        commit_lo = start if carry is None else start + O
        arm_idx = torch.as_tensor(_ARM_COLS, device=bp_out.device, dtype=torch.long)
        for w in range(n):
            f = start + w
            if f >= commit_lo:
                lh_out[f], rh_out[f] = lh_s[w], rh_s[w]
                bp_out[f, arm_idx] = bp_s[w, arm_idx]
        if end == N:
            break
        start = end - O
        carry = dict(k=torch.arange(O, device=bp_out.device),
                     lh=lh_out[start:start + O].clone(),
                     rh=rh_out[start:start + O].clone(),
                     arm=bp_out[start:start + O][:, arm_idx].clone())
    return lh_out, rh_out, bp_out


def build_hand_inputs(init_left_hand_poses, init_right_hand_poses, N, device, dtype):
    """Stack the per-frame WiLoR hand poses into (N,45) tensors (zeros where a frame lacks one).
    Returns (lh_init, rh_init, wilor_lh, wilor_rh) — the init to optimise from and the anchor
    target are both the WiLoR pose. hand_w_all (the 3D keypoint weights) is built in main.py."""
    def _stack_hand(poses):
        out = torch.zeros(N, 45, dtype=dtype, device=device)
        if poses is None:
            return out
        for i in range(N):
            if i < len(poses) and poses[i] is not None:
                out[i] = torch.as_tensor(np.asarray(poses[i], dtype=np.float32),
                                         dtype=dtype, device=device).reshape(-1)[:45]
        return out
    lh_all = _stack_hand(init_left_hand_poses)
    rh_all = _stack_hand(init_right_hand_poses)
    return lh_all.clone(), rh_all.clone(), lh_all, rh_all


def build_face_landmark_embedding(model_folder, gender, faces_tensor, device, dtype):
    """Load the SMPLX barycentric embedding of the 51 inner dlib face landmarks (17-67). Returns
    (lmk_faces_idx (51,), lmk_bary_coords (51,3), body_faces (F,3)) — used to interpolate the TRUE
    face landmarks on the mesh surface (they deform with expression, unlike the static face joints).
    Returns None if the SMPLX_{GENDER}.npz isn't found (head stage then falls back to model joints)."""
    npz = osp.join(osp.expandvars(model_folder), 'smplx', f'SMPLX_{gender.upper()}.npz')
    if not osp.isfile(npz):
        print(f"[stageB head] landmark embedding not found ({npz}) → falling back to model face joints")
        return None
    d = np.load(npz, allow_pickle=True)
    lfi = torch.as_tensor(np.asarray(d['lmk_faces_idx']), dtype=torch.long, device=device)          # (51,)
    lbc = torch.as_tensor(np.asarray(d['lmk_bary_coords'], dtype=np.float32), dtype=dtype, device=device)  # (51,3)
    bfl = faces_tensor.to(device=device).view(-1, 3).long()                                          # (F,3)
    return lfi, lbc, bfl


# ── Stage B: one window of head refinement (neck+head + jaw + expression + eyes) ───────────────
def refine_window_head(model_W, jaw_prior, gt_joints, face_w, betas, bp, go, tr,
                       jaw0, expr0, leye0, reye0, lmk_emb=None, carry=None, frame_lo=0):
    """Batched head refinement over one window. Optimises neck+head body_pose cols + jaw_pose +
    expression + eye pose to land the 51 inner face landmarks; go/tr and the rest of body_pose
    FIXED. If lmk_emb is given the landmarks are interpolated on the MESH SURFACE (barycentric,
    expression-deformable — the true dlib landmarks); else it falls back to the static model face
    joints (76:127). Face data (GMoF, annealed) + jaw/expr/eye L2 priors + neutral head prior +
    vel/accel coupling. Returns jaw, expr, leye, reye (W,·) and updated bp (W,63)."""
    device = bp.device
    hcols    = torch.as_tensor(_HEAD_COLS, device=device, dtype=torch.long)
    fkp      = torch.as_tensor(FACE_KP,    device=device, dtype=torch.long)
    bp_fixed = bp.clone()
    head     = bp[:, hcols].clone().requires_grad_(True)     # (W, 6) neck + head
    head_ref = bp[:, hcols].detach().clone()
    jaw  = jaw0.clone().requires_grad_(True)                 # (W, 3)
    expr = expr0.clone().requires_grad_(True)                # (W, E) expression blendshapes
    leye = leye0.clone().requires_grad_(True)                # (W, 3)
    reye = reye0.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([head, jaw, expr, leye, reye], lr=0.8, max_iter=20,
                            line_search_fn='strong_wolfe')

    use_bary = lmk_emb is not None
    if use_bary:
        _lfi, _lbc, _bfl = lmk_emb
        _tri_idx = _bfl[_lfi]                                 # (51, 3) vertex indices per landmark

    def _model_lmk():   # -> (W, 51, 3) the model's face-landmark positions
        bpf = bp_fixed.clone(); bpf[:, hcols] = head
        out = model_W(betas=betas, body_pose=bpf, global_orient=go, transl=tr, jaw_pose=jaw,
                      expression=expr, leye_pose=leye, reye_pose=reye, return_verts=use_bary)
        if use_bary:
            tri = out.vertices[:, _tri_idx]                   # (W,51,3,3) mesh-surface triangle verts
            return (tri * _lbc.view(1, 51, 3, 1)).sum(dim=2)  # (W,51,3) barycentric landmark
        return out.joints[:, fkp]

    if frame_lo == 0:   # DIAGNOSTIC: model landmarks vs gt (init)
        with torch.no_grad():
            ml = _model_lmk(); m = face_w[:, fkp] > 0
            d = (gt_joints[:, fkp] - ml).norm(dim=-1)
            print(f"  [head-init] {'barycentric' if use_bary else 'model-joint'} lmk  "
                  f"obs/frame={m.sum(1).float().mean().item():.1f}/51")
            if bool(m.any()):
                print(f"  [head-init] init dist mean={1000*d[m].mean().item():.1f}mm max={1000*d[m].max().item():.1f}mm")

    def closure(backward=True, rho=HEAD_RHO1):
        if backward:
            opt.zero_grad()
        mlmk = _model_lmk()
        d2  = (gt_joints[:, fkp] - mlmk).pow(2).sum(-1)                     # (W, 51)
        rob = rho ** 2 * d2 / (d2 + rho ** 2)
        L_face = (face_w[:, fkp] ** 2 * rob).sum(1).mean() * HEAD_FACE_W ** 2
        L_pose = HEAD_POSE_W ** 2 * head.pow(2).sum(-1).mean()
        L_jaw  = jaw_prior(jaw).mean() * HEAD_JAW_W ** 2
        L_expr = HEAD_EXPR_W ** 2 * expr.pow(2).sum(-1).mean()
        L_eye  = HEAD_EYE_W ** 2 * (leye.pow(2).sum(-1).mean() + reye.pow(2).sum(-1).mean())
        L_anc  = HEAD_ANCHOR * (head - head_ref).pow(2).sum(-1).mean()
        L_temp = (LAMBDA_HEAD_VEL * (_d1(head) + _d1(jaw) + _d1(expr) + _d1(leye) + _d1(reye))
                  + LAMBDA_HEAD_ACC * (_d2(head) + _d2(jaw) + _d2(expr) + _d2(leye) + _d2(reye)))
        L_bnd = go.new_zeros(())
        if carry is not None:
            k = carry['k']
            L_bnd = LAMBDA_BND * ((head[k] - carry['head']).pow(2).sum() + (jaw[k] - carry['jaw']).pow(2).sum()
                                  + (expr[k] - carry['expr']).pow(2).sum() + (leye[k] - carry['leye']).pow(2).sum()
                                  + (reye[k] - carry['reye']).pow(2).sum())
        total = (_cap(L_face) + _cap(L_pose) + _cap(L_jaw) + _cap(L_expr) + _cap(L_eye)
                 + _cap(L_anc) + _cap(L_temp) + _cap(L_bnd))
        if backward:
            total.backward()
            torch.nn.utils.clip_grad_norm_([head, jaw, expr, leye, reye], 10.0)
            print(f"  [head f{frame_lo:05d}] face={_f(L_face):8.3f} jaw={_f(L_jaw):6.3f} exp={_f(L_expr):6.3f} "
                  f"eye={_f(L_eye):6.3f} tmp={_f(L_temp):6.3f} tot={_f(total):8.3f}")
        return total

    params = [head, jaw, expr, leye, reye]
    best_loss = float('inf')
    best_state = [p.detach().clone() for p in params]
    for si in range(HEAD_STEPS):
        rho = HEAD_RHO0 * (HEAD_RHO1 / HEAD_RHO0) ** (si / max(HEAD_STEPS - 1, 1))   # anneal coarse→fine
        snapshot = [p.detach().clone() for p in params]
        loss = float(opt.step(lambda: closure(rho=rho)))
        if not (math.isfinite(loss) and all(bool(torch.isfinite(p).all()) for p in params)):
            print(f"  [head f{frame_lo:05d}] non-finite step → restoring best, stop")
            break
        if loss < best_loss:
            best_loss = loss; best_state = snapshot
    final = float(closure(backward=False))
    if not (math.isfinite(final) and final <= best_loss):
        with torch.no_grad():
            for p, s in zip(params, best_state):
                p.data.copy_(s)

    if frame_lo == 0:   # DIAGNOSTIC: face-landmark distance AFTER refinement
        with torch.no_grad():
            ml = _model_lmk(); m = face_w[:, fkp] > 0
            d = (gt_joints[:, fkp] - ml).norm(dim=-1)
            if bool(m.any()):
                print(f"  [head-final] dist mean={1000*d[m].mean().item():.1f}mm max={1000*d[m].max().item():.1f}mm")

    bp_out = bp.clone()
    bp_out[:, hcols] = head.detach()
    return jaw.detach(), expr.detach(), leye.detach(), reye.detach(), bp_out


def run_windowed_head(model_W, jaw_prior, gt_joints_all, face_w_all, betas1,
                      bp_all, go_all, tr_all, jaw_all, expr_all, leye_all, reye_all, lmk_emb=None):
    """Sliding-window Stage-B head refinement (neck+head + jaw + expression + eyes; go/tr + rest of
    body_pose FIXED). Returns jaw (N,3), expr (N,E), leye (N,3), reye (N,3) and updated bp (N,63)."""
    N = bp_all.shape[0]
    W, O = WIN_SIZE, WIN_OVERLAP
    bp_out  = bp_all.clone()
    jaw_out, expr_out = jaw_all.clone(), expr_all.clone()
    leye_out, reye_out = leye_all.clone(), reye_all.clone()
    betasW  = betas1.expand(W, -1).contiguous()
    hcols   = torch.as_tensor(_HEAD_COLS, device=bp_out.device, dtype=torch.long)

    def _pad(x, n):
        return x if n == W else torch.cat([x, x[-1:].expand(W - n, *x.shape[1:])], dim=0)

    carry, start = None, 0
    while start < N:
        end = min(start + W, N); n = end - start; sl = slice(start, end)
        fw = face_w_all[sl]
        if n < W:
            fw = torch.cat([fw, fw.new_zeros(W - n, fw.shape[1])], dim=0)
        jaw_s, expr_s, leye_s, reye_s, bp_s = refine_window_head(
            model_W, jaw_prior, _pad(gt_joints_all[sl], n), fw, betasW,
            _pad(bp_out[sl], n), _pad(go_all[sl], n), _pad(tr_all[sl], n),
            _pad(jaw_out[sl], n), _pad(expr_out[sl], n), _pad(leye_out[sl], n), _pad(reye_out[sl], n),
            lmk_emb=lmk_emb, carry=carry, frame_lo=start)

        commit_lo = start if carry is None else start + O
        for w in range(n):
            f = start + w
            if f >= commit_lo:
                jaw_out[f], expr_out[f] = jaw_s[w], expr_s[w]
                leye_out[f], reye_out[f] = leye_s[w], reye_s[w]
                bp_out[f, hcols] = bp_s[w, hcols]
        if end == N:
            break
        start = end - O
        carry = dict(k=torch.arange(O, device=bp_out.device),
                     head=bp_out[start:start + O][:, hcols].clone(),
                     jaw=jaw_out[start:start + O].clone(),
                     expr=expr_out[start:start + O].clone(),
                     leye=leye_out[start:start + O].clone(),
                     reye=reye_out[start:start + O].clone())
    return jaw_out, expr_out, leye_out, reye_out, bp_out
