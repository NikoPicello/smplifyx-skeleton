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
from smplx.lbs import blend_shapes, vertices2joints   # pelvis(betas) for the static-root reduction

from utils import aa_nearest
from fitting import build_camera_tensors, _project_to_pixels   # multi-view 2D term of the static root

# ── window geometry ──────────────────────────────────────────────────────────
WIN_SIZE    = 16      # frames optimised jointly (== batched model batch_size). >= N ⇒ full-seq.
WIN_OVERLAP = 4       # boundary frames pinned to the previous window's solve (>=2 ⇒ C1 seam)

# ── temporal smoothness (the new core) ───────────────────────────────────────
# Acceleration > velocity: penalise JERK, not motion, so fast-but-smooth moves aren't damped.
LAMBDA_VEL_BP, LAMBDA_ACC_BP = 20.0,  70.0
LAMBDA_VEL_TR, LAMBDA_ACC_TR = 60.0, 120.0
LAMBDA_VEL_GO, LAMBDA_ACC_GO = 60.0, 350.0

# ── anchors ──────────────────────────────────────────────────────────────────
LAMBDA_ROOT = 0.0     # 3D data observes the root; raise only if the trajectory drifts
LAMBDA_BND  = 1e3     # pin overlap frames to the previous window's committed solve
LAMBDA_GO_ANCHOR = 20.0  # observability-gated pull of go toward the window's well-observed
                        # orientation; self-gates to 0 on fully-observed frames (p0 untouched).
                        # Stops an underdetermined global_orient (p1: dropped-out arm) drifting.
LAMBDA_BP_STILL  = 15.0  # ABSOLUTE "keep in place" anchor: pull each frame's body_pose toward the
                        # window-mean pose. Smoothness only penalises jerk, so a small steady wobble
                        # survives; this pins the pose to a value and kills it. Raise until the
                        # vibration stops. NOTE it resists genuine motion — under FREEZE_ROOT the
                        # SPINE carries all trunk motion (leans/slouch), so _SPINE_COLS are EXCLUDED
                        # from this pin (they'd otherwise hold the window-mean posture rigidly).
                        # _HEAD_COLS excluded too: the real head/face 3D data should place it, not
                        # mamma's bp_ref (stale once betas no longer match mamma's own).
_SPINE_COLS = [6, 7, 8, 15, 16, 17, 24, 25, 26]   # spine1/2/3 — free to lean, no stillness pin
# Make neck and head SHARE their bend (full-vector difference — blocks both the one-joint kink
# and the ±70° opposing-twist candy-wrapper that pinched the neck mesh). Deliberately NO
# coupling to the spine (that pulled the chest forward).
LAMBDA_CERV  = 0.1                   # neck ↔ head bend sharing (full-vector difference)

# ── data / priors ────────────────────────────────────────────────────────────
DATA_RHO     = 0.25   # GMoF scale (metres), matches the per-frame fit
LAMBDA_POSE  = 0.5    # GMM body-pose prior (light: Stage A refines an already-plausible init)
LAMBDA_ANGLE = 0.3    # knee/elbow hyper-extension prior

# Coarse→fine: smooth + regularise first, then let the data sharpen.
STAGE_SCHEDULE = [
    dict(data=100.0, temporal=2.0, lbfgs_steps=10),
    dict(data=150.0, temporal=1.0, lbfgs_steps=10),
    dict(data=200.0, temporal=0.5, lbfgs_steps=10),
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
    # shoulder→hip. Not a rigid bone, but ~straight at seated spine curvatures (chord error <1%)
    # and the model side is measured at zero pose. THE missing constraint behind the arched back:
    # SMPLer-X betas gave a torso 7-13cm TOO LONG (model 55cm vs gt ~47cm; p0 has 4768 3D hip
    # obs), so with the shoulders pinned the pelvis sat 7-12cm BELOW the real hips (3D cloud and
    # both cameras agree) — the belly stretched forward, the back arched, and no root solve or
    # spine prior could ever fix it.
    'trunk':     [(5, 11), (6, 12)],
}
BETAS_NSAT     = 50     # samples at which a segment target reaches full fit weight: a robust
                        # median needs ~dozens of SAMPLES, not a high observation RATE — the old
                        # rate-based weight would zero p1's trunk (232 clean samples = 1% of
                        # frames) even though its torso is ~10cm off.
BETAS_LEN_W    = 1e3    # bone-length data term (m² residuals → O(1) loss)
BETAS_RHO      = 0.03   # GMoF (m) on segment residuals: a corrupt target (p1's 21cm L forearm,
                        # its only observed side) SATURATES instead of hijacking the coupled shape
                        # directions and dragging the whole arm short.
BETAS_ANCHOR_W = 0.05    # stay near the SMPLer-X init ALONG the bone-length directions (loose:
                        # the length data must win there — 1.0 under-converged the arm)
BETAS_NULL_W   = 5.0    # anchor ORTHOGONAL to the length Jacobian: 3 length targets constrain
                        # ≤3 of the 16 betas dims; the other 13 (girth — belly, neck thickness)
                        # have NO data and drifted |Δ|≈2σ under the loose anchor (doming abdomen).
BETAS_STEPS    = 3
BETAS_CONF_THR = 0.1    # a segment endpoint below this conf doesn't count as observed

# ── static root: solve ONE (go, tr) for the whole sequence, then FREEZE it ────────────────────
# The subjects are SEATED and their hips are table-occluded in almost every frame (005013/lego:
# p0 Rhip observed 4774/9345 frames, p1's hips ~100/9345), so the per-frame root is placed by the
# SHOULDERS — which genuinely lean (p95 ~8-13cm reaching over the table). Root-vs-spine is then
# underdetermined and the solved root wanders/jitters per frame (p0 even excursed 4.6m during a
# detection dropout). When the hips ARE observed they sit still (p0: 8.7mm median drift over the
# whole 5min video) → the pelvis is truly static. So: fit ONE (go, tr) against the trunk keypoints
# of a strided frame subsample (GMoF → transient leans saturate as outliers → a robust "resting"
# root), freeze it, and let Stage A express all motion through body_pose — leans go to the spine,
# where they anatomically belong. Kills root jitter identically to zero and the excursion failure
# class by construction.
FREEZE_ROOT   = True
ROOT_STRIDE   = 30            # fit every k-th frame (auto-lowered so short clips keep >=WIN_SIZE)
ROOT_TRUNK_KP = [5, 6, 11, 12]   # shoulders + hips (COCO ids in the mapped layout)
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
HAND_STEPS    = 15
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
# COCO head keypoints (nose/eyes/EARS): the 51 inner-face landmarks all lie on the face plane, so
# they position the FACE but leave the skull's rotation about it soft — the head stage was landing
# the face at ~10mm while the ears drifted 30mm+ (neck bent to chase the face). The lateral ears
# pin the skull orientation.
_HEAD_KP     = [0, 1, 2, 3, 4]
# Per-keypoint weights: the EARS exist only to break the face-plane rotation ambiguity, but
# their triangulated targets can carry cm-level bias (p1: 4-5cm, opposing directions — at a
# uniform weight 10 they DICTATED a ~8° skull rotation against the 51 landmarks, parking the
# face at 13mm when the rotation-optimal fit is 4mm). Nose/eyes stay strong; ears tie-break.
_HEAD_KP_W   = [10.0, 10.0, 10.0, 2.5, 2.5]   # nose, L/R eye, L/R ear
HEAD_EAR_RHO = 0.02   # fixed GMoF (m) for the ears (not annealed): an honest ear (~2cm, p0)
                      # sits at peak GMoF influence, a biased one (4-5cm, p1) saturates as an
                      # outlier instead of twisting the skull. Dials if a skull starts tilting
                      # about the face plane: raise ear weight toward 5 / rho toward 0.03.
# GMoF scale (m), ANNEALED coarse→fine: the face starts ~7cm off, so a fixed tight rho saturates
# the robustifier and starves the gradient; start wide to snap the head in, then tighten to refine.
HEAD_RHO0, HEAD_RHO1 = 0.20, 0.05
HEAD_FACE_W   = 60.0                         # face-landmark data weight
HEAD_JAW_W    = 1.0                          # jaw L2 prior
HEAD_POSE_W   = 0.1                          # keep neck/head near neutral
HEAD_ANCHOR   = 0.1                          # keep neck/head near the Stage-A value (don't wander)
HEAD_EXPR_W   = 1.0                          # L2 expression prior (keep blendshapes plausible)
HEAD_EYE_W    = 1.0                          # L2 eye-pose reg toward neutral (eyes barely rotate)
LAMBDA_HEAD_VEL, LAMBDA_HEAD_ACC = 5.0, 15.0 # temporal coupling on neck/head + jaw + expr + eyes
HEAD_STEPS    = 25

# ── static legs: hold the leg DOFs at the LEG_POSE_CAM SMPLer-X seated pose ───────────────────
# The legs have no 3D data (knees/ankles triangulate ~0% of the video, hips rarely) and only ONE
# camera — GB, the back view — sees them. Its per-camera SMPLer-X articulates them well and is
# near-constant over the clip (std <~2° per DOF), and body_pose is parent-relative so the angles
# transfer with no camera transform. main.py writes the median into the init's leg cols
# (load_static_leg_pose); FREEZE_LEGS then zeroes their gradient in Stage A so they never move —
# forced sitting by construction, same philosophy as FREEZE_ROOT. (The old SEATED_POSE template
# stays as the fallback, but its knee X = 74.5° vs the observed ~96-109° was the visible bug.)
# Known residual: GB's hip angles are relative to ITS pelvis estimate, not our frozen root — if
# the thighs visibly miss the GB image, the corrective is a one-shot 2D static-leg solve.
FREEZE_LEGS  = True    # hold the leg cols at their init through Stage A (data-free DOFs)
LEG_POSE_CAM = 'GB'    # the only view that sees the legs
_LEG_COLS  = [0, 1, 2,   3, 4, 5,        # L / R hip body_pose cols
              9, 10, 11, 12, 13, 14,     # L / R knee
              18, 19, 20, 21, 22, 23]    # L / R ankle

# ── Stage C — offline whole-sequence smoothing (runs AFTER all fitting, before the writer) ─────
# Whittaker–Eilers: x* = argmin Σ|x_t − y_t|² + λ Σ|x_{t+1} − 2 x_t + x_{t-1}|² per channel — the
# same acceleration prior as Stage A but GLOBAL (whole sequence coupled at once: no window seams)
# and with a closed-form banded solve (O(N), milliseconds for 9000 frames). Pure signal processing
# on the finished trajectories: no data-term tension, zero phase lag. The mesh is a deterministic
# function of the params, so smoothing the params smooths the mesh exactly.
# λ dials: higher = smoother but starts damping real motion. λ=0 disables a group.
# Rule of thumb: perceived cutoff ~ (1/λ)^(1/4) of Nyquist — λ 10→gentle, 100→strong, 1000→heavy.
SMOOTH_LAM_BP   = 50.0    # body_pose (63) — the visible body vibration
SMOOTH_LAM_LEG  = 1000.0  # leg body_pose cols (_LEG_COLS) — seated legs are near-static and their
                          # only data is 2D; smooth them much harder than the moving upper body
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
        if name == 'body_pose':   # leg cols get their own (much heavier) smoothing
            lcols = torch.as_tensor(_LEG_COLS, device=xs.device, dtype=torch.long)
            xs[:, lcols] = smooth_sequence(x[:, lcols], SMOOTH_LAM_LEG)
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


def aa_angle_deg(a, b):
    """Geodesic angle (deg) between two axis-angle rotations a, b of shape (..., 3)."""
    Ra, Rb = batch_rodrigues(a.reshape(-1, 3)), batch_rodrigues(b.reshape(-1, 3))
    tr = (Ra.transpose(1, 2) @ Rb).diagonal(dim1=1, dim2=2).sum(-1)
    return torch.rad2deg(torch.arccos(((tr - 1) / 2).clamp(-1.0, 1.0)))


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
        wseg[name] = min(1.0, float(len(v)) / BETAS_NSAT)   # count-saturated, not rate-based
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

    # Split the anchor by the length Jacobian at betas0: the fit may move freely ALONG the
    # directions the bone lengths actually constrain (row space of J), but girth & co. (the
    # null space — no data) stay pinned to SMPLer-X.
    b0g = betas0.clone().requires_grad_(True)
    Jrows = [torch.autograd.grad(v, b0g, retain_graph=True)[0].reshape(-1)
             for v in _seg_lengths(b0g).values()]
    Q, _ = torch.linalg.qr(torch.stack(Jrows).t())        # (B, G) orthonormal length directions
    P_len = (Q @ Q.t()).detach()                          # projector onto the length subspace

    betas = betas0.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([betas], lr=1.0, max_iter=20, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        lens = _seg_lengths(betas)
        L_len = betas.new_zeros(())
        for n in lens:
            d2 = (lens[n] - tgt[n]).pow(2)
            L_len = L_len + wseg[n] * BETAS_RHO ** 2 * d2 / (d2 + BETAS_RHO ** 2)   # GMoF
        d     = (betas - betas0).reshape(-1)
        d_par = P_len @ d
        L = (BETAS_LEN_W * L_len + BETAS_ANCHOR_W * d_par.pow(2).sum()
             + BETAS_NULL_W * (d - d_par).pow(2).sum())
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
    with torch.no_grad():
        d = (betas - betas0).reshape(-1)
        d_par = P_len @ d
    print(f"[stage0 betas] |Δbetas| along-lengths={float(d_par.norm()):.3f}  "
          f"null(girth)={float((d - d_par).norm()):.3f}")
    return betas.detach()


# ── static root: ONE (go, tr) for the whole sequence (FREEZE_ROOT) ─────────────────────────────
def solve_static_root(model_W, betas1, bp_all, go_all, tr_all, gt_joints_all, weights_all,
                      cams=None, gt2d_all=None, conf2d_all=None):
    """Solve the single frozen root. Rigid-root reduction: with body_pose fixed, changing only
    (go, tr) moves every output joint by  j → R(go)·(j₀ − pelvis₀) + pelvis₀ + tr,  where j₀ are
    the joints at go=0/tr=0 (precomputed once, chunked through the batch-W model) and pelvis₀
    depends on betas alone — so the optimisation loop never runs the body model.
        bp_all/go_all/tr_all: (N,·) SMPLer-X init (bp gives each frame's articulation; go/tr give
        the warm start via their strided component-wise median, go unwrapped first).
    Data: 3D trunk keypoints (GMoF, rho ANNEALED ROOT_RHO0→1) on every stride-th frame + the
    multi-view 2D trunk reprojection (_ROOT_2D_W mask, GMoF). Returns (go, tr), each (1,3)
    detached."""
    device, dt = go_all.device, go_all.dtype
    N, W = gt_joints_all.shape[0], WIN_SIZE
    tkp = torch.as_tensor(ROOT_TRUNK_KP, device=device, dtype=torch.long)

    stride = max(1, min(ROOT_STRIDE, N // W))          # short clips: use every frame
    idx = torch.arange(0, N, stride, device=device)
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

    def closure(backward=True, rho3=ROOT_RHO1, rho2=ROOT_RHO_PX1):
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
        per = '  '.join(f"{nm}={float(dj[wkp[:, i] > 0, i].median()):.0f}"
                        for i, nm in enumerate(('Lsho', 'Rsho', 'Lhip', 'Rhip'))
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
        print(f"[static root] fit {n} frames (stride {stride})  trunk resid p50={float(d.median()):5.1f} "
              f"p95={float(d.quantile(0.95)):5.1f} mm  Δinit-median {float(ang):.1f}° / "
              f"{float((tr - tr0).norm()) * 1000:.1f}mm  (rigid-model check {float(probe):.2f}mm)")
    return go.detach(), tr.detach()


# ── static legs: the seated leg pose from the LEG_POSE_CAM per-camera SMPLer-X ────────────────
def load_static_leg_pose(smpler_folder, person_id, device, dtype):
    """Median leg body_pose cols (_LEG_COLS) + median global_orient from the LEG_POSE_CAM
    SMPLer-X export ({smpler_folder}/{LEG_POSE_CAM}_smplx.npy, arr[i] = {person_id:
    {'body_pose'(21,3), 'global_orient'(1,3), ...}}). body_pose is parent-relative, so the
    articulation transfers with no camera transform; the median over frames is the robust static
    pose (per-DOF std is <~2° on these clips). The global_orient (CAMERA frame) is returned for
    align_hips_to_root — the hip angles only mean what they meant under SMPLer-X's OWN pelvis.
    Returns (leg (18,), go (3,)) tensors, or (None, None) if unavailable (caller keeps the
    SEATED_LEGS template)."""
    if not smpler_folder:
        return None, None
    p = osp.join(smpler_folder, f'{LEG_POSE_CAM}_smplx.npy')
    if not osp.isfile(p):
        print(f"[static legs] no {p} → keeping the SEATED_LEGS template")
        return None, None
    arr = np.load(p, allow_pickle=True)
    dets = [fr[person_id] for fr in arr
            if isinstance(fr, dict) and isinstance(fr.get(person_id), dict)
            and fr[person_id].get('body_pose') is not None
            and fr[person_id].get('global_orient') is not None]
    if not dets:
        print(f"[static legs] no person {person_id} in {p} → keeping the SEATED_LEGS template")
        return None, None
    bps = np.stack([np.asarray(d['body_pose'], np.float32).reshape(-1) for d in dets])
    gos = np.stack([np.asarray(d['global_orient'], np.float32).reshape(-1) for d in dets])
    med = torch.as_tensor(np.median(bps, 0), dtype=dtype, device=device)
    leg = med[torch.as_tensor(_LEG_COLS, dtype=torch.long, device=device)]
    go  = torch.as_tensor(np.median(gos, 0), dtype=dtype, device=device)
    hx, kx = torch.rad2deg(leg[[0, 3]]).tolist(), torch.rad2deg(leg[[6, 9]]).tolist()
    print(f"[static legs] {LEG_POSE_CAM} median over {len(dets)} frames  "
          f"hipX L/R={hx[0]:.0f}/{hx[1]:.0f}°  kneeX L/R={kx[0]:.0f}/{kx[1]:.0f}°")
    return leg, go


def align_hips_to_root(leg_pose, gb_go, R_cam, go_static):
    """Transport the LEG_POSE_CAM hip angles under OUR solved root. SMPLer-X's hip rotations are
    relative to ITS OWN pelvis, and monocular from one view it resolves the seated pelvis-pitch
    vs hip-flexion ambiguity its own way (40-47° from the multi-view root on 005013/lego).
    Pasting those local angles under a different pelvis rotates the whole lower body by that
    delta, so keep the WORLD thigh orientation GB saw instead:
        H_ours = D · H_gb,   D = R(go_static)ᵀ · R_camᵀ · R(gb_go)
    Knees/ankles are chain-relative and ride along unchanged. The matrix→axis-angle inverse is
    safe here: hip angles sit far from the θ=π singularity.
        leg_pose (18,) RAW loader output      gb_go (3,) loader median global_orient (cam frame)
        R_cam (3,3) world→cam extrinsic of LEG_POSE_CAM   go_static (1,3) solved root (world)
    Returns the corrected (18,) leg pose (new tensor). Call with the RAW leg_pose each time —
    the correction is absolute, not incremental."""
    import cv2
    rod = lambda v: cv2.Rodrigues(np.asarray(v, np.float64).reshape(3, 1))[0]
    D = (rod(go_static.detach().cpu().numpy().ravel()).T
         @ np.asarray(R_cam, np.float64).T
         @ rod(gb_go.detach().cpu().numpy().ravel()))
    out = leg_pose.detach().clone()
    leg = leg_pose.detach().cpu().numpy().astype(np.float64)
    for i in (0, 3):                                             # L / R hip triplets
        out[i:i + 3] = torch.as_tensor(cv2.Rodrigues(D @ rod(leg[i:i + 3]))[0].ravel(),
                                       dtype=leg_pose.dtype, device=leg_pose.device)
    ang = np.degrees(np.arccos(np.clip((np.trace(D) - 1) / 2, -1.0, 1.0)))
    hx = torch.rad2deg(out[[0, 3]]).tolist()
    print(f"[static legs] hips re-aligned to the solved root  (pelvis delta {ang:.1f}°, "
          f"hipX L/R → {hx[0]:.0f}/{hx[1]:.0f}°)")
    return out


# ── one window ───────────────────────────────────────────────────────────────
def refine_window_body(model_W, body_pose_prior, angle_prior,
                       gt_joints, weights, betas,
                       bp0, go0, tr0, go_ref, tr_ref,
                       bp_ref=None, carry=None, frame_lo=0):
    """Jointly fit one window of W frames. Shapes (W == WIN_SIZE, J mapped joints, B betas):
        gt_joints  (W, J, 3)   weights (W, J)    betas (W, B)   [betas frozen, shared]
        bp0        (W, 63)     go0/tr0 (W, 3)                   [SMPLer-X warm start]
        go_ref/tr_ref (W, 3)                                    [anchor targets]
        bp_ref     (W, 63) or None   [stillness-anchor target; None -> window's own mean (default)]
        carry: dict(k=LongTensor[O], bp=(O,63), go=(O,3), tr=(O,3)) or None (first window)
    Returns bp, go, tr each (W, ·), detached.
    """
    device = bp0.device
    bp = bp0.clone().requires_grad_(True)
    # FREEZE_ROOT: go/tr are constants (the pre-solved static root) — body_pose carries all motion.
    go = go0.clone().requires_grad_(not FREEZE_ROOT)
    tr = tr0.clone().requires_grad_(not FREEZE_ROOT)
    params = [bp] if FREEZE_ROOT else [bp, go, tr]
    leg_idx = torch.as_tensor(_LEG_COLS, device=device, dtype=torch.long)
    still_w = torch.ones(63, dtype=bp0.dtype, device=device)   # stillness mask: spine + head excluded
    still_w[torch.as_tensor(_SPINE_COLS + _HEAD_COLS, device=device, dtype=torch.long)] = 0.0

    for si, st in enumerate(STAGE_SCHEDULE):
        opt = torch.optim.LBFGS(params, lr=1.0, max_iter=20,
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
                     + angle_prior(bp).sum(-1).mean() * LAMBDA_ANGLE
                     + LAMBDA_CERV * (bp[:, 42:45] - bp[:, 33:36]).pow(2).sum(-1).mean())

            gou = _aa_unwrap(go)    # AA-continuous: still used by the root/seam anchors below
            go6 = _aa_to_6d(go)     # rotation-faithful 6D: drives the smoothness term (no unwrap)
            tw  = st['temporal']
            L_vel = tw * (LAMBDA_VEL_BP * _d1(bp) + LAMBDA_VEL_TR * _d1(tr) + LAMBDA_VEL_GO * _d1(go6))
            L_acc = tw * (LAMBDA_ACC_BP * _d2(bp) + LAMBDA_ACC_TR * _d2(tr) + LAMBDA_ACC_GO * _d2(go6))

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

            # Stillness anchor: pull body_pose toward a reference pose, an absolute pin that kills
            # the residual per-frame wobble smoothing leaves behind. Default reference is the
            # window's OWN mean pose (DETACHED, self-consistency only, no data); when a real
            # per-frame reference is available (bp_ref — e.g. mamma's occlusion-gated pose for
            # this window) anchor to THAT instead, so genuine motion isn't clamped to a constant.
            # Spine cols masked out (still_w): the spine is the motion carrier under FREEZE_ROOT.
            still_ref = bp.detach().mean(0, keepdim=True) if bp_ref is None else bp_ref
            L_still = LAMBDA_BP_STILL * ((bp - still_ref).pow(2) * still_w).sum(-1).mean()

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
            L_root, L_bnd, L_goanc = _cap(L_root), _cap(L_bnd), _cap(L_goanc)
            L_still = _cap(L_still)

            total = L_data + L_pri + L_vel + L_acc + L_root + L_bnd + L_goanc + L_still
            if backward:
                total.backward()
                if FREEZE_LEGS:
                    bp.grad[:, leg_idx] = 0.0   # no 3D leg data; hold the seated init (FREEZE_LEGS)
                torch.nn.utils.clip_grad_norm_(params, 10.0)
                # compact one-line log (fixed columns)
                print(f"  [win f{frame_lo:05d} s{si}] data={_f(L_data):7.3f} pri={_f(L_pri):6.3f} "
                      f"vel={_f(L_vel):6.3f} acc={_f(L_acc):6.3f} "
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
                 bp_init, go_init, tr_init, go_ref_all, tr_ref_all,
                 bp_ref_all=None):
    """Sliding-window Stage A over the full sequence. All *_all tensors are (N, ·) on device;
    betas1 is (1, B) shared+frozen. The final short window is padded to WIN_SIZE (replicated
    last frame, zero data weight, not committed). bp_ref_all (N, 63) or None: per-frame
    stillness-anchor reference (e.g. mamma's body_pose); None falls back to each window's own
    mean (see refine_window_body). Returns bp, go, tr each (N, ·).
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
        bp_ref_pad = None if bp_ref_all is None else _pad(bp_ref_all[sl], n)

        bp_s, go_s, tr_s = refine_window_body(
            model_W, body_pose_prior, angle_prior,
            gt_pad, w_pad, betasW,
            _pad(bp_out[sl], n), _pad(go_out[sl], n), _pad(tr_out[sl], n),
            _pad(go_ref_all[sl], n), _pad(tr_ref_all[sl], n),
            bp_ref=bp_ref_pad, carry=carry, frame_lo=start)

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
    place_opt = torch.optim.LBFGS([arm], lr=1.0, max_iter=25, line_search_fn='strong_wolfe')

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
    hkp      = torch.as_tensor(_HEAD_KP,   device=device, dtype=torch.long)
    kpw      = torch.as_tensor(_HEAD_KP_W, device=device, dtype=torch.long)
    bp_fixed = bp.clone()
    head     = bp[:, hcols].clone().requires_grad_(True)     # (W, 6) neck + head
    head_ref = bp[:, hcols].detach().clone()
    jaw  = jaw0.clone().requires_grad_(True)                 # (W, 3)
    expr = expr0.clone().requires_grad_(True)                # (W, E) expression blendshapes
    leye = leye0.clone().requires_grad_(True)                # (W, 3)
    reye = reye0.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([head, jaw, expr, leye, reye], lr=1.0, max_iter=50,
                            line_search_fn='strong_wolfe')

    use_bary = lmk_emb is not None
    if use_bary:
        _lfi, _lbc, _bfl = lmk_emb
        _tri_idx = _bfl[_lfi]                                 # (51, 3) vertex indices per landmark

    def _model_lmk():   # -> ((W,51,3) face landmarks, (W,5,3) nose/eyes/ears joints)
        bpf = bp_fixed.clone(); bpf[:, hcols] = head
        out = model_W(betas=betas, body_pose=bpf, global_orient=go, transl=tr, jaw_pose=jaw,
                      expression=expr, leye_pose=leye, reye_pose=reye, return_verts=use_bary)
        if use_bary:
            tri = out.vertices[:, _tri_idx]                   # (W,51,3,3) mesh-surface triangle verts
            lmk = (tri * _lbc.view(1, 51, 3, 1)).sum(dim=2)   # (W,51,3) barycentric landmark
        else:
            lmk = out.joints[:, fkp]
        return lmk, out.joints[:, hkp]

    def _diag(tag):     # DIAGNOSTIC: model landmarks + head keypoints vs gt
        with torch.no_grad():
            ml, mk = _model_lmk()
            m, mk_m = face_w[:, fkp] > 0, face_w[:, hkp] > 0
            d  = (gt_joints[:, fkp] - ml).norm(dim=-1)
            dk = (gt_joints[:, hkp] - mk).norm(dim=-1)
            if bool(m.any()):
                print(f"  [{tag}] lmk dist mean={1000*d[m].mean().item():.1f}mm max={1000*d[m].max().item():.1f}mm"
                      + (f"  head-kp mean={1000*dk[mk_m].mean().item():.1f}mm" if bool(mk_m.any()) else ""))

    if frame_lo == 0:
        print(f"  [head-init] {'barycentric' if use_bary else 'model-joint'} lmk  "
              f"obs/frame={(face_w[:, fkp] > 0).sum(1).float().mean().item():.1f}/51")
        _diag('head-init')

    def closure(backward=True, rho=HEAD_RHO1):
        if backward:
            opt.zero_grad()
        mlmk, mkp = _model_lmk()
        d2  = (gt_joints[:, fkp] - mlmk).pow(2).sum(-1)                     # (W, 51)
        rob = rho ** 2 * d2 / (d2 + rho ** 2)
        L_face = (face_w[:, fkp] ** 2 * rob).sum(1).mean() * HEAD_FACE_W ** 2
        d2k  = (gt_joints[:, hkp] - mkp).pow(2).sum(-1)                     # (W, 5)
        rhok = d2k.new_full((5,), rho)
        rhok[3:] = HEAD_EAR_RHO             # ears: fixed tight rho — biased targets saturate
        robk = rhok ** 2 * d2k / (d2k + rhok ** 2)
        L_kp = (face_w[:, hkp] ** 2 * kpw ** 2 * robk).sum(1).mean()
        # cervical sharing (see LAMBDA_CERV): neck and head carry the look-down TOGETHER — the
        # full-vector difference blocks both the one-joint kink and the opposing-twist candy-
        # wrapper. Deliberately NO coupling to the spine (that pulled the chest forward).
        L_pose = (HEAD_POSE_W ** 2 * head.pow(2).sum(-1).mean()
                  + LAMBDA_CERV * (head[:, 3:6] - head[:, 0:3]).pow(2).sum(-1).mean())
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
        total = (_cap(L_face) + _cap(L_kp) + _cap(L_pose) + _cap(L_jaw) + _cap(L_expr) + _cap(L_eye)
                 + _cap(L_anc) + _cap(L_temp) + _cap(L_bnd))
        if backward:
            total.backward()
            torch.nn.utils.clip_grad_norm_([head, jaw, expr, leye, reye], 10.0)
            print(f"  [head f{frame_lo:05d}] face={_f(L_face):8.3f} kp={_f(L_kp):6.3f} jaw={_f(L_jaw):6.3f} "
                  f"exp={_f(L_expr):6.3f} eye={_f(L_eye):6.3f} tmp={_f(L_temp):6.3f} tot={_f(total):8.3f}")
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

    if frame_lo == 0:   # DIAGNOSTIC: landmark + head-keypoint distance AFTER refinement
        _diag('head-final')

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
