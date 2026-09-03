# SMPL-X Fitting Pipeline — Technical Documentation

This document describes the **method and implementation** of the fitting pipeline in this
repository, at a level of detail intended for direct adaptation into a technical report
(Overleaf/LaTeX). It complements `README.md`, which covers installation and day-to-day usage
only. Every claim below was verified against the source in this repository as of the state
checked; file names, function names and constants are quoted verbatim so they can be
traced back to the code (`file.py:function_or_line`).

## Contents

1. [Overview](#1-overview)
2. [Repository map](#2-repository-map)
3. [Inputs and data representation](#3-inputs-and-data-representation)
4. [Body model parameterization](#4-body-model-parameterization)
5. [Fitting method](#5-fitting-method)
   - [5.1 Stage 0 — shape from bone lengths](#51-stage-0--shape-from-bone-lengths)
   - [5.2 Static root solve](#52-static-root-solve)
   - [5.3 Static legs](#53-static-legs)
   - [5.4 Stage A — windowed body fit](#54-stage-a--windowed-body-fit)
   - [5.5 Stage B — hands](#55-stage-b--hands)
   - [5.6 Stage B — head and face](#56-stage-b--head-and-face)
   - [5.7 Stage C — whole-sequence smoothing](#57-stage-c--whole-sequence-smoothing)
6. [Loss function reference](#6-loss-function-reference)
7. [Optimization mechanics](#7-optimization-mechanics)
8. [External reference fusion ("mamma")](#8-external-reference-fusion-mamma)
9. [Configuration schema](#9-configuration-schema)
10. [Orchestration and tooling](#10-orchestration-and-tooling)
11. [Legacy components (not exercised by the live pipeline)](#11-legacy-components-not-exercised-by-the-live-pipeline)
12. [Known inconsistencies and open items](#12-known-inconsistencies-and-open-items)

---

## 1. Overview

This pipeline fits [SMPL-X](https://smpl-x.is.tue.mpg.de/) body meshes — one mesh per person
per video frame — to a multi-view recording of two seated people interacting at a table. It
is a substantial evolution of [SMPLify-X](https://github.com/vchoutas/smplify-x): the
repository still contains SMPLify-X's original per-frame optimization vocabulary
(`fitting.py`, `optimizers/`, the loss-weight schedules in `cmd_parser.py`), but the pipeline
that actually runs today (`main.py` → `temporal_window.py`) replaces the per-frame solve with
a **batched, sliding-window temporal optimization**: instead of fitting each frame
independently and pulling frame *t* toward a frozen frame *t-1*, whole windows of frames are
optimized jointly, with two-sided velocity/acceleration coupling between neighboring frames'
free variables. Section 11 documents the legacy path for completeness, but the pipeline
described in Sections 4–8 is what produces the shipped output.

**Inputs**, per (session, activity, person):
- 3D keypoints (body, hands, face) triangulated from a calibrated multi-camera rig.
- Per-frame [SMPLer-X](https://github.com/caizhongang/SMPLer-X) pose estimates, used to
  initialize `body_pose`/`global_orient`/`transl`/`betas`.
- Per-camera 2D detections from RTMO, used only in the static-root solve (Section 5.2).
- A WiLoR hand-pose estimate, used as a soft anchor in the hand stage (Section 5.5).
- Optionally, an independent reference SMPL-X fit ("mamma", Section 8) over the same rig,
  used as a warm start and a temporal anchor, not as a hard constraint.

**Output**, per frame: a full SMPL-X parameter set (`betas`, `body_pose`, `left_hand_pose`,
`right_hand_pose`, `jaw_pose`, `expression`, `leye_pose`, `reye_pose`, `global_orient`,
`transl`) written as one JSON line to `body_smplx.json`, plus (optionally) a triangulated mesh
`.obj` per frame.

**Design constraints that shape the method.** The subjects are seated at a table for the
whole session: hips are frequently occluded (single view or none), the pelvis barely
translates, and the shoulders/torso genuinely lean toward the table. These observations,
documented directly in the code as design rationale, motivate three of the pipeline's
central decisions: (a) shape (`betas`) is fit once from pose-invariant bone lengths rather
than left to the pose solve to absorb, (b) global orientation and translation are solved
**once** for the entire session and frozen, so that all visible trunk motion is expressed
through the spine joints of `body_pose` instead of an underdetermined, occlusion-driven root
trajectory, and (c) leg pose is held at a static per-camera template because the legs are
essentially never triangulated in 3D.

---

## 2. Repository map

```
smplifyx-skeleton/
├── main.py                 # entry point: single (session, activity, person) fit
├── fitter_pipeline.py       # orchestration: discovers sessions/activities/persons, calls main.py in-process
├── run_parallel_sessions.py # subprocess-based multi-GPU scheduler around fitter_pipeline.py
├── cmd_parser.py            # configargparse schema (CLI flags + YAML config keys)
├── cvars.py                 # body-layout constants, seated-pose template, keypoint-gating constants
├── utils.py                 # GMoF robustifier, axis-angle utilities, JointMapper
├── data_parser.py           # dataset loaders (CustomDataset — production; ADT — benchmark/legacy)
├── mamma_loader.py           # loader/stitcher for the external "mamma" reference fit
├── prior.py                  # pose/shape priors (GMM, L2, angle prior)
├── fitting.py                 # legacy per-frame SMPLify-X-style optimizer and loss (§11)
├── temporal_window.py         # the live fitting engine — windowed/batched optimization (§5)
├── optimizers/                # legacy custom L-BFGS + strong-Wolfe line search (§11)
├── export_kit_amass.py         # body_smplx.json → KIT-AMASS .npz converter
├── visualization/              # interactive viewer, video overlay renderer, joint-mapping reference
├── cfg_files/                  # m1/m2/m3 production configs + an orphaned ablation grid (§9, §12)
├── priors/gmm_08.pkl            # pretrained 8-component GMM body-pose prior
├── models/smplx/                 # SMPL-X body model files (downloaded separately)
└── dependencies/                  # vendored smplx, human_body_prior (VPoser), torch-mesh-isect
```

The data the pipeline reads and writes (`resources/`) lives two directory levels above this
package and is shared with sibling pipelines in the same project (see `README.md` for the
full directory layout).

---

## 3. Inputs and data representation

### 3.1 Triangulated keypoints

`data_parser.CustomDataset` (the dataset class used in production) reads four `.npy` files
per person from the triangulation output directory: `body.npy`, `left_hand.npy`,
`right_hand.npy`, `face.npy`. Each stores a per-frame dict of `{'kpts_3d': (K,3),
'confidence': (K,)}`, densified into an `(N, K, 4)` array of `[x, y, z, confidence]`; any
point with a NaN coordinate or non-positive confidence is zeroed. A fifth file, `smpl.npy`,
holds the per-frame SMPLer-X pose estimate (`body_pose`, `global_orient`, `transl`) and a
global `betas` vector, already fused into the triangulation output by an upstream step. As a
consistency step, the body array's wrist keypoint is overwritten with the corresponding hand
file's wrist estimate whenever the hand file's own wrist confidence is higher, and vice
versa.

### 3.2 Keypoint layout ("data space")

All keypoint arrays are concatenated into one fixed column layout, indexed consistently
throughout the pipeline:

| Index range | Content | Source |
|---|---|---|
| 0–16 | Body, COCO-17 order (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles) | triangulated body |
| 17–37 | Left hand, 21 points (wrist + 20 finger joints) | triangulated left hand |
| 38–58 | Right hand, 21 points | triangulated right hand |
| 59–75 | dlib jaw/contour landmarks (17 points), present but not weighted by any loss term | triangulated face |
| 76–126 | 51 inner dlib face landmarks (eyebrows, eyes, nose, mouth) | triangulated face |

The mapping from this data-space layout back onto SMPL-X's native joint indices is built by
`CustomDataset.get_model2data()` and applied via `utils.JointMapper`, which performs a fixed
`torch.index_select` on the SMPL-X model's output joints so that model joints and ground-truth
keypoints line up index-for-index in every loss term.

### 3.3 Per-frame confidence and gap-filling

Before optimization, short dropouts (a keypoint missing for a few consecutive frames) are
linearly interpolated across the gap (`cvars.KP_FILL_MAX_GAP`, at a reduced confidence
`KP_FILL_CONF`) so that brief detector flicker does not have to be absorbed by the temporal
smoothness terms. Per-keypoint confidence is then sharpened by an exponent
(`cvars.KP_CONF_POWER = 2.0`) before being squared again inside the GMoF loss (i.e. an
effective confidence exponent of 4), so uncertain keypoints contribute disproportionately
less to the data term.

### 3.4 Camera calibration

Multi-view camera parameters (intrinsics, distortion, extrinsics) are loaded from OpenCV
`FileStorage` calibration files and used in two places: the optional 2D-reprojection term
inside the static-root solve (Section 5.2), and the visualization/rendering tools (Section
10.4). Cameras are referenced through a fixed logical-name mapping between the raw camera
serials/tags and the model used in code and configuration.

---

## 4. Body model parameterization

The body is [SMPL-X](https://smpl-x.is.tue.mpg.de/): a differentiable function mapping shape
and pose parameters to a triangulated mesh and a set of 3D joints, via linear blend skinning
on top of learned shape/pose blendshapes. The parameters fit by this pipeline are:

| Parameter | Dimensionality | Meaning |
|---|---|---|
| `betas` | 10 or 16 (config-dependent) | Body shape, in the model's learned PCA shape space; shared and frozen across an entire session per person |
| `body_pose` | 63 (21 joints × 3 axis-angle) | Torso/limb articulation, excluding hands |
| `global_orient` | 3 (axis-angle) | Root/pelvis orientation in world space |
| `transl` | 3 | Root/pelvis translation in world space |
| `left_hand_pose`, `right_hand_pose` | 45 each (15 joints × 3), full axis-angle (PCA disabled — `use_pca: False`) | Finger articulation |
| `jaw_pose` | 3 | Jaw articulation |
| `expression` | 10 (default) | Facial blendshape coefficients |
| `leye_pose`, `reye_pose` | 3 each | Eyeball orientation |

Two instances of the SMPL-X model are built for each run: a batch-size-1 instance used only
for the final per-frame mesh export, and a second instance built with
`batch_size = temporal_window.WIN_SIZE` (32), which is the instance the windowed stages
(Section 5.4–5.6) actually optimize through — every window's frames are evaluated as one
batched forward pass.

---

## 5. Fitting method

The pipeline runs as a fixed cascade of stages, executed once, in order, over the whole
session for a given (session, activity, person). All stage functions live in
`temporal_window.py` and are invoked in sequence from `main.py`.

| Stage | Function | Free parameters | Frozen | Purpose |
|---|---|---|---|---|
| 0 | `refine_betas_bone_lengths` | shared `betas` (once) | — | Fit shape to observed bone lengths |
| Root | `solve_static_root` | one `(global_orient, transl)` for the whole session | shape, pose | Freeze a single, non-jittering root pose |
| A | `run_windowed` | `body_pose` per frame | `betas`, root, legs | Main windowed torso/limb fit |
| B (hands) | `run_windowed_hands` | hand poses + arm-reach `body_pose` columns per frame | root, non-arm `body_pose` | Hand articulation and arm reach |
| B (head) | `run_windowed_head` | neck/head `body_pose` columns, `jaw_pose`, `expression`, eye poses, per frame | root, rest of `body_pose` | Face/head alignment |
| C | `smooth_all_outputs` | all output trajectories, whole sequence at once | — | Closed-form global smoothing |

### 5.1 Stage 0 — shape from bone lengths

**Rationale.** SMPLer-X's per-frame shape estimate is a reasonable initialization but can be
off by several centimeters on individual limb lengths — the code documents a case where a
subject's model arm length was roughly 8 cm shorter than the triangulated arm, which makes a
correct elbow fit impossible regardless of pose. Since skeletal segment lengths are
pose-invariant, `betas` can instead be fit directly against the *observed* segment lengths
before any pose optimization begins, avoiding a chicken-and-egg coupling between shape and
pose.

**Method.** For a fixed set of bone segments — upper arm, forearm, shoulder width, and a
shoulder-to-hip "trunk" segment (treated as approximately rigid at seated posture, since
chord error stays under 1% at typical seated spine curvature) — a confidence-weighted median
target length is computed over all frames where both segment endpoints are observed. Left/
right pairs are pooled into a single symmetric target, because SMPL-X's shape space is
bilaterally symmetric and cannot represent a length difference between the two sides; a
per-side target would over-specify the fit. Each segment's contribution is weighted by how
many valid samples it has (saturating at `BETAS_NSAT = 50` samples), rather than by
observation *rate*, since a segment seen in only 1% of frames can still contribute a robust,
useful median from tens of clean samples.

The shape is then optimized (with `body_pose` fixed at the rest pose, since bone lengths do
not depend on pose) to minimize a Geman-McClure–robustified squared error between the model's
segment lengths and these targets, plus a directional anchor to the SMPLer-X initial `betas`:
the anchor is split, via a QR-derived projection, into the subspace actually constrained by
the bone-length targets (weighted loosely, so the length data dominates) and its orthogonal
complement — shape dimensions like belly or neck girth that no bone-length target constrains
(weighted tightly, to prevent them drifting unconstrained). The fitted `betas` is then shared
and frozen for the remainder of the session.

### 5.2 Static root solve

**Rationale.** Because the subjects are seated with their hips occluded by the table for most
of a session (in one measured session, one subject's right hip was observed in roughly half
the frames, the other's in about 1%), a per-frame root pose is effectively placed by the
shoulders — which do genuinely lean toward the table, by several centimeters at the 95th
percentile. This makes root position and spine articulation mutually underdetermined on a
per-frame basis, and empirically causes the root to drift or jitter (in one case, several
meters during a brief detection dropout). Conversely, when the hips *are* observed, they are
measured to be essentially static (median drift under a centimeter over a five-minute
session), which supports fitting one rigid root for the whole sequence rather than one per
frame.

**Method.** With `body_pose` fixed, changing only `(global_orient, transl)` is a rigid
transform of every joint about the pelvis, so this sub-problem is solved without re-running
the full body model at every iteration. The target is a Geman-McClure–robustified fit of the
solved root to 3D trunk keypoints (shoulders and hips) from a strided subsample of frames,
with the robustifier's scale annealed from a coarse to a fine value over a few optimization
steps — annealing is used specifically because a fixed fine scale was found to saturate
before the root converges. Where available, a second term reprojects the same trunk keypoints
into each camera view and matches them against RTMO 2D detections, also with an annealed
pixel-space robustifier scale and hand-tuned per-keypoint weights (hips weighted higher, since
they set root position; knees/ankles set root pitch via the already-frozen leg template).
A light anchor toward the initial (SMPLer-X or mamma) root median is also included. Once
solved, the per-camera static leg template's hip angles are re-expressed under the newly
solved world root, since hip angles are parent-relative to whichever pelvis estimate produced
them.

A "root refit" pass — re-solving the static root a second time from the Stage-A-fitted trunk,
rather than the initial template, and re-running Stage A if the correction is large enough —
is implemented but currently disabled in `main.py` (present in the source, commented out).

### 5.3 Static legs

Because the legs are almost never observed in 3D (table occlusion), they are not optimized at
all: leg degrees of freedom are held at a per-camera median articulation taken from the one
camera with a usable view of the legs, and receive zero gradient throughout Stage A.

### 5.4 Stage A — windowed body fit

This is the core of the temporal method. A window of `W = 32` consecutive frames is optimized
**jointly**, as a single batched SMPL-X forward pass, rather than frame by frame. Consecutive
windows overlap by `O = 8` frames; only a window's non-overlap frames are newly committed to
the output — the overlap frames are re-solved for continuity but their values are taken from
whichever window already committed them, which by construction makes the transition between
windows continuous up to acceleration (no visible seam).

**Why windowed coupling replaces a per-frame temporal term.** In the legacy per-frame design
(Section 11), temporal smoothness is enforced by pulling frame *t*'s pose toward frame
*t-1*'s already-frozen value — a one-sided, backward-looking constraint. In the windowed
formulation, the coupling is between *free* variables on both sides of a frame, which means an
under-observed or low-confidence frame is naturally interpolated from both of its neighbors
rather than needing an explicit occlusion-hold rule, and there is no need for a special
frame-0 anchor, since the batched loss already couples every frame in the window to its
neighbors.

**Optimization schedule.** Each window is solved with a coarse-to-fine sequence of three
sub-stages, each running several outer optimizer steps: the data-term weight rises across
sub-stages (100 → 150 → 200) while the temporal-smoothness weight falls (2.0 → 1.0 → 0.5) —
i.e., the fit is smoothed and regularized first, then sharpened against the 3D data.

**Loss terms** (see Section 6 for exact formulas): a robustified 3D keypoint data term; the
GMM body-pose prior and the elbow/knee angle (hyperextension) prior; a "cervical sharing" term
coupling neck and head pose so they bend together rather than kinking at one joint or
twisting in opposite directions; velocity and acceleration smoothness on `body_pose`,
translation and (a rotation-safe 6D representation of) global orientation; an
observability-gated anchor that pulls a frame's orientation toward the window's consensus
orientation only when that frame is poorly observed; a "stillness" anchor pulling `body_pose`
toward a reference (the window's own mean pose, or the external mamma reference when
available) to remove residual per-frame wobble that acceleration smoothing alone does not
fully suppress; and the window-boundary anchor that pins overlap frames to the previous
window's committed solve. Legs remain in the forward pass and prior terms but are excluded
from the optimizer's parameter list (their gradient is zeroed), so they never move.

### 5.5 Stage B — hands

With the root and non-arm body pose frozen at their Stage-A values, this stage refines finger
articulation (`left_hand_pose`, `right_hand_pose`, each 45-dimensional) together with the
arm-reach columns of `body_pose` (shoulder/elbow/wrist), using the same sliding-window/
overlap/commit machinery as Stage A. Loss terms include a robustified 3D hand-keypoint data
term with an annealed scale, a soft anchor toward an external WiLoR hand-pose estimate, an L2/
GMM hand-pose prior, velocity/acceleration smoothness jointly over both hands and the arm
columns, an anchor keeping the arm-reach columns near their Stage-A value, and the same
window-boundary term. The stage is skipped entirely when no hand keypoints, WiLoR
initialization, or hand prior are available for a given sequence.

### 5.6 Stage B — head and face

With the root and non-head body pose frozen, this stage refines the neck/head columns of
`body_pose`, `jaw_pose`, `expression`, and the eye poses. The 51 inner face landmarks are
matched not against static model joints but against points interpolated directly on the
deforming mesh surface, via a precomputed barycentric embedding shipped with the SMPL-X model
files — so the landmark targets move correctly with expression. This is combined with a
robustified term on nose/eye/ear keypoints for coarse skull-orientation disambiguation, using
a fixed (non-annealed), tighter robustifier scale specifically for the ears, since triangulated
ear positions can carry several centimeters of systematic bias and, at a shared annealed
scale, would otherwise dominate and distort the fitted skull rotation relative to the more
reliable dense face landmarks. The same cervical-sharing, neutral-pose, prior, Stage-A anchor,
velocity/acceleration, and window-boundary terms as the other windowed stages are applied,
adapted to the head/jaw/expression/eye parameters. The stage runs only when a jaw prior and
non-zero face-term weights are configured.

### 5.7 Stage C — whole-sequence smoothing

After all windowed stages complete, every output trajectory (body pose, global orientation,
translation, hands, jaw, expression, eyes) is smoothed once more, but now over the **entire
session at once**, with no windows and therefore no seams. This step solves, independently per
output channel, the closed-form problem

$$
x^* = \arg\min_x \; \sum_t (x_t - y_t)^2 \;+\; \lambda \sum_t (x_{t+1} - 2x_t + x_{t-1})^2 ,
$$

i.e. a Whittaker–Eilers smoother: the same acceleration (jerk) penalty used inside the
windows, but solved exactly rather than iteratively. This reduces to a pentadiagonal,
symmetric positive-definite banded linear system, solved directly (not iteratively), which is
reported to take on the order of milliseconds even for a 9000-frame session. Because the mesh
is a deterministic function of its parameters, smoothing the parameters is equivalent to
smoothing the mesh directly, and — unlike a causal or windowed filter — introduces no phase
lag. Each output channel uses its own smoothing strength $\lambda$, chosen per the rule of
thumb that the perceived cutoff scales with $\lambda^{-1/4}$: legs (already static and only
weakly observed) are smoothed hardest, fingers and facial parameters (which move quickly)
are smoothed lightest, and body pose/root fall in between. Global orientation is smoothed on
an axis-angle-unwrapped trajectory to avoid the representation's discontinuity at
$\lvert\theta\rvert = \pi$.

---

## 6. Loss function reference

### 6.1 Robust data term (Geman-McClure)

Every 3D-keypoint data term in the pipeline uses the same robustifier, applied to the
per-coordinate residual between a ground-truth keypoint and the corresponding model joint (or
mesh point):

$$
\mathrm{GMoF}_\rho(r) = \frac{\rho^2 \, r^2}{r^2 + \rho^2}
$$

which behaves like an ordinary squared residual for $|r| \ll \rho$ but saturates to $\rho^2$
for large residuals, bounding the influence any single outlier keypoint can have on the fit.
The scale $\rho$ is stage-specific and, in several stages, **annealed** from a coarse to a
fine value across the stage's optimization steps (coarse-to-fine robustification): e.g. the
static-root solve anneals $\rho$ from 0.20 m to 0.05 m, the hand stage from 0.15 m to 0.05 m,
the head stage's face-landmark term from 0.20 m to 0.05 m. The main windowed body stage
(Stage A) uses a fixed $\rho = 0.25$ m. Every data term is additionally weighted per-keypoint
by the sharpened confidence described in Section 3.3, and every loss term (data and
regularization alike) is passed through a magnitude cap that rescales — rather than hard-clips
— any term exceeding a fixed ceiling, so a single corrupted target cannot destabilize the rest
of the optimization while still preserving a usable gradient direction.

### 6.2 Pose and shape priors

- **GMM body-pose prior.** A pretrained 8-component Gaussian mixture over SMPL's 69-dimensional
  body pose (23 joints), loaded from `priors/gmm_08.pkl`. Because SMPL-X's `body_pose` is only
  63-dimensional (21 joints — SMPL-X moves the two hand-root joints into the separate hand
  pose), the pose vector is zero-padded to 69 dimensions before evaluation, which is
  approximately equivalent to conditioning on a neutral hand pose. The prior is evaluated as a
  **max-mixture** approximation of the true negative log-likelihood: for each of the 8
  Gaussian components $k$ with mean $\mu_k$ and precision $\Sigma_k^{-1}$,

  $$
  \mathrm{NLL}_k(x) = \tfrac{1}{2}(x-\mu_k)^\top \Sigma_k^{-1} (x-\mu_k) \;-\; \log w_k ,
  $$

  and the loss is $\min_k \mathrm{NLL}_k(x)$ — i.e. the single best-fitting component, which
  keeps the prior's gradient well-behaved and avoids a log-sum-exp over all components (the
  same approximation used in the original SMPLify).

- **Angle (hyperextension) prior.** Penalizes physically implausible bending at the elbows and
  knees. For each of the four joints $j$ (with a per-joint sign $s_j \in \{+1,-1\}$ chosen so
  that hyperextension corresponds to a positive product),

  $$
  L_{\text{angle}} = \sum_{j \in \{\text{L elbow, R elbow, L knee, R knee}\}} \exp(s_j \, \theta_j)^2 ,
  $$

  an exponential barrier that is small and flat for normal bending and grows sharply as a
  joint hyperextends.

- **L2 shape/expression/jaw priors.** Simple zero-mean Gaussian shrinkage, $\sum \theta^2$,
  applied to `betas`, `expression`, and (in the legacy loss only) `jaw_pose`.

### 6.3 Temporal smoothness (windowed stages)

For a free trajectory $x_t$ within a window (body pose, translation, or a 6D rotation
representation of global orientation), two coupling terms are used together — acceleration
(jerk) is weighted more heavily than velocity, so that fast-but-smooth motion is not damped,
only genuine jitter:

$$
L_{\text{vel}} = \sum_t (x_{t+1} - x_t)^2, \qquad
L_{\text{acc}} = \sum_t (x_{t+1} - 2x_t + x_{t-1})^2 .
$$

Rotation-valued trajectories are smoothed in a 6D continuous rotation representation to avoid
the axis-angle discontinuity at $\pi$ radians distorting the velocity/acceleration metric,
while anchor terms on the same quantities instead use an axis-angle representation explicitly
*unwrapped* to be continuous with a reference (Section 6.4) — two different resolutions of the
same axis-angle degeneracy, applied to two different kinds of terms.

### 6.4 Anchors

Several terms pull a free variable toward a fixed or slowly-varying reference rather than
coupling it to its neighbors in time:

- **Window-boundary anchor.** Overlap frames of a new window are pulled strongly toward the
  values already committed by the previous window, which is what makes the windowed
  optimization produce a seamless (C¹-continuous) trajectory across window boundaries.
- **Observability-gated orientation anchor.** A per-frame "gate" measures how under-observed a
  frame is relative to the best-observed frame in its window; poorly observed frames are
  pulled toward a detached, observability-weighted consensus orientation for the window, while
  well-observed frames are left essentially untouched (the anchor's own weight is scaled by
  the gate, so it self-disables on good frames).
- **Stillness anchor.** Pulls `body_pose` toward a reference pose — either the window's own
  detached mean, or, when available, the corresponding frame of the external mamma reference
  fit (Section 8) — to remove small steady wobble that a jerk-only penalty does not, by
  construction, suppress (jerk penalizes *change*, not deviation from a set point). Spine and
  head columns are excluded from this anchor, since under the frozen-root design the spine is
  expected to carry genuine trunk motion, and the head is expected to be placed by its own 3D/
  face data rather than by a shape-fusion reference.
- **Cervical sharing.** A direct penalty on the difference between the neck and head pose
  vectors, encouraging them to bend together; this blocks both a single-joint kink and an
  opposing-twist mesh artifact at the neck, and is deliberately not coupled to the spine.

---

## 7. Optimization mechanics

- **Optimizer.** Every stage uses PyTorch's L-BFGS optimizer with a strong-Wolfe line search
  (`torch.optim.LBFGS(..., line_search_fn='strong_wolfe')`), constructed fresh for each
  optimization sub-stage.
- **Gradient clipping.** Gradients are clipped to a maximum norm before every optimizer step,
  for numerical stability.
- **Keep-best-snapshot robustness.** Every stage follows the same pattern: before each
  optimizer step, the current parameter state is snapshotted; if a step produces a non-finite
  loss, the stage aborts and restores the last known-good snapshot; at the end of a stage, the
  final state is evaluated once more and only kept if it is finite and at least as good as the
  best snapshot seen — otherwise the best snapshot is restored. This guarantees a stage's
  output is never worse (in its own loss) than its starting point.
- **Loss-term magnitude capping.** Every individual loss term is passed through a rescaling
  cap before being summed into the total loss, so that one badly-scaled or momentarily corrupt
  term cannot dominate the gradient of the whole objective; the rescaling preserves gradient
  direction rather than hard-clipping to zero gradient.
- **Windowing.** Stages A and B operate on sliding windows of `WIN_SIZE = 32` frames with
  `WIN_OVERLAP = 8` frames of overlap between consecutive windows (an overlap of at least 2 is
  required for the boundary term to guarantee first-derivative continuity across the seam).
  The final, possibly short window of a session is padded to the full window size by repeating
  the last frame with zero data weight, so it does not distort the fit, and padded frames are
  never committed to the output.
- **Per-window diagnostics.** For quality control, the pipeline logs the mean 3D residual (in
  millimeters) on observed keypoints per committed window, and, after the full pipeline
  completes, the output trajectories' jerk — used to distinguish "jitter has been removed" from
  "the output has been over-smoothed."

---

## 8. External reference fusion ("mamma")

"mamma" refers to an independent multi-view SMPL-X fitting pipeline run over the same
6-camera rig (a sibling project in the same repository group), which the fitter can
optionally use as extra supervision. The two pipelines' triangulated nose keypoints were
empirically found to agree to within 1.6–2.3 cm median distance over several thousand frames
in the same world coordinate frame, so no rigid re-alignment is needed to fuse mamma's output
into this pipeline.

**A person-identity complication.** The mamma output's own per-person track identifier is not
this pipeline's `person_id`, and is not even stable within a single video — mamma can
reassign which internal identifier corresponds to which physical person at internal
processing-chunk boundaries (roughly every 500 frames). The loader detects these
reassignments by comparing, at sampled frames, which of mamma's tracks lies nearest to this
pipeline's own triangulated nose position, refines each detected transition down to the exact
frame, and stitches a single, whole-video-consistent per-person trajectory across the
segments before returning it.

**How mamma data is used** — always as a soft prior or warm start, never as a hard
constraint:

1. **Pose warm start.** When available, mamma's `body_pose` (with legs always overridden by
   the SMPLer-X-derived static leg template, since mamma's own legs are similarly occluded/
   unreliable) is used as the per-frame initialization instead of SMPLer-X's own pose
   estimate.
2. **Static-root warm start.** The trunk/spine template fed into the static-root solve
   (Section 5.2) is likewise taken from mamma's pose/orientation/translation when available.
3. **Stillness-anchor reference.** Stage A's stillness anchor (Section 6.4) targets mamma's
   per-frame `body_pose` when available, rather than the window's own detached mean pose.
4. **Shape (`betas`).** Mamma's own shape parameters are *not* used to override this
   pipeline's `betas` — an earlier version of the code did this, but it is reverted (the
   corresponding code path is present but commented out), following the discovery that doing
   so previously mixed pose from one mamma processing segment with betas from a different
   segment. Shape is instead always sourced from SMPLer-X's own initialization, then refined
   against observed bone lengths in Stage 0.

---

## 9. Configuration schema

Configuration is parsed by `cmd_parser.py` using `configargparse`, so every setting can be
supplied either as a command-line flag or as a key in the YAML file passed via `-c/--config`
(required). The schema is broad — it still exposes the full legacy per-frame SMPLify-X
configuration surface (Section 11) alongside the handful of options the live windowed
pipeline actually reads. Broadly:

- **Data/IO paths** — input/output folder locations, keypoint/image/prior/model folders.
- **Dataset and model selection** — `dataset` (`custom` in production), `model_type`
  (`smplx`), `gender`, `num_betas`, `use_hands`/`use_face`/`use_face_contour`, `use_pca`
  (`False` in production — full-DOF hand pose rather than a reduced PCA basis).
- **Prior selection** — which prior class (`gmm`/`l2`/`angle`/`none`) backs the body, hand,
  jaw, expression and shape priors.
- **Camera/geometry** — legacy monocular camera parameters inherited from SMPLify-X's 2D
  formulation; still used by the static-root solve's optional 2D-reprojection term.
- **Interpenetration/collision settings** — parsed and configurable, but the corresponding
  loss is not computed anywhere in the live pipeline (Section 11).
- **Legacy staged loss-weight schedules** — five-stage arrays (`data_weights`,
  `body_pose_prior_weights`, `hand_joints_weights`, `temporal_weights`, etc.) that parameterize
  the legacy per-frame loss in `fitting.py`; not read by `temporal_window.py` (Section 11).
- **Keypoint gating (live)** — `joint_conf_threshold` and `hip_weight`, both consumed directly
  by the live pipeline in `main.py`.
- **Output control** — mesh export toggles, visualization flags.

### `cfg_files/fit_smplx_{m1,m2,m3}.yaml`

These are the three maintained, production configs (selected on the command line via
`-c/--config`; the trailing token in the filename, e.g. `m1`, becomes part of the output
directory name as `<session>_cfg<m1|m2|m3>`). `m2.yaml` and `m3.yaml` are currently identical.
`m1.yaml` differs from them in two settings: it uses `num_betas: 16` where `m2`/`m3` use
`num_betas: 10` (this is the one difference between the three that materially affects the live
pipeline, since it changes the dimensionality of the shared shape vector fit in Stage 0), and
it uses a lighter `body_pose_prior_weights` schedule (a legacy, currently-inert setting, see
Section 11). Settings shared by all three include: `use_hands`/`use_face`/
`use_face_contour: True`, `flat_hand_mean: True`, `body_prior_type: gmm`, `num_pca_comps: 45`,
`rho: 0.25`, `joint_conf_threshold: 0.1`, `hip_weight: 1.0`, and `joints_to_ign` set to the
knee/ankle indices (consistent with the pipeline's static-leg design). Note that
`fitter_pipeline.py` unconditionally overrides `gender` to `'neutral'` at dispatch time,
regardless of what any config file specifies.

---

## 10. Orchestration and tooling

### 10.1 `fitter_pipeline.py` — single-machine batch driver

Given a config and a set of filters, this script discovers every (session, activity, person)
combination with triangulated data available, and calls `main.py`'s entry function
**in-process** (not as a subprocess) once per combination, with paths and per-sequence options
(camera calibration, RTMO/mamma/mask folder locations, when present) filled in automatically.

- `--sid` (default: all sessions) — a **substring** filter on session id.
- `--activities` (default: all five task types recorded in the study) — an exact-match filter
  on activity name.
- `--max-frames` (default: unlimited) — caps the number of frames processed per sequence, for
  quick test runs.

Output for each combination is written to
`resources/fit_results/<session_id>_cfg<config_suffix>/<activity>/p<person_id>/`, where
`<config_suffix>` is extracted from the config's filename.

### 10.2 `run_parallel_sessions.py` — multi-GPU scheduler

A subprocess-based wrapper around `fitter_pipeline.py`, used to fit many sessions in parallel
across the GPUs available on a machine. It discovers candidate sessions from the triangulation
output directory, skips sessions that already have non-empty output under the target config's
output directory, and continuously assigns new sessions to any GPU with capacity — a GPU is
first considered free if no other process is using it, and a per-GPU concurrency cap (default
1) then limits how many of this script's own jobs share one GPU thereafter. Given no explicit
list of session ids, it runs as a long-lived polling daemon that picks up newly triangulated
sessions as they appear. Each session's logs are written under `run_logs/<run_id>/`.

### 10.3 `export_kit_amass.py` — mocap export

Converts a fitted sequence's `body_smplx.json` into a KIT-AMASS-format `.npz` mocap file,
mapping this pipeline's per-part pose fields onto AMASS's flat 165-dimensional pose vector
(root, body, jaw, eyes, hands, in that fixed order) plus `trans` and `betas`. Requires the
source fit to have used full-DOF (non-PCA) hand poses. Validates frame contiguity and shape
(`betas`) consistency across the sequence before exporting.

### 10.4 Visualization tools (`visualization/`)

- **`vis_fit_results_viser.py`** — an interactive 3D viewer (built on `viser`) that renders a
  fitted sequence's per-frame meshes alongside the triangulated 3D keypoints they were fit to,
  color-coded by body part, with a frame slider and playback controls — the primary tool for
  visually auditing fit quality.
- **`vis_fit_on_video.py`** — renders the fitted mesh, reprojected through each camera's
  calibration, as an overlay burned onto the original session video, one output video per
  camera; also overlays the raw multi-view 2D detections used by the static-root solve, to
  visualize what drove the root fit.
- **`vis_joint_mapping.py`** — a documentation/debugging utility that prints (and optionally
  plots) a full cross-reference of every joint-index space used in the pipeline: the data-space
  index, the raw SMPL-X joint index, the corresponding `body_pose` DOF slice, and (for the ADT
  benchmark format) the original skeleton index and name.

---

## 11. Legacy components (not exercised by the live pipeline)

This section documents machinery that remains in the repository, is fully implemented and
reachable via configuration, but is **not** on the code path `main.py` actually executes
today (verified by repository-wide reference search). It is included here because it explains
the origin and vocabulary of much of `cmd_parser.py`'s configuration surface, and because a
reader of the config files or of `fitting.py` could otherwise reasonably — but incorrectly —
assume it describes the pipeline's current behavior.

- **`fitting.py`** implements a classical, SMPLify-X-style per-frame optimizer: a
  `FittingMonitor` driving a staged L-BFGS loop with a random-perturbation escape mechanism for
  stuck fits, around a single `SMPLifyLoss` module that computes a full per-frame energy
  (robustified 2D/3D reprojection term, GMM or VPoser pose prior, shape/hand/expression/jaw
  priors, an interpenetration/collision term via a BVH mesh-intersection routine, a face
  landmark term, and simple frame-to-frame temporal anchor terms). Neither class is
  instantiated anywhere in the current pipeline; `temporal_window.py` imports only two
  stateless helper functions from this file (camera-projection utilities), used inside the
  static-root solve's optional 2D term.
- **`optimizers/lbfgs_ls.py`** and **`optimizers/optim_factory.py`** implement/select a custom
  L-BFGS optimizer with a strong-Wolfe line search, vendored at a time before PyTorch's own
  `LBFGS` supported strong-Wolfe line search natively. The live pipeline uses PyTorch's own
  built-in `LBFGS` with `line_search_fn='strong_wolfe'` directly, and does not import this
  package.
- **Most of `cmd_parser.py`'s staged loss-weight arrays** (`data_weights`,
  `body_pose_prior_weights`, `shape_weights`, `hand_joints_weights`, `expr_weights`,
  `temporal_weights`, `smpler_pose_weights`, `coll_loss_weights`, `silhouette_weights`, and the
  associated legacy tracking/IK parameters, optimizer-type selection, and
  `global_orient_mode`/`translation_mode` settings) parameterize `fitting.py`'s per-frame loss
  and optimizer only, and have no effect on the windowed pipeline, which sources its
  equivalent tuning from hardcoded module-level constants at the top of `temporal_window.py`
  instead.
- **`cfg_files/_generated/`** contains 118 auxiliary YAML files from an earlier ablation/grid
  study (naming pattern such as `fit_smplx_3_neck_spine_arms_init_global_orientTrue.yaml`).
  These target the ADT benchmark dataset (not the production `custom` dataset) and reference
  configuration keys (e.g. `cross_temp_weight`, `hand_wilor_weight`, `direct_refine_joints`)
  that do not exist in the current `cmd_parser.py` schema at all — no generator script for
  these files exists anywhere in the current repository history. They should be treated as
  historical artifacts, not as usable configuration.

---

## 12. Known inconsistencies and open items

Recorded here for completeness/traceability, since they are the kind of detail that is easy to
mis-describe in a write-up if not flagged explicitly:

- The `m1.yaml`/`m2.yaml`/`m3.yaml` configs each carry a header comment claiming to "match
  mamma's shape space (16 betas)," but only `m1.yaml` actually sets `num_betas: 16`; `m2.yaml`
  and `m3.yaml` set `num_betas: 10` while keeping the same comment.
- The root-refit pass described in Section 5.2 (re-solving the static root from the
  Stage-A-fitted trunk) is implemented in `temporal_window.py` but is currently disabled —
  the calling code in `main.py` is present but commented out.
- `cmd_parser.py`'s default `body_prior_type` is `'mog'`, a value `prior.create_prior` does not
  actually recognize (only `'gmm'`/`'l2'`/`'angle'`/`'none'` are implemented); all three
  production configs override this explicitly to `'gmm'`, so it has no practical effect, but a
  config that omitted the override would fail at startup.
- `visualization/vis_fit_on_video.py` defines an `activities` list at module scope that is
  inconsistent with its own `--activities` command-line default; the command-line flag is the
  one actually honored.
- `fitter_pipeline.py` copies the selected config file into each sequence's output directory
  for provenance before that directory necessarily exists, which can raise on a first run for a
  new (session, activity, person) combination; this is currently caught and only logged as a
  warning, not treated as fatal.

---

## Acknowledgements

Built on [SMPLify-X](https://github.com/vchoutas/smplify-x) (Pavlakos et al.); uses the
[SMPL-X](https://smpl-x.is.tue.mpg.de/) body model, [SMPLer-X](https://github.com/caizhongang/SMPLer-X)
for pose initialization, and (optionally) [human_body_prior](https://github.com/nghorbani/human_body_prior)
(VPoser) and [torch-mesh-isect](https://github.com/vchoutas/torch-mesh-isect) for the legacy
per-frame path.
