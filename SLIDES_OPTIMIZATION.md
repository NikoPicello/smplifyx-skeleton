# SMPLify-X Video Fitting — Optimization Overview

---

## Slide 1 — Goal

**Input:** Multi-view synchronized RGB video of a person  
**Output:** Per-frame SMPL-X body mesh parameters

**Key challenge:** Each frame is an independent non-linear optimization — naive per-frame fitting is slow, produces jitter, and ignores temporal structure.

**What we built on top of SMPLify-X:**
- Multi-view 3D keypoint joint loss (no 2D projection)
- 3D face landmark loss (51 dlib landmarks, barycentric interpolation on mesh)
- Silhouette loss via differentiable rasterization — nvdiffrast (currently disabled)
- Temporal consistency via **direct refinement** (cross-frame + intra-frame anchors)
- Warm-started fast per-frame solver: Jacobian IK → Direct refinement
- Temporal hand pose blending (WiLoR + previous frame carry-over)

---

## Slide 2 — SMPL-X Body Model

SMPL-X maps a compact parameter set to a body mesh M ∈ ℝ^{N×3}:

```
M(θ, β, ψ, t, R)
```

| Symbol | Meaning | Dim |
|--------|---------|-----|
| β | Shape (body proportions, PCA coefficients) | 10 |
| R | Global orientation (axis-angle) | 3 |
| t | Global translation | 3 |
| θ_b | Body pose: 21 joints × 3 DoF axis-angle | 63 |
| θ_h | Hand pose: left + right (full axis-angle, use_pca=False) | 2 × 45 |
| ψ | Facial expression | 10 |
| θ_j | Jaw pose | 3 |

**Full pose vector** concatenated: `full_pose = [R, θ_b, θ_j, θ_h_L, θ_h_R]`

**VPoser** is **not used** (`use_vposer=False`). Body pose `θ_b` is optimized directly as 63 axis-angle DOFs. The pose prior falls back to a mixture-of-Gaussians (L2 in current config).

---

## Slide 3 — LBFGS Stage Objective (Frame 0 Only)

Only frame 0 runs the full multi-stage LBFGS. At frame 0 we minimize:

```
L = λ_data  · L_joints
  + λ_pose  · L_pose
  + λ_β     · L_shape
  + λ_ang   · L_angle
  + λ_hand  · L_hand
  + λ_face  · L_face_lmk
  + λ_sil   · L_sil        ← currently disabled (λ_sil = 0)
  + λ_temp  · L_temp        ← currently inactive (see note)
  + λ_coll  · L_coll
```

### Joint loss
```
L_joints = Σ_j  w_j²  ·  ρ( Ĵ_j  −  J_j(θ, β, t, R) )
```
- Ĵ_j: 3D joint position from multi-view triangulation
- J_j: joint predicted by SMPL-X forward pass
- w_j: per-joint weight (arm joints [10→5], hand joints [15→2], face joints [80→20] by stage)
- ρ: Geman-McClure robustifier → ρ(x) = x² / (ρ₀² + x²) with ρ₀ = 150

### Pose prior
```
L_pose = E_GMM(θ_b, β)     (L2 in current config)
```

### Shape prior
```
L_shape = ||β||²            (L2 toward mean shape; β frozen after frame 0)
```

### Angle prior
```
L_angle = Σ_j max(0, θ_j − θ_j^max)   (prevents impossible elbow/knee bending)
```

### Hand prior
```
L_hand = E_hand(θ_h_L) + E_hand(θ_h_R)  (L2 toward neutral, weight ramps down)
```

### Face landmark loss (51 inner dlib landmarks)
```
L_face_lmk = Σ_{k=1}^{51}  v_k · || l̂_k − l_k(θ, β) ||²
```
- l̂_k: 3D landmark from head tracker (triangulated)
- l_k: landmark via barycentric interpolation on SMPL-X mesh faces
- v_k: binary validity (NaN landmarks excluded)
- Plain L2 (no robustifier — 3D triangulated landmarks are reliable)

### Silhouette loss (IoU-based, differentiable)
```
L_sil = 1 − |S_rendered ∩ S_gt| / |S_rendered ∪ S_gt|
```
Rendered via nvdiffrast. **Currently disabled** (all weights = 0).

### Temporal loss in LBFGS path
```
L_temp = || full_pose^(n) − full_pose^(n-1) ||²
```
**Currently inactive** — this fires only when `use_vposer=True` (which passes a pose embedding across frames). Since `use_vposer=False`, `prev_pose_embedding=None` and this term is zero. Temporal consistency is handled entirely in the **direct refinement stage** (Slide 6).

---

## Slide 4 — Weight Schedule (5 Stages, Coarse → Fine)

Config: `fit_smplx_10.yaml`. LBFGS runs 5 sequential stages with decreasing prior weights and increasing data fidelity (frame 0 only):

| Stage | data | pose | shape | arm_jt | hand_jt | hand_prior | face_lmk | sil | coll |
|-------|------|------|-------|--------|---------|------------|----------|-----|------|
| 0 | 100 | 4.0 | 10 | 10 | 15 | 4.0 | 80 | **0** | 1 |
| 1 | 80  | 2.0 | 20 | 8  | 8  | 2.0 | 40 | **0** | 2 |
| 2 | 15  | 1.0 | 50 | 6  | 5  | 1.0 | 30 | **0** | 3 |
| 3 | 20  | 1.0 | 25 | 4  | 3  | 0.4 | 20 | **0** | 4 |
| 4 | 15  | 0.5 | 10 | 5  | 2  | 0.1 | 20 | **0** | 5 |

**Loss contribution** = weight² × term. Stage 0 is prior-dominated (shape/orientation). Stage 4 is data-dominated (face detail, fine joint alignment).

`hand_jt` ramps **down** — WiLoR gives a good initialization so the data pull is relaxed as pose settles.

`face_lmk` ramps down from 80 to 20 — face orientation is most uncertain early; later stages let joint data and IK dominate.

---

## Slide 5 — Per-Frame Processing Strategy

Full LBFGS (5 stages × 200 iters) runs only for **frame 0** (with 3× more iterations = 600 iters total). All subsequent frames use a two-stage fast path:

```
Frame 0          → Full LBFGS (5 stages, 600 iters) → Direct Refinement
Frames 1, 2, … → Jacobian IK (warm-started)        → Direct Refinement
```

No periodic LBFGS re-runs are currently active. The Jacobian IK + direct refinement path handles all frames from 1 onward.

### Jacobian IK (Levenberg-Marquardt)
Directly minimizes joint position residuals via first-order linearization:

```
min_{R, θ_b_upper, t}  || J_model(params) − Ĵ ||²
```

Solved each iteration as a damped least-squares system:
```
[  J_fwd  ]         [  r  ]
[  λ · I  ]  δ  =  [  0  ]
```
- J_fwd: Jacobian of model joint positions w.r.t. [R(3), θ_b(63), t(3)] = 69 params (autograd)
- r = Ĵ − J_model: residual vector (NaN joints zeroed out via valid_mask)
- λ = 0.2: LM damping (prevents large steps)
- n_iters = 10, δ_tol = 1e-4
- **Lower-body DOFs frozen** (hips, knees, ankles, feet — seated assumption)
- Face/hand joint rows excluded from IK residual (not driven by model DOFs at this scale)

Updates are applied directly to model parameters via `.data` (no optimizer overhead).

---

## Slide 6 — Direct Refinement Stage

Runs after IK (or after LBFGS on frame 0) for **every** frame. This is where temporal consistency and fine head/shoulder adjustment happen.

### What is optimized
- **Free DOFs** (per-person configurable via `direct_refine_joints_p{id}`):  
  spine3, neck, left_collar, right_collar, head, left_shoulder, right_shoulder → **21 DOFs**
- **Jaw pose** (3 DOFs) — freed for face landmark alignment
- All other parameters frozen

### Objective
```
L_direct = λ_data · L_joints_upper
         + λ_pose · L_pose_direct
         + λ_face · L_face_lmk
         + λ_jaw  · L_jaw
         + λ_temp · L_intra
         + λ_cross · L_cross
```

| Term | Formula | Weight |
|------|---------|--------|
| `jloss` | Σ w_j² · ρ(Ĵ_j − J_j) for upper-body joints only (neck + arms, indices 3, 5-12) | λ_data = 15² |
| `ploss` | \|\|θ_free\|\|² | λ_pose = 0.05² |
| `floss` | Σ v_k · \|\|l̂_k − l_k\|\|² (51 landmarks, L2) | λ_face = 20² |
| `jploss` | jaw prior (from pose prior) | λ_jaw = 1² |
| `tloss` | \|\|θ_free^(n) − θ_free^(n)_IK\|\|² + \|\|θ_jaw^(n) − θ_jaw^(n)_IK\|\|² | λ_temp = 5² (0 at frame 0) |
| `closs` | \|\|θ_free^(n) − θ_free^(n-1)_final\|\|² | λ_cross = 10² (0 at frame 0) |

**L_intra** (tloss): prevents direct refinement from drifting far from what IK/LBFGS produced — an intra-frame anchor.  
**L_cross** (closs): anchors this frame's final pose to the previous frame's final refined pose — the main cross-frame temporal consistency term.

### Solver
LBFGS with strong Wolfe line search, lr = 1.2, max_iter = 10, run for 5 outer steps.

Upper body joint mask excludes head joint (index 4) — the GT head centroid is biased in front of/below the SMPL-X skeletal joint; face landmarks handle head orientation cleanly.

---

## Slide 7 — Temporal Warm-Starting

**Body pose carry-over (VPoser OFF):**  
`body_model.body_pose` is a shared object — its state persists in memory between frames. IK overwrites upper-body DOFs directly, and direct refinement writes back the final pose. Lower-body DOFs are effectively frozen throughout (IK skips them, direct refinement doesn't touch them).

**Translation:** initialized from the triangulated pelvis keypoint every frame:
```
t^(n)_init = Ĵ_pelvis^(n)
```
Falls back to centroid of all valid joints if pelvis is NaN.  
- IK frames: t is frozen (IK updates it via `.data` during LM solve)  
- LBFGS frames (frame 0): t is free — optimizer corrects pelvis initialization noise

**Hand pose blending:**

WiLoR provides a per-frame hand pose estimate θ̂_h^{WiLoR}. Initialization before each frame:
```
θ_h^(n)_init = α · θ_h^(n-1)_optimized  +  (1−α) · θ̂_h^{WiLoR,(n)}
```
- α = **0.80** (tunable via `hand_prev_alpha`) — carries 80% of the previous frame's pose
- If WiLoR has no estimate for frame n: carry θ_h^(n-1) directly
- Rationale: consecutive-frame hand motion is small; blending reduces jitter from WiLoR outliers

**Summary of temporal mechanisms:**

| Mechanism | What it anchors | Where |
|-----------|----------------|-------|
| Body pose carry-over | θ_b warm-start (lower body stays frozen) | IK initialization |
| Hand blending (α=0.80) | θ_h warm-start | Before IK |
| tloss (λ=5²) | Upper-body free DOFs vs. same-frame IK result | Direct refinement |
| closs (λ=10²) | Upper-body free DOFs vs. previous frame's refined pose | Direct refinement |

---

## Slide 8 — Bug Fixes

### Bug 1 — Injected betas bypassed frame-0 initialization
**Condition:** `if frame_idx == 0 and global_betas is None`

When pre-computed betas were provided (`init_betas`), `global_betas` was non-None before the loop. Frame 0 fell into the *else* (subsequent-frame) branch → translation was frozen from the start → full LBFGS on frame 0 could not optimize position.

**Fix:** Split into `if frame_idx == 0` (always reset) with a nested check for injected betas.

---

### Bug 2 — Translation frozen on LBFGS re-run frames
**What happened:** End-of-frame cleanup always set `transl.requires_grad_(False)`. On the next frame, the *else* branch never re-enabled it. So LBFGS re-run frames (10, 20, ...) started with translation frozen → the optimizer had no DOF to correct pelvis initialization noise → joint loss exploded (observed: ~13M at frame 10 vs ~33 at frame 9).

**Fix — state flow after the fix:**

```
End of IK frame N:
    transl.requires_grad = False   ← cleanup

Start of frame N+1 (else branch):
    transl.data ← pelvis keypoint
    transl.requires_grad = True    ← re-enable unconditionally

If IK path:
    transl.requires_grad = False   ← re-freeze before direct refinement
    _jacobian_ik(...)              ← updates transl via .data anyway
If LBFGS path:
    transl stays True → enters LBFGS optimizer → can correct pelvis noise
```

---

## Slide 9 — What's Next / Open Questions

- **Periodic LBFGS re-runs:** code has `lbfgs_rerun_interval=10` in config but it is disabled (`_do_lbfgs = (frame_idx == 0)`). Re-enabling every N frames could correct IK drift for long sequences.
- **Silhouette loss:** infrastructure is in place (nvdiffrast, camera tensors, mask loading) but weights are all 0. Enabling for a subset of cameras could improve global shape alignment.
- **LBFGS temporal loss:** `temporal_weights` config key exists [10, 7, 5, 3, 1] but is inactive. Could be activated for frame 0+ by passing `prev_pose_embedding=body_model.body_pose` explicitly.
- **Lower body:** frozen (seated assumption). Could relax for standing sequences with a DOF schedule.
- **Hand blending α:** fixed at 0.80. Could be adaptive (lower α when WiLoR confidence is high, higher when low).
- **Temporal loss on hands/face:** currently only `closs`/`tloss` cover upper-body free DOFs. Could add separate cross-frame anchor for hand pose and jaw.
- **Beta drift:** β frozen after frame 0. For long sequences with clothing/occlusion changes, could allow slow updates with a strong prior.
- **closs per-person tuning:** `cross_temp_weight_p0 = 10`, `cross_temp_weight_p1 = 10` — both equal now. Could differentiate if one person has more pose variation than the other.
