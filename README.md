# smplifyx-skeleton — SMPL-X fitting stage of the UBInteract 3D pipeline

This repository fits the **SMPL-X** body model (body, hands and face) to the UBInteract recordings: two people seated at a table, filmed by six calibrated cameras. It runs after the MKER triangulation stage and turns the triangulated 3D keypoints into one SMPL-X parameter set per person and frame.

It started as a fork of [smplifyx-skeleton](https://github.com/cseeyangchen/smplifyx-skeleton) (SMPLify-X adapted to 3D skeleton input), itself based on [SMPLify-X](https://github.com/vchoutas/smplify-x). The per-frame optimizer of those projects has been replaced by a **batched, sliding-window temporal fit** designed for the seated setting.

For every session, activity and person it:

1. loads the triangulated 3D keypoints (17 body, 2 × 21 hand, 68 face) and the fused SMPLer-X estimate used as initialization,
2. fits the body shape once, from the observed bone lengths,
3. holds the legs at the seated pose seen by the back camera and solves one static root (pelvis position and orientation) from the 3D shoulders and multi-view 2D detections,
4. fits body pose and root over the whole sequence in overlapping 32-frame windows, then refines hands and head the same way,
5. smooths every parameter trajectory over the whole sequence,
6. writes the SMPL-X parameters of every frame to a JSON-lines file (and optionally one mesh per frame).

| Path | Purpose |
|---|---|
| [`fitter_pipeline.py`](fitter_pipeline.py) | **Entry point.** Finds every (session, activity, person) with triangulation results and fits it. |
| [`run_parallel_sessions.py`](run_parallel_sessions.py) | Orchestrator: runs `fitter_pipeline.py` over many sessions, packed onto idle GPUs. |
| [`main.py`](main.py) | Fit of one sequence: loading, order of the stages, output writing. Called in-process by `fitter_pipeline.py`. |
| [`temporal_window/`](temporal_window/) | The optimization, one module per stage: `betas`, `legs`, `root`, `body`, `hands`, `head`, `smoothing`. `_common` holds the window geometry and shared helpers. |
| [`cmd_parser.py`](cmd_parser.py), [`cfg_files/`](cfg_files/) | Configuration schema and configuration file. |
| [`data_parser.py`](data_parser.py) | `CustomDataset`, the loader for this dataset. (The `ADT` loader comes from the upstream fork and does not work with the current `main.py`.) |
| [`cvars.py`](cvars.py), [`prior.py`](prior.py), [`utils.py`](utils.py), [`mamma_loader.py`](mamma_loader.py) | Keypoint pre-processing constants and seated leg template; GMM, angle and L2 priors; joint mapper and camera projection; optional mamma reference. |
| [`export_kit_amass.py`](export_kit_amass.py), [`visualization/`](visualization/) | AMASS export and viewers (§6). |

**Contents:** [1. Installation](#1-installation) · [2. Data layout](#2-data-layout) · [3. Usage](#3-usage) · [4. Method](#4-method) · [5. Outputs](#5-outputs) · [6. Visualization and export](#6-visualization-and-export) · [7. Limitations and known issues](#7-limitations-and-known-issues) · [8. Citation](#8-citation)

---

## 1. Installation

The repository is a submodule of UBInteract, at `3D_reconstruction/scripts/smplifyx-skeleton`, and expects `resources/` two levels up (§2).

```bash
git clone git@github.com:NikoPicello/smplifyx-skeleton.git
cd smplifyx-skeleton
```

```bash
conda create --name fitter python=3.12
conda activate fitter

pip install torch                      # pick the build matching your CUDA version
pip install numpy scipy opencv-python configargparse pyyaml smplx trimesh
pip install viser tyro imageio imageio-ffmpeg tqdm matplotlib    # visualization tools only
```

The local `fitter` environment runs Python 3.12, PyTorch 2.11 (CUDA 13.0), `smplx` 0.1.28 and NumPy 2.4. `smplx` is the PyPI package. The `dependencies/` folder only holds submodule pointers (`smplx`, `human_body_prior`, `torch-mesh-isect`) that the current fit does not use; nvdiffrast, VPoser and torch-mesh-isect are not needed.

### Body model and prior

Download `models_smplx_v1_1.zip` from the [SMPL-X website](https://smpl-x.is.tue.mpg.de/) and unzip it so that `models/smplx/SMPLX_NEUTRAL.npz` exists (`models/` is git-ignored). Only the neutral `.npz` is used: `fitter_pipeline.py` always fits the neutral model, and the head stage reads the face-landmark embedding (`lmk_faces_idx`, `lmk_bary_coords`) from that file.

The GMM body-pose prior, `priors/gmm_08.pkl`, is part of the repository.

### GPU

The fit runs on the default CUDA device; pick it with `CUDA_VISIBLE_DEVICES`. With `use_cuda: true` (the default) the run aborts if CUDA is not available.

---

## 2. Data layout

Paths are resolved relative to the repository: the scripts expect `smplifyx-skeleton/` two levels below the project root, next to `resources/`.

```text
3D_reconstruction/
├── resources/
│   ├── all_sessions/<sid>/
│   │   ├── session_data.txt                    # 2nd line: calib_date=YYYYMMDD
│   │   └── <activity>/                         # only the folder names are used here
│   ├── calibs/<calib_date>/                    # GC.yml HC.yml Z1.yml Z2.yml N1.yml N2.yml
│   ├── triangulation_results/<sid>/<activity>/p<0|1>/
│   │   └── body.npy  face.npy  left_hand.npy  right_hand.npy  smpl.npy
│   ├── rtmo_results/<sid>/<activity>/<camera>_rtmo.npy             # optional
│   ├── smpler_results/<sid>/<activity>/GB_smplx.npy                # optional
│   ├── mamma_results/<sid>/<activity>/smplx_params_body_id-*.npz   # optional (--use-mamma)
│   └── fit_results/                                                # output (§5)
└── scripts/
    ├── MKER/scripts/triangulation/             # writes triangulation_results/
    └── smplifyx-skeleton/                      # this repository
```

- **`<sid>`** is a 6-digit session id (e.g. `005013`). **`<activity>`** is one of `animals_task`, `gaze_task`, `ghost_task`, `lego_task`, `talk_task`. **`p0`/`p1`** are the two participants, as numbered by the triangulation stage.
- **Cameras.** `GB`, `GF`, `FC1`, `FC2`, `HA1`, `HA2`, calibrated in the files `GC`, `HC`, `Z1`, `Z2`, `N1`, `N2` respectively (OpenCV `FileStorage` with `K`, `D` and the world-to-camera `R`, `T`; image size 1280×720). The calibration folder must contain only these six `.yml` files: any other file name stops the loader.
- The triangulated keypoints and the output share the calibration's world frame; units are metres.

### 2.1 Inputs

| File | Written by | Content read by the fit | Used in |
|---|---|---|---|
| `body.npy` | `MKER/scripts/triangulation/body_triangulation.py` (RTMO) | per frame: `kpts_3d` (17, 3), `confidence` (17,), COCO-17 order | shape, root, Stages A and B |
| `face.npy` | `face_triangulation.py` (3DDFA-V3) | per frame: `kpts_3d` (68, 3), `confidence` (68,), 68-point order | head stage |
| `left_hand.npy`, `right_hand.npy` | `mano_triangulation.py` (WiLoR) | per frame: `kpts_3d` (21, 3), `confidence` (21,), `hand_pose` (45,) MANO axis-angle or `None` | hand stage |
| `smpl.npy` | `smpler_fusion.py` (SMPLer-X) | per frame: `body_pose` (63,), `global_orient` (3,), `transl` (3,); top-level `betas` (10,) | initialization |
| `<camera>_rtmo.npy` | `mmpose/rtmo_pipeline.py` | per frame: `{person_id: {keypoints (17, 2), keypoint_scores (17,)}}` | static root, hip seed |
| `GB_smplx.npy` | `MKER/smpler_pipeline.py` | per frame: `{person_id: {body_pose (21, 3), global_orient (1, 3)}}` from the GB camera | static legs |
| `smplx_params_body_id-*.npz` | `mamma/` | `smplx_pose` (T, 165), `smplx_translation`, `triangulated_3d_pts`, … | optional reference (§4.10) |

- The four triangulation files and `smpl.npy` are dicts keyed by integer frame index (string keys hold metadata and are ignored). The loader stacks the integer keys in order, so every file must hold every frame `0 … T−1`, which the triangulation stage does. The output frame index is therefore the video frame index.
- **All five files are required** for each person. An activity whose hand triangulation or fusion has not run yet (only `body.npy` and `face.npy`) cannot be fitted, and stops the run (§3.1).
- Without RTMO, the static root is solved from 3D only and the hip seed is skipped. Without `GB_smplx.npy`, the legs use a fixed seated template (§4.4).
- `sam_results/` is not read by the fit; it is used upstream, by the body triangulation.

### 2.2 Keypoint layout

The loader concatenates the four keypoint files into one array of 127 rows per frame, `[x, y, z, confidence]`. `JointMapper` (`utils.py`) reorders the SMPL-X joints the same way, so model row *i* and data row *i* are the same point:

| Rows | Points | SMPL-X counterpart | Fitted in |
|---|---|---|---|
| 0–16 | COCO-17: nose, L/R eye, L/R ear, L/R shoulder, L/R elbow, L/R wrist, L/R hip, L/R knee, L/R ankle | joints 55, 57, 56, 59, 58, 16–21, 1, 2, 4, 5, 7, 8 | shape, root, Stage A; elbows and wrists also in the hand stage, nose, eyes and ears in the head stage |
| 17–37 | left hand: wrist, then thumb, index, middle, ring, pinky (4 points each, base to tip) | wrist 20, finger joints 25–39, fingertips 66–70 | hand stage |
| 38–58 | right hand, same order | wrist 21, finger joints 40–54, fingertips 71–75 | hand stage |
| 59–75 | face contour (68-point indices 0–16) | — | not fitted |
| 76–126 | brows, nose, eyes, mouth (68-point indices 17–67) | points on the mesh surface (§4.8) | head stage |

- The body wrists (rows 9, 10) are replaced by the wrist of the hand files (rows 17, 38) whenever the latter is observed; otherwise the hand wrist is copied from the body.
- Points with a NaN coordinate or a non-positive confidence are set to zero and count as missing.

---

## 3. Usage

### 3.1 `fitter_pipeline.py`

Run from the repository root (`models/` and `priors/` are relative to the working directory):

```bash
conda activate fitter

# one activity of one session, on GPU 0
CUDA_VISIBLE_DEVICES=0 python fitter_pipeline.py -c cfg_files/fit_smplx_final.yaml --sid 005013 --activities lego_task

# quick test on the first 300 frames
CUDA_VISIBLE_DEVICES=0 python fitter_pipeline.py -c cfg_files/fit_smplx_final.yaml --sid 005013 --activities lego_task --max-frames 300

# every session in resources/all_sessions, all activities, with the mamma reference
CUDA_VISIBLE_DEVICES=0 python fitter_pipeline.py -c cfg_files/fit_smplx_final.yaml --use-mamma
```

| Flag | Default | Description |
|---|---|---|
| `-c`, `--config` | *required* | Configuration file, named `fit_smplx_<X>.yaml` (§3.3). |
| `--sid` | `all` | Session to process. Matched as a **substring** of the folder names in `resources/all_sessions/`. |
| `--activities` | all five | One or more activity folder names (exact match). |
| `--max-frames` | `-1` (all) | Fit only the first *N* frames. Stage 0, the hip seed and the static root still read the whole clip. |
| `--use-mamma` | off | Use `resources/mamma_results/<sid>/<activity>/` when it exists (§4.10). |

For each selected session the script loads the calibration, then, for each activity folder in `resources/all_sessions/<sid>/`:

1. looks for `p0` and `p1` in `triangulation_results/<sid>/<activity>/` (a missing person is skipped),
2. creates `fit_results/<sid>_cfg<X>/<activity>/p<i>/` and copies the configuration into it as `config_used.yaml`,
3. fits both people **at the same time**, in two spawned processes on the same GPU, each logging to its own `fit_log.txt`.

If either fit fails, the script raises once both processes have finished and **stops**: the remaining activities and sessions are not processed. Existing results are overwritten; there is no skip-existing option.

`main.py` cannot be run on its own: it needs per-sequence arguments (person id, cameras, input folders) that only `fitter_pipeline.py` fills in.

### 3.2 Multi-GPU batch run — `run_parallel_sessions.py`

Runs `fitter_pipeline.py -c <config> --sid <sid> --activities <activities> [forwarded args]` over many sessions, each pinned to one GPU with `CUDA_VISIBLE_DEVICES`. Launch it from the activated `fitter` environment; each job uses the same interpreter. Unlike the runners of the other stages, it stages nothing: it reads `resources/` directly.

```bash
python run_parallel_sessions.py -c cfg_files/fit_smplx_final.yaml                   # auto-discover lego_task sessions, keep watching
python run_parallel_sessions.py -c cfg_files/fit_smplx_final.yaml 005013 004115     # only these sessions, then exit
python run_parallel_sessions.py -c cfg_files/fit_smplx_final.yaml --activities lego_task gaze_task --gpus 0,1 --per-gpu 4
python run_parallel_sessions.py -c cfg_files/fit_smplx_final.yaml 005013 --fitter-args --max-frames 500 --use-mamma
python run_parallel_sessions.py -c cfg_files/fit_smplx_final.yaml --dry-run
```

| Flag | Default | Description |
|---|---|---|
| `sessions` (positional) | *auto-discover* | Explicit session ids, always run (even if already fitted). Without them, the script auto-discovers sessions (below) and keeps re-scanning for new ones. |
| `-c`, `--config` | *required* | Forwarded to `fitter_pipeline.py`; also sets the `<X>` of the already-fitted check. |
| `--activities` | `lego_task` | Activities to fit, forwarded to `fitter_pipeline.py --activities`: **only `lego_task` by default**, unlike `fitter_pipeline.py` (all five). Also selects the audit files used by auto-discovery. |
| `--gpus` | all GPUs (or `$GPUS`) | Comma-separated whitelist of GPU indices. |
| `--no-gpu-check` | off | Use the `--gpus` list as-is and never call `nvidia-smi`. Requires `--gpus`. |
| `--per-gpu` | `1` | Concurrent jobs per GPU. |
| `--poll-interval` | `15` s | Interval between GPU and candidate re-checks. |
| `--dry-run` | off | Log the planned runs without starting them. |
| `--fitter-args ...` | — | Everything after this flag is forwarded to `fitter_pipeline.py`, e.g. `--max-frames 500 --use-mamma` (give activities with `--activities`, not here). **Must be the last flag.** |

**Auto-discovery.** Candidates come from the per-activity audit files `resources/<activity>_audit.csv` (`lego_task` → `lego_audit.csv`), written by `../fill_status.py --activities <activity>`. A session is a candidate once its `triangulation_body`, `triangulation_face`, `triangulation_mano` and `fusion` columns all read `OK` in the file of at least one selected activity, and `fit_results/<sid>_cfg<X>/` is still empty or absent. mamma is not checked: add `--fitter-args --use-mamma` to use it where it exists. If an audit file is missing, the script exits and names it. `fill_status.py` writes the audit files under its own hard-coded data root (`RESOURCES_DIR = /data-fast/127-data1/npicello/3D_reconstruction`), so auto-discovery only works where that is the same folder as this repository's `resources/`.

A GPU is taken only if `nvidia-smi --query-compute-apps` shows no process on it; after that, up to `--per-gpu` of the script's own jobs share it. The already-fitted check only looks for *any* output under `fit_results/<sid>_cfg<X>/`, not for every activity and person.

**Logs:** `run_logs/<YYYYMMDD_HHMMSS>/<sid>.log` (output of each session) and `run_logs/<run_id>/summary.tsv` (`<sid>\t<exit_code>` per line).

### 3.3 Configuration

Settings are read from the YAML file passed with `-c` (`cmd_parser.py`, configargparse). `fitter_pipeline.py` passes only `-c` to the parser, so **every setting must be changed in the YAML file**. It also overrides, per sequence, `dataset` (`custom`), `data_folder`, `output_folder` and `gender` (`neutral`).

The file name must match `fit_smplx_<X>.yaml`: `<X>` names the output folder `<sid>_cfg<X>`, and `fitter_pipeline.py` fails on any other name. The shipped configuration is [`cfg_files/fit_smplx_final.yaml`](cfg_files/fit_smplx_final.yaml); new files in `cfg_files/` are git-ignored.

**Run and model keys** (values in `fit_smplx_final.yaml`):

| Key | Value | Description |
|---|---|---|
| `save_mesh` | `false` | Also write `meshes/<frame>_fit.obj`. Both visualization tools need it. |
| `num_betas` | `16` | Shape coefficients (the 10 SMPLer-X betas are zero-padded). |
| `root_frame_count` | `150` | The static root is solved on frames 0–149 only (§4.5). |
| `joints_to_ign` | `[13, 14, 15, 16]` | Body rows left out of the Stage A data term (knees, ankles). |
| `joint_conf_threshold` | `0.1` | Keypoints at or below this confidence count as missing for gap-filling (§4.2). |
| `hip_weight` | `1.0` | Multiplier on the hip rows in the Stage A data term. |
| `use_hands`, `use_face`, `use_face_contour` | `true` | Keep `true`: the row ranges in `main.py` assume the 127-row layout. |
| `use_pca`, `flat_hand_mean` | `false`, `true` | Full 45-D axis-angle hand poses; the zero pose is the flat hand. |
| `body_prior_type`, `num_gaussians` | `gmm`, `8` | GMM body-pose prior `priors/gmm_08.pkl`. |
| `left_hand_prior_type`, `right_hand_prior_type`, `jaw_prior_type` | `l2` | L2 priors of the hand and head stages. |
| `use_cuda`, `float_dtype` | `true`, `float32` | Run on the GPU (abort if CUDA is unavailable); floating-point precision. |
| `model_folder`, `prior_folder` | `models`, `priors` | Relative to the working directory. |

Parsed but with no effect on the fit: `gpu_id` and `max_persons` (use `CUDA_VISIBLE_DEVICES`; both people are always fitted), `use_vposer`, `expr_prior_type`, `shape_prior_type`, and `interpenetration` with its options (the collision term is not part of the windowed loss).

**Tuning keys.** Most constants of the method have a key whose default, `null`, keeps the value set at the top of its module (`cvars.py`, `temporal_window/*.py`). `cvars.configure()` and `temporal_window.configure()` apply the overrides once at start-up. Section 4 gives each stage's keys with their defaults. The constants without a key are changed in the code: the robustifier annealing ranges, the 2D joint weights of the root solve, the hip-seed thresholds, the temporal weights of the hand and head stages, and the L-BFGS iteration counts.

---

## 4. Method

### 4.1 Overview

| Step | Module | Optimized | Held fixed | Evidence |
|---|---|---|---|---|
| Input preparation | `main.py` | — | — | gap-filling, keypoint weights |
| Stage 0 — shape | `temporal_window/betas.py` | `betas`, once | pose (zero) | bone lengths: upper arm, forearm, shoulder width |
| Static legs | `temporal_window/legs.py` | — | hips, knees, ankles, set from the GB camera | GB SMPLer-X |
| Static root | `temporal_window/root.py` | one `global_orient` + `transl` | shape, pose | 3D shoulders and seeded hips, multi-view 2D |
| Stage A — body | `temporal_window/body.py` | `body_pose` (except legs), `global_orient`, `transl`, per frame | shape, legs | 3D body keypoints |
| Stage B — hands | `temporal_window/hands.py` | hand poses, shoulders, elbows, wrists | root, rest of the body | 3D hand and arm keypoints, WiLoR pose |
| Stage B — head | `temporal_window/head.py` | neck, head, jaw, expression, eyes | root, rest of the body | 3D face landmarks, nose, eyes, ears |
| Stage C — smoothing | `temporal_window/smoothing.py` | every trajectory, closed form | — | — |

The design follows from the recording setup. The participants sit at a table for the whole session: knees and ankles are essentially never triangulated and hips rarely (in `005013/lego_task`, p0's hips are observed in 11–16 % of the frames and the knees and ankles in none), the shoulders genuinely lean over the table, and the hips barely move when they are visible. So the shape is fitted from bone lengths before any pose, the legs are held at the seated pose seen by the only camera that sees them, the pelvis is anchored to one static solve, and frames are coupled to their neighbours inside windows instead of being fitted one by one.

**Notation.** θ = `body_pose`, φ = `global_orient` (axis-angle), t = `transl`, β = `betas`, c = keypoint confidence. Every data term uses the Geman–McClure robustifier

$$\mathrm{GMoF}_\rho(r) = \frac{\rho^2 r^2}{r^2 + \rho^2},$$

quadratic for |r| ≪ ρ and saturating at ρ² for outliers. Several stages **anneal** ρ geometrically from a coarse to a fine value over their L-BFGS steps: a fixed fine ρ gives almost no gradient to a target that starts far away, while a coarse ρ lets real outliers pull the fit. D₁(x) and D₂(x) are the mean squared first and second differences of x over the frames of a window, and R₆ is the 6D rotation representation (first two columns of the rotation matrix).

### 4.2 Input preparation

1. **Frames.** N is the number of frames in `body.npy`, limited by `--max-frames` and by the length of `smpl.npy`.
2. **Gap-filling** (Stages A and B; Stage 0 and the static root read the raw keypoints). For each keypoint, a run of at most 3 missing frames (`kp_fill_max_gap`) between two observed frames is filled by linear interpolation, with confidence 0.3 (`kp_fill_conf`). A keypoint counts as missing if its confidence is ≤ `joint_conf_threshold`. Longer gaps are left to the priors and the temporal terms. This removes the observed/unobserved flicker that would otherwise tug a limb every time a detection reappears.
3. **Weights.** Stage A uses the body rows only, with weight w = c² (`kp_conf_power` = 2) for an observed keypoint. The loss squares the weight, so low-confidence detections are suppressed as c⁴. Knees and ankles are excluded (`joints_to_ign`) and the hips multiplied by `hip_weight`. The hand and head stages use w = c.
4. **Initialization.** `smpl.npy` gives β (zero-padded to `num_betas`) and the per-frame θ, φ, t. The leg values of θ are replaced by the static leg pose (§4.4), and φ, t by the static root (§4.5).

### 4.3 Stage 0 — shape from bone lengths

SMPLer-X's shape is a good start, but its limb lengths can be off by several centimetres, which no pose can compensate. Bone lengths do not depend on the pose, so β is fitted to them once, before any pose fitting, and then shared by all frames.

- **Targets.** For each segment group (upper arm: shoulder–elbow, forearm: elbow–wrist, shoulder width), the target is the confidence-weighted median length over all frames of the clip where both end points have confidence > 0.1 (`betas_conf_thr`). Left and right are pooled into one target, because SMPL-X's shape space is left–right symmetric. A shoulder–hip "trunk" group is defined but disabled: hips are rarely triangulated in these sessions, and the few confident samples give a target that is biased short.
- **Group weight** w_g = min(1, n / 50) for n valid samples (`betas_nsat`): it depends on the number of samples, not on the fraction of frames.
- **Loss.** With ℓ_g(β) the model's segment length at zero pose, ℓ̂_g the target and β₀ the SMPLer-X shape:

$$L(\beta) = 1000 \sum_g w_g\, \mathrm{GMoF}_\rho\big(\ell_g(\beta) - \hat\ell_g\big) + 0.05\,\lVert P(\beta-\beta_0)\rVert^2 + 5\,\lVert (I-P)(\beta-\beta_0)\rVert^2$$

  P projects onto the few directions of β that the bone lengths constrain (the span of their gradients at β₀, from a QR decomposition). The shape moves freely along them, while the other directions (girth: belly, neck), which no target constrains, stay at SMPLer-X's values. Keys: `betas_len_w` = 1000, `betas_anchor_w` = 0.05, `betas_null_w` = 5.
- **Optimization.** 3 L-BFGS steps (`betas_steps`), ρ annealed from 0.15 m to 0.03 m (`betas_rho0`, `betas_rho1`). A group still more than 2ρ = 6 cm from its target at the end is reported as `[saturated …]`: the fit treated it as an outlier. If no segment is observed, or the result is not finite, the SMPLer-X shape is kept.

### 4.4 Static legs

The leg values of θ (hips, knees and ankles: 18 values) are not fitted. Only the GB (back) camera sees the legs, and its own SMPLer-X estimate articulates them well and stays almost constant over a clip. So the legs are set to the per-value median of GB's `body_pose` over the clip (`GB_smplx.npy`; camera set by `leg_pose_cam`).

SMPLer-X's hip angles are relative to its own pelvis estimate, which splits the seated ambiguity between pelvis pitch and hip flexion differently from the solved root. Once the static root is solved (§4.5), the hip rotations H are re-expressed under it so that the thighs keep the world direction GB saw: H′ = D·H, with D = R(φ_static)ᵀ · R_GBᵀ · R(φ_GB), where R_GB is the GB extrinsic rotation and φ_GB the median SMPLer-X root. Knees and ankles are relative to the thigh and stay as they are.

Without `GB_smplx.npy` a fixed template is used: hip flexion −1.1 rad, knee flexion 1.3 rad (`seated_hip_x`, `seated_knee_x`). The legs get zero gradient in Stage A (`freeze_legs: true`) and the strongest smoothing in Stage C.

### 4.5 Static root

With the hips hidden, a per-frame root would be placed by the shoulders, which lean over the table. Root position and spine bend then cannot be told apart frame by frame, and the root drifts or jitters. When the hips are visible, they barely move. So one root (φ_static, t_static) is solved for the whole sequence, and Stage A keeps every frame close to it.

**Hip seed** (`lift_hip_seed()`). Hips are almost never triangulated, but most frames have a confident 2D hip detection in at least one camera, and the ray through that detection passes within a few millimetres of the true hip: only the depth along the ray is unknown. For the frames of the solve window:

1. Cameras are kept if the projected triangulated shoulders lie within 15 px (median) of their own RTMO shoulders (GB fails this check in every session checked so far).
2. Each confident 2D hip (score ≥ 0.3) is back-projected as a ray, and its depth is solved so that the hip lies at the measured distance from both shoulders, same side and opposite side (Gauss–Newton from the far ray–sphere intersection; the two distances also resolve the near/far ambiguity). The distances are the confidence-weighted median over the clip when at least 20 frames have both shoulders and the hip observed, otherwise the rest-pose distances of the fitted shape.
3. The per-frame points (best camera per frame) are pooled into one median 3D point per side, written with confidence 0.5 into every window frame whose own hip confidence is lower.

The seed only feeds the static root solve; Stage A does not see it.

**Solve** (`solve_static_root()`).

- **Frames.** Frames `root_frame_start` … `root_frame_start + root_frame_count − 1` (the whole clip if `root_frame_count` is unset; 0–149 in the shipped configuration), every k-th frame with k = min(30, pool size / 32), at least 1 (`root_stride`), keeping the frames with at least one observed shoulder or hip. Each frame uses its initial θ with the static legs.
- **Rigid reduction.** With θ fixed, changing (φ, t) only moves the posed skeleton rigidly about the rest-pose pelvis p₀: x = R(φ)(x₀ − p₀) + p₀ + t. The zero-root joints x₀ are computed once, so the optimization never runs the body model.
- **Loss.**

| Term | Loss | Key = default |
|---|---|---|
| 3D trunk | w_D² · mean over frames of Σ w² GMoF_ρ(‖x̂ − x‖), over shoulders and hips, w = c² | `root_data_w` = 100; ρ: 0.20 → 0.05 m (code) |
| 2D | w_2D² · (1/n) Σ over cameras, frames and joints of (c · m_j)² · Σ_uv GMoF_ρ(û − u) / f² | w_2D = 100 (code); ρ: 200 → 50 px (code); detections with score < `root_conf_floor` = 0.3 ignored |
| Anchor | λ_φ ‖R₆(φ) − R₆(φ₀)‖² + λ_t ‖t − t₀‖², toward the median initial root (φ₀, t₀) | `root_go_anchor_w` = 3, `root_tr_anchor_w` = 3 |

  f is the camera's focal length and n the number of frames. The 2D joint weights m_j let the hips (2.0) set the pelvis position and the knees and ankles (1.0), which hang from the frozen legs, set its pitch; the shoulders weigh 1.0, the nose, eyes and ears 0.1, the elbows and wrists 0.05.
- **Optimization.** 3 L-BFGS steps (`root_steps`), both ρ annealed. When the GB leg pose and calibration are available, the legs are then re-aligned to the solved root (§4.4), the root is solved once more with the corrected legs, and the legs are re-aligned again.

**Free or frozen root.** By default (`freeze_root: false`), the static root is the starting point of every frame and the target of the root anchor in Stage A, which lets the pelvis follow the data by a few centimetres. With `freeze_root: true` it is used as-is for every frame; after Stage A the root is then solved once more on the fitted pose, and Stage A is re-run if the root moved by more than 5 mm or 0.5° (`root_refit`, `root_refit_thr_mm`, `root_refit_thr_deg`). With `solve_static_root: false` the per-frame SMPLer-X root is used instead.

### 4.6 Stage A — windowed body fit

Stage A fits θ, φ and t of all frames. Instead of solving frame by frame, it optimizes **windows of 32 consecutive frames jointly**, in one batched SMPL-X forward pass, with velocity and acceleration terms coupling each frame to its neighbours on both sides. Unlike a per-frame fit pulled toward the already-fitted previous frame, a poorly observed frame is interpolated from the frames around it.

**Windows.** W = 32 frames (`win_size`), overlapping by O = 8 (`win_overlap`), so windows start at frames 0, 24, 48, … The first window commits all its frames; each later window re-solves its first 8 frames, pinned by the boundary term to the values already committed, and commits the other 24. The last window is padded to 32 frames by repeating its last frame with zero data weight; padded frames are not committed. The hand and head stages use the same windows.

**Variables.** θ (63 values per frame; the 18 leg values are held fixed) and, with `freeze_root: false`, φ and t (3 + 3 per frame). β is shared and fixed.

**Loss.** Body rows 0–16 only; hands and face are fitted in Stage B.

| Term | Loss | Key = default |
|---|---|---|
| 3D data | s_d² · mean over frames of Σ_j w_j² Σ_xyz GMoF_ρ(x̂ − x) (per coordinate) | `stage_data_weights` = [100, 150, 200]; `data_rho` = 0.25 m |
| Pose prior | λ · mean GMM negative log-likelihood of θ (best of 8 components) | `lambda_pose` = 0.5 |
| Angle prior | λ · exponential barrier against elbow and knee hyper-extension | `lambda_angle` = 0.3 |
| Cervical sharing | λ · mean ‖θ_head − θ_neck‖²: neck and head bend together, without a kink or opposite twists | `lambda_cerv` = 0.1 |
| Velocity | s_t · (λ_θ D₁(θ) + λ_t D₁(t) + λ_φ D₁(R₆(φ))) | `lambda_vel_bp` / `_tr` / `_go` = 20 / 60 / 60; `stage_temporal_weights` = [2.0, 1.0, 0.5] |
| Acceleration | s_t · (λ_θ D₂(θ) + λ_t D₂(t) + λ_φ D₂(R₆(φ))) | `lambda_acc_bp` / `_tr` / `_go` = 70 / 120 / 350 |
| Root anchor | λ · (mean ‖φ − φ_static‖² + mean ‖t − t_static‖²) | `lambda_root` = 2000 |
| Orientation consensus | λ · mean of g · ‖R₆(φ) − R̄₆‖² (below) | `lambda_go_anchor` = 20 |
| Stillness | λ · mean of Σ_d m_d (θ_d − θ̄_d)² (below) | `lambda_bp_still` = 15 |
| Window boundary | λ · Σ over the 8 overlap frames of ‖θ − θ′‖² + ‖φ − φ′‖² + ‖t − t′‖² (′ = committed value) | `lambda_bnd` = 1000 |

- **Schedule.** Each window is solved in 3 sub-stages of 10 L-BFGS steps each (`stage_lbfgs_steps`): s_d and s_t move from smooth and regularized to data-driven. Acceleration is weighted more than velocity so that fast but smooth motion is not damped, only jitter.
- **Orientation consensus.** The gate g = max(0, 1 − n / n_win), with n the number of observed body keypoints in the frame and n_win the number observed anywhere in the window, is 0 for a fully observed frame and grows as keypoints drop out. The target R̄₆ is the window's mean orientation weighted by n and held constant, so poorly observed frames are pulled toward the well-observed ones and never the reverse.
- **Stillness.** Pulls θ toward a reference: the window's mean pose (held constant), or mamma's pose for that frame with `--use-mamma`. It removes the small steady wobble that the acceleration term, which only penalizes change, lets through. The neck and head are excluded (m_d = 0), since their own 3D data places them. With `freeze_root: true` the spine is excluded too, because it then carries all trunk motion; with a free root, pinning the spine is what keeps root and spine from explaining the same shoulder motion twice.
- **Rotations.** The velocity and acceleration of φ are measured on R₆(φ), so the 2π wrap of axis-angle at |φ| = π does not distort them. The anchor and boundary terms use φ unwrapped to the 2π-equivalent vector nearest the previous frame.

For every window the log prints the mean 3D error of the observed keypoints on the committed frames (`[win f…] committed … resid=… mm`); at the end it prints the mean over all frames and the jerk of the output.

### 4.7 Stage B — hands

Root and body stay at their Stage A values, except the shoulder, elbow and wrist values of θ (18), which are refined together with the two hand poses (45 values each), so that the arm can bring the hand onto its keypoints.

| Term | Loss | Key = default |
|---|---|---|
| 3D data | w_D² · mean over frames of Σ_j w_j² GMoF_ρ(‖x̂ − x‖), over the 42 hand keypoints and the elbows and wrists (w = 2c for these, so that 4 arm points are not outvoted by 40 finger points) | `hand_data_w` = 50; ρ: 0.15 → 0.05 m (code) |
| WiLoR anchor | w² · Σ over hands of mean ‖h − h_WiLoR‖² | `hand_wilor_w` = 0.8 |
| Hand prior | w² · Σ over hands and frames of ‖h‖² | `hand_prior_w` = 0.1 |
| Arm anchor | λ · mean ‖θ_arm − θ_arm(Stage A)‖² | `hand_arm_anchor` = 0.5 |
| Temporal | 5 Σ D₁ + 15 Σ D₂, over both hands and θ_arm | code |
| Window boundary | as in Stage A, on the hands and θ_arm | `lambda_bnd` = 1000 |

The hand poses start from the per-frame WiLoR MANO pose stored in the hand files, which is also the anchor target; frames without one start from, and are pulled toward, the flat hand. Each window runs 15 L-BFGS steps (`hand_steps`) with ρ annealed. The stage is skipped if there are no hand keypoints and no WiLoR poses. An arm-only placement phase before the joint solve exists but is disabled (`hand_place_steps: 0`).

### 4.8 Stage B — head

Root and body stay fixed, except the neck and head values of θ (6), which are refined together with the jaw pose, the 10 expression coefficients and the two eye poses.

The 51 inner face landmarks (rows 76–126) are compared with points on the mesh surface, interpolated with the landmark embedding of the SMPL-X model file (`lmk_faces_idx`, `lmk_bary_coords`), so they follow jaw and expression; the face contour is not used. These landmarks lie roughly on one plane and leave the rotation of the skull about it loose; the nose, eyes and ears fix it. Triangulated ears can be biased by several centimetres, so they get a lower weight and a fixed, tight robustifier: a biased ear saturates as an outlier instead of rotating the skull.

| Term | Loss | Key = default |
|---|---|---|
| Face landmarks | w² · mean over frames of Σ c² GMoF_ρ(‖x̂ − x‖), 51 landmarks | `head_face_w` = 60; ρ: 0.20 → 0.05 m (code) |
| Head keypoints | mean over frames of Σ_j c² k_j² GMoF_ρj(‖x̂ − x‖), nose, eyes and ears | k = 10 (nose, eyes), 2.5 (ears, see §7); ρ annealed as above for nose and eyes, `head_ear_rho` = 0.02 m fixed for the ears |
| Neutral head | w² · mean ‖θ_neck,head‖² | `head_pose_w` = 0.1 |
| Cervical sharing | λ · mean ‖θ_head − θ_neck‖² | `lambda_cerv` = 0.1 |
| Stage A anchor | λ · mean ‖θ_neck,head − θ_neck,head(Stage A)‖² | `head_anchor` = 0.1 |
| Jaw prior | w² · Σ over frames of ‖jaw‖² | `head_jaw_w` = 1 |
| Expression prior | w² · mean ‖expression‖² | `head_expr_w` = 1 |
| Eye prior | w² · (mean ‖left eye‖² + mean ‖right eye‖²) | `head_eye_w` = 1 |
| Temporal | 5 Σ D₁ + 15 Σ D₂, over neck/head, jaw, expression and eyes | code |
| Window boundary | as in Stage A, on all head variables | `lambda_bnd` = 1000 |

Neck and head start from Stage A; jaw, expression and eyes start from zero. Each window runs 25 L-BFGS steps (`head_steps`). The stage is skipped without face keypoints. For the first window, the log prints the landmark error before and after the stage (`[head-init]`, `[head-final]`).

### 4.9 Stage C — whole-sequence smoothing

After the windowed stages, every trajectory is smoothed once more over the whole sequence, without windows and therefore without seams. Each channel (one parameter over time) of the fitted trajectory y is replaced by the Whittaker–Eilers smoother

$$x^\star = \arg\min_x \sum_t (x_t - y_t)^2 + \lambda \sum_t (x_{t+1} - 2x_t + x_{t-1})^2,$$

i.e. the banded linear system (I + λ D₂ᵀD₂) x = y, solved directly with `scipy.linalg.solveh_banded`. It has no phase lag, and since the mesh is a function of the parameters, smoothing the parameters smooths the mesh. Larger λ is smoother: the perceived cut-off scales roughly as λ^(−1/4) of the Nyquist frequency. λ = 0 disables a group.

| Trajectory | λ (key = default) |
|---|---|
| `body_pose` | `smooth_lam_bp` = 50 |
| leg values of `body_pose` | `smooth_lam_leg` = 1000 |
| `global_orient` (unwrapped) | `smooth_lam_go` = 200 |
| `transl` | `smooth_lam_tr` = 200 |
| hand poses | `smooth_lam_hand` = 20 |
| jaw, expression, eyes | `smooth_lam_head` = 20 |

### 4.10 Optional reference — mamma

mamma (`../mamma/`) is an independent multi-view SMPL-X fit of the same recordings, in the same world frame. With `--use-mamma`, `mamma_loader.py` loads it as a soft reference, never as a constraint:

- mamma's track ids are not this pipeline's person ids, and they can swap between mamma's processing chunks. Every 50 frames the loader picks the track whose triangulated points are nearest to this person's triangulated nose, refines each switch to the exact frame and stitches one trajectory. Tracks shorter than the video are rejected.
- mamma's θ replaces SMPLer-X's as the Stage A initialization, its (θ, φ, t) are the template and starting point of the static root solve, and its θ is the stillness reference of Stage A.
- Its legs are never used (the GB pose is), nor its shape (SMPLer-X's shape refined by Stage 0 is).

### 4.11 Optimization details

- **Optimizer.** PyTorch L-BFGS with strong-Wolfe line search and learning rate 1, created anew for each stage and sub-stage. One step runs at most 10 iterations in Stage A and in the head stage, 12 in the hand stage, and 20 in Stage 0 and the static root.
- **Keep-best.** The parameters are saved before every step. A non-finite loss ends the stage; at the end, the final state is kept only if it is finite and no worse than the best state seen, otherwise the best state is restored.
- **Loss caps.** Each loss term above 10⁵ is scaled down to 10⁵, keeping its gradient direction, so that a spike stays finite and the line search can reject the step.
- **Gradient clipping** at norm 10 in the windowed stages.
- **Models.** Two SMPL-X instances: batch size 1 (Stage 0, hip seed, diagnostics, output) and batch size 32 (static root and windowed stages).
- **Determinism.** Seeds are fixed (42) and deterministic algorithms are requested (`warn_only`). Some CUDA operations (e.g. `median`) have no deterministic version, so results can differ slightly between machines.

---

## 5. Outputs

### 5.1 Files

```text
resources/fit_results/<sid>_cfg<X>/<activity>/
├── p0/
│   ├── body_smplx.json        # SMPL-X parameters: one JSON object per line, one line per frame
│   ├── meshes/                # <frame:06d>_fit.obj, only with save_mesh: true (the folder is always created)
│   ├── config_used.yaml       # copy of the configuration
│   ├── fit_log.txt            # full log of this person's fit
│   ├── timing.json            # Stage A wall time and per-frame mesh-export time
│   └── failed_frames.txt      # only if some frames could not be written
├── p1/
│   └── …
└── <camera>_fit_render.mp4    # written by visualization/vis_fit_on_video.py
```

`<X>` comes from the configuration file name: `fit_smplx_final.yaml` writes to `<sid>_cfgfinal`.

### 5.2 `body_smplx.json`

Each line is one frame:

| Key | Shape | Description |
|---|---|---|
| `frame_idx` | int | Frame index in the source video (0-based). |
| `betas` | (16,) | Shape (`num_betas`), identical in every frame. |
| `global_orient` | (3,) | Root rotation, axis-angle, world frame. Unwrapped so that consecutive frames are continuous. |
| `transl` | (3,) | Root translation, metres, world frame. |
| `body_pose` | (63,) | 21 body joints × axis-angle, relative to the parent joint. |
| `left_hand_pose`, `right_hand_pose` | (45,) each | 15 finger joints × axis-angle; zero is the flat hand (`flat_hand_mean`). |
| `jaw_pose`, `leye_pose`, `reye_pose` | (3,) each | Axis-angle. |
| `expression` | (10,) | Expression coefficients. |

To get the mesh, re-create the model with the settings of the fit (neutral, no hand PCA, flat hand mean). From the repository root:

```python
import json, torch, smplx

rows = [json.loads(l) for l in open('../../resources/fit_results/005013_cfgfinal/lego_task/p0/body_smplx.json')]
model = smplx.create('models', model_type='smplx', gender='neutral',
                     num_betas=len(rows[0]['betas']), use_pca=False, flat_hand_mean=True)
frame = rows[0]
out = model(**{k: torch.tensor([v], dtype=torch.float32) for k, v in frame.items() if k != 'frame_idx'})
vertices = out.vertices[0]        # (10475, 3), metres, world frame; same as meshes/<frame>_fit.obj
```

### 5.3 Reading the log

`fit_log.txt` reports, stage by stage, the quantities worth checking:

| Line | What it tells |
|---|---|
| `[kp gapfill] interpolated N keypoint-frame(s)` | How much gap-filling was needed. |
| `[static legs] GB median over N frames hipX … kneeX …` | Source and angles of the leg pose; `keeping the SEATED_LEGS template` means the fixed template was used. |
| `[stage0 betas] <group> tgt=… (n=…) model A → B cm` | Bone-length target, and model length before → after. `[saturated …]`: the target stayed more than 6 cm away and was treated as an outlier. |
| `[hip seed] … good_cams=[…] window=[a,b)` | Cameras used for the hip seed. |
| `[static root] per-joint 3D resid (median mm)` and `fit N frames … trunk resid p50/p95 … Δinit-median …` | Fit of the static root on the shoulders and hips, and how far it moved from the initial root. |
| `[static legs] hips re-aligned … (pelvis delta X°)` | Rotation between GB's pelvis and the solved root. |
| `[win fNNNNN] committed n resid=… mm` | Mean 3D error of the observed body keypoints, per window. |
| `[stageA] mean committed residual`, `output accel` | Stage A summary. A steady error with lower jerk means jitter was removed, not motion damped. |
| `[arm-resid] after …`, `[R-arm bones]` | Elbow and wrist errors (mm) after Stage A, the hand stage and Stage C; model vs triangulated right-arm bone lengths. |
| `[head-init]`, `[head-final] lmk dist mean/max` | Face-landmark error before and after the head stage (first window). |
| `[stageC smooth] <trajectory> λ=… accel A → B` | Effect of the final smoothing. |
| `[timing] …` | Wall time per stage. |

Every `log_every` (50) loss evaluations, the optimizations also print their loss terms (`[win …]`, `[hand …]`, `[head …]`, `[root]`).

---

## 6. Visualization and export

The two visualization tools read the per-frame meshes, so the fit must run with **`save_mesh: true`** (the shipped configuration has `false`).

- **`visualization/vis_fit_results_viser.py`** — interactive 3D viewer (viser) at http://localhost:8080: both people's meshes over the triangulated keypoints they were fitted to (body red, left hand green, right hand blue, face yellow), with a frame slider and playback.
  ```bash
  python visualization/vis_fit_results_viser.py --sid 005013 --activity lego_task --cfg final
  ```
  Other options: `--max-frames`, `--fps`, `--autoplay`, `--mesh-opacity`, `--point-size`, `--up`. If you use `--scene-dir` with a `<sid>_cfg<X>` folder, also pass `--cfg ""`: otherwise a second `_cfg<X>` suffix is appended (the default `--cfg` is `m1`).
- **`visualization/vis_fit_on_video.py`** — draws the fitted meshes, the projected triangulated keypoints and the RTMO hip detections over the GF and GB videos (`resources/all_sessions/<sid>/<activity>/<camera>.mp4`), and writes `<camera>_fit_render.mp4` next to `p0/` and `p1/`.
  ```bash
  python visualization/vis_fit_on_video.py --sid 005013 --activities lego_task --cfg final --max-frames 600
  ```
- **`visualization/vis_joint_mapping.py`** — prints the joint-index tables of the upstream ADT skeleton format (not the layout of §2.2).
- **`export_kit_amass.py`** — converts `body_smplx.json` to a KIT-AMASS `.npz`: `poses` (F, 165) = root, body, jaw, eyes, hands; `trans`; `betas` (16,); the separate pose fields and `expression`. It checks that the frames are contiguous and warns if `betas` change.
  ```bash
  python export_kit_amass.py ../../resources/fit_results/005013_cfgfinal/lego_task/p0 --fps 30    # -> p0/p0_stageii.npz
  ```
  `--fps` is required (the fit stores no frame rate); `-o` sets the output path, and `--strict` turns frame gaps into errors.

---

## 7. Limitations and known issues

### Limitations

- **Seated setting by design.** The legs are never fitted, and the pelvis is held near one static root by a soft anchor that is stiff beyond about 5 cm. Standing up, walking or moving the chair cannot be represented. With the shipped configuration the static root comes from the first 150 frames, so a participant who moves later is still pulled toward the initial position.
- **Partial shape.** Only arm and shoulder lengths constrain β; trunk length and girth stay at the SMPLer-X estimate. Neutral model only.
- **3D data only after the root.** Stages A and B fit the triangulated 3D points; multi-view 2D detections are used only by the static root. Triangulation errors (e.g. biased ears) reach the fit, damped by the robustifier and the weights.
- **Hands.** Frames without a WiLoR pose start from, and are pulled toward, the flat hand.
- **Face.** No contour landmarks and 10 expression coefficients; the eyes have no data and stay near neutral.
- **No contact or collision terms.** Nothing prevents hands from going through the table or body parts from intersecting.
- **No identity handling.** `p0`/`p1` are used as numbered by the triangulation stage.
- **No resume.** Re-running overwrites, and a failed person stops `fitter_pipeline.py` (§3.1).

### Known issues (October 2026)

- `term_cap` has no effect: `_cap()` binds its default cap when it is defined (`temporal_window/_common.py:117`), so 10⁵ is always used.
- The head-stage keypoint weights are converted to integers (`dtype=torch.long`, `temporal_window/head.py:97`), so the ears weigh 2 instead of 2.5.

---

## 8. Citation

This stage builds on SMPLify-X and the SMPL-X body model:

```bibtex
@inproceedings{SMPL-X:2019,
  title     = {Expressive Body Capture: {3D} Hands, Face, and Body from a Single Image},
  author    = {Pavlakos, Georgios and Choutas, Vasileios and Ghorbani, Nima and Bolkart, Timo and Osman, Ahmed A. A. and Tzionas, Dimitrios and Black, Michael J.},
  booktitle = {Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2019}
}
```

The repository is forked from [cseeyangchen/smplifyx-skeleton](https://github.com/cseeyangchen/smplifyx-skeleton).
