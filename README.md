# SMPL-X Fitting Pipeline

Multi-view, multi-person, temporal SMPL-X fitting from triangulated 3D keypoints, built
on [SMPLify-X](https://github.com/vchoutas/smplify-x). Given 3D body/hand/face keypoints
triangulated from a multi-camera rig — plus per-frame [SMPLer-X](https://github.com/caizhongang/SMPLer-X)
pose estimates for initialization — it fits one SMPL-X mesh per person per frame across
a whole session, with temporal smoothing/anchoring across frames so the result doesn't
jitter. It can optionally fuse in an independent reference fit ("mamma") for extra
shape/root supervision.

## Contents
- [Setup](#setup)
- [Data layout](#data-layout)
- [Running a fit](#running-a-fit)
- [Visualization](#visualization)
- [Other tools](#other-tools)

## Setup

### Environment
```bash
conda create -n fitter python=3.12
conda activate fitter
```

### Core dependencies
```bash
pip install torch numpy opencv-python configargparse "smplx[all]"
pip install nvdiffrast   # see https://github.com/NVlabs/nvdiffrast for build prerequisites
```

### Optional dependencies
Both are off by default and only needed if you enable the feature they support:

- **[torch-mesh-isect](https://github.com/vchoutas/torch-mesh-isect)** — mesh
  self-intersection/collision loss, only imported when a config sets
  `interpenetration: true`.
  ```bash
  git clone https://github.com/vchoutas/torch-mesh-isect dependencies/torch-mesh-isect
  cd dependencies/torch-mesh-isect && python setup.py install && cd -
  ```
- **[human_body_prior](https://github.com/nghorbani/human_body_prior)** (VPoser) — a
  learned pose-VAE prior. The pipeline runs with `use_vposer: false` by default (the
  body pose prior is a GMM instead — see `priors/gmm_08.pkl`); only needed if you flip
  that flag on.
  ```bash
  git clone https://github.com/nghorbani/human_body_prior dependencies/human_body_prior
  cd dependencies/human_body_prior && git checkout cvpr19 && python setup.py develop && cd -
  ```

<details>
<summary>Common torch-mesh-isect build errors</summary>

Verified against a working local build — these are the fixes actually applied, not
just the error text:

1. `error: 'AT_CHECK' was not declared in this scope` — newer PyTorch removed it. In
   `src/bvh.cpp`, replace `AT_CHECK` with `TORCH_CHECK` in the `CHECK_CUDA`/
   `CHECK_CONTIGUOUS` macros. Also drop the `torch::autograd::make_variable(...)` call
   in `bvh_forward` (removed too) — just `return collisionTensor;`.
2. `fatal error: helper_math.h: No such file or directory` — download `helper_math.h`
   from the [CUDA Samples repo](https://github.com/NVIDIA/cuda-samples/tree/master/Common)
   and place it in `torch-mesh-isect/include/` (matches the build's actual include path
   — not `src/`).
3. `no suitable conversion function from "const at::DeprecatedTypeProperties" to
   "c10::ScalarType"`, or a `data<scalar_t>()` error — in `src/bvh_cuda_op.cu`, change
   `triangles.type()` to `triangles.scalar_type()` and `triangles.data<scalar_t>()` to
   `triangles.data_ptr<scalar_t>()` (both renamed in newer PyTorch).
4. Build can't find `include/`, or chokes on an unset `$CUDA_SAMPLES_INC` — in
   `setup.py`, resolve the include dir relative to `setup.py` itself
   (`osp.join(here, 'include')`) and only add `$CUDA_SAMPLES_INC` to the include path
   when that env var is actually set to a real directory.
5. On newer Thrust/CUDA, `is_valid_cnt : public thrust::unary_function<long2, int>` may
   fail to compile — drop the inheritance and add `typedef long2 argument_type;` /
   `typedef int result_type;` directly inside the struct.

</details>

### Body model
Download `models_smplx_v1_1.zip` from the [SMPL-X website](https://smpl-x.is.tue.mpg.de/)
and unzip it into `models/smplx/` (expects `SMPLX_{FEMALE,MALE,NEUTRAL}.{npz,pkl}`).

## Data layout

The pipeline reads and writes a `resources/` directory shared with the rest of the
project, two levels above this package (`../../resources/`) — not anything inside this
repo:

```
resources/
├── sessions/                # raw session data (per-camera video, session_data.txt)
├── calibs/                  # camera calibration
├── triangulation_results/   # triangulated 3D body/hand/face keypoints (input)
├── smpler_results/          # per-frame SMPLer-X pose estimates (init/prior)
├── sam_results/              # SAM segmentation masks
├── rtmo_results/             # RTMO 2D keypoints
├── mamma_results/            # external reference SMPL-X fits (optional fusion)
└── fit_results/               # OUTPUT: this pipeline's fitted meshes + params
```

## Running a fit

```bash
python fitter_pipeline.py -c cfg_files/<config>.yaml \
    --sid 005013 \
    --activities lego_task \
    --max-frames -1
```

- `-c/--config` — a config from `cfg_files/` (loss weights, which refinement stages run, etc.)
- `--sid` — session id substring, or `all` to run every session (default: `all`)
- `--activities` — one or more activity names to run (default: all five task types)
- `--max-frames` — cap frames per sequence for a quick test run; `-1` runs the full sequence

For every (session, activity, person) found under `resources/triangulation_results/`,
this fits SMPL-X and writes `body_smplx.json` + `meshes/*.obj` into
`resources/fit_results/<session>_cfg<X>/<activity>/`, where `<X>` is taken from the
config filename (`fit_smplx_<X>.yaml`).

To fit many sessions in parallel across multiple GPUs (one session per idle GPU), see
`python run_parallel_sessions.py --help`.

## Visualization

Scripts below live under `visualization/`; run them from the package root as shown.

- **`visualization/vis_fit_results_viser.py`** — interactive 3D viewer: fitted mesh
  alongside the triangulated keypoints it was fit to, colour-coded by body part.
  ```bash
  python visualization/vis_fit_results_viser.py --scene-dir ../../resources/fit_results/005013_cfg7/lego
  ```
- **`visualization/vis_fit_on_video.py`** — overlays the fitted mesh back onto the source
  session videos, one output `.mp4` per camera.
- **`visualization/vis_joint_mapping.py`** — prints a cross-reference table of every
  joint-index space used across the pipeline (raw skeleton index, SMPL-X index,
  body_pose DOF slice, ...).

## Other tools

- **`export_kit_amass.py`** — converts a fit's `body_smplx.json` to the KIT-AMASS `.npz`
  mocap format.

## Acknowledgements
Based on [SMPLify-X](https://github.com/vchoutas/smplify-x). Thanks to the authors for
their foundational work.
