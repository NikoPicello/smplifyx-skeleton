# -*- coding: utf-8 -*-
"""
temporal_window.legs — static legs: hold the leg DOFs at the LEG_POSE_CAM SMPLer-X seated pose.

The legs have no 3D data (knees/ankles triangulate ~0% of the video, hips rarely) and only ONE
camera — GB, the back view — sees them. Its per-camera SMPLer-X articulates them well and is
near-constant over the clip (std <~2° per DOF), and body_pose is parent-relative so the angles
transfer with no camera transform. main.py writes the median into the init's leg cols
(load_static_leg_pose); FREEZE_LEGS then zeroes their gradient in Stage A so they never move —
forced sitting by construction, same philosophy as FREEZE_ROOT. (The old SEATED_POSE template
stays as the fallback, but its knee X = 74.5° vs the observed ~96-109° was the visible bug.)
Known residual: GB's hip angles are relative to ITS pelvis estimate, not our frozen root — if
the thighs visibly miss the GB image, the corrective is a one-shot 2D static-leg solve.

Fully self-contained: no dependency on any other temporal_window submodule.
"""
from __future__ import absolute_import, print_function, division

import os.path as osp

import numpy as np
import torch

FREEZE_LEGS  = True    # hold the leg cols at their init through Stage A (data-free DOFs)
LEG_POSE_CAM = 'GB'    # the only view that sees the legs
_LEG_COLS  = [0, 1, 2,   3, 4, 5,        # L / R hip body_pose cols
              9, 10, 11, 12, 13, 14,     # L / R knee
              18, 19, 20, 21, 22, 23]    # L / R ankle

_CFG_OVERRIDABLE = ['FREEZE_LEGS', 'LEG_POSE_CAM']


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
