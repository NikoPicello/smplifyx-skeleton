#!/usr/bin/env python3
"""Convert SMPLify-X fit output (body_smplx.json) to the KIT-AMASS .npz format.

The fitter writes one JSON object per frame to `body_smplx.json` (see main.py).
KIT-AMASS instead expects stacked (F, ...) NumPy arrays inside a single .npz:

    trans        (F, 3)    root translation, meters
    poses        (F, 165)  [root_orient(3) | pose_body(63) | pose_jaw(3)
                            | pose_eye(6) | pose_hand(90)]
    betas        (16,)     shape coefficients (constant across the sequence)
    root_orient  (F, 3)
    pose_body    (F, 63)
    pose_jaw     (F, 3)
    pose_eye     (F, 6)    [leye(3) | reye(3)]
    pose_hand    (F, 90)   [left(45) | right(45)]   (requires use_pca: False)
    gender / surface_model_type / mocap_frame_rate / mocap_time_length / num_betas

Assumptions (matching the project cfg_files):
  * use_pca: False        -> left/right_hand_pose are full 45-d axis-angle each.
  * SMPL-X (model_type: 'smplx').
  * betas are frozen across the sequence (global betas), so a single (16,) vector
    is emitted (padded with zeros if the model used fewer than 16).

Marker fields (markers_latent / latent_labels / markers_latent_vids) are NOT
emitted: they come from the source mocap and have no counterpart in a fit.

Usage:
    python export_kit_amass.py body_smplx.json --fps 30 --gender male
    python export_kit_amass.py <sequence_dir> --fps 30 -o out_stageii.npz
"""
from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import sys

import numpy as np

# Expected per-component axis-angle dimensions for SMPL-X with use_pca=False.
DIM_BODY = 63
DIM_HAND = 45   # per hand, full axis-angle (use_pca: False)
DIM_AA = 3      # root_orient / jaw / each eye


def _load_rows(json_path):
    """Read the JSON-lines `body_smplx.json` into a list of per-frame dicts."""
    rows = []
    with open(json_path, 'r') as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{json_path}:{ln}: malformed JSON line: {e}")
    if not rows:
        raise SystemExit(f"{json_path}: no frames found.")
    return rows


def _order_and_check(rows, strict):
    """Sort rows by frame_idx and report any gaps / duplicates."""
    rows = sorted(rows, key=lambda r: r['frame_idx'])
    idxs = [r['frame_idx'] for r in rows]
    if len(set(idxs)) != len(idxs):
        raise SystemExit("Duplicate frame_idx values found; aborting.")
    expected = list(range(idxs[0], idxs[0] + len(idxs)))
    if idxs != expected:
        missing = sorted(set(expected) - set(idxs))
        msg = (f"Frames are not contiguous: {len(missing)} gap(s), "
               f"e.g. {missing[:10]}{' ...' if len(missing) > 10 else ''}. "
               f"Output arrays will pack the present frames densely (no padding).")
        if strict:
            raise SystemExit("ERROR: " + msg + "  (remove --strict to proceed anyway)")
        print("WARNING: " + msg, file=sys.stderr)
    return rows


def _stack(rows, key, dim):
    """Stack one field across frames into (F, dim) float32, validating width."""
    out = np.empty((len(rows), dim), dtype=np.float32)
    for i, r in enumerate(rows):
        if key not in r:
            raise KeyError(key)
        v = np.asarray(r[key], dtype=np.float32).reshape(-1)
        if v.shape[0] != dim:
            raise SystemExit(
                f"frame {r['frame_idx']}: '{key}' has length {v.shape[0]}, "
                f"expected {dim}. (If hands look PCA-sized, the fit used "
                f"use_pca=True; this exporter needs use_pca: False.)")
        out[i] = v
    return out


def _stack_optional(rows, key, dim, label):
    """Like _stack but zero-fills (with a warning) if the field is absent.

    Older fits predate jaw/eye being written to body_smplx.json.
    """
    if key in rows[0]:
        return _stack(rows, key, dim)
    print(f"WARNING: '{key}' not present in input; filling {label} with zeros. "
          f"Re-run the fit with the updated main.py to capture real values.",
          file=sys.stderr)
    return np.zeros((len(rows), dim), dtype=np.float32)


def convert(json_path, out_path, fps, gender, num_betas, strict):
    rows = _order_and_check(_load_rows(json_path), strict)
    F = len(rows)

    root_orient = _stack(rows, 'global_orient', DIM_AA)
    pose_body = _stack(rows, 'body_pose', DIM_BODY)
    pose_jaw = _stack_optional(rows, 'jaw_pose', DIM_AA, 'pose_jaw')
    leye = _stack_optional(rows, 'leye_pose', DIM_AA, 'leye')
    reye = _stack_optional(rows, 'reye_pose', DIM_AA, 'reye')
    pose_eye = np.concatenate([leye, reye], axis=1)                  # (F, 6)
    lhand = _stack(rows, 'left_hand_pose', DIM_HAND)
    rhand = _stack(rows, 'right_hand_pose', DIM_HAND)
    pose_hand = np.concatenate([lhand, rhand], axis=1)              # (F, 90)
    trans = _stack(rows, 'transl', 3)

    poses = np.concatenate(
        [root_orient, pose_body, pose_jaw, pose_eye, pose_hand], axis=1)  # (F,165)
    assert poses.shape == (F, 165), poses.shape

    # Betas are frozen across the sequence -> emit one (num_betas,) vector,
    # padding with zeros if the model used fewer coefficients.
    b0 = np.asarray(rows[0]['betas'], dtype=np.float32).reshape(-1)
    betas = np.zeros((num_betas,), dtype=np.float32)
    betas[:min(num_betas, b0.shape[0])] = b0[:num_betas]
    # Sanity check: warn if betas actually drift between frames.
    last = np.asarray(rows[-1]['betas'], dtype=np.float32).reshape(-1)
    if b0.shape == last.shape and not np.allclose(b0, last, atol=1e-4):
        print("WARNING: betas differ between first and last frame; the AMASS "
              "format stores a single (16,) vector, using frame-0 betas.",
              file=sys.stderr)

    # Expression is preserved as a non-standard extra (not part of the 165 pose).
    expression = (_stack(rows, 'expression', len(rows[0]['expression']))
                  if 'expression' in rows[0] else None)

    payload = dict(
        gender=str(gender),
        surface_model_type='smplx',
        mocap_frame_rate=np.float64(fps),
        mocap_time_length=np.float64(F) / np.float64(fps),
        trans=trans,
        poses=poses,
        betas=betas,
        num_betas=np.int64(num_betas),
        root_orient=root_orient,
        pose_body=pose_body,
        pose_jaw=pose_jaw,
        pose_eye=pose_eye,
        pose_hand=pose_hand,
    )
    if expression is not None:
        payload['expression'] = expression  # extra, non-AMASS

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez(out_path, **payload)

    print(f"Wrote {out_path}")
    print(f"  frames F = {F}   fps = {fps}   duration = {F / fps:.3f}s")
    print(f"  gender = {gender}   num_betas = {num_betas}")
    print(f"  poses {poses.shape}  trans {trans.shape}  "
          f"pose_hand {pose_hand.shape}  betas {betas.shape}")


def _resolve_input(path):
    """Accept either a body_smplx.json file or a sequence directory holding it."""
    if os.path.isdir(path):
        cand = os.path.join(path, 'body_smplx.json')
        if not os.path.isfile(cand):
            raise SystemExit(f"No body_smplx.json in directory: {path}")
        return cand
    if not os.path.isfile(path):
        raise SystemExit(f"Input not found: {path}")
    return path


def main():
    ap = argparse.ArgumentParser(
        description="Convert body_smplx.json -> KIT-AMASS stageii .npz")
    ap.add_argument('input',
                    help="body_smplx.json file, or a sequence dir containing it")
    ap.add_argument('-o', '--out', default=None,
                    help="output .npz (default: <input_dir>/<seq>_stageii.npz)")
    ap.add_argument('--fps', type=float, required=True,
                    help="mocap frame rate (Hz); sets mocap_frame_rate and "
                         "mocap_time_length = F/fps. Not tracked by the fitter.")
    ap.add_argument('--gender', default='neutral',
                    help="SMPL-X model gender used for the fit (your cfg_files "
                         "use 'male'); the KIT-AMASS copy happens to be 'neutral'.")
    ap.add_argument('--num-betas', type=int, default=16,
                    help="beta count in the output (default 16; fit uses 10, "
                         "padded with zeros).")
    ap.add_argument('--strict', action='store_true',
                    help="error out on missing/non-contiguous frames")
    args = ap.parse_args()

    json_path = _resolve_input(args.input)
    if args.out is None:
        seq_dir = os.path.dirname(os.path.abspath(json_path))
        seq_name = os.path.basename(seq_dir) or 'sequence'
        args.out = os.path.join(seq_dir, f"{seq_name}_stageii.npz")

    convert(json_path, args.out, args.fps, args.gender, args.num_betas, args.strict)


if __name__ == '__main__':
    main()
