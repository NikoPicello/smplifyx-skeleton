#!/usr/bin/env python3
"""
build_skeletons_json.py

For every (session, activity, person) found in:
  resources/triangulation_results/{session}/{activity}/

Loads:
  body/p{i}_triangulated.npy        -> kpts_3d  shape (22, 3)
  mano/p{i}_left_triangulated.npy   -> kpts_3d  shape (21, 3)
  mano/p{i}_right_triangulated.npy  -> kpts_3d  shape (21, 3)

Assembles a 51-joint skeleton per frame using the index mapping defined in
samples/idx_mapping.txt (format: <out_idx> : <source><src_joint_idx>
where source is b=body, r=right hand, l=left hand).

Applies a Y-axis flip (y *= -1) to align the body coordinate frame
(triangulated SMPL-X, which comes out Y-down) with the hand coordinate frame
(MANO kpts_3d, which are Y-up).  Without this flip the body appears rotated
180 degrees around the X-axis relative to the hands.

Output: one JSONL file per (session, activity, person):
  resources/skeleton_jsons/{session}/{activity}/p{i}/skeletons.json

Each line is a JSON object: {"frame_idx": <int>, "joints": [[x,y,z], ...]}
"""

import os
import re
import sys
import glob
import json
import numpy as np
from pathlib import Path

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Parse idx_mapping.txt
# ---------------------------------------------------------------------------

def parse_idx_mapping(mapping_path: str) -> list[tuple[str, int]]:
    """Return a list of (source, joint_idx) indexed by output joint index."""
    pattern = re.compile(r'(\d+)\s*:\s*([brl])(\d+)')
    mapping = {}
    with open(mapping_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                out_idx = int(m.group(1))
                source  = m.group(2)   # 'b', 'r', or 'l'
                src_idx = int(m.group(3))
                mapping[out_idx] = (source, src_idx)
    # Validate contiguous
    n = max(mapping.keys()) + 1
    for i in range(n):
        if i not in mapping:
            raise ValueError(f"idx_mapping missing output index {i}")
    return [mapping[i] for i in range(n)]


# ---------------------------------------------------------------------------
# Per-person skeleton assembly
# ---------------------------------------------------------------------------

def assemble_skeletons(body_data, left_data, right_data, idx_mapping) -> list[dict]:
    """
    Returns a list of frame dicts with 'frame_idx' and 'joints'.

    Body and hand kpts_3d are both triangulated in the same camera world
    coordinate system (as verified in mano_triangulation.py), so no
    coordinate transformation is applied here.
    """
    frame_idxs = sorted(body_data.keys())
    records = []

    for fidx in frame_idxs:
        b_kpts = np.array(body_data[fidx]['kpts_3d'], dtype=np.float64)  # (22,3)

        l_frame = left_data.get(fidx) if left_data else None
        r_frame = right_data.get(fidx) if right_data else None
        l_kpts = (np.array(l_frame['kpts_3d'], dtype=np.float64)
                  if l_frame is not None else None)   # (21,3) or None
        r_kpts = (np.array(r_frame['kpts_3d'], dtype=np.float64)
                  if r_frame is not None else None)   # (21,3) or None

        joints = []
        for source, src_idx in idx_mapping:
            if source == 'b':
                pt = b_kpts[src_idx]
            elif source == 'r':
                pt = r_kpts[src_idx] if r_kpts is not None else np.zeros(3)
            else:  # 'l'
                pt = l_kpts[src_idx] if l_kpts is not None else np.zeros(3)
            joints.append(pt.tolist())

        records.append({'frame_idx': fidx, 'joints': joints})

    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    resources_path = os.path.normpath(
        os.path.join(_SCRIPT_DIR, '..', '..', 'resources'))
    trig_root = os.path.join(resources_path, 'triangulation_results')
    out_root  = os.path.join(resources_path, 'skeleton_jsons')

    mapping_path = os.path.join(_SCRIPT_DIR, 'cfg_files', 'idx_mapping.txt')
    idx_mapping = parse_idx_mapping(mapping_path)
    print(f"Loaded idx_mapping: {len(idx_mapping)} output joints")

    session_dirs = sorted(glob.glob(os.path.join(trig_root, '*')))
    if not session_dirs:
        print(f"No sessions found under {trig_root}")
        sys.exit(1)

    for sid_path in session_dirs:
        session_id = Path(sid_path).stem

        # Discover activities (any subdirectory that has a body/ folder)
        for activity_path in sorted(glob.glob(os.path.join(sid_path, '*'))):
            body_dir = os.path.join(activity_path, 'body')
            if not os.path.isdir(body_dir):
                continue
            activity = Path(activity_path).stem
            hand_dir = os.path.join(activity_path, 'mano')

            for person_id in range(10):  # up to p9
                body_file = os.path.join(body_dir, f'p{person_id}_triangulated.npy')
                if not os.path.isfile(body_file):
                    continue

                print(f"[{session_id}] {activity} / person {person_id}", end=' ... ')

                body_data = np.load(body_file, allow_pickle=True).item()

                lhand_file = os.path.join(hand_dir, f'p{person_id}_left_triangulated.npy')
                rhand_file = os.path.join(hand_dir, f'p{person_id}_right_triangulated.npy')
                left_data  = (np.load(lhand_file, allow_pickle=True).item()
                              if os.path.isfile(lhand_file) else None)
                right_data = (np.load(rhand_file, allow_pickle=True).item()
                              if os.path.isfile(rhand_file) else None)

                records = assemble_skeletons(
                    body_data, left_data, right_data, idx_mapping,
                )

                out_dir = os.path.join(out_root, session_id, activity, f'p{person_id}')
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, 'skeletons.json')

                with open(out_path, 'w') as f:
                    for rec in records:
                        f.write(json.dumps(rec) + '\n')

                print(f"{len(records)} frames -> {out_path}")


if __name__ == '__main__':
    main()
