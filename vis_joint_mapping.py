#!/usr/bin/env python3
"""
vis_joint_mapping.py  —  Cross-reference all joint index spaces in the pipeline.

Spaces printed in the table:
  DATA    : gt_joints / out.joints row index (0-52)
  SMPLX   : raw SMPL-X output joint index (before JointMapper), 0-126
  ADT51   : raw skeleton JSON index (0-50) from idx_mapping.txt
  ADT_SYM : symbolic name from idx_mapping.txt  (bN=body, lN=left hand, rN=right hand)
  BPOSE   : body_pose DOF slice  [start:end]  (body joints only, DOFs 0-62)
  SEGMENT : human-readable ADT segment name + SMPL-X joint name

Usage:
  conda run -n fitter python vis_joint_mapping.py
  conda run -n fitter python vis_joint_mapping.py --model-folder models/
  conda run -n fitter python vis_joint_mapping.py --model-folder models/ --gender male
"""

import argparse
import numpy as np

# ─── SMPL-X raw output joint names ────────────────────────────────────────────
# These are the joint indices in the SMPL-X output tensor BEFORE the JointMapper.
# The model outputs 127 joints total; only the ones referenced in MODEL2DATA matter.
SMPLX_NAMES = {
    # body (0-21) — also appear as kinematic DOFs in body_pose
    0:  'pelvis',
    1:  'l_hip',      2:  'r_hip',
    3:  'spine1',     4:  'l_knee',     5:  'r_knee',
    6:  'spine2',     7:  'l_ankle',    8:  'r_ankle',
    9:  'spine3',     10: 'l_foot',     11: 'r_foot',
    12: 'neck',
    13: 'l_collar',   14: 'r_collar',
    15: 'head',
    16: 'l_shoulder', 17: 'r_shoulder',
    18: 'l_elbow',    19: 'r_elbow',
    20: 'l_wrist',    21: 'r_wrist',
    # extra body (jaw, eyes, toes come from extra joint regressor)
    22: 'jaw',        23: 'l_eye',      24: 'r_eye',
    60: 'l_toe',      63: 'r_toe',
    # left hand finger joints (25-39)
    25: 'l_index1',   26: 'l_index2',   27: 'l_index3',
    28: 'l_middle1',  29: 'l_middle2',  30: 'l_middle3',
    31: 'l_pinky1',   32: 'l_pinky2',   33: 'l_pinky3',
    34: 'l_ring1',    35: 'l_ring2',    36: 'l_ring3',
    37: 'l_thumb1',   38: 'l_thumb2',   39: 'l_thumb3',
    # right hand finger joints (40-54)
    40: 'r_index1',   41: 'r_index2',   42: 'r_index3',
    43: 'r_middle1',  44: 'r_middle2',  45: 'r_middle3',
    46: 'r_pinky1',   47: 'r_pinky2',   48: 'r_pinky3',
    49: 'r_ring1',    50: 'r_ring2',    51: 'r_ring3',
    52: 'r_thumb1',   53: 'r_thumb2',   54: 'r_thumb3',
}

# ─── body_pose DOF order (fit_single_frame.py lines 40-45) ────────────────────
# body_pose[i*3 : i*3+3] = local axis-angle of BPOSE_ORDER[i]
# NOTE: this ordering is NOT the same as the SMPL-X output joint order above.
BPOSE_ORDER = [
    'l_hip',    'r_hip',    'spine1',
    'l_knee',   'r_knee',   'spine2',
    'l_ankle',  'r_ankle',  'spine3',
    'l_foot',   'r_foot',   'neck',
    'l_collar', 'r_collar', 'head',
    'l_shoulder','r_shoulder',
    'l_elbow',  'r_elbow',
    'l_wrist',  'r_wrist',
]
BPOSE_DOF = {name: f'{i*3}:{i*3+3}' for i, name in enumerate(BPOSE_ORDER)}

# ─── model2data mapping (data_parser.py  get_model2data, adt51+smplx+hands) ───
# MODEL2DATA[data_idx] = SMPL-X output joint index
# This is what JointMapper uses: out.joints is reindexed so that
#   out.joints[:, data_idx, :] == raw_smplx_joints[:, MODEL2DATA[data_idx], :]
BODY_MAP  = [0, 3, 9, 12, 15, 13, 16, 18, 20, 14, 17, 19, 21, 1, 4, 7, 60, 2, 5, 8, 63]
LHAND_MAP = [20, 37, 38, 39, 25, 26, 27, 28, 29, 30, 34, 35, 36, 31, 32, 33]
RHAND_MAP = [21, 52, 53, 54, 40, 41, 42, 43, 44, 45, 49, 50, 51, 46, 47, 48]
MODEL2DATA = BODY_MAP + LHAND_MAP + RHAND_MAP   # length 53

# Duplicate wrist entries: data[8]  and data[21] both → SMPL-X 20 (l_wrist)
#                          data[12] and data[37] both → SMPL-X 21 (r_wrist)

# ─── ADT51 raw indices → symbolic name (from idx_mapping.txt) ─────────────────
# bN  = body joint, references SMPL-X joint index N directly
# lN  = left  hand, mediapipe-style index N (0=wrist, 1-3=thumb, 5-7=index, …)
# rN  = right hand, same convention
ADT51_SYM = {
     0: 'b0',   1: 'b3',   2: 'b9',   3: 'b12',  4: 'b15',
     5: 'b13',  6: 'b16',  7: 'b18',
     8: 'l0',   9: 'l1',  10: 'l2',  11: 'l3',
    12: 'l5',  13: 'l6',  14: 'l7',
    15: 'l9',  16: 'l10', 17: 'l11',
    18: 'l13', 19: 'l14', 20: 'l15',
    21: 'l17', 22: 'l18', 23: 'l19',
    24: 'b14', 25: 'b17', 26: 'b19',
    27: 'r0',  28: 'r1',  29: 'r2',  30: 'r3',
    31: 'r5',  32: 'r6',  33: 'r7',
    34: 'r9',  35: 'r10', 36: 'r11',
    37: 'r13', 38: 'r14', 39: 'r15',
    40: 'r17', 41: 'r18', 42: 'r19',
    43: 'b1',  44: 'b4',  45: 'b7',  46: 'b10',
    47: 'b2',  48: 'b5',  49: 'b8',  50: 'b11',
}

# ─── ADT segment names (data_parser.py comment, in data-space order 0-20) ─────
# ADT uses anatomical segment names; the SMPL-X equivalent is in parentheses.
ADT_BODY_SEG = [
    'Skeleton (pelvis)',    'Ab (spine1)',         'Chest (spine3)',
    'Neck',                 'Head',
    'LShoulder (l_collar)', 'LUArm (l_shoulder)',  'LFArm (l_elbow)',  'LHand (l_wrist)',
    'RShoulder (r_collar)', 'RUArm (r_shoulder)',  'RFArm (r_elbow)',  'RHand (r_wrist)',
    'LThigh (l_hip)',       'LShin (l_knee)',       'LFoot (l_ankle)',  'LToe',
    'RThigh (r_hip)',       'RShin (r_knee)',       'RFoot (r_ankle)',  'RToe',
]

# Mediapipe-style hand landmark names for lN / rN indices
HAND_LM = {
     0: 'wrist',
     1: 'thumb_cmc',  2: 'thumb_mcp',  3: 'thumb_ip',   4: 'thumb_tip',
     5: 'index_mcp',  6: 'index_pip',  7: 'index_dip',  8: 'index_tip',
     9: 'middle_mcp', 10: 'middle_pip',11: 'middle_dip', 12: 'middle_tip',
    13: 'ring_mcp',   14: 'ring_pip',  15: 'ring_dip',  16: 'ring_tip',
    17: 'pinky_mcp',  18: 'pinky_pip', 19: 'pinky_dip', 20: 'pinky_tip',
}

# ─── ADT51 raw → data space (read_item logic in data_parser.py) ───────────────
# body:   raw[0:9]   + raw[24:28] + raw[43:51] → data[0:21]
# l_hand: raw[8:24]                             → data[21:37]
# r_hand: raw[27:43]                            → data[37:53]
_body_raw = list(range(0, 9)) + list(range(24, 28)) + list(range(43, 51))
ADT51_TO_DATA: dict = {}
for _di, _ri in enumerate(_body_raw):
    ADT51_TO_DATA[_ri] = _di
for _i, _ri in enumerate(range(8, 24)):
    ADT51_TO_DATA[_ri] = 21 + _i
for _i, _ri in enumerate(range(27, 43)):
    ADT51_TO_DATA[_ri] = 37 + _i
DATA_TO_ADT51 = {v: k for k, v in ADT51_TO_DATA.items()}

# ─── skeleton connections in DATA space (for 3-D visualisation) ───────────────
BODY_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # pelvis→spine1→spine3→neck→head
    (2, 5), (5, 6), (6, 7), (7, 8),            # spine3 → left arm chain
    (2, 9), (9, 10), (10, 11), (11, 12),        # spine3 → right arm chain
    (0, 13), (13, 14), (14, 15), (15, 16),      # pelvis → left leg chain
    (0, 17), (17, 18), (18, 19), (19, 20),      # pelvis → right leg chain
]
# Per-hand finger chains — indices are offsets within the hand block.
# Left hand: add 21; right hand: add 37.
HAND_BONES = [
    (0, 1), (1, 2),  (2, 3),              # thumb  (wrist→cmc→mcp→ip)
    (0, 4), (4, 5),  (5, 6),              # index
    (0, 7), (7, 8),  (8, 9),              # middle
    (0, 10),(10, 11),(11, 12),             # ring
    (0, 13),(13, 14),(14, 15),             # pinky
]


# ──────────────────────────────────────────────────────────────────────────────
def build_rows() -> list[dict]:
    rows = []
    for data_idx, smplx_idx in enumerate(MODEL2DATA):
        smplx_name = SMPLX_NAMES.get(smplx_idx, f'?{smplx_idx}')
        bpose_dof  = BPOSE_DOF.get(smplx_name, '–')

        adt51_raw = DATA_TO_ADT51.get(data_idx)
        adt51_sym = ADT51_SYM.get(adt51_raw, '–') if adt51_raw is not None else '–'
        raw_str   = str(adt51_raw) if adt51_raw is not None else '–'

        if data_idx < 21:
            seg = ADT_BODY_SEG[data_idx]
        else:
            try:
                n   = int(adt51_sym[1:])
                pre = 'l_' if data_idx < 37 else 'r_'
                seg = pre + HAND_LM[n]
            except (ValueError, KeyError):
                seg = '–'

        region = 'body' if data_idx < 21 else ('l_hand' if data_idx < 37 else 'r_hand')

        note = ''
        if data_idx == 21 and smplx_idx == 20:
            note = '← same joint as data[8]'
        elif data_idx == 37 and smplx_idx == 21:
            note = '← same joint as data[12]'

        rows.append(dict(
            data_idx=data_idx, smplx_idx=smplx_idx, smplx_name=smplx_name,
            adt51_raw=raw_str, adt51_sym=adt51_sym, seg=seg,
            bpose_dof=bpose_dof, region=region, note=note,
        ))
    return rows


def print_table(rows: list[dict]):
    H = (f"{'DATA':>4}  {'SMPLX':>5}  {'SMPL-X name':<16}"
         f"  {'ADT51raw':>8}  {'ADT51sym':>8}  {'segment / joint':<28}"
         f"  {'bpose DOFs':<11}  NOTE")
    SEP = '─' * 100
    print(SEP)
    print(H)
    print(SEP)
    prev = None
    for r in rows:
        if r['region'] != prev:
            if prev is not None:
                print()
            print(f"  ── {r['region'].upper()} ──")
            prev = r['region']
        print(
            f"{r['data_idx']:>4}  {r['smplx_idx']:>5}  {r['smplx_name']:<16}"
            f"  {r['adt51_raw']:>8}  {r['adt51_sym']:>8}  {r['seg']:<28}"
            f"  {r['bpose_dof']:<11}  {r['note']}"
        )
    print(SEP)
    print()
    print("Columns:")
    print("  DATA     – index into gt_joints / out.joints (after JointMapper)")
    print("  SMPLX    – raw SMPL-X output joint index (before JointMapper)")
    print("  ADT51raw – index in the raw skeleton JSON / idx_mapping.txt (0-50)")
    print("  ADT51sym – symbolic name: bN=body SMPL-X joint N, lN/rN=mediapipe hand joint N")
    print("  bpose    – body_pose DOF slice controlling this joint's local rotation")
    print("             (only body joints; hand fingers are in left/right_hand_pose)")
    print()
    print("Key duplicates:")
    print("  data[8]  and data[21] both map to SMPL-X 20 (l_wrist) — wrist appears")
    print("  once in the body block and once as the root of the left hand block.")
    print("  data[12] and data[37] both map to SMPL-X 21 (r_wrist) — same reason.")
    print()
    print("Missing from data space (present in SMPL-X output, no ADT keypoint):")
    print("  spine2 (SMPL-X 6), jaw (22), l_eye (23), r_eye (24)")


def visualize_3d(model_folder: str, gender: str = 'neutral', save_png: str = 'joint_mapping.png'):
    import torch
    import smplx
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    print(f'Loading SMPL-X ({gender}) from {model_folder} …')
    model = smplx.create(
        model_folder, model_type='smplx', gender=gender,
        use_pca=True, num_pca_comps=12,
        flat_hand_mean=True, batch_size=1,
    )
    model.eval()
    with torch.no_grad():
        raw = model().joints[0].numpy()   # (127, 3) raw SMPL-X joint order

    # Reindex to data space (same operation as JointMapper.forward)
    pts = np.array([raw[si] for si in MODEL2DATA])   # (53, 3)

    rows  = build_rows()
    CMAP  = {'body': '#2980b9', 'l_hand': '#c0392b', 'r_hand': '#27ae60'}
    LCOL  = {'body': '#1a5276', 'l_hand': '#922b21', 'r_hand': '#1d8348'}

    fig = plt.figure(figsize=(22, 8))
    fig.suptitle(
        'Joint mapping — label format:  DATA_idx : smplx_name  (ADT51_raw)\n'
        'blue = body  |  red = left hand  |  green = right hand',
        fontsize=9,
    )

    def draw_bones(ax, bones, offset=0, color='#aab7b8', lw=0.9):
        for a, b in bones:
            p1, p2 = pts[a + offset], pts[b + offset]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                    color=color, lw=lw, zorder=1)

    def label(r):
        raw = r['adt51_raw']
        raw_part = f' ({raw})' if raw != '–' else ''
        return f" {r['data_idx']}:{r['smplx_name']}{raw_part}"

    # ── subplot 1: full body front view ──────────────────────────────────────
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.set_title('Body  (data 0-20)\nfront view  —  rotate to explore', fontsize=8)
    draw_bones(ax1, BODY_BONES, offset=0, color='#aab7b8')
    for r in rows:
        if r['region'] != 'body':
            continue
        x, y, z = pts[r['data_idx']]
        c = CMAP['body']
        ax1.scatter(x, y, z, c=c, s=30, zorder=5, depthshade=False)
        ax1.text(x, y, z, label(r), fontsize=4.2, color=LCOL['body'], zorder=6)
    ax1.set_xlabel('X', fontsize=6); ax1.set_ylabel('Y', fontsize=6); ax1.set_zlabel('Z', fontsize=6)
    ax1.tick_params(labelsize=5)
    ax1.view_init(elev=5, azim=-90)

    # ── subplot 2: left hand ──────────────────────────────────────────────────
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.set_title(
        'Left hand  (data 21-36)\n'
        'data[21]=l_wrist = same 3-D point as data[8]',
        fontsize=7.5,
    )
    draw_bones(ax2, HAND_BONES, offset=21, color='#e8b4b8')
    for r in rows:
        if r['region'] != 'l_hand':
            continue
        x, y, z = pts[r['data_idx']]
        ax2.scatter(x, y, z, c=CMAP['l_hand'], s=28, zorder=5, depthshade=False)
        ax2.text(x, y, z, label(r), fontsize=4.5, color=LCOL['l_hand'], zorder=6)
    ax2.set_xlabel('X', fontsize=6); ax2.set_ylabel('Y', fontsize=6); ax2.set_zlabel('Z', fontsize=6)
    ax2.tick_params(labelsize=5)

    # ── subplot 3: right hand ─────────────────────────────────────────────────
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.set_title(
        'Right hand  (data 37-52)\n'
        'data[37]=r_wrist = same 3-D point as data[12]',
        fontsize=7.5,
    )
    draw_bones(ax3, HAND_BONES, offset=37, color='#a9dfbf')
    for r in rows:
        if r['region'] != 'r_hand':
            continue
        x, y, z = pts[r['data_idx']]
        ax3.scatter(x, y, z, c=CMAP['r_hand'], s=28, zorder=5, depthshade=False)
        ax3.text(x, y, z, label(r), fontsize=4.5, color=LCOL['r_hand'], zorder=6)
    ax3.set_xlabel('X', fontsize=6); ax3.set_ylabel('Y', fontsize=6); ax3.set_zlabel('Z', fontsize=6)
    ax3.tick_params(labelsize=5)

    plt.tight_layout()
    if save_png:
        plt.savefig(save_png, dpi=180, bbox_inches='tight')
        print(f'Saved {save_png}')
    plt.show()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--model-folder', default=None,
                    help='SMPL-X model folder (e.g. models/)  — enables 3-D visualisation')
    ap.add_argument('--gender', default='neutral', choices=['neutral', 'male', 'female'])
    ap.add_argument('--save-png', default='joint_mapping.png',
                    help='Output PNG path for the 3-D figure (default: joint_mapping.png)')
    args = ap.parse_args()

    rows = build_rows()
    print_table(rows)

    if args.model_folder:
        visualize_3d(args.model_folder, args.gender, args.save_png)
    else:
        print('Tip: add --model-folder models/ to also produce 3-D joint position plots.')


if __name__ == '__main__':
    main()
