# body_layout.py — single source of truth for the SMPLX body layout.
# Leaf module: imported by fitting / fit_single_frame / temporal_window / main; imports none of them.

# ── body_pose DOF groups (21 joints × 3 axis-angle) ──────────────────────────
SMPLX_BODY_JOINTS = ['left_hip','right_hip','spine1', ..., 'right_wrist']   # canonical order
JOINT_DOF = {n: range(3*k, 3*k+3) for k, n in enumerate(SMPLX_BODY_JOINTS)}

LOWER_BODY_POSE_DOFS = [
    0, 1, 2,   # left_hip
    3, 4, 5,   # right_hip
    9, 10, 11, # left_knee
    12, 13, 14,# right_knee
    18, 19, 20,# left_ankle
    21, 22, 23,# right_ankle
    27, 28, 29,# left_foot
    30, 31, 32,# right_foot
]
UPPER_BODY_POSE_DOFS = [d for d in range(63) if d not in set(LOWER_BODY_POSE_DOFS)]

# ── keypoint index ranges (51-joint skeleton) ────────────────────────────────
LH_FINGER_KPTS = list(range(18, 38))
RH_FINGER_KPTS = list(range(39, 59))

# ── seated-task pose (this task's CHOICE, not a structural fact) ──────────────
SEATED_HIP_X, SEATED_KNEE_X, SEATED_SPINE_X = -1.1, 1.3, 0.07
SEATED_POSE = {0:SEATED_HIP_X, 3:SEATED_HIP_X, 9:SEATED_KNEE_X, 12:SEATED_KNEE_X,
               6:SEATED_SPINE_X, 15:SEATED_SPINE_X, 24:SEATED_SPINE_X}


TEMPORAL_HOLD_SUPPORT = {
    0: [13, 15], 3: [15], 6: [15], 9: [15],     # left  hip / knee / ankle / foot
    1: [14, 16], 4: [16], 7: [16], 10: [16],    # right hip / knee / ankle / foot
    15: [7, 9], 17: [9], 19: [9],               # left  shoulder / elbow / wrist
    16: [8, 10], 18: [10], 20: [10],            # right shoulder / elbow / wrist
}
TEMPORAL_HOLD_MIN_MISSES = 1
TEMPORAL_BOOST = 8.
TEMPORAL_MISS_COUNT = {}

# Neck (11) + collars (12,13) sit between the frozen spine3 and the moving head/shoulders, so
# they absorb that motion and jitter. They must FOLLOW the body (not freeze), so smooth them to
# the PREVIOUS frame with this boost instead of anchoring the neck to per-frame SMPLer-X (that
# anchor is removed in fitting.py _ANCHOR_JOINT_W). Higher = smoother but laggier.
NECK_COLLAR_JOINTS = [11, 12, 13]