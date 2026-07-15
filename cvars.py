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
# Split: the windowed path templates the LEGS only (fallback when the per-camera SMPLer-X legs
# are unavailable). The SPINE must NOT be templated there: SMPLer-X sees a ~50° lumbar slouch on
# this task, and stamping +4° into the static-root template mis-pitches the frozen root — the
# spine then kinks (S-shape) or dumps the bend into the neck to reach the shoulders.
SEATED_HIP_X, SEATED_KNEE_X, SEATED_SPINE_X = -1.1, 1.3, 0.07
SEATED_LEGS  = {0: SEATED_HIP_X, 3: SEATED_HIP_X, 9: SEATED_KNEE_X, 12: SEATED_KNEE_X}
SEATED_SPINE = {6: SEATED_SPINE_X, 15: SEATED_SPINE_X, 24: SEATED_SPINE_X}
SEATED_POSE  = {**SEATED_LEGS, **SEATED_SPINE}   # full template (per-frame fit path only)


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


# ── keypoint preprocessing for Stage A (kill the flickering-detection limb jitter) ───────────
# A body keypoint that toggles observed<->unobserved (e.g. p1's left elbow, seen ~88% of frames)
# YANKS its limb toward the noisy detection each time it reappears, then releases it — the main
# source of p1's arm jitter. Two mitigations, both applied when building weights_all in main.py:
KP_CONF_POWER   = 2.0   # data weight = jw * valid * conf**this. The loss squares the weight, so
                        # the EFFECTIVE sharpening is 2*KP_CONF_POWER: a low-conf flickering
                        # detection is suppressed while a high-conf one is barely touched. 1.0 =
                        # previous behaviour (loss already effectively conf**2).
KP_FILL_MAX_GAP = 3     # linearly interpolate a keypoint across missing runs up to this many frames
                        # (removes the observed<->unobserved TOGGLE); longer gaps — e.g. a fully
                        # unseen arm — are left to the prior/smoothing. 0 disables gap-fill.
KP_FILL_CONF    = 0.3   # confidence stamped on interpolated points: guides gently, never dominates.