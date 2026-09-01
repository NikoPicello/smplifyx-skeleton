# body_layout.py — single source of truth for the SMPLX body layout.
# Leaf module: imported by fitting / main; imports none of them.

# ── body_pose DOF groups (21 joints × 3 axis-angle) ──────────────────────────
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

# ── seated-task pose (this task's CHOICE, not a structural fact) ──────────────
# Split: the windowed path templates the LEGS only (fallback when the per-camera SMPLer-X legs
# are unavailable). The SPINE must NOT be templated there: SMPLer-X sees a ~50° lumbar slouch on
# this task, and stamping +4° into the static-root template mis-pitches the frozen root — the
# spine then kinks (S-shape) or dumps the bend into the neck to reach the shoulders.
SEATED_HIP_X, SEATED_KNEE_X, SEATED_SPINE_X = -1.1, 1.3, 0.07
SEATED_LEGS  = {0: SEATED_HIP_X, 3: SEATED_HIP_X, 9: SEATED_KNEE_X, 12: SEATED_KNEE_X}
SEATED_SPINE = {6: SEATED_SPINE_X, 15: SEATED_SPINE_X, 24: SEATED_SPINE_X}
SEATED_POSE  = {**SEATED_LEGS, **SEATED_SPINE}   # full template (per-frame fit path only)


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