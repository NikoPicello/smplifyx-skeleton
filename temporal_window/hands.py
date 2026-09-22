# -*- coding: utf-8 -*-
"""
temporal_window.hands — Stage B: batched, refine hand pose + arm REACH only (go/tr fixed).

Fits the 3D hand keypoints by moving the hand poses + the arm cols (shoulder/elbow/wrist) of
body_pose that carry the hand into place; go/tr and the rest of body_pose stay FIXED. WiLoR init
is the anchor + a hand prior regularises, and vel/accel coupling keeps hands smooth across the
window. (Single coherent batched solve — the per-frame version splits this into place/articulate
/snap phases; batching + coupling lets one annealed solve do the job.)
"""
from __future__ import absolute_import, print_function, division

import math

import numpy as np
import torch

from . import _common
from ._common import _d1, _d2, _cap, _f

HAND_L_KP = list(range(17, 38))            # left  hand keypoints (root + 20 fingers) in the mapping
HAND_R_KP = list(range(38, 59))            # right hand keypoints
_ARM_COLS = [45, 46, 47, 51, 52, 53, 57, 58, 59,     # L shoulder / elbow / wrist body_pose cols
             48, 49, 50, 54, 55, 56, 60, 61, 62]     # R shoulder / elbow / wrist
HAND_RHO0, HAND_RHO1 = 0.15, 0.05          # GMoF scale (m): anneal coarse→fine over the steps
HAND_DATA_W   = 50.0                        # 3D hand-keypoint data weight
HAND_WILOR_W  = 0.8                         # pull hand pose toward the WiLoR init
HAND_PRIOR_W  = 0.1                         # L2 hand-pose prior (plausible fingers)
HAND_ARM_ANCHOR = 0.5                       # keep the arm cols near the Stage-A reach (don't wander)
LAMBDA_HAND_VEL, LAMBDA_HAND_ACC = 5.0, 15.0   # temporal coupling on hand pose + arm cols
HAND_STEPS    = 15
# PLACE sub-phase (DISABLED, kept behind the flag): fit ONLY the arm keypoints with the arm cols
# first, no fingers — was added to escape a local min that turned out to be a SHAPE problem (arm
# too short from the SMPLer-X betas; see Stage 0). With betas corrected, the single mixed solve
# reaches the same arm residuals without it (A/B tested), at half the hand-stage time.
_ARM_KP = [7, 8, 9, 10, 17, 38]             # elbow(7,8) + wrist(9,10) + hand-root(17,38) keypoints
HAND_PLACE_RHO0, HAND_PLACE_RHO1 = 0.20, 0.05
HAND_PLACE_STEPS = 0                        # 0 disables the place phase

_CFG_OVERRIDABLE = [
    'HAND_DATA_W', 'HAND_WILOR_W', 'HAND_PRIOR_W', 'HAND_ARM_ANCHOR',
    'HAND_STEPS', 'HAND_PLACE_STEPS',
]


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


# ── Stage B: one window of hand refinement (hand pose + arm reach; go/tr + non-arm body fixed) ─
def refine_window_hands(model_W, left_hand_prior, right_hand_prior,
                        gt_joints, hand_w, betas, bp, go, tr,
                        lh0, rh0, wilor_lh, wilor_rh, carry=None, frame_lo=0):
    """Batched hand refinement over one window. Optimises left/right hand pose (W,45 each) + the
    arm cols of body_pose (shoulder/elbow/wrist) that reach the hand into place; go/tr and the
    non-arm body_pose stay FIXED. Fits the 3D hand keypoints (GMoF, rho annealed) with a WiLoR
    anchor, an L2 hand prior, and vel/accel temporal coupling. Returns lh, rh, bp (W,·) detached."""
    device = bp.device
    arm_idx = torch.as_tensor(_ARM_COLS, device=device, dtype=torch.long)
    bp_fixed = bp.clone()                                    # non-arm cols stay at Stage-A value
    arm      = bp[:, arm_idx].clone().requires_grad_(True)   # (W, 18) arm reach
    arm_ref  = bp[:, arm_idx].detach().clone()
    lh = lh0.clone().requires_grad_(True)                    # (W, 45)
    rh = rh0.clone().requires_grad_(True)

    # ── PLACE phase: position the whole arm from the arm keypoints ONLY (no fingers), so the
    # elbow lands on its keypoint before the finger fit can tug the wrist and stall it. ──
    arm_kp = torch.as_tensor(_ARM_KP, device=device, dtype=torch.long)
    place_opt = torch.optim.LBFGS([arm], lr=1.0, max_iter=25, line_search_fn='strong_wolfe')

    def _place(rho):
        place_opt.zero_grad()
        bpf = bp_fixed.clone(); bpf[:, arm_idx] = arm
        out = model_W(betas=betas, body_pose=bpf, global_orient=go, transl=tr,
                      left_hand_pose=lh, right_hand_pose=rh, return_verts=False)
        d2  = (gt_joints[:, arm_kp] - out.joints[:, arm_kp]).pow(2).sum(-1)
        rob = rho ** 2 * d2 / (d2 + rho ** 2)
        L = ((hand_w[:, arm_kp] ** 2 * rob).sum(1).mean() * HAND_DATA_W ** 2
             + HAND_ARM_ANCHOR * (arm - arm_ref).pow(2).sum(-1).mean()
             + LAMBDA_HAND_VEL * _d1(arm) + LAMBDA_HAND_ACC * _d2(arm))
        L.backward()
        return L

    for si in range(HAND_PLACE_STEPS):
        rho_p = HAND_PLACE_RHO0 * (HAND_PLACE_RHO1 / HAND_PLACE_RHO0) ** (si / max(HAND_PLACE_STEPS - 1, 1))
        place_opt.step(lambda: _place(rho_p))

    opt = torch.optim.LBFGS([arm, lh, rh], lr=1.0, max_iter=12, line_search_fn='strong_wolfe')
    _call_i = 0

    def closure(backward=True, rho=HAND_RHO1):
        nonlocal _call_i
        if backward:
            opt.zero_grad()
        bpf = bp_fixed.clone()
        bpf[:, arm_idx] = arm
        out = model_W(betas=betas, body_pose=bpf, global_orient=go, transl=tr,
                      left_hand_pose=lh, right_hand_pose=rh, return_verts=False)
        d2  = (gt_joints - out.joints).pow(2).sum(-1)                       # (W, J)
        rob = rho ** 2 * d2 / (d2 + rho ** 2)
        L_data = (hand_w ** 2 * rob).sum(1).mean() * HAND_DATA_W ** 2
        L_wilor = go.new_zeros(())
        if wilor_lh is not None:
            L_wilor = L_wilor + (lh - wilor_lh).pow(2).sum(-1).mean() * HAND_WILOR_W ** 2
        if wilor_rh is not None:
            L_wilor = L_wilor + (rh - wilor_rh).pow(2).sum(-1).mean() * HAND_WILOR_W ** 2
        L_prior = (left_hand_prior(lh).mean() + right_hand_prior(rh).mean()) * HAND_PRIOR_W ** 2
        L_temp  = (LAMBDA_HAND_VEL * (_d1(lh) + _d1(rh) + _d1(arm))
                   + LAMBDA_HAND_ACC * (_d2(lh) + _d2(rh) + _d2(arm)))
        L_arm   = HAND_ARM_ANCHOR * (arm - arm_ref).pow(2).sum(-1).mean()
        L_bnd = go.new_zeros(())
        if carry is not None:
            k = carry['k']
            L_bnd = _common.LAMBDA_BND * ((lh[k] - carry['lh']).pow(2).sum() + (rh[k] - carry['rh']).pow(2).sum()
                                  + (arm[k] - carry['arm']).pow(2).sum())
        total = (_cap(L_data) + _cap(L_wilor) + _cap(L_prior) + _cap(L_temp)
                 + _cap(L_arm) + _cap(L_bnd))
        if backward:
            total.backward()
            torch.nn.utils.clip_grad_norm_([arm, lh, rh], 10.0)
            _call_i += 1
            if _call_i % _common.LOG_EVERY == 0:
              print(f"  [hand f{frame_lo:05d}] data={_f(L_data):8.3f} wil={_f(L_wilor):6.3f} "
                    f"pri={_f(L_prior):6.3f} tmp={_f(L_temp):6.3f} arm={_f(L_arm):6.3f} tot={_f(total):8.3f}")
        return total

    best_loss = float('inf')
    best_state = [p.detach().clone() for p in (arm, lh, rh)]
    for si in range(HAND_STEPS):
        rho = HAND_RHO0 * (HAND_RHO1 / HAND_RHO0) ** (si / max(HAND_STEPS - 1, 1))   # anneal coarse→fine
        snapshot = [p.detach().clone() for p in (arm, lh, rh)]
        loss = float(opt.step(lambda: closure(rho=rho)))
        if not (math.isfinite(loss) and all(bool(torch.isfinite(p).all()) for p in (arm, lh, rh))):
            print(f"  [hand f{frame_lo:05d}] non-finite step → restoring best, stop")
            break
        if loss < best_loss:
            best_loss = loss; best_state = snapshot
    final = float(closure(backward=False))
    if not (math.isfinite(final) and final <= best_loss):
        with torch.no_grad():
            for p, s in zip((arm, lh, rh), best_state):
                p.data.copy_(s)

    bp_out = bp.clone()
    bp_out[:, arm_idx] = arm.detach()
    return lh.detach(), rh.detach(), bp_out


def run_windowed_hands(model_W, left_hand_prior, right_hand_prior,
                       gt_joints_all, hand_w_all, betas1, bp_all, go_all, tr_all,
                       lh_all, rh_all, wilor_lh_all, wilor_rh_all):
    """Sliding-window Stage-B hand refinement. Refines hand poses + arm reach with go/tr and the
    non-arm body_pose FIXED. Returns lh, rh each (N,45) and the updated bp (N,63)."""
    N = bp_all.shape[0]
    W, O = _common.WIN_SIZE, _common.WIN_OVERLAP
    bp_out, lh_out, rh_out = bp_all.clone(), lh_all.clone(), rh_all.clone()
    betasW = betas1.expand(W, -1).contiguous()

    def _pad(x, n):
        return x if n == W else torch.cat([x, x[-1:].expand(W - n, *x.shape[1:])], dim=0)

    carry, start = None, 0
    while start < N:
        end = min(start + W, N); n = end - start; sl = slice(start, end)
        hw = hand_w_all[sl]
        if n < W:
            hw = torch.cat([hw, hw.new_zeros(W - n, hw.shape[1])], dim=0)
        lh_s, rh_s, bp_s = refine_window_hands(
            model_W, left_hand_prior, right_hand_prior,
            _pad(gt_joints_all[sl], n), hw, betasW,
            _pad(bp_out[sl], n), _pad(go_all[sl], n), _pad(tr_all[sl], n),
            _pad(lh_out[sl], n), _pad(rh_out[sl], n),
            _pad(wilor_lh_all[sl], n), _pad(wilor_rh_all[sl], n),
            carry=carry, frame_lo=start)

        commit_lo = start if carry is None else start + O
        arm_idx = torch.as_tensor(_ARM_COLS, device=bp_out.device, dtype=torch.long)
        for w in range(n):
            f = start + w
            if f >= commit_lo:
                lh_out[f], rh_out[f] = lh_s[w], rh_s[w]
                bp_out[f, arm_idx] = bp_s[w, arm_idx]
        if end == N:
            break
        start = end - O
        carry = dict(k=torch.arange(O, device=bp_out.device),
                     lh=lh_out[start:start + O].clone(),
                     rh=rh_out[start:start + O].clone(),
                     arm=bp_out[start:start + O][:, arm_idx].clone())
    return lh_out, rh_out, bp_out


def build_hand_inputs(init_left_hand_poses, init_right_hand_poses, N, device, dtype):
    """Stack the per-frame WiLoR hand poses into (N,45) tensors (zeros where a frame lacks one).
    Returns (lh_init, rh_init, wilor_lh, wilor_rh) — the init to optimise from and the anchor
    target are both the WiLoR pose. hand_w_all (the 3D keypoint weights) is built in main.py."""
    def _stack_hand(poses):
        out = torch.zeros(N, 45, dtype=dtype, device=device)
        if poses is None:
            return out
        for i in range(N):
            if i < len(poses) and poses[i] is not None:
                out[i] = torch.as_tensor(np.asarray(poses[i], dtype=np.float32),
                                         dtype=dtype, device=device).reshape(-1)[:45]
        return out
    lh_all = _stack_hand(init_left_hand_poses)
    rh_all = _stack_hand(init_right_hand_poses)
    return lh_all.clone(), rh_all.clone(), lh_all, rh_all
