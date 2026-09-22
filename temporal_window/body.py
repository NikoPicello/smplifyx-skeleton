# -*- coding: utf-8 -*-
"""
Windowed temporal SMPLX fitting — Stage A: batched body + root.

Replaces the per-frame main body solve (fit_single_frame's LBFGS body loop) for the
bulk of a sequence. A window of W frames is optimised JOINTLY from the SMPLer-X warm
start, with a 3D keypoint data term + GMM/angle priors per frame and velocity +
acceleration smoothness COUPLING neighbouring frames.

Why this subsumes the per-frame temporal band-aids: today's temporal term pulls frame
t toward a *frozen* frame t-1 (a constant). Here the coupling is two-sided between
*free* variables, so:
  - an unobserved/low-conf frame is INTERPOLATED from BOTH neighbours (no occlusion
    hold / neck-collar boost needed),
  - drift is removed globally (no frame-0 root/leg anchor needed; the 3D data already
    observes the root absolutely),
  - window seams are C1 by construction — the OVERLAP frames are pinned to the
    previous window's committed solve, and acceleration couples across the seam.

Speed: batching amortises the GPU launch + LBFGS line-search overhead that dominates
batch=1, and we drop the collision BVH + nvdiffrast silhouette here. Windowing is also
what makes a 9000-frame clip fit in memory (a single batch would not).

Tuning constants for THIS stage live here as module globals (overridden via configure(),
see temporal_window/__init__.py). Window geometry (WIN_SIZE/WIN_OVERLAP), the seam-pin
weight (LAMBDA_BND) and the neck/head cervical-sharing weight (LAMBDA_CERV, also used by
head.py's Stage-B refinement of the same cols) are shared across stages and live in
_common.py instead — main.py reads WIN_SIZE from there to build the batched model.
"""
from __future__ import absolute_import, print_function, division

import math

import torch

from . import _common, root, legs, head
from ._common import _d1, _d2, _aa_unwrap, _aa_to_6d, _cap, _f

# ── temporal smoothness (the new core) ───────────────────────────────────────
# Acceleration > velocity: penalise JERK, not motion, so fast-but-smooth moves aren't damped.
LAMBDA_VEL_BP, LAMBDA_ACC_BP = 20.0,  70.0
LAMBDA_VEL_TR, LAMBDA_ACC_TR = 60.0, 120.0
LAMBDA_VEL_GO, LAMBDA_ACC_GO = 60.0, 350.0

# ── anchors ──────────────────────────────────────────────────────────────────
# THE root freeze<->free trade-off dial (only active when FREEZE_ROOT is False). Quadratic spring
# holding (go, tr) at the static-root solve: L_root = LAMBDA_ROOT * (‖go-go_ref‖² + ‖tr-tr_ref‖²),
# in rad² and m². Calibrated against the observed loss scale (data ≈ 40-90, vel/acc ≈ 0.1-1.7):
# at 2000 a 1cm deviation costs 0.2, 5cm costs 5, 10cm costs 20 — i.e. free to follow the data by
# a couple of cm, stiff beyond ~5cm. It is also the dropout guard: when a frame loses its data the
# anchor is the only term left on tr, so the root relaxes back to the static solve instead of
# excursing. Sweep DOWN for a more data-driven pelvis, UP toward frozen behaviour.
LAMBDA_ROOT = 2000.0
# LAMBDA_BND (pin overlap frames to the previous window's committed solve) is shared by every
# windowed stage and lives in _common.py — see root.FREEZE_ROOT-style cross-module note there.
LAMBDA_GO_ANCHOR = 20.0  # observability-gated pull of go toward the window's well-observed
                        # orientation; self-gates to 0 on fully-observed frames (p0 untouched).
                        # Stops an underdetermined global_orient (p1: dropped-out arm) drifting.
LAMBDA_BP_STILL  = 15.0  # ABSOLUTE "keep in place" anchor: pull each frame's body_pose toward the
                        # window-mean pose. Smoothness only penalises jerk, so a small steady wobble
                        # survives; this pins the pose to a value and kills it. Raise until the
                        # vibration stops. NOTE it resists genuine motion — under FREEZE_ROOT the
                        # SPINE carries all trunk motion (leans/slouch), so _SPINE_COLS are EXCLUDED
                        # from this pin (they'd otherwise hold the window-mean posture rigidly).
                        # With a FREE root the exclusion is DROPPED: the root now carries trunk
                        # translation, and an unpinned spine would give the two of them redundant
                        # ways to explain the same shoulder motion — that ambiguity is what makes
                        # a free root wander/jitter. Free one, pin the other (see refine_window_body).
                        # _HEAD_COLS excluded too: the real head/face 3D data should place it, not
                        # mamma's bp_ref (stale once betas no longer match mamma's own).
_SPINE_COLS = [6, 7, 8, 15, 16, 17, 24, 25, 26]   # spine1/2/3 — unpinned only under FREEZE_ROOT
# LAMBDA_CERV (neck/head cervical-bend sharing) is shared with head.py's Stage-B refinement of
# the same cols and lives in _common.py — see the cross-module note there.

# ── data / priors ────────────────────────────────────────────────────────────
DATA_RHO     = 0.25   # GMoF scale (metres), matches the per-frame fit
LAMBDA_POSE  = 0.5    # GMM body-pose prior (light: Stage A refines an already-plausible init)
LAMBDA_ANGLE = 0.3    # knee/elbow hyper-extension prior

# Coarse→fine: smooth + regularise first, then let the data sharpen.
STAGE_SCHEDULE = [
    dict(data=100.0, temporal=2.0, lbfgs_steps=10),
    dict(data=150.0, temporal=1.0, lbfgs_steps=10),
    dict(data=200.0, temporal=0.5, lbfgs_steps=10),
]

_CFG_OVERRIDABLE = [
    'LAMBDA_VEL_BP', 'LAMBDA_ACC_BP', 'LAMBDA_VEL_TR', 'LAMBDA_ACC_TR',
    'LAMBDA_VEL_GO', 'LAMBDA_ACC_GO',
    'LAMBDA_ROOT', 'LAMBDA_GO_ANCHOR', 'LAMBDA_BP_STILL',
    'DATA_RHO', 'LAMBDA_POSE', 'LAMBDA_ANGLE',
]


def configure(args):
    """Overwrite the tuning constants named in _CFG_OVERRIDABLE from a parsed cfg/CLI
    args dict (see cmd_parser.py), plus STAGE_SCHEDULE from its three parallel
    stage_* list args. Called once by temporal_window.configure(), before any fitting
    stage runs."""
    g = globals()
    for name in _CFG_OVERRIDABLE:
        key = name.lower()
        if args.get(key) is not None:
            g[name] = args[key]

    d = args.get('stage_data_weights')
    t = args.get('stage_temporal_weights')
    s = args.get('stage_lbfgs_steps')
    if d is not None or t is not None or s is not None:
        d = d if d is not None else [st['data'] for st in STAGE_SCHEDULE]
        t = t if t is not None else [st['temporal'] for st in STAGE_SCHEDULE]
        s = s if s is not None else [st['lbfgs_steps'] for st in STAGE_SCHEDULE]
        assert len(d) == len(t) == len(s), (
            f"stage_data_weights ({len(d)}), stage_temporal_weights ({len(t)}) and "
            f"stage_lbfgs_steps ({len(s)}) must all have the same length")
        g['STAGE_SCHEDULE'] = [dict(data=dd, temporal=tt, lbfgs_steps=int(ss))
                               for dd, tt, ss in zip(d, t, s)]


# ── one window ───────────────────────────────────────────────────────────────
def refine_window_body(model_W, body_pose_prior, angle_prior,
                       gt_joints, weights, betas,
                       bp0, go0, tr0, go_ref, tr_ref,
                       bp_ref=None, carry=None, frame_lo=0):
    """Jointly fit one window of W frames. Shapes (W == WIN_SIZE, J mapped joints, B betas):
        gt_joints  (W, J, 3)   weights (W, J)    betas (W, B)   [betas frozen, shared]
        bp0        (W, 63)     go0/tr0 (W, 3)                   [SMPLer-X warm start]
        go_ref/tr_ref (W, 3)                                    [anchor targets]
        bp_ref     (W, 63) or None   [stillness-anchor target; None -> window's own mean (default)]
        carry: dict(k=LongTensor[O], bp=(O,63), go=(O,3), tr=(O,3)) or None (first window)
    Returns bp, go, tr each (W, ·), detached.
    """
    device = bp0.device
    bp = bp0.clone().requires_grad_(True)
    # FREEZE_ROOT: go/tr are constants (the pre-solved static root) — body_pose carries all motion.
    # Free: they are optimised too, held near the static solve by L_root (LAMBDA_ROOT) and damped
    # by the root velocity/acceleration terms.
    go = go0.clone().requires_grad_(not root.FREEZE_ROOT)
    tr = tr0.clone().requires_grad_(not root.FREEZE_ROOT)
    params = [bp] if root.FREEZE_ROOT else [bp, go, tr]
    leg_idx = torch.as_tensor(legs._LEG_COLS, device=device, dtype=torch.long)
    # Stillness mask. Head is ALWAYS excluded (the real face 3D data should place it). The spine is
    # excluded only under FREEZE_ROOT, where it is the sole carrier of trunk motion; with a free
    # root, pinning it is what breaks the root-vs-spine ambiguity that would otherwise jitter both.
    still_w = torch.ones(63, dtype=bp0.dtype, device=device)
    _still_free = list(head._HEAD_COLS) + (list(_SPINE_COLS) if root.FREEZE_ROOT else [])
    still_w[torch.as_tensor(_still_free, device=device, dtype=torch.long)] = 0.0

    _call_i = 0
    for si, st in enumerate(STAGE_SCHEDULE):
        opt = torch.optim.LBFGS(params, lr=1.0, max_iter=10, line_search_fn='strong_wolfe')

        def closure(backward=True):
            nonlocal _call_i
            if backward:
                opt.zero_grad()
            out = model_W(betas=betas, body_pose=bp, global_orient=go, transl=tr,
                          return_verts=False)
            r   = gt_joints - out.joints                                   # (W, J, 3)
            rob = DATA_RHO ** 2 * r.pow(2) / (r.pow(2) + DATA_RHO ** 2)     # GMoF
            L_data = (weights.unsqueeze(-1) ** 2 * rob).sum(dim=(1, 2)).mean() * st['data'] ** 2

            L_pri = (body_pose_prior(bp, betas).mean() * LAMBDA_POSE
                     + angle_prior(bp).sum(-1).mean() * LAMBDA_ANGLE
                     + _common.LAMBDA_CERV * (bp[:, 42:45] - bp[:, 33:36]).pow(2).sum(-1).mean())

            gou = _aa_unwrap(go)    # AA-continuous: still used by the root/seam anchors below
            go6 = _aa_to_6d(go)     # rotation-faithful 6D: drives the smoothness term (no unwrap)
            tw  = st['temporal']
            L_vel = tw * (LAMBDA_VEL_BP * _d1(bp) + LAMBDA_VEL_TR * _d1(tr) + LAMBDA_VEL_GO * _d1(go6))
            L_acc = tw * (LAMBDA_ACC_BP * _d2(bp) + LAMBDA_ACC_TR * _d2(tr) + LAMBDA_ACC_GO * _d2(go6))

            L_root = LAMBDA_ROOT * ((gou - go_ref).pow(2).sum(-1).mean()
                                    + (tr - tr_ref).pow(2).sum(-1).mean())

            # Observability-gated go anchor. Gate = how many body joints THIS frame is missing
            # vs. the joints seen SOMEWHERE in the window: 0 when the frame is as observed as the
            # best (all of p0, and p1's good frames), rising as joints drop out (p1's arm flicker).
            # Target = the observability-weighted mean 6D orientation of the window (DETACHED, so
            # the pull is one-way: sparse frames -> the well-observed consensus, never the reverse).
            n_obs   = (weights > 0).float().sum(1)                            # (W,) observed joints/frame
            n_used  = (weights.max(0).values > 0).float().sum().clamp(min=1)  # joints seen anywhere in win
            gate    = (1.0 - n_obs / n_used).clamp(min=0.0)                   # (W,)
            cw      = (n_obs / n_obs.sum().clamp(min=1)).unsqueeze(1)         # (W,1) trust well-observed
            go6_ref = (go6.detach() * cw).sum(0, keepdim=True)               # (1,6) consensus orientation
            L_goanc = LAMBDA_GO_ANCHOR * (gate.unsqueeze(1) * (go6 - go6_ref).pow(2)).sum(-1).mean()

            # Stillness anchor: pull body_pose toward a reference pose, an absolute pin that kills
            # the residual per-frame wobble smoothing leaves behind. Default reference is the
            # window's OWN mean pose (DETACHED, self-consistency only, no data); when a real
            # per-frame reference is available (bp_ref — e.g. mamma's occlusion-gated pose for
            # this window) anchor to THAT instead, so genuine motion isn't clamped to a constant.
            # still_w masks the head always, and the spine only under FREEZE_ROOT (see above).
            still_ref = bp.detach().mean(0, keepdim=True) if bp_ref is None else bp_ref
            L_still = LAMBDA_BP_STILL * ((bp - still_ref).pow(2) * still_w).sum(-1).mean()

            L_bnd = bp.new_zeros(())
            if carry is not None:
                k = carry['k']
                L_bnd = _common.LAMBDA_BND * ((bp[k] - carry['bp']).pow(2).sum()
                                      + (gou[k] - carry['go']).pow(2).sum()
                                      + (tr[k] - carry['tr']).pow(2).sum())

            # Cap each term so a single spike stays finite — strong-Wolfe can then reject the
            # step and back off (as it does for finite spikes) instead of hitting inf/nan.
            L_data, L_pri = _cap(L_data), _cap(L_pri)
            L_vel,  L_acc = _cap(L_vel),  _cap(L_acc)
            L_root, L_bnd, L_goanc = _cap(L_root), _cap(L_bnd), _cap(L_goanc)
            L_still = _cap(L_still)

            total = L_data + L_pri + L_vel + L_acc + L_root + L_bnd + L_goanc + L_still
            if backward:
                total.backward()
                if legs.FREEZE_LEGS:
                    bp.grad[:, leg_idx] = 0.0   # no 3D leg data; hold the seated init (FREEZE_LEGS)
                torch.nn.utils.clip_grad_norm_(params, 10.0)
                # compact one-line log (fixed columns)
                _call_i += 1
                if _call_i % _common.LOG_EVERY == 0:
                  print(f"  [win f{frame_lo:05d} s{si}] data={_f(L_data):7.3f} pri={_f(L_pri):6.3f} "
                        f"vel={_f(L_vel):6.3f} acc={_f(L_acc):6.3f} "
                        f"stl={_f(L_still):6.3f} bnd={_f(L_bnd):7.3f} tot={_f(total):7.3f}")
            return total

        # Keep-best (consistent pairing). L-BFGS.step() returns the PRE-step loss, so snapshot
        # the PRE-step params to pair with it (matches fitting.run_fitting). A non-finite step
        # stops the stage. After the loop, score the final post-step state once (no backward)
        # and keep it only if it is finite and actually <= the best snapshot; else restore best.
        best_loss = float('inf')
        best_state = [p.detach().clone() for p in (bp, go, tr)]   # stage-start fallback (finite)
        for _ in range(st['lbfgs_steps']):
            snapshot = [p.detach().clone() for p in (bp, go, tr)]   # pre-step: matches `loss`
            loss = float(opt.step(closure))
            if not (math.isfinite(loss) and all(bool(torch.isfinite(p).all()) for p in (bp, go, tr))):
                print(f"  [win f{frame_lo:05d} s{si}] non-finite step → restoring best, stop stage")
                break
            if loss < best_loss:
                best_loss = loss
                best_state = snapshot
        final_loss = float(closure(backward=False))
        if not (math.isfinite(final_loss) and final_loss <= best_loss):
            with torch.no_grad():
                for p, s in zip((bp, go, tr), best_state):
                    p.data.copy_(s)

    return bp.detach(), go.detach(), tr.detach()


# ── slide windows across the whole sequence ──────────────────────────────────
def run_windowed(model_W, body_pose_prior, angle_prior,
                 gt_joints_all, weights_all, betas1,
                 bp_init, go_init, tr_init, go_ref_all, tr_ref_all,
                 bp_ref_all=None):
    """Sliding-window Stage A over the full sequence. All *_all tensors are (N, ·) on device;
    betas1 is (1, B) shared+frozen. The final short window is padded to WIN_SIZE (replicated
    last frame, zero data weight, not committed). bp_ref_all (N, 63) or None: per-frame
    stillness-anchor reference (e.g. mamma's body_pose); None falls back to each window's own
    mean (see refine_window_body). Returns bp, go, tr each (N, ·).
    """
    N = gt_joints_all.shape[0]
    W, O = _common.WIN_SIZE, _common.WIN_OVERLAP
    bp_out, go_out, tr_out = bp_init.clone(), go_init.clone(), tr_init.clone()
    betasW = betas1.expand(W, -1).contiguous()

    def _pad(x, n):
        return x if n == W else torch.cat([x, x[-1:].expand(W - n, *x.shape[1:])], dim=0)

    carry, start = None, 0
    resid_sum, resid_cnt = 0.0, 0   # guardrail: mean per-joint 3D error over COMMITTED frames
    while start < N:
        end = min(start + W, N)
        n   = end - start
        sl  = slice(start, end)

        w_pad  = weights_all[sl]
        if n < W:   # padded frames carry no data weight
            w_pad = torch.cat([w_pad, w_pad.new_zeros(W - n, w_pad.shape[1])], dim=0)
        gt_pad = _pad(gt_joints_all[sl], n)
        bp_ref_pad = None if bp_ref_all is None else _pad(bp_ref_all[sl], n)

        bp_s, go_s, tr_s = refine_window_body(
            model_W, body_pose_prior, angle_prior,
            gt_pad, w_pad, betasW,
            _pad(bp_out[sl], n), _pad(go_out[sl], n), _pad(tr_out[sl], n),
            _pad(go_ref_all[sl], n), _pad(tr_ref_all[sl], n),
            bp_ref=bp_ref_pad, carry=carry, frame_lo=start)

        commit_lo = start if carry is None else start + O   # never recommit the overlap

        # Guardrail: smoothness alone can't tell over-damping from good tracking. Report the mean
        # per-joint 3D error (mm) on the frames we actually commit, over OBSERVED joints only.
        with torch.no_grad():
            outj  = model_W(betas=betasW, body_pose=bp_s, global_orient=go_s,
                            transl=tr_s, return_verts=False).joints
            obs   = (w_pad > 0)
            pf_mm = 1000.0 * ((gt_pad - outj).norm(dim=-1) * obs).sum(1) / obs.sum(1).clamp(min=1)
        cw = torch.tensor([w for w in range(n) if start + w >= commit_lo],
                          device=pf_mm.device, dtype=torch.long)
        if len(cw):
            print(f"  [win f{start:05d}] committed {len(cw):2d}  resid={float(pf_mm[cw].mean()):6.1f} mm")
            resid_sum += float(pf_mm[cw].sum()); resid_cnt += int(len(cw))

        for w in range(n):
            f = start + w
            if f >= commit_lo:
                bp_out[f], go_out[f], tr_out[f] = bp_s[w], go_s[w], tr_s[w]

        if end == N:
            break
        start = end - O
        carry = dict(k=torch.arange(O, device=bp_out.device),
                     bp=bp_out[start:start + O].clone(),
                     go=_aa_unwrap(go_out[start:start + O]).clone(),
                     tr=tr_out[start:start + O].clone())

    if resid_cnt:
        # Companion to the residual: output jerk on the full committed trajectory (across seams).
        # global_orient in the SAME 6D metric the loss damps, so it's the knob feedback for
        # LAMBDA_ACC_GO. Resid steady + this dropping = jitter fixed, not over-damped.
        with torch.no_grad():
            bp_acc = _d2(bp_out).item()
            go_acc = _d2(_aa_to_6d(go_out)).item()
            tr_acc = _d2(tr_out).item()
        print(f"[stageA] mean committed residual: {resid_sum / resid_cnt:6.1f} mm  ({resid_cnt} frames)")
        print(f"[stageA] output accel  body_pose={bp_acc:.2e}  global_orient(6D)={go_acc:.2e}  transl={tr_acc:.2e}")
    return bp_out, go_out, tr_out
