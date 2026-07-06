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

All tuning lives here as module constants (no cross-module arg plumbing). main.py reads
WIN_SIZE to build the batched model.
"""
from __future__ import absolute_import, print_function, division

import math

import torch

from smplx.lbs import batch_rodrigues   # axis-angle -> rotation matrix (exp map, smooth for all theta)

from utils import aa_nearest
from cvars import LOWER_BODY_POSE_DOFS   # legs + spine DOFs (the seated anchor set)

# ── window geometry ──────────────────────────────────────────────────────────
WIN_SIZE    = 16      # frames optimised jointly (== batched model batch_size). >= N ⇒ full-seq.
WIN_OVERLAP = 8       # boundary frames pinned to the previous window's solve (>=2 ⇒ C1 seam)

# ── temporal smoothness (the new core) ───────────────────────────────────────
# Acceleration > velocity: penalise JERK, not motion, so fast-but-smooth moves aren't damped.
LAMBDA_VEL_BP, LAMBDA_ACC_BP = 8.0,  20.0
LAMBDA_VEL_TR, LAMBDA_ACC_TR = 60.0, 120.0
LAMBDA_VEL_GO, LAMBDA_ACC_GO = 60.0, 350.0

# ── anchors ──────────────────────────────────────────────────────────────────
LAMBDA_LEG  = 5.0     # under-observed seated legs/spine → hold near the SMPLer-X seated ref
LAMBDA_ROOT = 0.0     # 3D data observes the root; raise only if the trajectory drifts
LAMBDA_BND  = 1e3     # pin overlap frames to the previous window's committed solve

# ── data / priors ────────────────────────────────────────────────────────────
DATA_RHO     = 0.25   # GMoF scale (metres), matches the per-frame fit
LAMBDA_POSE  = 1.0    # GMM body-pose prior (light: Stage A refines an already-plausible init)
LAMBDA_ANGLE = 0.3    # knee/elbow hyper-extension prior

# Coarse→fine: smooth + regularise first, then let the data sharpen.
STAGE_SCHEDULE = [
    dict(data=100.0, temporal=2.0, lbfgs_steps=2),
    dict(data=150.0, temporal=1.0, lbfgs_steps=2),
    dict(data=200.0, temporal=0.5, lbfgs_steps=2),
]


# ── helpers ──────────────────────────────────────────────────────────────────
def _d1(x):
    """Mean squared velocity over the window (sum over the last/param dim, mean over time)."""
    return (x[1:] - x[:-1]).pow(2).sum(-1).mean()


def _d2(x):
    """Mean squared acceleration (second difference) over the window."""
    return (x[2:] - 2.0 * x[1:-1] + x[:-2]).pow(2).sum(-1).mean()


def _aa_unwrap(go):
    """Make a (W,3) axis-angle trajectory continuous across the |theta|=pi boundary so the
    vel/accel terms measure the true rotation change, not a spurious ~2*pi vector jump.
    Reuses utils.aa_nearest; the chosen 2*pi*m offset is constant w.r.t. the variable, so
    gradients pass straight through (same trick the per-frame global_orient anchor uses)."""
    rows = [go[:1]]
    for w in range(1, go.shape[0]):
        rows.append(aa_nearest(go[w:w + 1], rows[-1].detach()))
    return torch.cat(rows, dim=0)


def _aa_to_6d(aa):
    """(W,3) axis-angle -> (W,6) 6D rotation rep (first two columns of R), for the smoothness
    term ONLY. AA->matrix is the exponential map (batch_rodrigues), analytic for ALL theta, so
    gradients flow cleanly; the |theta|=pi pathology lives in the INVERSE (matrix->AA) map, which
    we never take. Measuring vel/accel here penalises the TRUE orientation change, free of the
    axis-angle metric distortion + 2*pi wrap near theta=pi. `go` itself stays axis-angle (this is
    a read-only metric), so the saved output needs no conversion and hits no singularity."""
    R = batch_rodrigues(aa)                       # (W,3,3)
    return R[:, :, :2].reshape(aa.shape[0], 6)


TERM_CAP = 1e5   # per-term clamp: keeps one spike finite so the line search can reject it


def _f(x):
    """Safe scalar for logging (avoids the autograd warning from float() on a grad tensor)."""
    return x.item() if torch.is_tensor(x) else float(x)


def _cap(x, cap=TERM_CAP):
    """Scale a loss term down if it exceeds `cap`, preserving gradient direction. Mirrors
    SMPLifyLoss._clamp_term — stops a finite-but-huge term (e.g. the GMM prior when a frame
    wanders out of support) from compounding to inf/nan and breaking the line search."""
    if torch.is_tensor(x):
        v = x.detach()
        if bool(torch.isfinite(v)) and float(v) > cap:
            return x * (cap / float(v))
    return x


# ── one window ───────────────────────────────────────────────────────────────
def refine_window_body(model_W, body_pose_prior, angle_prior,
                       gt_joints, weights, betas,
                       bp0, go0, tr0, bp_leg_ref, go_ref, tr_ref,
                       carry=None, frame_lo=0):
    """Jointly fit one window of W frames. Shapes (W == WIN_SIZE, J mapped joints, B betas):
        gt_joints  (W, J, 3)   weights (W, J)    betas (W, B)   [betas frozen, shared]
        bp0        (W, 63)     go0/tr0 (W, 3)                   [SMPLer-X warm start]
        bp_leg_ref (W, |static|)   go_ref/tr_ref (W, 3)         [anchor targets]
        carry: dict(k=LongTensor[O], bp=(O,63), go=(O,3), tr=(O,3)) or None (first window)
    Returns bp, go, tr each (W, ·), detached.
    """
    device = bp0.device
    bp = bp0.clone().requires_grad_(True)
    go = go0.clone().requires_grad_(True)
    tr = tr0.clone().requires_grad_(True)
    static = torch.as_tensor(LOWER_BODY_POSE_DOFS, device=device, dtype=torch.long)

    for si, st in enumerate(STAGE_SCHEDULE):
        opt = torch.optim.LBFGS([bp, go, tr], lr=1.0, max_iter=20,
                                line_search_fn='strong_wolfe')

        def closure(backward=True):
            if backward:
                opt.zero_grad()
            out = model_W(betas=betas, body_pose=bp, global_orient=go, transl=tr,
                          return_verts=False)
            r   = gt_joints - out.joints                                   # (W, J, 3)
            rob = DATA_RHO ** 2 * r.pow(2) / (r.pow(2) + DATA_RHO ** 2)     # GMoF
            L_data = (weights.unsqueeze(-1) ** 2 * rob).sum(dim=(1, 2)).mean() * st['data'] ** 2

            L_pri = (body_pose_prior(bp, betas).mean() * LAMBDA_POSE
                     + angle_prior(bp).sum(-1).mean() * LAMBDA_ANGLE)

            gou = _aa_unwrap(go)    # AA-continuous: still used by the root/seam anchors below
            go6 = _aa_to_6d(go)     # rotation-faithful 6D: drives the smoothness term (no unwrap)
            tw  = st['temporal']
            L_vel = tw * (LAMBDA_VEL_BP * _d1(bp) + LAMBDA_VEL_TR * _d1(tr) + LAMBDA_VEL_GO * _d1(go6))
            L_acc = tw * (LAMBDA_ACC_BP * _d2(bp) + LAMBDA_ACC_TR * _d2(tr) + LAMBDA_ACC_GO * _d2(go6))

            L_leg  = LAMBDA_LEG * (bp[:, static] - bp_leg_ref).pow(2).sum(-1).mean()
            L_root = LAMBDA_ROOT * ((gou - go_ref).pow(2).sum(-1).mean()
                                    + (tr - tr_ref).pow(2).sum(-1).mean())

            L_bnd = bp.new_zeros(())
            if carry is not None:
                k = carry['k']
                L_bnd = LAMBDA_BND * ((bp[k] - carry['bp']).pow(2).sum()
                                      + (gou[k] - carry['go']).pow(2).sum()
                                      + (tr[k] - carry['tr']).pow(2).sum())

            # Cap each term so a single spike stays finite — strong-Wolfe can then reject the
            # step and back off (as it does for finite spikes) instead of hitting inf/nan.
            L_data, L_pri = _cap(L_data), _cap(L_pri)
            L_vel,  L_acc = _cap(L_vel),  _cap(L_acc)
            L_leg,  L_root, L_bnd = _cap(L_leg), _cap(L_root), _cap(L_bnd)

            total = L_data + L_pri + L_vel + L_acc + L_leg + L_root + L_bnd
            if backward:
                total.backward()
                torch.nn.utils.clip_grad_norm_([bp, go, tr], 10.0)
                # compact one-line log (fixed columns)
                print(f"  [win f{frame_lo:05d} s{si}] data={_f(L_data):7.3f} pri={_f(L_pri):6.3f} "
                      f"vel={_f(L_vel):6.3f} acc={_f(L_acc):6.3f} leg={_f(L_leg):6.3f} "
                      f"bnd={_f(L_bnd):7.3f} tot={_f(total):7.3f}")
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
                 bp_init, go_init, tr_init, leg_ref_all, go_ref_all, tr_ref_all):
    """Sliding-window Stage A over the full sequence. All *_all tensors are (N, ·) on device;
    betas1 is (1, B) shared+frozen. The final short window is padded to WIN_SIZE (replicated
    last frame, zero data weight, not committed). Returns bp, go, tr each (N, ·).
    """
    N = gt_joints_all.shape[0]
    W, O = WIN_SIZE, WIN_OVERLAP
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

        bp_s, go_s, tr_s = refine_window_body(
            model_W, body_pose_prior, angle_prior,
            gt_pad, w_pad, betasW,
            _pad(bp_out[sl], n), _pad(go_out[sl], n), _pad(tr_out[sl], n),
            _pad(leg_ref_all[sl], n), _pad(go_ref_all[sl], n), _pad(tr_ref_all[sl], n),
            carry=carry, frame_lo=start)

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
