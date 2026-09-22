# -*- coding: utf-8 -*-
"""
temporal_window.betas — Stage 0: fit the SHAPE to the observed bone lengths.

SMPLer-X betas are a good INIT but their limb lengths can be off by several cm (p0's model arm
was ~8cm shorter than the triangulated one → the elbow could NEVER fit, whatever the pose).
Bone lengths are pose-INVARIANT (skeleton segments move rigidly), so betas can be fit directly
to the median observed segment lengths — no pose estimate needed, no chicken-and-egg with
Stage A. Solved ONCE, anchored to the SMPLer-X init, then shared+frozen for the whole sequence
(cross-frame consistency preserved). Segments = COCO-17 mapped-joint pairs; each is weighted by
how often both endpoints are observed, so an unseen limb (p1's Rwrist) just drops out.
Symmetric groups: L/R pooled into ONE target. SMPL-X's shape space is bilaterally symmetric —
it CANNOT give the two arms different lengths — so per-side targets over-specify and force the
fit to chase the impossible. Both people's LEFT arm also triangulates ~3cm shorter than the
right (systematic bias), so within a pool the samples are conf-weighted: the better-observed
side arbitrates. Fully self-contained: no dependency on any other temporal_window submodule.
"""
from __future__ import absolute_import, print_function, division

import torch

BETAS_SEGMENTS = {
    'upperarm':  [(5, 7), (6, 8)],
    'forearm':   [(7, 9), (8, 10)],
    'shoulders': [(5, 6)],
    # shoulder→hip. Not a rigid bone, but ~straight at seated spine curvatures (chord error <1%)
    # and the model side is measured at zero pose. THE missing constraint behind the arched back:
    # SMPLer-X betas gave a torso 7-13cm TOO LONG (model 55cm vs gt ~47cm; p0 has 4768 3D hip
    # obs), so with the shoulders pinned the pelvis sat 7-12cm BELOW the real hips (3D cloud and
    # both cameras agree) — the belly stretched forward, the back arched, and no root solve or
    # spine prior could ever fix it.
    # 'trunk':     [(5, 11), (6, 12)],
}
BETAS_NSAT     = 50     # samples at which a segment target reaches full fit weight: a robust
                        # median needs ~dozens of SAMPLES, not a high observation RATE — the old
                        # rate-based weight would zero p1's trunk (232 clean samples = 1% of
                        # frames) even though its torso is ~10cm off.
BETAS_LEN_W    = 1e3    # bone-length data term (m² residuals → O(1) loss)
# GMoF (m) on segment residuals, ANNEALED coarse→fine over BETAS_STEPS -- same rho-saturation
# trap as ROOT_RHO0/ROOT_RHO1 (see that comment): a FIXED tight rho can't tell "genuinely
# corrupt" from "real but large" apart -- a well-supported target (high n, full BETAS_NSAT
# weight) sitting >2*rho from the model gets near-zero gradient either way and never moves,
# even across all BETAS_STEPS (seen directly: a 296-sample trunk target 10+cm from the model,
# full fit weight, moved under 1cm in 3 steps at a fixed 0.03). Start wide enough that a real
# discrepancy of that size can actually pull the fit; finish at the old fixed value so a truly
# corrupt target (p1's 21cm L forearm, its only observed side) still SATURATES at the end
# instead of hijacking the coupled shape directions and dragging the whole arm short.
BETAS_RHO0, BETAS_RHO1 = 0.15, 0.03
BETAS_ANCHOR_W = 0.05    # stay near the SMPLer-X init ALONG the bone-length directions (loose:
                        # the length data must win there — 1.0 under-converged the arm)
BETAS_NULL_W   = 5.0    # anchor ORTHOGONAL to the length Jacobian: 3 length targets constrain
                        # ≤3 of the 16 betas dims; the other 13 (girth — belly, neck thickness)
                        # have NO data and drifted |Δ|≈2σ under the loose anchor (doming abdomen).
BETAS_STEPS    = 3
BETAS_CONF_THR = 0.1    # a segment endpoint below this conf doesn't count as observed

_CFG_OVERRIDABLE = [
    'BETAS_STEPS', 'BETAS_NSAT', 'BETAS_LEN_W', 'BETAS_RHO0', 'BETAS_RHO1', 'BETAS_ANCHOR_W',
    'BETAS_NULL_W', 'BETAS_CONF_THR',
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


def refine_betas_bone_lengths(model_1, betas0, gt_joints_all, conf_all):
    """Refine the shared betas so the model's skeleton matches the OBSERVED bone lengths.
        model_1:      batch-1 SMPLX model (used at zero pose; lengths are pose-invariant)
        betas0:       (1, B) SMPLer-X init — warm start AND anchor
        gt_joints_all (N, J, 3), conf_all (N, J): triangulated keypoints + validity*conf
    Target per symmetric group = conf-WEIGHTED MEDIAN over all (frame, side) samples; residuals
    are GMoF-robustified so one corrupt segment can't hijack the coupled shape directions.
    Returns the refined betas (1, B), detached. Prints per-group model-vs-gt lengths."""
    device, dt = betas0.device, betas0.dtype

    def _wmedian(v, w):
        i = torch.argsort(v)
        v, w = v[i], w[i]
        c = torch.cumsum(w, 0)
        return float(v[min(int(torch.searchsorted(c, c[-1] * 0.5)), len(v) - 1)])

    # Pooled conf-weighted target per group + its share of possible observations (fit weight).
    tgt, wseg, nobs = {}, {}, {}
    for name, pairs in BETAS_SEGMENTS.items():
        vs, ws = [], []
        for a, b in pairs:
            ok = (conf_all[:, a] > BETAS_CONF_THR) & (conf_all[:, b] > BETAS_CONF_THR)
            if int(ok.sum()) == 0:
                continue
            vs.append((gt_joints_all[ok, a] - gt_joints_all[ok, b]).norm(dim=-1))
            ws.append(torch.minimum(conf_all[ok, a], conf_all[ok, b]))
        if not vs:
            continue
        v, w = torch.cat(vs), torch.cat(ws)
        tgt[name]  = _wmedian(v, w)
        wseg[name] = min(1.0, float(len(v)) / BETAS_NSAT)   # count-saturated, not rate-based
        nobs[name] = len(v)
    if not tgt:
        print("[stage0 betas] no observed segments → keeping SMPLer-X betas")
        return betas0

    zero_bp = torch.zeros(1, 63, dtype=dt, device=device)
    zero_3  = torch.zeros(1, 3,  dtype=dt, device=device)

    def _seg_lengths(b):
        J = model_1(betas=b, body_pose=zero_bp, global_orient=zero_3, transl=zero_3,
                    return_verts=False).joints[0]
        # model is L/R-symmetric; average the pair's sides anyway for numerical safety
        return {n: torch.stack([(J[a] - J[bb]).norm() for a, bb in pairs]).mean()
                for n, pairs in BETAS_SEGMENTS.items() if n in tgt}

    # Split the anchor by the length Jacobian at betas0: the fit may move freely ALONG the
    # directions the bone lengths actually constrain (row space of J), but girth & co. (the
    # null space — no data) stay pinned to SMPLer-X.
    b0g = betas0.clone().requires_grad_(True)
    Jrows = [torch.autograd.grad(v, b0g, retain_graph=True)[0].reshape(-1)
             for v in _seg_lengths(b0g).values()]
    Q, _ = torch.linalg.qr(torch.stack(Jrows).t())        # (B, G) orthonormal length directions
    P_len = (Q @ Q.t()).detach()                          # projector onto the length subspace

    betas = betas0.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([betas], lr=1.0, max_iter=20, line_search_fn='strong_wolfe')

    def closure(rho=BETAS_RHO1):
        opt.zero_grad()
        lens = _seg_lengths(betas)
        L_len = betas.new_zeros(())
        for n in lens:
            d2 = (lens[n] - tgt[n]).pow(2)
            L_len = L_len + wseg[n] * rho ** 2 * d2 / (d2 + rho ** 2)   # GMoF
        d     = (betas - betas0).reshape(-1)
        d_par = P_len @ d
        L = (BETAS_LEN_W * L_len + BETAS_ANCHOR_W * d_par.pow(2).sum()
             + BETAS_NULL_W * (d - d_par).pow(2).sum())
        L.backward()
        return L

    with torch.no_grad():
        before = {n: float(v) for n, v in _seg_lengths(betas0).items()}
    for si in range(BETAS_STEPS):
        t = si / max(BETAS_STEPS - 1, 1)                       # anneal coarse → fine
        rho = BETAS_RHO0 * (BETAS_RHO1 / BETAS_RHO0) ** t
        opt.step(lambda: closure(rho=rho))
    if not bool(torch.isfinite(betas).all()):
        print("[stage0 betas] non-finite result → keeping SMPLer-X betas")
        return betas0
    with torch.no_grad():
        after = {n: float(v) for n, v in _seg_lengths(betas).items()}
    for n in sorted(tgt):
        sat = "  [saturated: target >2*rho from model — likely corrupt]" \
            if abs(after[n] - tgt[n]) > 2 * BETAS_RHO1 else ""
        print(f"[stage0 betas] {n:10s} tgt={100*tgt[n]:5.1f}cm (n={nobs[n]:2d})  "
              f"model {100*before[n]:5.1f} → {100*after[n]:5.1f}cm{sat}")
    with torch.no_grad():
        d = (betas - betas0).reshape(-1)
        d_par = P_len @ d
    print(f"[stage0 betas] |Δbetas| along-lengths={float(d_par.norm()):.3f}  "
          f"null(girth)={float((d - d_par).norm()):.3f}")
    return betas.detach()
