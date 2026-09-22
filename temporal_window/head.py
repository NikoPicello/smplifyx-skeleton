# -*- coding: utf-8 -*-
"""
temporal_window.head — Stage B: batched, re-aim neck+head + jaw onto the face landmarks.

Refines the neck+head body_pose cols + jaw_pose to land the 51 inner face landmarks; go/tr and
the rest of body_pose stay FIXED. Uses the model's mapped face joints (indices 76:127) directly
as the landmark set, or the barycentric mesh-surface embedding when available (see
build_face_landmark_embedding). vel/accel coupling + a light anchor to Stage A keep the head
smooth.
"""
from __future__ import absolute_import, print_function, division

import math
import os.path as osp

import numpy as np
import torch

from . import _common
from ._common import _d1, _d2, _cap, _f

_HEAD_COLS   = [33, 34, 35, 42, 43, 44]     # neck (joint 11) + head (joint 14) body_pose cols
FACE_KP      = list(range(76, 127))         # 51 inner face landmarks in the mapped-joint layout
# COCO head keypoints (nose/eyes/EARS): the 51 inner-face landmarks all lie on the face plane, so
# they position the FACE but leave the skull's rotation about it soft — the head stage was landing
# the face at ~10mm while the ears drifted 30mm+ (neck bent to chase the face). The lateral ears
# pin the skull orientation.
_HEAD_KP     = [0, 1, 2, 3, 4]
# Per-keypoint weights: the EARS exist only to break the face-plane rotation ambiguity, but
# their triangulated targets can carry cm-level bias (p1: 4-5cm, opposing directions — at a
# uniform weight 10 they DICTATED a ~8° skull rotation against the 51 landmarks, parking the
# face at 13mm when the rotation-optimal fit is 4mm). Nose/eyes stay strong; ears tie-break.
_HEAD_KP_W   = [10.0, 10.0, 10.0, 2.5, 2.5]   # nose, L/R eye, L/R ear
HEAD_EAR_RHO = 0.02   # fixed GMoF (m) for the ears (not annealed): an honest ear (~2cm, p0)
                      # sits at peak GMoF influence, a biased one (4-5cm, p1) saturates as an
                      # outlier instead of twisting the skull. Dials if a skull starts tilting
                      # about the face plane: raise ear weight toward 5 / rho toward 0.03.
# GMoF scale (m), ANNEALED coarse→fine: the face starts ~7cm off, so a fixed tight rho saturates
# the robustifier and starves the gradient; start wide to snap the head in, then tighten to refine.
HEAD_RHO0, HEAD_RHO1 = 0.20, 0.05
HEAD_FACE_W   = 60.0                         # face-landmark data weight
HEAD_JAW_W    = 1.0                          # jaw L2 prior
HEAD_POSE_W   = 0.1                          # keep neck/head near neutral
HEAD_ANCHOR   = 0.1                          # keep neck/head near the Stage-A value (don't wander)
HEAD_EXPR_W   = 1.0                          # L2 expression prior (keep blendshapes plausible)
HEAD_EYE_W    = 1.0                          # L2 eye-pose reg toward neutral (eyes barely rotate)
LAMBDA_HEAD_VEL, LAMBDA_HEAD_ACC = 5.0, 15.0 # temporal coupling on neck/head + jaw + expr + eyes
HEAD_STEPS    = 25

_CFG_OVERRIDABLE = [
    'HEAD_FACE_W', 'HEAD_JAW_W', 'HEAD_POSE_W', 'HEAD_ANCHOR', 'HEAD_EXPR_W',
    'HEAD_EYE_W', 'HEAD_EAR_RHO', 'HEAD_STEPS',
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


def build_face_landmark_embedding(model_folder, gender, faces_tensor, device, dtype):
    """Load the SMPLX barycentric embedding of the 51 inner dlib face landmarks (17-67). Returns
    (lmk_faces_idx (51,), lmk_bary_coords (51,3), body_faces (F,3)) — used to interpolate the TRUE
    face landmarks on the mesh surface (they deform with expression, unlike the static face joints).
    Returns None if the SMPLX_{GENDER}.npz isn't found (head stage then falls back to model joints)."""
    npz = osp.join(osp.expandvars(model_folder), 'smplx', f'SMPLX_{gender.upper()}.npz')
    if not osp.isfile(npz):
        print(f"[stageB head] landmark embedding not found ({npz}) → falling back to model face joints")
        return None
    d = np.load(npz, allow_pickle=True)
    lfi = torch.as_tensor(np.asarray(d['lmk_faces_idx']), dtype=torch.long, device=device)          # (51,)
    lbc = torch.as_tensor(np.asarray(d['lmk_bary_coords'], dtype=np.float32), dtype=dtype, device=device)  # (51,3)
    bfl = faces_tensor.to(device=device).view(-1, 3).long()                                          # (F,3)
    return lfi, lbc, bfl


# ── Stage B: one window of head refinement (neck+head + jaw + expression + eyes) ───────────────
def refine_window_head(model_W, jaw_prior, gt_joints, face_w, betas, bp, go, tr,
                       jaw0, expr0, leye0, reye0, lmk_emb=None, carry=None, frame_lo=0):
    """Batched head refinement over one window. Optimises neck+head body_pose cols + jaw_pose +
    expression + eye pose to land the 51 inner face landmarks; go/tr and the rest of body_pose
    FIXED. If lmk_emb is given the landmarks are interpolated on the MESH SURFACE (barycentric,
    expression-deformable — the true dlib landmarks); else it falls back to the static model face
    joints (76:127). Face data (GMoF, annealed) + jaw/expr/eye L2 priors + neutral head prior +
    vel/accel coupling. Returns jaw, expr, leye, reye (W,·) and updated bp (W,63)."""
    device = bp.device
    hcols    = torch.as_tensor(_HEAD_COLS, device=device, dtype=torch.long)
    fkp      = torch.as_tensor(FACE_KP,    device=device, dtype=torch.long)
    hkp      = torch.as_tensor(_HEAD_KP,   device=device, dtype=torch.long)
    kpw      = torch.as_tensor(_HEAD_KP_W, device=device, dtype=torch.long)
    bp_fixed = bp.clone()
    head     = bp[:, hcols].clone().requires_grad_(True)     # (W, 6) neck + head
    head_ref = bp[:, hcols].detach().clone()
    jaw  = jaw0.clone().requires_grad_(True)                 # (W, 3)
    expr = expr0.clone().requires_grad_(True)                # (W, E) expression blendshapes
    leye = leye0.clone().requires_grad_(True)                # (W, 3)
    reye = reye0.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([head, jaw, expr, leye, reye], lr=1.0, max_iter=10, line_search_fn='strong_wolfe')
    _call_i = 0

    use_bary = lmk_emb is not None
    if use_bary:
        _lfi, _lbc, _bfl = lmk_emb
        _tri_idx = _bfl[_lfi]                                 # (51, 3) vertex indices per landmark

    def _model_lmk():   # -> ((W,51,3) face landmarks, (W,5,3) nose/eyes/ears joints)
        bpf = bp_fixed.clone(); bpf[:, hcols] = head
        out = model_W(betas=betas, body_pose=bpf, global_orient=go, transl=tr, jaw_pose=jaw,
                      expression=expr, leye_pose=leye, reye_pose=reye, return_verts=use_bary)
        if use_bary:
            tri = out.vertices[:, _tri_idx]                   # (W,51,3,3) mesh-surface triangle verts
            lmk = (tri * _lbc.view(1, 51, 3, 1)).sum(dim=2)   # (W,51,3) barycentric landmark
        else:
            lmk = out.joints[:, fkp]
        return lmk, out.joints[:, hkp]

    def _diag(tag):     # DIAGNOSTIC: model landmarks + head keypoints vs gt
        with torch.no_grad():
            ml, mk = _model_lmk()
            m, mk_m = face_w[:, fkp] > 0, face_w[:, hkp] > 0
            d  = (gt_joints[:, fkp] - ml).norm(dim=-1)
            dk = (gt_joints[:, hkp] - mk).norm(dim=-1)
            if bool(m.any()):
                print(f"  [{tag}] lmk dist mean={1000*d[m].mean().item():.1f}mm max={1000*d[m].max().item():.1f}mm"
                      + (f"  head-kp mean={1000*dk[mk_m].mean().item():.1f}mm" if bool(mk_m.any()) else ""))

    if frame_lo == 0:
        print(f"  [head-init] {'barycentric' if use_bary else 'model-joint'} lmk  "
              f"obs/frame={(face_w[:, fkp] > 0).sum(1).float().mean().item():.1f}/51")
        _diag('head-init')

    def closure(backward=True, rho=HEAD_RHO1):
        nonlocal _call_i
        if backward:
            opt.zero_grad()
        mlmk, mkp = _model_lmk()
        d2  = (gt_joints[:, fkp] - mlmk).pow(2).sum(-1)                     # (W, 51)
        rob = rho ** 2 * d2 / (d2 + rho ** 2)
        L_face = (face_w[:, fkp] ** 2 * rob).sum(1).mean() * HEAD_FACE_W ** 2
        d2k  = (gt_joints[:, hkp] - mkp).pow(2).sum(-1)                     # (W, 5)
        rhok = d2k.new_full((5,), rho)
        rhok[3:] = HEAD_EAR_RHO             # ears: fixed tight rho — biased targets saturate
        robk = rhok ** 2 * d2k / (d2k + rhok ** 2)
        L_kp = (face_w[:, hkp] ** 2 * kpw ** 2 * robk).sum(1).mean()
        # cervical sharing (see LAMBDA_CERV): neck and head carry the look-down TOGETHER — the
        # full-vector difference blocks both the one-joint kink and the opposing-twist candy-
        # wrapper. Deliberately NO coupling to the spine (that pulled the chest forward).
        L_pose = (HEAD_POSE_W ** 2 * head.pow(2).sum(-1).mean()
                  + _common.LAMBDA_CERV * (head[:, 3:6] - head[:, 0:3]).pow(2).sum(-1).mean())
        L_jaw  = jaw_prior(jaw).mean() * HEAD_JAW_W ** 2
        L_expr = HEAD_EXPR_W ** 2 * expr.pow(2).sum(-1).mean()
        L_eye  = HEAD_EYE_W ** 2 * (leye.pow(2).sum(-1).mean() + reye.pow(2).sum(-1).mean())
        L_anc  = HEAD_ANCHOR * (head - head_ref).pow(2).sum(-1).mean()
        L_temp = (LAMBDA_HEAD_VEL * (_d1(head) + _d1(jaw) + _d1(expr) + _d1(leye) + _d1(reye))
                  + LAMBDA_HEAD_ACC * (_d2(head) + _d2(jaw) + _d2(expr) + _d2(leye) + _d2(reye)))
        L_bnd = go.new_zeros(())
        if carry is not None:
            k = carry['k']
            L_bnd = _common.LAMBDA_BND * ((head[k] - carry['head']).pow(2).sum() + (jaw[k] - carry['jaw']).pow(2).sum()
                                  + (expr[k] - carry['expr']).pow(2).sum() + (leye[k] - carry['leye']).pow(2).sum()
                                  + (reye[k] - carry['reye']).pow(2).sum())
        total = (_cap(L_face) + _cap(L_kp) + _cap(L_pose) + _cap(L_jaw) + _cap(L_expr) + _cap(L_eye)
                 + _cap(L_anc) + _cap(L_temp) + _cap(L_bnd))
        if backward:
            total.backward()
            torch.nn.utils.clip_grad_norm_([head, jaw, expr, leye, reye], 10.0)
            _call_i += 1
            if _call_i % _common.LOG_EVERY == 0:
              print(f"  [head f{frame_lo:05d}] face={_f(L_face):8.3f} kp={_f(L_kp):6.3f} jaw={_f(L_jaw):6.3f} "
                    f"exp={_f(L_expr):6.3f} eye={_f(L_eye):6.3f} tmp={_f(L_temp):6.3f} tot={_f(total):8.3f}")
        return total

    params = [head, jaw, expr, leye, reye]
    best_loss = float('inf')
    best_state = [p.detach().clone() for p in params]
    for si in range(HEAD_STEPS):
        rho = HEAD_RHO0 * (HEAD_RHO1 / HEAD_RHO0) ** (si / max(HEAD_STEPS - 1, 1))   # anneal coarse→fine
        snapshot = [p.detach().clone() for p in params]
        loss = float(opt.step(lambda: closure(rho=rho)))
        if not (math.isfinite(loss) and all(bool(torch.isfinite(p).all()) for p in params)):
            print(f"  [head f{frame_lo:05d}] non-finite step → restoring best, stop")
            break
        if loss < best_loss:
            best_loss = loss; best_state = snapshot
    final = float(closure(backward=False))
    if not (math.isfinite(final) and final <= best_loss):
        with torch.no_grad():
            for p, s in zip(params, best_state):
                p.data.copy_(s)

    if frame_lo == 0:   # DIAGNOSTIC: landmark + head-keypoint distance AFTER refinement
        _diag('head-final')

    bp_out = bp.clone()
    bp_out[:, hcols] = head.detach()
    return jaw.detach(), expr.detach(), leye.detach(), reye.detach(), bp_out


def run_windowed_head(model_W, jaw_prior, gt_joints_all, face_w_all, betas1,
                      bp_all, go_all, tr_all, jaw_all, expr_all, leye_all, reye_all, lmk_emb=None):
    """Sliding-window Stage-B head refinement (neck+head + jaw + expression + eyes; go/tr + rest of
    body_pose FIXED). Returns jaw (N,3), expr (N,E), leye (N,3), reye (N,3) and updated bp (N,63)."""
    N = bp_all.shape[0]
    W, O = _common.WIN_SIZE, _common.WIN_OVERLAP
    bp_out  = bp_all.clone()
    jaw_out, expr_out = jaw_all.clone(), expr_all.clone()
    leye_out, reye_out = leye_all.clone(), reye_all.clone()
    betasW  = betas1.expand(W, -1).contiguous()
    hcols   = torch.as_tensor(_HEAD_COLS, device=bp_out.device, dtype=torch.long)

    def _pad(x, n):
        return x if n == W else torch.cat([x, x[-1:].expand(W - n, *x.shape[1:])], dim=0)

    carry, start = None, 0
    while start < N:
        end = min(start + W, N); n = end - start; sl = slice(start, end)
        fw = face_w_all[sl]
        if n < W:
            fw = torch.cat([fw, fw.new_zeros(W - n, fw.shape[1])], dim=0)
        jaw_s, expr_s, leye_s, reye_s, bp_s = refine_window_head(
            model_W, jaw_prior, _pad(gt_joints_all[sl], n), fw, betasW,
            _pad(bp_out[sl], n), _pad(go_all[sl], n), _pad(tr_all[sl], n),
            _pad(jaw_out[sl], n), _pad(expr_out[sl], n), _pad(leye_out[sl], n), _pad(reye_out[sl], n),
            lmk_emb=lmk_emb, carry=carry, frame_lo=start)

        commit_lo = start if carry is None else start + O
        for w in range(n):
            f = start + w
            if f >= commit_lo:
                jaw_out[f], expr_out[f] = jaw_s[w], expr_s[w]
                leye_out[f], reye_out[f] = leye_s[w], reye_s[w]
                bp_out[f, hcols] = bp_s[w, hcols]
        if end == N:
            break
        start = end - O
        carry = dict(k=torch.arange(O, device=bp_out.device),
                     head=bp_out[start:start + O][:, hcols].clone(),
                     jaw=jaw_out[start:start + O].clone(),
                     expr=expr_out[start:start + O].clone(),
                     leye=leye_out[start:start + O].clone(),
                     reye=reye_out[start:start + O].clone())
    return jaw_out, expr_out, leye_out, reye_out, bp_out
