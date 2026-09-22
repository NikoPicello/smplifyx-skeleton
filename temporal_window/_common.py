# -*- coding: utf-8 -*-
"""
temporal_window._common — helpers and cross-stage constants shared by every windowed
fitting stage (root/body/hands/head) and by Stage C smoothing.

Cross-module access rule: a stage module that reads one of THIS module's cfg-overridable
constants (WIN_SIZE, WIN_OVERLAP, LAMBDA_BND, LOG_EVERY) does so via `from . import
_common` + `_common.CONST` at the point of use, never a top-level `from ._common import
CONST` -- the latter would snapshot today's default at package-import time, before
temporal_window.configure() has run, and never see a CLI/cfg override again. The plain
functions below (_d1, _d2, _aa_unwrap, _aa_to_6d, aa_angle_deg, _f, _cap) ARE safe to
import bare (`from ._common import _cap`): a function always resolves its free variables
against the module it was DEFINED in, regardless of who imports/calls it or when.
"""
from __future__ import absolute_import, print_function, division

import math

import torch

from smplx.lbs import batch_rodrigues   # axis-angle -> rotation matrix (exp map, smooth for all theta)

# ── window geometry (shared: every windowed stage slides over the sequence in WIN_SIZE
#    chunks with WIN_OVERLAP boundary frames) ─────────────────────────────────────────
WIN_SIZE    = 32      # frames optimised jointly (== batched model batch_size). >= N ⇒ full-seq.
WIN_OVERLAP = 8       # boundary frames pinned to the previous window's solve (>=2 ⇒ C1 seam)

LAMBDA_BND  = 1e3     # pin overlap frames to the previous window's committed solve. Shared by
                      # every windowed stage (body/hands/head each add an L_bnd term).

# Make neck and head SHARE their bend (full-vector difference — blocks both the one-joint kink
# and the ±70° opposing-twist candy-wrapper that pinched the neck mesh). Deliberately NO
# coupling to the spine (that pulled the chest forward). Shared by body.py (Stage A's own
# neck/head cols, bp[:,33:36]/bp[:,42:45]) and head.py (Stage B's refined head[:,0:3]/[:,3:6]).
LAMBDA_CERV  = 0.1                   # neck ↔ head bend sharing (full-vector difference)

TERM_CAP = 1e5   # per-term clamp: keeps one spike finite so the line search can reject it
LOG_EVERY = 50

_CFG_OVERRIDABLE = ['WIN_SIZE', 'WIN_OVERLAP', 'LAMBDA_BND', 'LAMBDA_CERV', 'TERM_CAP', 'LOG_EVERY']


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


def _d1(x):
    """Mean squared velocity over the window (sum over the last/param dim, mean over time)."""
    return (x[1:] - x[:-1]).pow(2).sum(-1).mean()


def _d2(x):
    """Mean squared acceleration (second difference) over the window."""
    return (x[2:] - 2.0 * x[1:-1] + x[:-2]).pow(2).sum(-1).mean()


_TWO_PI = 2.0 * math.pi
def _aa_unwrap(go):
     """Frame w becomes (theta_w + 2*pi*m) k_w, the equivalent vector nearest the already-unwrapped
     frame w-1 (an aa_nearest chain); the nearest one is closed-form, m = round((k_w . prev -
     theta_w) / 2*pi) clamped to [-2, 2]. The chain's only carried state is m_{w-1} in {-2..2},
     so each frame is a 5-entry table m_{w-1} -> m_w and the tables compose associatively: all W-1
     steps resolve in log2(W) gather rounds, not a per-frame python loop (~2800 tiny kernel
     launches per fwd+bwd at W=32, paid on EVERY LBFGS closure eval). m is a discrete choice
     computed under no_grad; the 2*pi*m offset is constant w.r.t. the variable, so gradients pass
     straight through (same trick the per-frame global_orient anchor uses)."""
     with torch.no_grad():
         g = go.detach()
         th = g.norm(dim=-1, keepdim=True).clamp_min(1e-8)                               # (W,1)
         k = g / th                                                                      # (W,3)
         j = torch.arange(-2, 3, device=g.device, dtype=g.dtype)                         # candidate m_{w-1}
         prev = g[:-1, None, :] + _TWO_PI * j[None, :, None] * k[:-1, None, :]           # (W-1,5,3) frame w-1
         # A transient NaN/Inf in `go` is a real, expected input here: strong_wolfe's line
         # search evaluates intermediate trial points before any finiteness check runs.
         ratio = torch.nan_to_num(((prev * k[1:, None, :]).sum(-1) - th[1:]) / _TWO_PI,
                                  nan=0.0, posinf=0.0, neginf=0.0)
         F = torch.round(ratio).clamp_(-2, 2).long() + 2
         n, d = g.shape[0] - 1, 1                                                        # F[w-1]: state(m_{w-1}) -> state(m_w)
         while d < n:                                                                    # prefix-compose: F[w] <- F[w] o F[w-d]
             F = torch.cat([F[:d], F[d:].gather(1, F[:-d])], dim=0)
             d *= 2
         m = torch.cat([g.new_zeros(1), (F[:, 2] - 2).to(g.dtype)])                      # frame 0: m=0 (state 2)
     return go + (_TWO_PI * m)[:, None] * (go / go.norm(dim=-1, keepdim=True).clamp_min(1e-8))


def _aa_to_6d(aa):
    """(W,3) axis-angle -> (W,6) 6D rotation rep (first two columns of R), for the smoothness
    term ONLY. AA->matrix is the exponential map (batch_rodrigues), analytic for ALL theta, so
    gradients flow cleanly; the |theta|=pi pathology lives in the INVERSE (matrix->AA) map, which
    we never take. Measuring vel/accel here penalises the TRUE orientation change, free of the
    axis-angle metric distortion + 2*pi wrap near theta=pi. `go` itself stays axis-angle (this is
    a read-only metric), so the saved output needs no conversion and hits no singularity."""
    R = batch_rodrigues(aa)                       # (W,3,3)
    return R[:, :, :2].reshape(aa.shape[0], 6)


def aa_angle_deg(a, b):
    """Geodesic angle (deg) between two axis-angle rotations a, b of shape (..., 3)."""
    Ra, Rb = batch_rodrigues(a.reshape(-1, 3)), batch_rodrigues(b.reshape(-1, 3))
    tr = (Ra.transpose(1, 2) @ Rb).diagonal(dim1=1, dim2=2).sum(-1)
    return torch.rad2deg(torch.arccos(((tr - 1) / 2).clamp(-1.0, 1.0)))


def _f(x):
    """Safe scalar for logging (avoids the autograd warning from float() on a grad tensor)."""
    return x.item() if torch.is_tensor(x) else float(x)


def _cap(x, cap=TERM_CAP):
    """Scale a loss term down if it exceeds `cap`, preserving gradient direction. Mirrors
    SMPLifyLoss._clamp_term — stops a finite-but-huge term (e.g. the GMM prior when a frame
    wanders out of support) from compounding to inf/nan and breaking the line search. Pure-tensor
    (no .item()/bool()/float() on a CUDA tensor) — those force a device sync, and this runs on
    every loss term of every closure evaluation, so it was the dominant per-iteration
    host<->device overhead under strong_wolfe's repeated closure calls."""
    if not torch.is_tensor(x):
        return x
    v = x.detach()
    scale = torch.where(torch.isfinite(v) & (v > cap), cap / v.clamp(min=cap), v.new_ones(()))
    return x * scale
