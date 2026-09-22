# -*- coding: utf-8 -*-
"""
temporal_window.smoothing — Stage C: offline whole-sequence smoothing (runs AFTER all
fitting, before the writer).

Whittaker–Eilers: x* = argmin Σ|x_t − y_t|² + λ Σ|x_{t+1} − 2 x_t + x_{t-1}|² per channel — the
same acceleration prior as Stage A but GLOBAL (whole sequence coupled at once: no window seams)
and with a closed-form banded solve (O(N), milliseconds for 9000 frames). Pure signal processing
on the finished trajectories: no data-term tension, zero phase lag. The mesh is a deterministic
function of the params, so smoothing the params smooths the mesh exactly.
λ dials: higher = smoother but starts damping real motion. λ=0 disables a group.
Rule of thumb: perceived cutoff ~ (1/λ)^(1/4) of Nyquist — λ 10→gentle, 100→strong, 1000→heavy.
"""
from __future__ import absolute_import, print_function, division

import numpy as np
import torch

from . import legs
from ._common import _aa_unwrap, _d2

SMOOTH_LAM_BP   = 50.0    # body_pose (63) — the visible body vibration
SMOOTH_LAM_LEG  = 1000.0  # leg body_pose cols (_LEG_COLS) — seated legs are near-static and their
                          # only data is 2D; smooth them much harder than the moving upper body
SMOOTH_LAM_GO   = 200.0   # global_orient — smoothed on the UNWRAPPED (aa_nearest) trajectory
SMOOTH_LAM_TR   = 200.0   # translation
SMOOTH_LAM_HAND = 20.0    # hand poses (fingers move fast; keep light)
SMOOTH_LAM_HEAD = 20.0    # jaw + expression + eyes

_CFG_OVERRIDABLE = [
    'SMOOTH_LAM_BP', 'SMOOTH_LAM_LEG', 'SMOOTH_LAM_GO', 'SMOOTH_LAM_TR',
    'SMOOTH_LAM_HAND', 'SMOOTH_LAM_HEAD',
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


def smooth_sequence(y, lam):
    """Whittaker–Eilers smoother over the whole sequence. y: (N, D) tensor; returns (N, D).
    Solves the pentadiagonal SPD system (I + λ D2ᵀD2) x = y once for all D channels."""
    N = y.shape[0]
    if lam <= 0 or N < 3:
        return y
    from scipy.linalg import solveh_banded
    yn = y.detach().cpu().numpy().astype(np.float64)
    d0 = np.ones(N)                       # I  +  λ·(D2ᵀD2) diagonals
    d0[0:N - 2] += lam; d0[1:N - 1] += 4.0 * lam; d0[2:N] += lam
    d1 = np.zeros(N - 1)
    d1[0:N - 2] += -2.0 * lam; d1[1:N - 1] += -2.0 * lam
    d2 = np.full(N - 2, lam)
    ab = np.zeros((3, N))                 # upper banded form for solveh_banded
    ab[0, 2:] = d2
    ab[1, 1:] = d1
    ab[2, :]  = d0
    x = solveh_banded(ab, yn, lower=False)
    return torch.as_tensor(x, dtype=y.dtype, device=y.device)


def smooth_all_outputs(bp, go, tr, lh, rh, jaw, expr, leye, reye):
    """Stage C over every saved trajectory. go is unwrapped (aa_nearest chain) BEFORE smoothing so
    the filter never sees a 2π rep jump; the returned go stays on that continuous branch (the
    writer's saved-output unwrap is a no-op on it). Prints accel before → after per group."""
    gou = _aa_unwrap(go)
    groups = [('body_pose', bp,  SMOOTH_LAM_BP), ('global_orient', gou, SMOOTH_LAM_GO),
              ('transl',    tr,  SMOOTH_LAM_TR), ('left_hand',     lh,  SMOOTH_LAM_HAND),
              ('right_hand', rh, SMOOTH_LAM_HAND), ('jaw',         jaw, SMOOTH_LAM_HEAD),
              ('expr',      expr, SMOOTH_LAM_HEAD), ('leye',       leye, SMOOTH_LAM_HEAD),
              ('reye',      reye, SMOOTH_LAM_HEAD)]
    out = []
    for name, x, lam in groups:
        xs = smooth_sequence(x, lam)
        if name == 'body_pose':   # leg cols get their own (much heavier) smoothing
            lcols = torch.as_tensor(legs._LEG_COLS, device=xs.device, dtype=torch.long)
            xs[:, lcols] = smooth_sequence(x[:, lcols], SMOOTH_LAM_LEG)
        if name in ('body_pose', 'global_orient', 'transl'):
            print(f"[stageC smooth] {name:13s} λ={lam:6.0f}  accel {_d2(x).item():.2e} → {_d2(xs).item():.2e}")
        out.append(xs)
    return out
