# -*- coding: utf-8 -*-
"""
temporal_window — windowed temporal SMPLX fitting, split by stage:
    _common     shared helpers + cross-stage constants (window geometry, boundary-seam
                weight, cervical-sharing weight, per-term loss cap, log cadence)
    betas       Stage 0: fit betas to observed bone lengths (once, pose-invariant)
    legs        static seated-leg template
    root        static-root solve (+ the hip-lift seed that feeds it)
    hands       Stage B: windowed hand refinement
    head        Stage B: windowed head/face refinement
    body        Stage A: the windowed body_pose (+root) fit
    smoothing   Stage C: offline whole-sequence smoothing

configure(args) is the single entry point main.py calls (`import temporal_window;
temporal_window.configure(args)`) BEFORE the first `from temporal_window.<stage> import
...` -- see each submodule's own configure() for the per-stage cfg keys it reads (from
cmd_parser.py's parsed args).

Cross-module reads: a submodule that reads ANOTHER submodule's cfg-overridable constant
does so via `from . import <module>` + `<module>.CONST` at the point of use, never a
top-level `from .<module> import CONST` -- the latter would snapshot today's default at
PACKAGE-IMPORT time (i.e. while `import temporal_window` is still running this file,
before main.py has had a chance to call configure() at all) and never see an override
again. See e.g. body.py's use of root.FREEZE_ROOT / legs.FREEZE_LEGS / legs._LEG_COLS /
head._HEAD_COLS / _common.LOG_EVERY / _common.WIN_SIZE / _common.LAMBDA_BND /
_common.LAMBDA_CERV. Plain function imports across stages ARE safe bare (e.g. `from
._common import _cap`) since a function always resolves its free variables against the
module it was DEFINED in, regardless of who imports or calls it, or when.
"""
from __future__ import absolute_import, print_function, division

from . import _common, betas, legs, root, hands, head, body, smoothing


def configure(args):
    """Overwrite every stage's tuning constants from a parsed cfg/CLI args dict (see
    cmd_parser.py). Call once, before the first fitting stage runs."""
    _common.configure(args)
    betas.configure(args)
    legs.configure(args)
    root.configure(args)
    hands.configure(args)
    head.configure(args)
    body.configure(args)
    smoothing.configure(args)
