"""Diagnostic: how much of the fitting wall-clock goes into LBFGS's strong-Wolfe
line search, and how many closure() calls (= forced GPU<->CPU syncs) each
.step() actually spends. Answers "is the sync overhead real, and where" with
numbers instead of guesswork, before touching WIN_SIZE / STAGE_SCHEDULE / max_eval.

Monkey-patches torch.optim.LBFGS.step at import time -- does NOT touch
fitter_pipeline.py, main.py or temporal_window.py, so it's safe to run against
a locally-modified fitter_pipeline.py and safe to delete when you're done.
Grouping is by call SITE (caller function + source line of the .step() call),
found via stack introspection, e.g.:
    refine_window_body:862    <- the 3 STAGE_SCHEDULE passes, aggregated
    refine_window_hands:1015  <- place_opt (arm pre-phase)
    refine_window_hands:1035  <- opt (hand refinement)      [line numbers approximate]
so the 6 LBFGS call sites in temporal_window.py separate out on their own.

Usage -- drop-in replacement for the normal fitter_pipeline.py invocation
(forwards argv unchanged, just runs fitter_pipeline.py's __main__ under the patch):
    python lbfgs_profile.py -c cfg_files/fit_smplx_9.yaml --sid 005013 \\
        --activities lego_task --max-frames 200

Prints one summary table at process exit, aggregated over EVERY (session,
activity, person) the process fit -- fine for a first pass, but if you want a
breakdown per fit instead of one number for the whole process, add right after
each `main(**args)` call in fitter_pipeline.py:
    import lbfgs_profile; lbfgs_profile.report(reset=True)
"""
import atexit
import collections
import runpy
import sys
import time

import torch

_stats = collections.defaultdict(lambda: [0, 0, 0.0])  # "fn:lineno" -> [step_calls, closure_calls, wall_s]

_orig_step = torch.optim.LBFGS.step


def _counting_step(self, closure):
    frame = sys._getframe(1)
    site = f"{frame.f_code.co_name}:{frame.f_lineno}"

    n_calls = [0]

    def counted_closure(*a, **kw):
        n_calls[0] += 1
        return closure(*a, **kw)

    sync = torch.cuda.is_available()
    if sync:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = _orig_step(self, counted_closure)
    if sync:
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    row = _stats[site]
    row[0] += 1
    row[1] += n_calls[0]
    row[2] += dt
    return out


torch.optim.LBFGS.step = _counting_step


def report(reset=False, title="LBFGS profile"):
    """Print the summary table built so far. reset=True clears counters after
    printing -- call this once per fit for a per-(session/activity/person)
    breakdown instead of one aggregate at process exit."""
    if not _stats:
        print(f"\n=== {title}: no LBFGS .step() calls observed ===")
        return
    rows = sorted(_stats.items(), key=lambda kv: -kv[1][2])
    print(f"\n=== {title} (closure calls = forced GPU<->CPU sync points) ===")
    print(f"{'call site':<40}{'.step() calls':>14}{'closure calls':>15}{'wall s':>10}{'closures/step':>15}{'ms/closure':>12}")
    tot = [0, 0, 0.0]
    for site, (steps, closures, wall) in rows:
        per_step = closures / max(steps, 1)
        ms_per_closure = 1000 * wall / max(closures, 1)
        print(f"{site:<40}{steps:>14}{closures:>15}{wall:>10.2f}{per_step:>15.2f}{ms_per_closure:>12.2f}")
        tot[0] += steps; tot[1] += closures; tot[2] += wall
    per_step = tot[1] / max(tot[0], 1)
    ms_per_closure = 1000 * tot[2] / max(tot[1], 1)
    print(f"{'TOTAL':<40}{tot[0]:>14}{tot[1]:>15}{tot[2]:>10.2f}{per_step:>15.2f}{ms_per_closure:>12.2f}")
    print("(compare TOTAL wall s against fitter_pipeline.py's own "
          "'[timing/pipeline] main() for ...' line -- if they're close, "
          "LBFGS.step() really is where the time goes; if not, look elsewhere.)")
    if reset:
        _stats.clear()


atexit.register(report)


if __name__ == "__main__":
    sys.argv[0] = "fitter_pipeline.py"
    runpy.run_path("fitter_pipeline.py", run_name="__main__")
