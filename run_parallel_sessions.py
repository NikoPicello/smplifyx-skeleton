#!/usr/bin/env python3
"""Run fitter_pipeline.py across many sessions, packed onto idle GPUs.

Run this from inside the container, from the already-activated fitter conda
env (e.g. `python run_parallel_sessions.py ...`) -- each per-session
subprocess is launched with the same interpreter (sys.executable), not a
fresh env activation, so the env this script itself runs under is the env
fitter_pipeline.py runs under too.

Unlike the sibling run_parallel_sessions.py scripts (mamma/WiLoR/3DDFA-V3),
this one does NOT stage anything. Those pipelines decode raw multi-camera
video, which is large enough to need a copy into local scratch before
processing and a cleanup after. fitter_pipeline.py never touches raw video --
per session it only reads resources/all_sessions/<sid>/session_data.txt (tiny)
plus lists that session's activity folder names, and otherwise reads already-
small derived outputs (triangulation_results, rtmo_results, mamma_results,
smpler_results). So there's nothing worth copying in/out; each worker just
runs fitter_pipeline.py directly against resources/ and cleans up nothing.

Candidate discovery walks resources/triangulation_results/<sid> (NOT
resources/all_sessions) -- triangulation is upstream of this fit, so not
every raw session has triangulated data yet. Discovering from all_sessions
would repeatedly hand out sessions with nothing to fit, and since a no-op run
produces no fit_results output, auto-discovery mode would just re-select the
same not-yet-triangulated session forever. resources/all_sessions is still
required at run time -- fitter_pipeline.py itself reads
<all_sessions>/<sid>/session_data.txt and lists activity folders from there --
this script just doesn't use it for the candidate list.

Because output is config-dependent (resources/fit_results/<sid>_cfg<X>/, where
<X> comes from the config filename, same fit_smplx_(\\w+).yaml rule
fitter_pipeline.py itself uses), --config is required here and the
already-processed check is keyed on <sid>_cfg<X>, not just <sid> -- rerunning
the same sessions under a different config is expected to redo the work, not
be skipped as "already done".

With explicit session ids on the command line, the candidate list is fixed and
the script exits once they're all done (no already-processed filtering --
if you name a session, it runs, matching the sibling scripts' behavior). With
no ids, it auto-discovers candidates from resources/triangulation_results and
keeps re-scanning for newly arrived ones, so it runs indefinitely (Ctrl-C to
stop; a session already mid-run finishes before the process exits).

"Free GPU" means zero processes on it right now (any user, any container --
checked via `nvidia-smi --query-compute-apps`), not just low utilization. This
gates the FIRST job we put on a GPU only -- we never take a GPU someone else
is already using. Once we have a foothold there, --per-gpu (default 1) caps
how many of OUR OWN fitter_pipeline.py jobs run on that GPU concurrently,
without re-checking nvidia-smi -- after our first job starts, nvidia-smi will
report the GPU busy (that's our own process), so re-checking "free" would
wrongly block further packing onto a GPU that is, from our side, still wide
open.

Bump --per-gpu when one instance leaves a lot of headroom. fitter_pipeline.py
is a per-frame gradient-descent fit, not a big vision model -- a single run
can sit around ~2-3% memory and ~25% utilization on a 24 GB card, so several
can usually share one GPU. There's no auto-detection here: memory scales with
how many people/frames/refinement stages a session needs, and packed jobs
contend for memory bandwidth and kernel-launch scheduling in ways a single
isolated run's numbers don't capture (and heavy --fitter-args data loading
can turn this CPU-bound before it's GPU-bound). Watch `nvidia-smi` under a
real multi-job run (both memory AND utilization, not just one instance) and
raise --per-gpu from there rather than assuming linear scaling from one job's
numbers.

The pool re-checks continuously: as soon as a GPU has room (either a fresh
idle GPU or one of ours under --per-gpu) a new candidate is started on it --
no waiting for the whole batch to finish. Our own per-GPU counts are tracked
in-process so a not-yet-CUDA-initialized subprocess can't be double-booked
during its startup lag.

Usage:
    python run_parallel_sessions.py --config cfg_files/fit_smplx_9.yaml
    python run_parallel_sessions.py --config cfg_files/fit_smplx_9.yaml 005013 004115
    python run_parallel_sessions.py --config cfg_files/fit_smplx_9.yaml --gpus 0,1,2,3
    python run_parallel_sessions.py --config cfg_files/fit_smplx_9.yaml --per-gpu 8
    python run_parallel_sessions.py --config cfg_files/fit_smplx_9.yaml \\
        --fitter-args --activities lego_task --max-frames 500
    python run_parallel_sessions.py --config cfg_files/fit_smplx_9.yaml --dry-run

--fitter-args grabs every token after it (argparse REMAINDER), so it must come
LAST -- session ids or other flags after it are swallowed as fitter_pipeline.py
args instead. Put session ids first: `... 005013 004115 --fitter-args --max-frames 500`.

Note --sid is a substring match (fitter_pipeline.py's own semantics) -- session
ids passed here should be the full session id (they're fixed-width, e.g.
"005013") so one can't accidentally match another as a substring.

The already-processed / discovery check only looks at whether
resources/fit_results/<sid>_cfg<X> has *any* output -- it does not check that
every activity/person you actually wanted was fit. Re-run explicit session ids
if a run was only partially completed.

Logs: ./run_logs/<run_id>/<session_id>.log (one per session attempt), plus a
summary.tsv of "<session_id>\\t<exit_code>" lines.
"""
from __future__ import annotations

import argparse
import datetime
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

FITTER_ROOT   = Path(__file__).resolve().parent
RESOURCES_DIR = FITTER_ROOT.parent.parent / "resources"
TRIG_ROOT     = RESOURCES_DIR / "triangulation_results"
FIT_ROOT      = RESOURCES_DIR / "fit_results"

DEFAULT_POLL_INTERVAL = 15.0


def cfg_suffix(config_path: str) -> str:
    """Same run tag fitter_pipeline.py itself derives from the config filename
    (fit_smplx_<X>.yaml -> X), used to name/detect fit_results/<sid>_cfg<X>."""
    m = re.search(r'fit_smplx_(\w+)\.yaml', os.path.basename(config_path))
    if m:
        return m.group(1)
    return re.sub(r'[^A-Za-z0-9_]', '_', os.path.splitext(os.path.basename(config_path))[0])


def free_gpu_indices(whitelist: set[int] | None = None) -> list[int]:
    """GPU indices with zero processes right now (any user), via nvidia-smi."""
    try:
        gpus_out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=15,
        ).stdout
        apps_out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=15,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        raise SystemExit(f"nvidia-smi query failed: {e}") from e

    busy_uuids = {line.strip() for line in apps_out.strip().splitlines() if line.strip()}
    free = []
    for line in gpus_out.strip().splitlines():
        idx_s, uuid = (p.strip() for p in line.split(",", 1))
        idx = int(idx_s)
        if whitelist is not None and idx not in whitelist:
            continue
        if uuid not in busy_uuids:
            free.append(idx)
    return sorted(free)


def already_processed(sid: str, cfg_x: str) -> bool:
    d = FIT_ROOT / f"{sid}_cfg{cfg_x}"
    return d.is_dir() and any(d.iterdir())


def discover_candidates(known: set[str], cfg_x: str) -> list[str]:
    if not TRIG_ROOT.is_dir():
        return []
    ids = sorted(p.name for p in TRIG_ROOT.iterdir() if p.is_dir())
    return [sid for sid in ids if sid not in known and not already_processed(sid, cfg_x)]


class Runner:
    def __init__(self, log_dir: Path, config: str, fitter_args: list[str], dry_run: bool):
        self.log_dir = log_dir
        self.config = config
        self.fitter_args = fitter_args
        self.dry_run = dry_run

        self.lock = threading.Lock()
        self.active_counts: dict[int, int] = {}
        self.threads: list[threading.Thread] = []
        self.summary: list[tuple[str, int]] = []

    def _log_path(self, sid: str) -> Path:
        return self.log_dir / f"{sid}.log"

    def _run_pipeline(self, sid: str, gpu: int, log_fh) -> int:
        # Uses the same interpreter this orchestrator is running under (sys.executable),
        # so it must itself already be launched from the right conda env's python --
        # no env activation/wrapping happens here.
        cmd = [sys.executable, "fitter_pipeline.py", "-c", self.config,
               "--sid", sid, *self.fitter_args]
        print(f"[{sid}] running: {' '.join(cmd)} (CUDA_VISIBLE_DEVICES={gpu})", file=log_fh)
        if self.dry_run:
            print(f"[{sid}] DRY-RUN: skipping actual run", file=log_fh)
            return 0
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        log_fh.flush()
        proc = subprocess.run(cmd, cwd=str(FITTER_ROOT), env=env,
                              stdout=log_fh, stderr=subprocess.STDOUT)
        return proc.returncode

    def _worker(self, sid: str, gpu: int) -> None:
        print(f"[{sid}] starting on GPU {gpu} (log: {self._log_path(sid)})")
        try:
            with open(self._log_path(sid), "w") as log_fh:
                rc = self._run_pipeline(sid, gpu, log_fh)

            with self.lock:
                self.summary.append((sid, rc))

            if rc == 0:
                print(f"[{sid}] done (GPU {gpu})")
            else:
                print(f"[{sid}] FAILED rc={rc} (GPU {gpu}) — see {self._log_path(sid)}",
                      file=sys.stderr)
        finally:
            with self.lock:
                self.active_counts[gpu] -= 1
                if self.active_counts[gpu] <= 0:
                    del self.active_counts[gpu]

    def launch(self, sid: str, gpu: int) -> None:
        with self.lock:
            self.active_counts[gpu] = self.active_counts.get(gpu, 0) + 1
        t = threading.Thread(target=self._worker, args=(sid, gpu), daemon=True)
        t.start()
        self.threads.append(t)

    def active_snapshot(self) -> dict[int, int]:
        with self.lock:
            return dict(self.active_counts)

    def join_all(self) -> None:
        for t in self.threads:
            t.join()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run fitter_pipeline.py across sessions, packed onto idle GPUs.")
    ap.add_argument("sessions", nargs="*",
                    help="explicit session ids to run (default: auto-discover from "
                         "resources/triangulation_results, re-scanning forever)")
    ap.add_argument("-c", "--config", required=True,
                    help="config yaml forwarded to fitter_pipeline.py as -c, e.g. "
                         "cfg_files/fit_smplx_9.yaml. Also determines the output "
                         "suffix (fit_results/<sid>_cfg<X>) used for the "
                         "already-processed check.")
    ap.add_argument("--gpus", default=os.environ.get("GPUS"),
                    help="comma-separated GPU index whitelist (default: all GPUs "
                         "reported by nvidia-smi). Still only used when actually idle.")
    ap.add_argument("--per-gpu", type=int, default=1,
                    help="concurrent fitter_pipeline.py jobs per GPU (default: 1). "
                         "fitter_pipeline.py is lightweight enough that several "
                         "usually fit on one GPU -- see the module docstring before "
                         "raising this past the default.")
    ap.add_argument("--fitter-args", nargs=argparse.REMAINDER, default=[],
                    help="remaining args forwarded to fitter_pipeline.py, e.g. "
                         "--activities lego_task --max-frames 500. Must be LAST on "
                         "the command line -- it swallows everything after it, "
                         "including session ids.")
    ap.add_argument("--poll-interval", type=float, default=DEFAULT_POLL_INTERVAL,
                    help=f"seconds between nvidia-smi/candidate re-checks (default: "
                         f"{DEFAULT_POLL_INTERVAL})")
    ap.add_argument("--dry-run", action="store_true",
                    help="log planned run actions without launching anything")
    args = ap.parse_args()

    whitelist = None
    if args.gpus:
        whitelist = {int(g) for g in args.gpus.replace(",", " ").split()}

    cfg_x = cfg_suffix(args.config)

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = FITTER_ROOT / "run_logs" / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"logs: {log_dir}/<session_id>.log")

    explicit = bool(args.sessions)
    known: set[str] = set()
    candidates: deque[str] = deque()
    if explicit:
        candidates.extend(args.sessions)
        known.update(args.sessions)
    else:
        new = discover_candidates(known, cfg_x)
        candidates.extend(new)
        known.update(new)
        print(f"auto-discovery mode: watching {TRIG_ROOT} forever (Ctrl-C to stop)")

    runner = Runner(log_dir, args.config, args.fitter_args, args.dry_run)

    try:
        while candidates or runner.active_snapshot() or not explicit:
            if not explicit:
                new = discover_candidates(known, cfg_x)
                if new:
                    print(f"discovered {len(new)} new session(s): {', '.join(new)}")
                    candidates.extend(new)
                    known.update(new)

            # Slots on GPUs we already have a foothold on: pack up to --per-gpu
            # without re-checking nvidia-smi (our own job makes it look "busy").
            active = runner.active_snapshot()
            slots = []
            for gpu, n in active.items():
                slots.extend([gpu] * max(0, args.per_gpu - n))
            # Slots on GPUs we don't hold yet: only ones nvidia-smi reports idle,
            # so we never take a GPU someone else is actively using.
            for gpu in free_gpu_indices(whitelist):
                if gpu not in active:
                    slots.extend([gpu] * args.per_gpu)

            for gpu in slots:
                if not candidates:
                    break
                runner.launch(candidates.popleft(), gpu)

            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        print("\ninterrupted — waiting for active sessions to finish "
              "(Ctrl-C again to force)...", file=sys.stderr)

    try:
        runner.join_all()
    except KeyboardInterrupt:
        print("\nforced exit — active sessions were left running in the background.",
              file=sys.stderr)
        return 130

    summary = runner.summary
    summary_path = log_dir / "summary.tsv"
    with open(summary_path, "w") as f:
        for sid, rc in summary:
            f.write(f"{sid}\t{rc}\n")

    total = len(summary)
    ok = sum(1 for _, rc in summary if rc == 0)
    print(f"\ndone: {ok}/{total} sessions ok")
    if ok < total:
        print("failed sessions:")
        for sid, rc in summary:
            if rc != 0:
                print(f"  {sid} (rc={rc})")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
