#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Parallel fitting-sweep runner for fitter_pipeline.py.

Launches several SMPLX fits at once, each with a different parameterization, so
you can compare many configs and pick the best. Designed for a multi-GPU server:
each run is an isolated subprocess pinned to one GPU (round-robin), which sidesteps
all the module-level / CUDA-context state that in-process parallelism would share.

The exact config that produced each run is saved in three places:
  * cfg_files/_generated/fit_smplx_{run}.yaml   (the resolved config)
  * <fit_results>/{session}_cfg{run}/.../config_used.yaml   (copied by the pipeline)
  * <fit_results>/_sweeps/{sweep}/manifest.json            (index of all runs)

------------------------------------------------------------------------------
Three ways to declare the set of runs
------------------------------------------------------------------------------
1) Sweep file (base + named variants and/or a grid):

     python run_sweep.py --sweep cfg_files/sweep_example.yaml --gpus 0,1,2,3

   sweep_example.yaml:
     base: cfg_files/fit_smplx_9.yaml
     name: head_dof_sweep                 # optional label (logs/manifest folder)
     variants:
       free_neck:
         direct_refine_joints_p0: [neck]
         direct_refine_joints_p1: [neck]
       neck_spine:
         direct_refine_joints_p0: [neck, spine3]
     grid:                                # optional; crossed with variants
       direct_temporal_weight: [0.0, 5.0]

2) Explicit list of full configs:

     python run_sweep.py --configs cfg_files/fit_smplx_9.yaml cfg_files/fit_smplx_10.yaml

3) A directory of full configs:

     python run_sweep.py --dir cfg_files/sweep_dir --gpus 0,1

Add --dry-run to generate the configs and print the schedule without launching.
"""

from __future__ import absolute_import, print_function, division

import os
import re
import sys
import json
import time
import glob
import argparse
import itertools
import subprocess

try:
    import yaml
except ImportError:
    sys.exit("run_sweep.py needs PyYAML (`pip install pyyaml`).")


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_RESOURCES  = os.path.normpath(os.path.join(_SCRIPT_DIR, '..', '..', 'resources'))
_FIT_ROOT   = os.path.join(_RESOURCES, 'fit_results')
_GEN_DIR    = os.path.join(_SCRIPT_DIR, 'cfg_files', '_generated')
_PIPELINE   = os.path.join(_SCRIPT_DIR, 'fitter_pipeline.py')


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------

def _sanitize(name):
    """Run names must match fitter_pipeline's regex (fit_smplx_(\\w+).yaml) and be
    filesystem-safe, so collapse everything outside [A-Za-z0-9_] to '_'."""
    return re.sub(r'[^A-Za-z0-9_]', '_', str(name)).strip('_') or 'run'


def _fmt_val(v):
    """Compact a single grid value into a name fragment."""
    if isinstance(v, (list, tuple)):
        return '-'.join(_fmt_val(x) for x in v)
    return _sanitize(v)


def _expand_grid(grid):
    """grid: {key: [v1, v2, ...]} -> list of (name, {key: value}) cartesian combos."""
    if not grid:
        return [(None, {})]
    keys = list(grid.keys())
    value_lists = [grid[k] if isinstance(grid[k], list) else [grid[k]] for k in keys]
    combos = []
    for values in itertools.product(*value_lists):
        override = dict(zip(keys, values))
        name = '_'.join(f"{k}{_fmt_val(v)}" for k, v in override.items())
        combos.append((name, override))
    return combos


def _load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def resolve_runs(args):
    """Return a list of (run_name, resolved_config_dict, sweep_label)."""
    # Mode 2/3: explicit configs or a directory — run each file as-is.
    if args.configs or args.dir:
        paths = list(args.configs or [])
        if args.dir:
            paths += sorted(glob.glob(os.path.join(args.dir, '*.yaml')))
        if not paths:
            sys.exit(f"No configs found (configs={args.configs}, dir={args.dir}).")
        runs = []
        for p in paths:
            m = re.search(r'fit_smplx_(\w+)\.yaml', os.path.basename(p))
            name = m.group(1) if m else _sanitize(os.path.splitext(os.path.basename(p))[0])
            runs.append((name, _load_yaml(p), 'explicit'))
        return runs

    # Mode 1: sweep file. `base:` (one) or `bases:` (several) + variants and/or
    # grid. With several bases the same variant/grid set is applied to each, and
    # run names are prefixed with the base tag so outputs stay distinguishable.
    spec = _load_yaml(args.sweep)
    bases = spec.get('bases') or ([spec['base']] if spec.get('base') else None)
    if not bases:
        sys.exit(f"Sweep file {args.sweep} must define `base:` or `bases:`.")
    multi_base = len(bases) > 1
    sweep_label = _sanitize(spec.get('name') or os.path.splitext(os.path.basename(args.sweep))[0])

    variant_items = list((spec.get('variants') or {}).items()) or [(None, {})]
    grid_items    = _expand_grid(spec.get('grid') or {})

    runs = []
    seen = set()
    for base_path in bases:
        bp = base_path if os.path.isabs(base_path) else os.path.join(_SCRIPT_DIR, base_path)
        base = _load_yaml(bp)
        m = re.search(r'fit_smplx_(\w+)\.yaml', os.path.basename(bp))
        base_tag = m.group(1) if m else _sanitize(os.path.splitext(os.path.basename(bp))[0])
        for vname, vov in variant_items:
            for gname, gov in grid_items:
                parts = [p for p in (vname, gname) if p]
                vg = _sanitize('_'.join(parts)) if parts else 'base'
                name = f"{base_tag}_{vg}" if multi_base else vg
                # disambiguate accidental name clashes
                stem_name, n = name, 1
                while name in seen:
                    n += 1
                    name = f"{stem_name}_{n}"
                seen.add(name)
                cfg = dict(base)
                cfg.update(vov)   # variant overrides
                cfg.update(gov)   # grid overrides (win over variant on key clash)
                runs.append((name, cfg, sweep_label))
    return runs


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------

def detect_gpus():
    """Best-effort GPU id list: nvidia-smi -> CUDA_VISIBLE_DEVICES -> [0]."""
    try:
        out = subprocess.check_output(['nvidia-smi', '-L'],
                                      stderr=subprocess.DEVNULL).decode()
        ids = [str(i) for i, _ in enumerate(out.strip().splitlines()) if _.strip()]
        if ids:
            return ids
    except Exception:
        pass
    env = os.environ.get('CUDA_VISIBLE_DEVICES')
    if env:
        return [x for x in env.split(',') if x != '']
    return ['0']


def main():
    ap = argparse.ArgumentParser(description='Parallel fitting-sweep runner.')
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--sweep', help='Sweep file (base + variants/grid).')
    src.add_argument('--configs', nargs='+', help='Explicit list of full config yamls.')
    src.add_argument('--dir', help='Directory of full config yamls (*.yaml).')

    ap.add_argument('--device', choices=['cuda', 'cpu'], default='cuda',
                    help='Run on GPUs (pinned round-robin) or CPU cores.')
    ap.add_argument('--gpus', default=None,
                    help='Comma-separated GPU ids (default: auto-detect).')
    ap.add_argument('--jobs', type=int, default=None,
                    help='Max concurrent runs (default: #gpus for cuda, #cpus for cpu).')
    ap.add_argument('--per-gpu', type=int, default=1,
                    help='Runs to pack per GPU (jobs = #gpus * per-gpu if --jobs unset).')
    ap.add_argument('--python', default=sys.executable, help='Python interpreter.')
    ap.add_argument('--extra', nargs=argparse.REMAINDER, default=[],
                    help='Extra args appended verbatim to every fitter_pipeline call.')
    ap.add_argument('--dry-run', action='store_true',
                    help='Generate configs + print the schedule, do not launch.')
    args = ap.parse_args()

    runs = resolve_runs(args)
    sweep_label = runs[0][2] if runs else 'sweep'

    # Device / concurrency. For cuda we build an explicit slot->gpu list so runs
    # pack onto GPUs and rebalance correctly as they finish (assignment by free
    # slot, not by launch count). With 44 GB cards, --per-gpu 2..4 is reasonable;
    # --jobs overrides and is spread round-robin across the GPUs.
    if args.device == 'cuda':
        gpus = args.gpus.split(',') if args.gpus else detect_gpus()
        jobs = args.jobs or (len(gpus) * max(1, args.per_gpu))
        slot_gpus = [gpus[i % len(gpus)] for i in range(jobs)]
    else:
        gpus = []
        jobs = args.jobs or min(len(runs), os.cpu_count() or 4)
        slot_gpus = [None] * jobs

    os.makedirs(_GEN_DIR, exist_ok=True)
    log_dir = os.path.join(_FIT_ROOT, '_sweeps', sweep_label)
    os.makedirs(log_dir, exist_ok=True)

    # Materialize each run's resolved config (this is the saved parameterization).
    plan = []
    for name, cfg, _ in runs:
        if args.device == 'cpu':
            cfg = dict(cfg, use_cuda=False)
        gen_path = os.path.join(_GEN_DIR, f'fit_smplx_{name}.yaml')
        with open(gen_path, 'w') as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
        plan.append({'name': name, 'config': gen_path, 'out_suffix': f'_cfg{name}',
                     'log': os.path.join(log_dir, f'{name}.log')})

    print(f"[sweep] '{sweep_label}': {len(plan)} run(s), device={args.device}, "
          f"jobs={jobs}" + (f", gpus={gpus}" if gpus else ""))
    for p in plan:
        print(f"   - {p['name']:<24} -> {os.path.relpath(p['config'], _SCRIPT_DIR)}")
    manifest_path = os.path.join(log_dir, 'manifest.json')

    if args.dry_run:
        with open(manifest_path, 'w') as f:
            json.dump({'sweep': sweep_label, 'device': args.device, 'jobs': jobs,
                       'gpus': gpus, 'runs': plan, 'dry_run': True}, f, indent=2)
        print(f"[sweep] dry-run: wrote {len(plan)} configs + {manifest_path}")
        return

    # Subprocess scheduler: a free-list of GPU slots; each launch claims a slot and
    # each completion returns it, so packing stays balanced even with uneven runtimes.
    pending = list(plan)
    free_gpus = list(slot_gpus)   # multiset of gpu ids (or [None, ...] on cpu)
    running = []                  # list of dicts with proc/log_fh/meta/gpu
    t0 = time.time()

    def _launch(meta):
        gpu = free_gpus.pop(0)
        cmd = [args.python, _PIPELINE, '-c', meta['config']] + (args.extra or [])
        env = dict(os.environ)
        if gpu is not None:
            env['CUDA_VISIBLE_DEVICES'] = gpu
        meta['gpu'] = gpu if gpu is not None else 'cpu'
        log_fh = open(meta['log'], 'w')
        log_fh.write(f"# cmd: {' '.join(cmd)}\n# gpu: {meta['gpu']}\n\n")
        log_fh.flush()
        proc = subprocess.Popen(cmd, cwd=_SCRIPT_DIR, env=env,
                                stdout=log_fh, stderr=subprocess.STDOUT)
        meta.update(start=time.time(), status='running')
        running.append({'proc': proc, 'log_fh': log_fh, 'meta': meta, 'gpu': gpu})
        print(f"[sweep] ▶ {meta['name']} (gpu={meta['gpu']}, pid={proc.pid})")

    def _write_manifest():
        with open(manifest_path, 'w') as f:
            json.dump({'sweep': sweep_label, 'device': args.device, 'jobs': jobs,
                       'gpus': gpus, 'elapsed_s': round(time.time() - t0, 1),
                       'runs': plan}, f, indent=2)

    while pending or running:
        while pending and len(running) < jobs:
            _launch(pending.pop(0))
        time.sleep(1.0)
        for r in list(running):
            ret = r['proc'].poll()
            if ret is None:
                continue
            meta = r['meta']
            meta.update(status='ok' if ret == 0 else 'FAILED', returncode=ret,
                        elapsed_s=round(time.time() - meta['start'], 1))
            r['log_fh'].close()
            running.remove(r)
            free_gpus.append(r['gpu'])   # return the slot for the next pending run
            flag = '✓' if ret == 0 else '✗'
            print(f"[sweep] {flag} {meta['name']} ({meta['status']}, "
                  f"{meta['elapsed_s']}s) — log: {os.path.relpath(meta['log'], _SCRIPT_DIR)}")
            _write_manifest()

    _write_manifest()
    ok = sum(1 for p in plan if p.get('status') == 'ok')
    print(f"\n[sweep] done: {ok}/{len(plan)} ok in {round(time.time()-t0,1)}s")
    print(f"[sweep] manifest: {manifest_path}")
    print(f"[sweep] results : {_FIT_ROOT}/<session>_cfg<run>/...")
    if ok < len(plan):
        sys.exit(1)


if __name__ == '__main__':
    main()
