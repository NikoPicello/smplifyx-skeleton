###################################################################
##################### modified from SMPLify-X  ####################
###                   SMPLX fitting pipeline                   ####
###################################################################
#
# For every (session, activity, person) found in:
#   resources/triangulation_results/{session}/{activity}/
#
# 1. Fits SMPLX and writes body_smplx.json + meshes/ into that same
#    folder.
#
# Run exactly like main.py:
#   python fitter_pipeline.py -c cfg_files/fit_smplx_9.yaml
#
# The data_folder / dataset entries in the config are ignored;
# they are overridden per-sequence by this script.
###################################################################

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import re
import sys
import glob
import json
import time
import shutil
import argparse
import multiprocessing

import numpy as np
from pathlib import Path

import cv2 as cv

from cmd_parser import parse_config
from main import main


_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_RESOURCES    = os.path.normpath(os.path.join(_SCRIPT_DIR, '..', '..', 'resources'))
SESS_ROOT     = os.path.join(_RESOURCES, 'all_sessions')
TRIG_ROOT     = os.path.join(_RESOURCES, 'triangulation_results')
FIT_ROOT      = os.path.join(_RESOURCES, 'fit_results')
SMPLER_ROOT   = os.path.join(_RESOURCES, 'smpler_results')
CALIBS_ROOT   = os.path.join(_RESOURCES, 'calibs')
SAM_ROOT      = os.path.join(_RESOURCES, 'sam_results')
RTMO_ROOT     = os.path.join(_RESOURCES, 'rtmo_results')
MAMMA_ROOT    = os.path.join(_RESOURCES, 'mamma_results')

cam_map = {
  'GC' : 'GB',
  'HC' : 'GF',
  'Z1' : 'FC1',
  'Z2' : 'FC2',
  'N1' : 'HA1',
  'N2' : 'HA2'
}

# ---------------------------------------------------------------------------
# Camera calibration loading
# ---------------------------------------------------------------------------
def load_session_cameras(sid_path, calibs_root, cam_map, image_size):
    """
    Load OpenCV-style camera calibrations for one session.

    Reads calib_date from {sid_path}/session_data.txt (line index 1, chars 11+),
    then loads every calibration file under {calibs_root}/{calib_date}/ via
    cv2.FileStorage.

    Args:
        sid_path    : path to the session folder (e.g. resources/sessions/S001)
        calibs_root : root folder containing per-date calibration sub-folders
        cam_map     : dict mapping calibration file stem → logical camera name
                      e.g. {"GC": "GB", "HC": "GF", "Z1": "FC1", ...}
        image_size  : (H, W) — pixel dimensions of the camera images

    Returns:
        dict {logical_cam_name: {K, D, R, T, image_size}} with keys in sorted order,
        or None if the calibration folder cannot be found.
    """
    session_data_path = os.path.join(sid_path, 'session_data.txt')
    with open(session_data_path) as f:
        lines = f.readlines()
    calib_date = lines[1][11:].strip()

    calib_dir = os.path.join(calibs_root, calib_date)
    if not os.path.isdir(calib_dir):
        print(f"  [cameras] calibration dir not found: {calib_dir}")
        return None

    cam_dict = {}
    for cam_calib in glob.glob(os.path.join(calib_dir, '*')):
        stem = os.path.splitext(os.path.basename(cam_calib))[0]
        if stem not in cam_map:
            continue
        logical_name = cam_map[stem]
        fs = cv.FileStorage(cam_calib, cv.FILE_STORAGE_READ)
        K = fs.getNode('K').mat()
        D = fs.getNode('D').mat()
        R = fs.getNode('R').mat()
        T = fs.getNode('T').mat().ravel()
        fs.release()
        cam_dict[logical_name] = {'K': K, 'D': D, 'R': R, 'T': T, 'image_size': image_size}

    missing = [name for name in cam_map.values() if name not in cam_dict]
    if missing:
        print(f"  [cameras] missing calibration for: {missing} — skipping silhouette for this session")
        return None

    # Return sorted by logical name for consistent ordering
    return dict(sorted(cam_dict.items()))


# ---------------------------------------------------------------------------
# Per-person worker (run in its own process)
# ---------------------------------------------------------------------------
def _fit_person(args, label, log_path):
    """multiprocessing target: fit one person's SMPLX sequence.

    Runs in a separate process (spawned, not forked/threaded) rather than
    just a separate thread because main() calls cvars.configure(args) /
    temporal_window.configure(args), which mutate module-level globals --
    two concurrent in-process calls would race on that state. A separate
    process also gets its own CUDA context; both people's jobs still land
    on the same (default) GPU, which has plenty of headroom for two runs.

    stdout/stderr are redirected to their own per-person log file (line
    buffered, so `tail -f` sees live progress) -- otherwise two people's
    fits print concurrently and interleave into an unreadable mess. The
    log file is deliberately never closed here: if main() raises,
    multiprocessing's own handler prints the traceback to (redirected)
    sys.stderr right after, and closing first would make that write fail.
    The OS closes the fd for us when the process exits.
    """
    sys.stdout = sys.stderr = open(log_path, 'w', buffering=1)
    _t_main = time.perf_counter()
    main(**args)
    print(f"  [timing/pipeline] main() for {label}: {time.perf_counter()-_t_main:.1f}s")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    curr_parser = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
    curr_parser.add_argument('-c', '--config', required=True)
    curr_parser.add_argument('--sid', default='all',
                              help="session id substring, or 'all' (default: all)")
    curr_parser.add_argument('--activities', nargs='+', default=['animals_task', 'gaze_task', 'ghost_task', 'lego_task', 'talk_task'])
    curr_parser.add_argument('--max-frames', type=int, default=-1,
                              help='cap frames; -1 for all available (default: -1)')
    curr_parser.add_argument('--use-mamma', action='store_true')

    curr_args = curr_parser.parse_args()

    # Only forward -c to cmd_parser's own parser — it doesn't know about --sid/
    # --activities/--max-frames, so passing the raw sys.argv here would fail with
    # "unrecognized arguments" on those.
    base_args = parse_config(argv=['-c', curr_args.config])

    cfg_path = curr_args.config
    m = re.search(r'fit_smplx_(\w+)\.yaml', cfg_path)
    if m: cfg_x = m.group(1)

    sess_root = os.path.abspath(SESS_ROOT)
    trig_root = os.path.abspath(TRIG_ROOT)
    fit_root  = os.path.abspath(FIT_ROOT)

    session_dirs = sorted(glob.glob(os.path.join(sess_root, '*')))
    if not session_dirs:
        print(f"No sessions found under {sess_root}")
        raise SystemExit(1)

    camera_image_size = (720, 1280)

    # spawn (not fork): child processes must get a fresh CUDA context, not a
    # forked copy of one that may already be initialized in this process.
    mp_ctx = multiprocessing.get_context('spawn')

    for sid_path in session_dirs:
        session_id = Path(sid_path).stem
        if curr_args.sid != 'all' and curr_args.sid not in session_id:
          continue
        with open(os.path.join(sid_path, 'session_data.txt')) as f:
          lines = f.readlines()
          calib_date = lines[1][11:].strip()
        curr_calib_path = os.path.join(CALIBS_ROOT, calib_date)
        cam_calibs = glob.glob(curr_calib_path + '/*')
        silhouette_cameras = {}
        for cam_calib in cam_calibs:
          cam_name = Path(cam_calib).stem
          fs = cv.FileStorage(os.path.join(curr_calib_path, f"{cam_name}.yml"), cv.FILE_STORAGE_READ)
          K = fs.getNode('K').mat()
          D = fs.getNode('D').mat()
          R = fs.getNode('R').mat()
          T = fs.getNode('T').mat()
          fs.release()
          silhouette_cameras[cam_map[cam_name]] = {'K': K, 'D': D, 'R': R, 'T': T, 'image_size' : camera_image_size}

        silhouette_cameras = dict(sorted(silhouette_cameras.items()))

        for activity_path in sorted(glob.glob(os.path.join(sid_path, '*'))):
            activity = Path(activity_path).stem
            if activity not in curr_args.activities:
              continue
            # Build each present person's args first (cheap, no CUDA), then fit
            # both people at once in separate processes instead of one after
            # the other -- see _fit_person for why processes rather than threads.
            person_args = {}
            for person_id in [0, 1]:
                trig_path = os.path.join(trig_root, session_id, activity, f"p{person_id}")
                if not os.path.isdir(os.path.join(trig_path)):
                    continue
                out_session = f'{session_id}_cfg{cfg_x}' if cfg_x else session_id
                seq_dir = os.path.join(fit_root, out_session, activity, f'p{person_id}')
                sam_dir = os.path.join(SAM_ROOT, session_id, activity)
                rtmo_dir = os.path.join(RTMO_ROOT, session_id, activity)
                mamma_dir = os.path.join(MAMMA_ROOT, session_id, activity)

                print(f"\n[pipeline] {session_id} / {activity} / p{person_id}")

                os.makedirs(seq_dir, exist_ok=True)
                if cfg_path and os.path.isfile(cfg_path):
                    try:
                        shutil.copy(cfg_path, os.path.join(seq_dir, 'config_used.yaml'))
                    except OSError as e:
                        print(f"  [pipeline] could not save config copy: {e}")

                args = base_args.copy()
                args['dataset']       = 'custom'
                args['data_folder']   = trig_path
                args['output_folder'] = os.path.dirname(seq_dir)
                args['person_id']     = person_id
                args['gender'] = 'neutral'
                args['max_frames'] = curr_args.max_frames


                if silhouette_cameras is not None:
                    args['silhouette_cameras'] = silhouette_cameras

                if os.path.isdir(rtmo_dir):
                    args['rtmo_folder'] = rtmo_dir

                if os.path.isdir(mamma_dir) and curr_args.use_mamma:
                    args['mamma_folder'] = mamma_dir
                else:
                    print('MAMMA not included')

                smpler_dir = os.path.join(SMPLER_ROOT, session_id, activity)
                if os.path.isdir(smpler_dir):
                    args['smpler_folder'] = smpler_dir

                if os.path.isdir(sam_dir):
                    args['mask_folder'] = sam_dir
                    args['mask_person_id'] = person_id

                person_args[person_id] = (args, os.path.join(seq_dir, 'fit_log.txt'))

            if not person_args:
                continue

            procs = []
            for person_id, (args, log_path) in person_args.items():
                label = f"{session_id}/{activity}/p{person_id}"
                print(f"  [{label}] fitting SMPLX ... (log: {log_path})")
                p = mp_ctx.Process(target=_fit_person, args=(args, label, log_path), name=label)
                p.start()
                procs.append((person_id, p, log_path))

            failed = []
            for person_id, p, log_path in procs:
                p.join()
                if p.exitcode != 0:
                    failed.append((person_id, p.exitcode, log_path))
            if failed:
                raise RuntimeError(
                    f"[pipeline] {session_id}/{activity}: fit failed for "
                    + ', '.join(f"p{pid} (exit={ec}, log={lp})" for pid, ec, lp in failed))

    print('\n[pipeline] Done.')
