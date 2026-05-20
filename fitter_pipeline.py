###################################################################
##################### modified from SMPLify-X  ####################
###  combined skeleton assembly + SMPLX fitting pipeline       ####
###################################################################
#
# For every (session, activity, person) found in:
#   resources/triangulation_results/{session}/{activity}/
#
# 1. Assembles a skeletons.json using cfg_files/idx_mapping.txt and
#    saves it to:
#      resources/fit_results/{session}/{activity}/p{i}/skeletons.json
#
# 2. Fits SMPLX and writes body_smplx.json + meshes/ into that same
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
import glob
import json

import numpy as np
from pathlib import Path

import cv2 as cv

from cmd_parser import parse_config
from main import main


_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_RESOURCES    = os.path.normpath(os.path.join(_SCRIPT_DIR, '..', '..', 'resources'))
SESS_ROOT     = os.path.join(_RESOURCES, 'sessions')
TRIG_ROOT     = os.path.join(_RESOURCES, 'triangulation_results')
FIT_ROOT      = os.path.join(_RESOURCES, 'fit_results')
SMPLER_ROOT   = os.path.join(_RESOURCES, 'smpler_results')
CALIBS_ROOT   = os.path.join(_RESOURCES, 'calibs')
SAM_ROOT      = os.path.join(_RESOURCES, 'sam_results')
MAPPING_PATH  = os.path.join(_SCRIPT_DIR, 'cfg_files', 'idx_mapping.txt')

cam_map = {
  'GC' : 'GB',
  'HC' : 'GF',
  'Z1' : 'FC1',
  'Z2' : 'FC2',
  'N1' : 'HA1',
  'N2' : 'HA2'
}

# Close-up cameras that face the *other* person — SMPLer-X estimates from
# these views are unreliable for the target person and are excluded from fusion.
_EXCLUDE_CAMS = {0: {'FC2', 'HA2'}, 1: {'FC1', 'HA1'}}


def _geodesic_mean_aa(aa_list):
    """Average a list of axis-angles via SVD on SO(3)."""
    Rs = np.stack([cv.Rodrigues(np.asarray(aa).reshape(3))[0] for aa in aa_list])
    R_sum = Rs.sum(axis=0)
    U, _, Vt = np.linalg.svd(R_sum)
    R_mean = U @ Vt
    if np.linalg.det(R_mean) < 0:
        U[:, -1] *= -1
        R_mean = U @ Vt
    return cv.Rodrigues(R_mean)[0].reshape(3).astype(np.float32)


def fuse_smpler_poses(session_id, activity, person_id, silhouette_cameras, n_frames):
    """
    Load per-camera SMPLer-X pose estimates and fuse them into a single
    per-frame initialisation dict.

    body_pose    : averaged in axis-angle space (body-relative, consistent across views)
    global_orient: each view's estimate is rotated from camera frame to world
                   frame, then geodesic-averaged across views.

    Returns a list of length n_frames; entries are dicts or None when no
    SMPLer-X data is available for that frame.
    """
    excluded = _EXCLUDE_CAMS.get(person_id, set())

    # Pre-load all usable camera files once
    cam_data = {}
    for cam_name in sorted(silhouette_cameras.keys()):
        if cam_name in excluded:
            continue
        npy = os.path.join(SMPLER_ROOT, session_id, activity, f'{cam_name}_smplx.npy')
        if not os.path.isfile(npy):
            continue
        try:
            cam_data[cam_name] = np.load(npy, allow_pickle=True)
        except Exception as e:
            print(f"  [smpler poses] failed to load {npy}: {e}")

    if not cam_data:
        return None

    result = []
    for fidx in range(n_frames):
        body_poses, global_orients_world = [], []

        for cam_name, data in cam_data.items():
            if fidx >= len(data):
                continue
            frame = data[fidx]
            if not isinstance(frame, dict) or person_id not in frame:
                continue
            p = frame[person_id]
            if 'body_pose' not in p or 'global_orient' not in p:
                continue

            body_poses.append(
                np.asarray(p['body_pose'], dtype=np.float32).reshape(63))

            # global_orient is in camera frame; rotate to world frame
            R_cam = np.asarray(silhouette_cameras[cam_name]['R'], dtype=np.float64)
            aa_cam = np.asarray(p['global_orient'], dtype=np.float64).reshape(3)
            R_smplx = cv.Rodrigues(aa_cam)[0]          # cam-frame rotation matrix
            R_world = R_cam.T @ R_smplx                 # world-frame rotation matrix
            global_orients_world.append(cv.Rodrigues(R_world)[0].reshape(3))

        if not body_poses:
            result.append(None)
            continue

        result.append({
            'body_pose':     np.mean(body_poses, axis=0).astype(np.float32),   # (63,)
            'global_orient': _geodesic_mean_aa(global_orients_world),           # (3,)
        })

    n_valid = sum(1 for x in result if x is not None)
    print(f"  [smpler poses] {session_id}/{activity}/p{person_id}: "
          f"{n_valid}/{n_frames} frames fused from {len(cam_data)} views")
    return result


# ---------------------------------------------------------------------------
# SMPLer-X beta injection
# ---------------------------------------------------------------------------

def load_smpler_betas(session_id: str, activity: str, person_id: int):
    """Return averaged betas (np.float32, shape (10,)) for this person from
    SMPLer-X outputs, or None if no usable file exists.

    SMPLer-X stores one .npy per person. By convention FC{person_id+1}_smplx.npy
    holds that person's frames, and each frame dict has an inner key equal to
    person_id. We average across frames (betas are near-constant per subject).
    """
    candidate = os.path.join(SMPLER_ROOT, session_id, activity, f'FC{person_id + 1}_smplx.npy')
    if not os.path.isfile(candidate):
        return None
    try:
        data = np.load(candidate, allow_pickle=True)
    except Exception as e:
        print(f"  [smpler betas] failed to load {candidate}: {e}")
        return None

    frames = []
    for frame in data:
        if isinstance(frame, dict) and person_id in frame and 'betas' in frame[person_id]:
            frames.append(np.asarray(frame[person_id]['betas'], dtype=np.float32).reshape(-1))
    if not frames:
        print(f"  [smpler betas] no betas for person {person_id} in {candidate}")
        return None
    return np.stack(frames).mean(axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Skeleton assembly (mirrors build_skeletons_json.py)
# ---------------------------------------------------------------------------

def parse_idx_mapping(mapping_path: str) -> list:
    """Return [(source, joint_idx), ...] indexed by output joint index."""
    pattern = re.compile(r'(\d+)\s*:\s*([brl])(\d+)')
    mapping = {}
    with open(mapping_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                mapping[int(m.group(1))] = (m.group(2), int(m.group(3)))
    n = max(mapping.keys()) + 1
    for i in range(n):
        if i not in mapping:
            raise ValueError(f"idx_mapping missing output index {i}")
    return [mapping[i] for i in range(n)]


def assemble_skeletons(body_data, left_data, right_data, idx_mapping):
    """Return (records, left_poses, right_poses).

    left_poses / right_poses: {frame_idx: np.ndarray (45,)} — hand pose from
    WiLoR, extracted when the hand npy stores dicts with 'kpts_3d'/'hand_pose'.
    """
    records = []
    left_poses  = {}
    right_poses = {}
    fidxs = [k for k in body_data.keys() if not isinstance(k, str)]

    for fidx in sorted(fidxs):
        b_kpts = np.array(body_data[fidx], dtype=np.float64)

        def _unpack(data, fidx, pose_store):
            if data is None:
                return None
            frame = data.get(fidx)
            if frame is None:
                return None
            if isinstance(frame, dict):
                kpts = np.array(frame['kpts_3d'], dtype=np.float64)
                if 'hand_pose' in frame and frame['hand_pose'] is not None:
                    pose_store[fidx] = np.array(frame['hand_pose'], dtype=np.float32).ravel()
            else:
                kpts = np.array(frame, dtype=np.float64)
            return kpts

        l_kpts = _unpack(left_data,  fidx, left_poses)
        r_kpts = _unpack(right_data, fidx, right_poses)

        joints = []
        for source, src_idx in idx_mapping:
            if source == 'b':
                pt = b_kpts[src_idx]
            elif source == 'r':
                pt = r_kpts[src_idx] if r_kpts is not None else np.zeros(3)
            else:
                pt = l_kpts[src_idx] if l_kpts is not None else np.zeros(3)
            joints.append(pt.tolist())

        records.append({'frame_idx': fidx, 'joints': joints})

    return records, left_poses, right_poses


def build_skeleton(session_id, activity, person_id, activity_path, out_dir, idx_mapping):
    """Assemble and write skeletons.json; return path or None on failure."""
    body_file = os.path.join(activity_path, 'body', f'p{person_id}_triangulated.npy')
    if not os.path.isfile(body_file):
        return None

    body_data  = np.load(body_file, allow_pickle=True).item()
    hand_dir   = os.path.join(activity_path, 'mano')
    head_dir   = os.path.join(activity_path, 'head')
    lhand_file = os.path.join(hand_dir, f'p{person_id}_left_triangulated.npy')
    rhand_file = os.path.join(hand_dir, f'p{person_id}_right_triangulated.npy')
    head_file  = os.path.join(head_dir, f'p{person_id}_triangulated.npy')
    left_data  = np.load(lhand_file, allow_pickle=True).item() if os.path.isfile(lhand_file) else None
    right_data = np.load(rhand_file, allow_pickle=True).item() if os.path.isfile(rhand_file) else None
    # head data: plain (frames, 68, 3) array — not a dict
    head_data  = np.load(head_file, allow_pickle=True) if os.path.isfile(head_file) else None

    betas = body_data.get('betas', None)

    records, left_poses, right_poses = assemble_skeletons(body_data, left_data, right_data, idx_mapping)

    # Build per-frame pose lists aligned to the body frame order.
    body_fidxs = sorted(k for k in body_data.keys() if not isinstance(k, str))
    def _make_pose_list(pose_dict):
        if pose_dict is None:
            return None
        result = [
            np.asarray(pose_dict[fi], dtype=np.float32).reshape(45)
            if (fi in pose_dict and pose_dict[fi] is not None) else None
            for fi in body_fidxs
        ]
        return result if any(p is not None for p in result) else None

    init_left_hand_poses  = _make_pose_list(left_poses)
    init_right_hand_poses = _make_pose_list(right_poses)

    n_lh = sum(1 for p in init_left_hand_poses  if p is not None) if init_left_hand_poses  else 0
    n_rh = sum(1 for p in init_right_hand_poses if p is not None) if init_right_hand_poses else 0

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'skeletons.json')
    with open(out_path, 'w') as f:
        for rec in records:
            f.write(json.dumps(rec) + '\n')

    # Build per-frame pose lists aligned to frame order in records
    frame_order = [rec['frame_idx'] for rec in records]
    init_left_hand_poses  = [left_poses.get(fi)  for fi in frame_order]
    init_right_hand_poses = [right_poses.get(fi) for fi in frame_order]

    n_lh = sum(1 for p in init_left_hand_poses  if p is not None)
    n_rh = sum(1 for p in init_right_hand_poses if p is not None)
    print(f"  [{session_id}/{activity}/p{person_id}] {len(records)} frames -> {out_path}"
          f"  (left_hand={n_lh}/{len(records)}, right_hand={n_rh}/{len(records)}, head={head_data is not None})")
    return out_path, betas, head_data, len(records), init_left_hand_poses, init_right_hand_poses


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
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    base_args = parse_config()

    idx_mapping = parse_idx_mapping(MAPPING_PATH)
    print(f"Loaded idx_mapping: {len(idx_mapping)} output joints")

    sess_root = os.path.abspath(SESS_ROOT)
    trig_root = os.path.abspath(TRIG_ROOT)
    fit_root  = os.path.abspath(FIT_ROOT)

    session_dirs = sorted(glob.glob(os.path.join(sess_root, '*')))
    if not session_dirs:
        print(f"No sessions found under {sess_root}")
        raise SystemExit(1)

    # Camera calibration setup (optional — only active when config supplies cam_map).
    # cam_map: {calib_file_stem: logical_cam_name}, e.g. {"GC": "GB", "Z1": "FC1", ...}
    # camera_image_size: [H, W], e.g. [720, 1280]
    camera_image_size = (720, 1280)

    for sid_path in session_dirs:
        session_id = Path(sid_path).stem
        if '005013' not in session_id: continue

        # Load camera calibrations for this session once (shared across activities/persons)
        # silhouette_cameras = None
        # if cam_map is not None and camera_image_size is not None:
        #     silhouette_cameras = load_session_cameras(
        #         sid_path, CALIBS_ROOT, cam_map, camera_image_size)
        #     if silhouette_cameras is not None:
        #         print(f"  [cameras] loaded {len(silhouette_cameras)} views for session {session_id}")
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
            if 'lego' not in activity: continue
            trig_path = os.path.join(trig_root, session_id, activity)
            if not os.path.isdir(os.path.join(trig_path, 'body')):
                continue

            for person_id in [0, 1]:
                seq_dir = os.path.join(fit_root, session_id, activity, f'p{person_id}')

                print(f"\n[pipeline] {session_id} / {activity} / p{person_id}")

                # Step 1 — build skeleton
                result = build_skeleton(
                    session_id, activity, person_id,
                    trig_path, seq_dir, idx_mapping,
                )
                if result is None:
                    continue
                skeleton_path, init_betas, head_data, n_frames, init_left_hand_poses, init_right_hand_poses = result

                # Step 2 — fit SMPLX
                # data_folder  = seq_dir  (contains skeletons.json)
                # output_folder = parent  so main() writes into seq_dir/
                print(f"  [{session_id}/{activity}/p{person_id}] fitting SMPLX ...")
                args = base_args.copy()
                args['dataset']       = 'custom'
                args['data_folder']   = seq_dir
                args['output_folder'] = os.path.dirname(seq_dir)
                args['person_id']     = person_id

                if silhouette_cameras is not None:
                    args['silhouette_cameras'] = silhouette_cameras
                # head_data: (frames, 68, 3) triangulated face landmarks — passed in-memory
                if head_data is not None:
                    args['head_data'] = head_data
                # SAM masks: sam_results/{session_id}/{activity}/{logical_cam_name}/f{idx:05d}.png
                # Pixel values: 0=person0, 1=person1, 255=background.
                sam_dir = os.path.join(SAM_ROOT, session_id, activity)
                if os.path.isdir(sam_dir):
                    args['mask_folder'] = sam_dir
                    args['mask_person_id'] = person_id

                # SMPLer-X body pose initialisation (fused across views)
                smpler_init = fuse_smpler_poses(
                    session_id, activity, person_id, silhouette_cameras, n_frames)
                if smpler_init is not None:
                    args['smpler_init'] = smpler_init

                # WiLoR hand pose initialisation (fused geodesic mean across views)
                if init_left_hand_poses is not None:
                    args['init_left_hand_poses'] = init_left_hand_poses
                if init_right_hand_poses is not None:
                    args['init_right_hand_poses'] = init_right_hand_poses

                # smpler_betas = load_smpler_betas(session_id, activity, person_id)
                if init_betas is not None:
                    args['init_betas'] = init_betas
                    print(f"  [{session_id}/{activity}/p{person_id}] seeding betas "
                          f"from SMPLer-X (β₀={init_betas[0]:+.3f})")
                else:
                    print(f"  [{session_id}/{activity}/p{person_id}] no SMPLer-X betas "
                          f"available; will optimize shape from skeleton")

                args['init_left_hand_poses']  = init_left_hand_poses
                args['init_right_hand_poses'] = init_right_hand_poses

                main(**args)

    print('\n[pipeline] Done.')
