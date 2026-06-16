"""Overlay fitted SMPL-X meshes on top of session videos.

For every (session, activity, camera) present under resources/, loads the
fitted p0/p1 meshes from fit_results/, the calibration for that camera, and
the matching mp4 from sessions/, then writes an overlay video into
fit_results/<session>/<activity>/<camera>_fit_render.mp4.

Follows the same iteration schema as MKER/smpler_pipeline.py (no CLI args;
module-level flags and hardcoded activity list).
"""

import glob
import json
import os
import os.path as osp
import re
import sys

import argparse

import cv2 as cv
import imageio
import numpy as np
import trimesh
from tqdm import trange


undistort = False
alpha = 0.75

# activities = ['animals', 'gaze', 'ghost', 'lego', 'talk']
activities = ['lego']

cam_map = {
    'GC': 'GB',
    'HC': 'GF',
    'Z1': 'FC1',
    'Z2': 'FC2',
    'N1': 'HA1',
    'N2': 'HA2',
}

PERSON_COLORS = {
    0: (237, 149, 100),
    1: (14, 127, 255),
}

FRAME_W, FRAME_H = 1280, 720


def _project(points_world, R, T, K, D):
    """Project Nx3 world-space points to Nx2 pixel coords with distortion."""
    rvec, _ = cv.Rodrigues(R)
    pts, _ = cv.projectPoints(points_world.astype(np.float64), rvec,
                               T.reshape(3, 1), K, D)
    return pts.reshape(-1, 2)


def render_mesh_simple(img, meshes_by_person, camera_dict, alpha=0.55, is_backview=False):
    """Overlay each person's world-space SMPL-X mesh on `img`.

    Uses cv.projectPoints so lens distortion is correctly applied whether or
    not the frame has been undistorted (caller passes the right K/D).
    """
    K = np.asarray(camera_dict['K'], dtype=np.float64)
    D = np.asarray(camera_dict['D'], dtype=np.float64)
    R = np.asarray(camera_dict['R'], dtype=np.float64)
    T = np.asarray(camera_dict['T'], dtype=np.float64).reshape(3,)
    rvec, _ = cv.Rodrigues(R)

    overlay = img.copy()
    for pid, (vertices, faces) in meshes_by_person.items():
        # depth in camera space for painter's sort and back-face cull
        cam = vertices @ R.T + T.reshape(1, 3)
        z = cam[:, 2]

        proj, _ = cv.projectPoints(vertices.astype(np.float64), rvec,
                                    T.reshape(3, 1), K, D)
        proj = proj.reshape(-1, 2)

        face_z = z[faces]
        in_front = (face_z > 0).all(axis=1)
        if not in_front.any():
            continue
        valid = faces[in_front]
        if is_backview:
          order = np.argsort(face_z[in_front].mean(axis=1))  # painter's algo
        else:
          order = np.argsort(-face_z[in_front].mean(axis=1))  # painter's algo
        tri_pts = proj[valid[order]].astype(np.int32)
        cv.fillPoly(overlay, tri_pts, PERSON_COLORS.get(pid, (200, 200, 200)))

    return cv.addWeighted(overlay, alpha, img, 1 - alpha, 0)


# GT keypoint segments — the same triangulated 3D points the optimizer fits
# (body coco17 + hands + face), read straight from triangulation_results,
# mirroring vis_fit_results_viser.py. Colored by segment so you can see which
# group the projected mesh fails to match.
SEG_FILES = [
    ('body',  'body.npy'),
    ('lhand', 'left_hand.npy'),
    ('rhand', 'right_hand.npy'),
    ('face',  'face.npy'),
]
SEG_BGR = {            # OpenCV is BGR
    'body':  ( 60,  60, 255),   # red
    'lhand': ( 60, 230,  60),   # green
    'rhand': (255, 120,  60),   # blue
    'face':  ( 40, 215, 255),   # yellow
}


def _read_kpts(path):
    """Per-frame (K, 4) [x, y, z, conf] from a triangulation .npy (dict int->frame)."""
    raw = np.load(path, allow_pickle=True).item()
    out = []
    for k in raw:
        if not isinstance(k, int):
            continue
        v = raw[k]
        kp = np.asarray(v['kpts_3d'], dtype=np.float64)
        cf = np.asarray(v['confidence'], dtype=np.float64).reshape(-1, 1)
        out.append(np.hstack([kp, cf]))
    return out


def load_gt_keypoints(gt_dir):
    """Per-frame (pts (K,4), colors (K,3) BGR) for body+hands+face in read_item
    order (dict order, aligned with the mesh index and video frame). Returns []
    if the GT folder / body file is missing."""
    loaded = {}
    for seg, fname in SEG_FILES:
        p = osp.join(gt_dir, fname)
        if osp.isfile(p):
            loaded[seg] = _read_kpts(p)
    if 'body' not in loaded:
        return []
    n = min(len(v) for v in loaded.values())
    frames = []
    for i in range(n):
        pts_list, col_list = [], []
        for seg, _ in SEG_FILES:
            if seg not in loaded:
                continue
            arr = loaded[seg][i]
            pts_list.append(arr)
            col_list.append(np.tile(np.array(SEG_BGR[seg], np.int32), (arr.shape[0], 1)))
        frames.append((np.vstack(pts_list), np.vstack(col_list)))
    return frames


def draw_gt_keypoints(img, pts4, cols, camera_dict, radius=4):
    """Project world-space GT keypoints and draw them on img in-place, colored by
    segment. Points with NaN coords or conf<=0 are skipped."""
    K = np.asarray(camera_dict['K'], dtype=np.float64)
    D = np.asarray(camera_dict['D'], dtype=np.float64)
    R = np.asarray(camera_dict['R'], dtype=np.float64)
    T = np.asarray(camera_dict['T'], dtype=np.float64)
    rvec, _ = cv.Rodrigues(R)

    xyz = pts4[:, :3]
    conf = pts4[:, 3]
    valid = np.isfinite(xyz).all(axis=1) & (conf > 0)
    if not valid.any():
        return
    proj, _ = cv.projectPoints(xyz[valid], rvec, T.reshape(3, 1), K, D)
    proj = proj.reshape(-1, 2)
    vcols = cols[valid]

    h, w = img.shape[:2]
    for (px, py), c in zip(proj, vcols):
        x, y = int(round(px)), int(round(py))
        if 0 <= x < w and 0 <= y < h:
            cv.circle(img, (x, y), radius, (int(c[0]), int(c[1]), int(c[2])), -1, cv.LINE_AA)


def index_meshes(person_dir):
    out = {}
    pattern = re.compile(r'(\d+)_fit\.obj$')
    for p in glob.glob(osp.join(person_dir, 'meshes', '*.obj')):
        m = pattern.search(osp.basename(p))
        if m:
            out[int(m.group(1))] = p
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='', help='cfg suffix, e.g. 2 → looks in session_cfg2/')
    args = parser.parse_args()

    main_path = '/'.join(osp.abspath(__file__).split('/')[:-3]) + '/'
    resources_path = osp.join(main_path, 'resources')
    calibs_path = osp.join(resources_path, 'calibs')
    sessions_path = osp.join(resources_path, 'sessions')
    fit_root = osp.join(resources_path, 'fit_results')

    sid_paths = sorted(glob.glob(sessions_path + '/*'))
    if not sid_paths:
        print(f"No sessions under {sessions_path}")
        return

    for sid_path in sid_paths:
        session_id = osp.basename(sid_path.rstrip('/'))
        if '005013' not in session_id: continue
        with open(osp.join(sid_path, 'session_data.txt')) as f:
            lines = f.readlines()
            calib_date = lines[1][11:].strip()
        curr_calib_path = osp.join(calibs_path, calib_date)

        cam_dict = {}
        for cam_calib in glob.glob(curr_calib_path + '/*'):
            cam_name = osp.splitext(osp.basename(cam_calib))[0]
            fs = cv.FileStorage(cam_calib, cv.FILE_STORAGE_READ)
            K = fs.getNode('K').mat()
            D = fs.getNode('D').mat()
            R = fs.getNode('R').mat()
            T = fs.getNode('T').mat()
            fs.release()
            cam_dict[cam_map[cam_name]] = {'K': K, 'D': D, 'R': R, 'T': T}

        for activity in activities:
            activity_dir = osp.join(sid_path, activity)
            if not osp.isdir(activity_dir):
                continue

            cfg_sfx = f'_cfg{args.cfg}' if args.cfg else ''
            scene_fit_dir = osp.join(fit_root, f'{session_id}{cfg_sfx}', activity)
            if not osp.isdir(scene_fit_dir):
                print(f"[{session_id}/{activity}] no fit_results, skipping")
                continue

            person_frames, faces = {}, None
            for pid in (0, 1):
                frames = index_meshes(osp.join(scene_fit_dir, f'p{pid}'))
                if not frames:
                    continue
                person_frames[pid] = frames
                if faces is None:
                    sample = next(iter(frames.values()))
                    faces = np.asarray(trimesh.load(sample, force='mesh').faces,
                                       dtype=np.int32)
            if not person_frames:
                print(f"[{session_id}/{activity}] no meshes for either person")
                continue

            # GT 3D keypoints the optimizer was fit to (body coco17 + hands + face),
            # straight from triangulation_results — same source as
            # vis_fit_results_viser.py. Per person: list of (pts, colors) by frame,
            # in read_item/dict order (aligned with the mesh index and video frame).
            trig_root = osp.join(resources_path, 'triangulation_results')
            gt_by_person = {}
            for pid in person_frames:
                gt_dir = osp.join(trig_root, session_id, activity, f'p{pid}')
                gt = load_gt_keypoints(gt_dir)
                if gt:
                    gt_by_person[pid] = gt
                    print(f"  [p{pid}] {len(gt)} GT keypoint frames  (gt_dir={gt_dir})")

            vid_paths = sorted(glob.glob(osp.join(activity_dir, '*.mp4')))
            # vid_paths = [v for v in vid_paths if not ('E1.mp4' in v or 'E2.mp4' in v)]
            vid_paths = [v for v in vid_paths if ('GF.mp4' in v or 'GB.mp4' in v)]

            for vid_path in vid_paths:
                video_name = osp.splitext(osp.basename(vid_path))[0]
                is_backview = True if video_name == 'GB' else False
                if video_name not in cam_dict:
                    print(f"[{session_id}/{activity}/{video_name}] no calib, skipping")
                    continue

                K = cam_dict[video_name]['K']
                D = cam_dict[video_name]['D']
                R = cam_dict[video_name]['R']
                T = cam_dict[video_name]['T']

                cap = cv.VideoCapture(vid_path)
                fps = int(cap.get(cv.CAP_PROP_FPS)) or 30
                total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
                total_frames = 24

                if undistort:
                    new_K, _ = cv.getOptimalNewCameraMatrix(K, D, (FRAME_W, FRAME_H), 1)
                    K_used = new_K
                    D_used = np.zeros_like(D)  # image already undistorted
                    out_vid_path = osp.join(scene_fit_dir, f"{video_name}_fit_render_und.mp4")
                else:
                    K_used = K
                    D_used = D
                    out_vid_path = osp.join(scene_fit_dir, f"{video_name}_fit_render.mp4")

                cam_for_render = {'K': K_used, 'D': D_used, 'R': R, 'T': T}

                print(f"[{session_id}/{activity}/{video_name}] {total_frames} frames -> {out_vid_path}")
                writer = imageio.get_writer(
                    out_vid_path, fps=fps, mode='I', format='FFMPEG', macro_block_size=1
                )

                try:
                    for fidx in trange(total_frames):
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame = cv.resize(frame, (FRAME_W, FRAME_H))
                        if undistort:
                            frame = cv.undistort(frame, K, D, None, K_used)

                        meshes_this_frame = {}
                        for pid, frames in person_frames.items():
                            mesh_path = frames.get(fidx)
                            if mesh_path is None:
                                continue
                            verts = np.asarray(
                                trimesh.load(mesh_path, force='mesh').vertices,
                                dtype=np.float64,
                            )
                            meshes_this_frame[pid] = (verts, faces)

                        if meshes_this_frame:
                            frame = render_mesh_simple(
                                frame, meshes_this_frame, cam_for_render, alpha=alpha, is_backview=is_backview
                            )

                        # overlay GT 3D keypoints (projected), colored by segment:
                        # body=red, left hand=green, right hand=blue, face=yellow
                        for pid, gt in gt_by_person.items():
                            if fidx >= len(gt):
                                continue
                            pts4, cols = gt[fidx]
                            draw_gt_keypoints(frame, pts4, cols, cam_for_render, radius=4)

                        writer.append_data(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
                finally:
                    cap.release()
                    writer.close()

    print('\n[vis_fit_on_video] Done.')


if __name__ == '__main__':
    main()
