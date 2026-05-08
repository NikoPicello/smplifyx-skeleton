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

import cv2 as cv
import imageio
import numpy as np
import trimesh
from tqdm import trange


undistort = False
alpha = 0.55

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


# Body skeleton edges (SMPL-X joint order as assembled in skeletons.json).
# Indices follow idx_mapping.txt: 0-7 body, 8-23 left hand, 24-26 body lower,
# 27-42 right hand, 43-50 body lower.
_BODY_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),   # pelvis→spine→neck→head
    (2, 5), (5, 6), (6, 7), (7, 8),   # left arm
    (0, 43),(43,44),(44,45),           # left leg
    (2, 24),(24,25),(25,26),(26,27),   # right arm (wrist at 27)
    (0, 47),(47,48),(48,49),           # right leg
]

# dlib 68-point face landmark connections
_FACE_EDGES = (
    [(i, i + 1) for i in range(0, 16)] +           # jawline
    [(17, 18), (18, 19), (19, 20), (20, 21)] +      # right eyebrow
    [(22, 23), (23, 24), (24, 25), (25, 26)] +      # left eyebrow
    [(27, 28), (28, 29), (29, 30)] +                # nose bridge
    [(30, 31), (31, 32), (32, 33), (33, 34), (34, 35), (35, 30)] +  # nose tip
    [(36, 37), (37, 38), (38, 39), (39, 40), (40, 41), (41, 36)] +  # right eye
    [(42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 42)] +  # left eye
    [(48, 49), (49, 50), (50, 51), (51, 52), (52, 53), (53, 54),
     (54, 55), (55, 56), (56, 57), (57, 58), (58, 59), (59, 48)] +  # outer lips
    [(60, 61), (61, 62), (62, 63), (63, 64), (64, 65), (65, 66), (66, 67), (67, 60)]  # inner lips
)


def draw_face_landmarks(img, face_pts3d, camera_dict, color, radius=2):
    """Project world-space 68 face landmarks and draw them on img in-place."""
    K = np.asarray(camera_dict['K'], dtype=np.float64)
    D = np.asarray(camera_dict['D'], dtype=np.float64)
    R = np.asarray(camera_dict['R'], dtype=np.float64)
    T = np.asarray(camera_dict['T'], dtype=np.float64)
    rvec, _ = cv.Rodrigues(R)

    pts3d = np.array(face_pts3d, dtype=np.float64)
    valid = ~np.isnan(pts3d).any(axis=1)

    proj2d = np.full((len(pts3d), 2), np.nan)
    if valid.any():
        p, _ = cv.projectPoints(pts3d[valid], rvec, T.reshape(3, 1), K, D)
        proj2d[valid] = p.reshape(-1, 2)

    h, w = img.shape[:2]

    def in_frame(pt):
        return 0 <= int(pt[0]) < w and 0 <= int(pt[1]) < h

    for i, j in _FACE_EDGES:
        if i < len(proj2d) and j < len(proj2d):
            if not (np.isnan(proj2d[i]).any() or np.isnan(proj2d[j]).any()):
                pi = tuple(proj2d[i].astype(int))
                pj = tuple(proj2d[j].astype(int))
                if in_frame(pi) and in_frame(pj):
                    cv.line(img, pi, pj, color, 1, cv.LINE_AA)

    for pt in proj2d:
        if np.isnan(pt).any():
            continue
        px, py = int(pt[0]), int(pt[1])
        if in_frame((px, py)):
            cv.circle(img, (px, py), radius, color, -1, cv.LINE_AA)


def draw_keypoints(img, joints_world, camera_dict, color, radius=4,
                   draw_edges=True):
    """Project world-space skeleton joints and draw them on img in-place."""
    K = np.asarray(camera_dict['K'], dtype=np.float64)
    D = np.asarray(camera_dict['D'], dtype=np.float64)
    R = np.asarray(camera_dict['R'], dtype=np.float64)
    T = np.asarray(camera_dict['T'], dtype=np.float64)
    rvec, _ = cv.Rodrigues(R)

    pts3d = np.array(joints_world, dtype=np.float64)
    valid = ~np.isnan(pts3d).any(axis=1)

    proj2d = np.full((len(pts3d), 2), np.nan)
    if valid.any():
        p, _ = cv.projectPoints(pts3d[valid], rvec, T.reshape(3, 1), K, D)
        proj2d[valid] = p.reshape(-1, 2)

    h, w = img.shape[:2]
    def in_frame(pt):
        return 0 <= int(pt[0]) < w and 0 <= int(pt[1]) < h

    if draw_edges:
        for i, j in _BODY_EDGES:
            if i < len(proj2d) and j < len(proj2d):
                if not (np.isnan(proj2d[i]).any() or np.isnan(proj2d[j]).any()):
                    pi, pj = tuple(proj2d[i].astype(int)), tuple(proj2d[j].astype(int))
                    if in_frame(pi) and in_frame(pj):
                        cv.line(img, pi, pj, color, 1, cv.LINE_AA)

    for pt in proj2d:
        if np.isnan(pt).any():
            continue
        px, py = int(pt[0]), int(pt[1])
        if in_frame((px, py)):
            cv.circle(img, (px, py), radius, color, -1, cv.LINE_AA)


def index_meshes(person_dir):
    out = {}
    pattern = re.compile(r'(\d+)_fit\.obj$')
    for p in glob.glob(osp.join(person_dir, 'meshes', '*.obj')):
        m = pattern.search(osp.basename(p))
        if m:
            out[int(m.group(1))] = p
    return out


def main():
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

            scene_fit_dir = osp.join(fit_root, session_id, activity)
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

            # Load triangulated face landmarks per person.
            # head_data shape: (frames, 68, 3) in world space.
            # Map sequence index → video frame_idx using skeletons.json order.
            trig_root = osp.join(resources_path, 'triangulation_results')
            face_by_person = {}
            for pid in person_frames:
                head_file = osp.join(trig_root, session_id, activity,
                                     'head', f'p{pid}_triangulated.npy')
                if not osp.isfile(head_file):
                    continue
                head_data = np.load(head_file, allow_pickle=True)
                if not isinstance(head_data, np.ndarray) or head_data.ndim != 3:
                    continue
                skel_path = osp.join(scene_fit_dir, f'p{pid}', 'skeletons.json')
                if not osp.isfile(skel_path):
                    continue
                face_frame_map = {}
                with open(skel_path) as sf:
                    for seq_idx, line in enumerate(sf):
                        if seq_idx >= head_data.shape[0]:
                            break
                        rec = json.loads(line)
                        face_frame_map[rec['frame_idx']] = head_data[seq_idx]  # (68, 3)
                face_by_person[pid] = face_frame_map

            vid_paths = sorted(glob.glob(osp.join(activity_dir, '*.mp4')))
            vid_paths = [v for v in vid_paths if not ('E1.mp4' in v or 'E2.mp4' in v)]

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
                total_frames = 5

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

                # load skeleton keypoints (world-space 3D) for each person
                skeletons_by_person = {}
                for pid in person_frames:
                    skel_path = osp.join(scene_fit_dir, f'p{pid}', 'skeletons.json')
                    if not osp.isfile(skel_path):
                        continue
                    skel_by_frame = {}
                    with open(skel_path) as sf:
                        for line in sf:
                            rec = json.loads(line)
                            skel_by_frame[rec['frame_idx']] = rec['joints']
                    skeletons_by_person[pid] = skel_by_frame

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

                        # overlay skeleton keypoints on top of mesh
                        for pid, skel_frames in skeletons_by_person.items():
                            joints = skel_frames.get(fidx)
                            if joints is None:
                                continue
                            kpt_color = tuple(
                                max(0, c - 80) for c in PERSON_COLORS.get(pid, (200, 200, 200))
                            )
                            draw_keypoints(frame, joints, cam_for_render,
                                           color=kpt_color, radius=4)

                        # overlay triangulated face landmarks
                        for pid, face_frames in face_by_person.items():
                            face_pts = face_frames.get(fidx)
                            if face_pts is None:
                                continue
                            face_color = tuple(
                                max(0, c - 120) for c in PERSON_COLORS.get(pid, (200, 200, 200))
                            )
                            draw_face_landmarks(frame, face_pts, cam_for_render,
                                                color=face_color, radius=2)

                        writer.append_data(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
                finally:
                    cap.release()
                    writer.close()

    print('\n[vis_fit_on_video] Done.')


if __name__ == '__main__':
    main()
