"""Visualize fitted SMPL-X meshes together with the GT 3D keypoints, using viser.

For a scene under `fit_results/<session>_cfg<X>/<activity>/`, this loads each
person's fitted `meshes/*.obj` and overlays the *same* triangulated 3D keypoints
that the optimizer was fit to (body + hands + face, read straight from
`triangulation_results/<session>/<activity>/p<i>/{body,left_hand,right_hand,face}.npy`).

Keypoints are colored by segment so you can see which group the mesh fails to
match: body=red, left hand=green, right hand=blue, face=yellow. The mesh is drawn
semi-transparent so keypoints behind the surface remain visible — if the cloud
and the surface don't overlap, that's your bad fit.

Example:
    python vis_fit_results_viser.py \\
        --scene-dir ../../resources/fit_results/005013_cfg7/lego
"""

import glob
import json
import os.path as osp
import re
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh
import tyro
import viser

PERSON_IDS = ['p0', 'p1']

# mesh color per person
PERSON_MESH_COLORS = [
    (100, 149, 237),  # p0: cornflower blue
    (255, 127,  14),  # p1: orange
]

# GT keypoint colors by segment (uint8 RGB)
SEG_COLORS = {
    'body':  (255,  60,  60),   # red
    'lhand': ( 60, 230,  60),   # green
    'rhand': ( 60, 120, 255),   # blue
    'face':  (255, 215,  40),   # yellow
}
SEG_FILES = [
    ('body',  'body.npy'),
    ('lhand', 'left_hand.npy'),
    ('rhand', 'right_hand.npy'),
    ('face',  'face.npy'),
]


def _read_kpts(path: str) -> List[np.ndarray]:
    """Per-frame (K, 4) [x, y, z, conf] from a triangulation .npy (dict of int->frame)."""
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


def load_gt_keypoints(gt_dir: str) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Per-frame (points (K,4), colors (K,3)) for body+hands+face, in read_item order.

    Returns [] if the GT folder / body file is missing.
    """
    loaded: Dict[str, List[np.ndarray]] = {}
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
            arr = loaded[seg][i]  # (Kseg, 4)
            pts_list.append(arr)
            col_list.append(np.tile(np.array(SEG_COLORS[seg], np.uint8), (arr.shape[0], 1)))
        frames.append((np.vstack(pts_list), np.vstack(col_list)))
    return frames


def derive_gt_dir(scene_dir: str, pid: str, gt_root: str = "") -> str:
    """fit_results/<session>_cfg<X>/<activity> + pid  ->  triangulation_results/<session>/<activity>/pid."""
    scene_dir = scene_dir.rstrip('/')
    activity = osp.basename(scene_dir)
    session = re.sub(r'_cfg.*$', '', osp.basename(osp.dirname(scene_dir)))
    if not gt_root:
        fit_root = osp.dirname(osp.dirname(scene_dir))          # .../resources/fit_results
        gt_root = osp.join(osp.dirname(fit_root), 'triangulation_results')
    return osp.join(gt_root, session, activity, pid)


def load_person(scene_dir: str, pid: str, gt_root: str = "") \
        -> Tuple[List[trimesh.Trimesh], List[Tuple[np.ndarray, np.ndarray]]]:
    person_dir = osp.join(scene_dir, pid)
    mesh_paths = sorted(glob.glob(osp.join(person_dir, 'meshes', '*.obj')))
    meshes = [trimesh.load(p, force='mesh') for p in mesh_paths]
    gt_dir = derive_gt_dir(scene_dir, pid, gt_root)
    gt = load_gt_keypoints(gt_dir)
    print(f"[{pid}] {len(meshes)} meshes | {len(gt)} GT keypoint frames  (gt_dir={gt_dir})")
    return meshes, gt


def set_person_frame(
    server: viser.ViserServer,
    pid: str,
    mesh: trimesh.Trimesh,
    gt_frame: Optional[Tuple[np.ndarray, np.ndarray]],
    mesh_color: Tuple[int, int, int],
    mesh_opacity: float,
    show_keypoints: bool,
    point_size: float,
) -> None:
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)

    # opacity is supported on modern viser; fall back gracefully if not.
    try:
        server.scene.add_mesh_simple(
            f"/{pid}/mesh", vertices=vertices, faces=faces,
            flat_shading=False, wireframe=False, color=mesh_color,
            opacity=mesh_opacity,
        )
    except TypeError:
        server.scene.add_mesh_simple(
            f"/{pid}/mesh", vertices=vertices, faces=faces,
            flat_shading=False, wireframe=False, color=mesh_color,
        )

    node = f"/{pid}/gt_keypoints"
    if show_keypoints and gt_frame is not None:
        pts, cols = gt_frame
        xyz = pts[:, :3]
        conf = pts[:, 3]
        valid = np.isfinite(xyz).all(axis=1) & (conf > 0)
        server.scene.add_point_cloud(
            node, points=xyz[valid].astype(np.float32),
            colors=cols[valid], point_size=point_size,
        )
    else:
        try:
            server.scene.remove_scene_node(node)
        except Exception:
            pass


def main(
    scene_dir: str = "../../resources/fit_results/005013_cfg1/lego",
    cfg: str = "",
    gt_root: str = "",
    fps: float = 10.0,
    autoplay: bool = False,
    mesh_opacity: float = 0.5,
    point_size: float = 0.02,
    up: str = "+z",
):
    if cfg:
        head, activity = osp.split(scene_dir.rstrip('/'))
        head2, session = osp.split(head)
        scene_dir = osp.join(head2, f'{session}_cfg{cfg}', activity)

    people: Dict[str, Tuple[list, list]] = {}
    for pid in PERSON_IDS:
        meshes, gt = load_person(scene_dir, pid, gt_root)
        if not meshes:
            print(f"[{pid}] no meshes, skipping")
            continue
        people[pid] = (meshes, gt)

    if not people:
        print(f"No data found under {scene_dir}")
        return

    # Align meshes and GT by index; fall back to mesh count if GT is missing.
    num_frames = min(
        len(meshes) if not gt else min(len(meshes), len(gt))
        for meshes, gt in people.values()
    )
    print(f"Animating {num_frames} frames")

    server = viser.ViserServer()
    server.scene.world_axes.visible = True
    server.scene.set_up_direction(up)

    frame_slider = server.gui.add_slider("Frame", min=0, max=max(num_frames - 1, 0), step=1, initial_value=0)
    play_btn = server.gui.add_button("Play / Pause")
    kp_toggle = server.gui.add_checkbox("Show GT keypoints", initial_value=True)

    playing = [autoplay]
    current = [0]

    def render(frame_idx: int):
        for pid, (meshes, gt) in people.items():
            idx = PERSON_IDS.index(pid)
            mesh_color = PERSON_MESH_COLORS[idx % len(PERSON_MESH_COLORS)]
            gt_frame = gt[frame_idx] if gt and frame_idx < len(gt) else None
            set_person_frame(
                server, pid, meshes[frame_idx], gt_frame,
                mesh_color=mesh_color, mesh_opacity=mesh_opacity,
                show_keypoints=kp_toggle.value, point_size=point_size,
            )

    @play_btn.on_click
    def _(_):
        playing[0] = not playing[0]

    @frame_slider.on_update
    def _(_):
        current[0] = frame_slider.value
        render(current[0])

    @kp_toggle.on_update
    def _(_):
        render(current[0])

    render(0)

    dt = 1.0 / fps
    print("\nViser server running. Open http://localhost:8080 in your browser.")
    print("Press Ctrl+C to exit.\n")

    try:
        while True:
            if playing[0]:
                current[0] = (current[0] + 1) % num_frames
                frame_slider.value = current[0]
                render(current[0])
                time.sleep(dt)
            else:
                time.sleep(0.05)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    tyro.cli(main)
