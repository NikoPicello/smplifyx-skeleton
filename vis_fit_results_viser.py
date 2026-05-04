"""Visualize fitted SMPL-X OBJ meshes and skeleton joints from fit_results/ using viser.

For a given scene under `fit_results/<session>/<scene>/`, this script loads
both `p0` and `p1` subfolders (each with `meshes/*.obj` and `skeletons.json`)
and renders them together with a frame slider and play/pause control.

Example:
    python vis_fit_results_viser.py \\
        --scene-dir ../../resources/fit_results/004096/animals
"""

import glob
import json
import os.path as osp
import time
from typing import List, Tuple

import numpy as np
import trimesh
import tyro
import viser


BONES: List[Tuple[int, int]] = [
    (4, 3), (3, 2), (2, 1), (1, 0), (0, 43), (43, 44), (44, 45),
    (45, 46), (0, 47), (47, 48), (48, 49), (49, 50), (2, 5), (5, 6),
    (6, 7), (7, 8), (2, 24), (24, 25), (25, 26), (26, 27), (8, 9),
    (9, 10), (10, 11), (8, 12), (12, 13), (13, 14), (8, 15), (15, 16),
    (16, 17), (8, 18), (18, 19), (19, 20), (8, 21), (21, 22), (22, 23),
    (27, 28), (28, 29), (29, 30), (27, 31), (27, 32), (27, 33), (27, 34),
    (27, 35), (27, 36), (27, 37), (27, 38), (27, 39), (27, 40), (27, 41), (27, 42),
]

JOINT_NAMES = [
    'Skeleton', 'Ab', 'Chest', 'Neck', 'Head', 'LShoulder', 'LUArm', 'LFArm', 'LHand',
    'LThumb1', 'LThumb2', 'LThumb3', 'LIndex1', 'LIndex2', 'LIndex3', 'LMiddle1', 'LMiddle2',
    'LMiddle3', 'LRing1', 'LRing2', 'LRing3', 'LPinky1', 'LPinky2', 'LPinky3', 'RShoulder', 'RUArm',
    'RFArm', 'RHand', 'RThumb1', 'RThumb2', 'RThumb3', 'RIndex1', 'RIndex2', 'RIndex3', 'RMiddle1',
    'RMiddle2', 'RMiddle3', 'RRing1', 'RRing2', 'RRing3', 'RPinky1', 'RPinky2', 'RPinky3', 'LThigh',
    'LShin', 'LFoot', 'LToe', 'RThigh', 'RShin', 'RFoot', 'RToe',
]

PERSON_IDS = ['p0', 'p1']

# (mesh_color, joint_color, bone_color) per person
PERSON_COLORS = [
    ((100, 149, 237), (255,  80,  80), (220, 220, 220)),  # p0: blue mesh / red joints
    ((255, 127,  14), ( 80, 200,  80), (220, 220, 220)),  # p1: orange mesh / green joints
]


def load_skeleton_frames(path: str) -> List[np.ndarray]:
    frames = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            joints = np.array(obj['joints'], dtype=np.float64).reshape(-1, 3)
            frames.append(joints)
    return frames


def load_person(scene_dir: str, person_id: str) -> Tuple[List[trimesh.Trimesh], List[np.ndarray]]:
    person_dir = osp.join(scene_dir, person_id)
    mesh_paths = sorted(glob.glob(osp.join(person_dir, 'meshes', '*.obj')))
    meshes = [trimesh.load(p, force='mesh') for p in mesh_paths]
    skeletons = load_skeleton_frames(osp.join(person_dir, 'skeletons.json'))
    if len(meshes) != len(skeletons):
        print(f"[{person_id}] warning: {len(meshes)} meshes vs {len(skeletons)} skeleton frames")
    return meshes, skeletons


def set_person_frame(
    server: viser.ViserServer,
    person_id: str,
    mesh: trimesh.Trimesh,
    joints: np.ndarray,
    mesh_color: Tuple[int, int, int],
    joint_color: Tuple[int, int, int],
    bone_color: Tuple[int, int, int],
    show_labels: bool,
) -> None:
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)

    server.scene.add_mesh_simple(
        f"/{person_id}/mesh",
        vertices=vertices,
        faces=faces,
        flat_shading=False,
        wireframe=False,
        color=mesh_color,
    )

    server.scene.add_point_cloud(
        f"/{person_id}/skeleton/joints",
        points=joints,
        colors=np.tile(np.array(joint_color, dtype=np.uint8), (joints.shape[0], 1)),
        point_size=0.015,
    )

    starts = np.array([joints[a] for a, _ in BONES])
    ends = np.array([joints[b] for _, b in BONES])
    segments = np.stack([starts, ends], axis=1)  # (N, 2, 3)
    server.scene.add_line_segments(
        f"/{person_id}/skeleton/bones",
        points=segments,
        colors=np.tile(np.array(bone_color, dtype=np.uint8), (len(BONES), 2, 1)),
        line_width=2.0,
    )

    for i, pt in enumerate(joints):
        node = f"/{person_id}/skeleton/labels/j{i}"
        if show_labels and i < len(JOINT_NAMES):
            server.scene.add_label(node, text=JOINT_NAMES[i], position=pt)
        else:
            try:
                server.scene.remove_scene_node(node)
            except Exception:
                pass


def main(
    scene_dir: str = "../../resources/fit_results/004096/lego",
    fps: float = 10.0,
    autoplay: bool = False,
    up: str = "+z",
):
    people = {}
    for pid in PERSON_IDS:
        meshes, skeletons = load_person(scene_dir, pid)
        print(f"[{pid}] loaded {len(meshes)} meshes and {len(skeletons)} skeleton frames")
        if not meshes or not skeletons:
            print(f"[{pid}] nothing to show, skipping")
            continue
        people[pid] = (meshes, skeletons)

    if not people:
        print(f"No data found under {scene_dir}")
        return

    num_frames = min(min(len(m), len(s)) for m, s in people.values())
    print(f"Animating {num_frames} frames")

    server = viser.ViserServer()
    server.scene.world_axes.visible = True
    server.scene.set_up_direction(up)

    frame_slider = server.gui.add_slider("Frame", min=0, max=num_frames - 1, step=1, initial_value=0)
    play_btn = server.gui.add_button("Play / Pause")
    label_toggle = server.gui.add_checkbox("Show labels", initial_value=False)

    playing = [autoplay]
    current = [0]

    def render(frame_idx: int):
        for pid, (meshes, skeletons) in people.items():
            idx = PERSON_IDS.index(pid)
            mesh_color, joint_color, bone_color = PERSON_COLORS[idx % len(PERSON_COLORS)]
            set_person_frame(
                server,
                pid,
                meshes[frame_idx],
                skeletons[frame_idx],
                mesh_color=mesh_color,
                joint_color=joint_color,
                bone_color=bone_color,
                show_labels=label_toggle.value,
            )

    @play_btn.on_click
    def _(_):
        playing[0] = not playing[0]

    @frame_slider.on_update
    def _(_):
        current[0] = frame_slider.value
        render(current[0])

    @label_toggle.on_update
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
