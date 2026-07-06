from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import nvdiffrast.torch as dr


def aa_nearest(v, ref):
    """Axis-angle vector equivalent to `v` (same rotation) but closest in L2 to `ref`.

    Axis-angle is degenerate near |theta|=pi: a rotation by theta about k equals one by
    (theta + 2*pi*m) about k, so the principal vector flips sign/axis when the rotation
    crosses pi even though the orientation barely moves. A naive L2 term `(v - ref)^2`
    therefore spikes there. This returns the representation of `v` nearest to a reference
    (the anchor target, or the previous frame) so the anchor / saved trajectory stays
    continuous. Last dim must be 3; leading dims broadcast.
    """
    n = v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    axis = v / n
    best = v
    best_d = (v - ref).pow(2).sum(dim=-1, keepdim=True)
    for m in (-2, -1, 1, 2):
        cand = (n + 2.0 * math.pi * m) * axis
        d = (cand - ref).pow(2).sum(dim=-1, keepdim=True)
        closer = d < best_d
        best = torch.where(closer, cand, best)
        best_d = torch.where(closer, d, best_d)
    return best



def to_tensor(tensor, dtype=torch.float32):
    if torch.Tensor == type(tensor):
        return tensor.clone().detach()
    else:
        return torch.tensor(tensor, dtype)

def rel_change(prev_val, curr_val):
    return (prev_val - curr_val) / max([np.abs(prev_val), np.abs(curr_val), 1])


def max_grad_change(grad_arr):
    return grad_arr.abs().max()

class JointMapper(nn.Module):
    def __init__(self, joint_maps=None):
        super(JointMapper, self).__init__()
        if joint_maps is None:
            self.joint_maps = joint_maps
        else:
            self.register_buffer('joint_maps',
                                 torch.tensor(joint_maps, dtype=torch.long))

    def forward(self, joints, **kwargs):
        if self.joint_maps is None:
            return joints
        else:
            return torch.index_select(joints, 1, self.joint_maps)

class GMoF(nn.Module):
    def __init__(self, rho=1):
        super(GMoF, self).__init__()
        self.rho = rho

    def extra_repr(self):
        return 'rho = {}'.format(self.rho)

    def forward(self, residual):
        squared_res = residual ** 2
        dist = torch.div(squared_res, squared_res + self.rho ** 2)
        return self.rho ** 2 * dist


# ----------------------------------------------------------------------
# Silhouette rendering helpers (used by the placement-refinement stage in
# fit_single_frame; kept here so the silhouette term is no longer part of
# the main SMPLifyLoss in fitting.py).
# ----------------------------------------------------------------------
def _project_to_clip(verts, cam):
    """
    Project world-space vertices to nvdiffrast clip space, matching cv.projectPoints.

    Applies the full OpenCV radial+tangential distortion model so the rendered
    silhouette lands on the same distorted image plane as the GT masks.

    verts : (1, V, 3) float32 world space
    cam   : dict with K (3x3), D (N,), R (3x3), T (3,), H (int), W (int)
    Returns (1, V, 4) float32 clip space
    """
    v = verts[0]                              # (V, 3)
    K, D, R, T = cam['K'], cam['D'], cam['R'], cam['T']
    H, W = cam['H'], cam['W']

    # --- camera space ---
    v_cam = v @ R.T + T                       # (V, 3)
    z = v_cam[:, 2].clamp(min=1e-4)

    # --- normalised (undistorted pinhole) coords ---
    x_n = v_cam[:, 0] / z
    y_n = v_cam[:, 1] / z

    # --- OpenCV distortion model (matches cv.projectPoints) ---
    k1 = D[0]; k2 = D[1]
    p1 = D[2]; p2 = D[3]
    k3 = D[4] if D.shape[0] > 4 else torch.zeros(1, device=D.device, dtype=D.dtype).squeeze()

    r2 = x_n ** 2 + y_n ** 2
    r4 = r2 ** 2
    r6 = r2 ** 3
    radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
    x_d = x_n * radial + 2.0 * p1 * x_n * y_n + p2 * (r2 + 2.0 * x_n ** 2)
    y_d = y_n * radial + p1 * (r2 + 2.0 * y_n ** 2) + 2.0 * p2 * x_n * y_n

    # --- pixel coords on the distorted image plane ---
    u   = K[0, 0] * x_d + K[0, 2]
    v_p = K[1, 1] * y_d + K[1, 2]

    # --- clip space for nvdiffrast (y-up / OpenGL convention) ---
    x_clip = (2.0 * u / W - 1.0) * z
    y_clip = (1.0 - 2.0 * v_p / H) * z
    z_clip = z
    w      = z

    clip = torch.stack([x_clip, y_clip, z_clip, w], dim=-1)  # (V, 4)
    return clip.unsqueeze(0)                  # (1, V, 4)


def silhouette_term(verts, gt_silhouettes, cameras, glctx, body_faces_sil):
    """Soft-IoU silhouette loss summed over camera views (unweighted).

    Lives here (not in SMPLifyLoss) since the silhouette term is no longer part of
    the main optimization — it is only used by the placement-refinement stage in
    fit_single_frame. The caller applies its own weight.

    verts          : (1, V, 3) world-space vertices
    gt_silhouettes : list of (H, W) binary masks, one per camera (None to skip)
    cameras        : list of camera dicts (see build_camera_tensors in fitting.py)
    glctx          : nvdiffrast rasterization context
    body_faces_sil : (F, 3) int face index tensor
    Returns scalar tensor = sum_v (1 - IoU_v).
    """
    verts = verts.float()
    faces = body_faces_sil.to(verts.device)
    V = verts.shape[1]
    alpha_vtx = torch.ones(1, V, 1, device=verts.device, dtype=torch.float32)
    sil = verts.new_zeros(())
    for v_idx in range(min(len(cameras), len(gt_silhouettes))):
        gt = gt_silhouettes[v_idx]
        if gt is None:
            continue
        cam = cameras[v_idx]
        H, W = cam['H'], cam['W']
        clip = _project_to_clip(verts, cam)
        rast, _ = dr.rasterize(glctx, clip, faces, resolution=[H, W])
        alpha, _ = dr.interpolate(alpha_vtx, rast, faces)
        rendered = dr.antialias(alpha, rast, clip, faces)[..., 0].clamp(0.0, 1.0)
        rendered = rendered.flip(dims=[1])  # OpenGL row-0=bottom → cv2 row-0=top
        gt_f = gt.to(rendered.device).float()
        if gt_f.dim() == 2:
            gt_f = gt_f.unsqueeze(0)
        if gt_f.sum() < 1.0:
            continue
        inter = (rendered * gt_f).sum()
        union = (rendered + gt_f - rendered * gt_f).sum()
        sil = sil + (1.0 - inter / (union + 1e-6))
    return sil


def visualize_stage(verts, gt_silhouettes, cameras, glctx, body_faces_sil,
                    stage_idx, frame_idx, out_dir='./log/sil_vis', cam_names=None):
    """
    Render the current mesh silhouette for every camera and save overlay
    PNGs to out_dir/f{frame_idx:04d}/stage{stage_idx:02d}_v{i:02d}.png.

    verts          : (1, V, 3) world-space vertices
    gt_silhouettes : list of (H, W) binary masks, one per camera (None to skip)
    cameras        : list of camera dicts (see build_camera_tensors in fitting.py)
    glctx          : nvdiffrast rasterization context
    body_faces_sil : (F, 3) int face index tensor

    Colour key in the saved image (BGR):
        Green  : GT mask only   (model missing)
        Red    : rendered only  (model too big / wrong place)
        Yellow : both           (correct overlap)
        Black  : neither
    """
    if gt_silhouettes is None:
        return
    save_dir = os.path.join(out_dir, f'f{frame_idx:04d}')
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        verts_f = verts.float()
        faces   = body_faces_sil.to(verts_f.device)
        V       = verts_f.shape[1]
        alpha_vtx = torch.ones(1, V, 1, device=verts_f.device, dtype=torch.float32)

        for v_idx in range(min(len(cameras), len(gt_silhouettes))):
            gt = gt_silhouettes[v_idx]
            if gt is None:
                continue
            cam  = cameras[v_idx]
            H, W = cam['H'], cam['W']

            clip = _project_to_clip(verts_f, cam)
            rast, _  = dr.rasterize(glctx, clip, faces, resolution=[H, W])
            alpha, _ = dr.interpolate(alpha_vtx, rast, faces)
            rendered = dr.antialias(alpha, rast, clip, faces)[..., 0].clamp(0, 1)
            rendered = rendered.flip(dims=[1])  # OpenGL row-0=bottom → cv2 row-0=top

            rend_np = (rendered[0].cpu().numpy() * 255).astype(np.uint8)
            gt_np   = (gt.cpu().numpy() * 255).astype(np.uint8)
            recall_val = 0.0
            if gt_np.sum() > 0:
                inter = np.minimum(rend_np, gt_np).sum()
                recall_val = inter / (gt_np.sum() + 1e-6)

            # BGR colour overlay
            img = np.zeros((H, W, 3), dtype=np.uint8)
            img[:, :, 1] = gt_np                        # green  = GT
            img[:, :, 2] = rend_np                      # red    = rendered
            # where both are present the channels add → yellow

            label = cam_names[v_idx] if (cam_names and v_idx < len(cam_names)) else f'v{v_idx}'
            fname = os.path.join(save_dir,
                                 f'stage{stage_idx:02d}_{label}_recall{recall_val:.2f}.png')
            cv2.imwrite(fname, img)

    print(f"  [vis] stage {stage_idx} → {save_dir}/")


def load_gt_silhouettes(mask_folder, cam_names, frame_idx, person_id, device):
    """Per-camera binary silhouettes for one frame, aligned with `cam_names`.

    Masks live at {mask_folder}/{cam_name}/f{frame_idx:05d}.png with pixel
    values 0=person0, 1=person1, 255=background. Returns a list (same order as
    `cam_names`) of (H, W) float32 tensors — 1.0 where this person, else 0.0 —
    or None for any view whose mask file is missing. The fitting helper skips
    None views and all-zero masks (person not visible in that view).
    """
    sils = []
    for cam_name in cam_names:
        mpath = os.path.join(mask_folder, cam_name, f'f{frame_idx:05d}.png')
        m = cv2.imread(mpath, cv2.IMREAD_UNCHANGED)
        if m is None:
            sils.append(None)
            continue
        sil = (m == person_id).astype(np.float32)
        sils.append(torch.from_numpy(sil).to(device=device, dtype=torch.float32))
    return sils