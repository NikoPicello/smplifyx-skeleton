from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math

import numpy as np
import torch
import torch.nn as nn


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


def rel_change(prev_val, curr_val):
    return (prev_val - curr_val) / max([np.abs(prev_val), np.abs(curr_val), 1])


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


def build_camera_tensors(camera_params, device):
    """
    Convert OpenCV camera parameters to tensors for nvdiffrast projection.

    camera_params keys:
        K         : (3, 3) OpenCV intrinsics
        D         : (4–8,) OpenCV distortion coefficients (k1,k2,p1,p2[,k3,...])
        R         : (3, 3) world-to-cam rotation
        T         : (3,)   world-to-cam translation
        image_size: (H, W)
    """
    K = torch.from_numpy(np.asarray(camera_params['K'], dtype=np.float32)).to(device)
    D = torch.from_numpy(np.asarray(camera_params['D'], dtype=np.float32).ravel()).to(device)
    R = torch.from_numpy(np.asarray(camera_params['R'], dtype=np.float32)).to(device)
    T = torch.from_numpy(np.asarray(camera_params['T'], dtype=np.float32).ravel()).to(device)
    H, W = camera_params['image_size']
    return {'K': K, 'D': D, 'R': R, 'T': T, 'H': H, 'W': W}


def _project_to_pixels(points, cam, z_min=0.05, norm_clamp=20.0):
    """
    Project world-space points to distorted pixel coords, matching cv.projectPoints.

    Returns pixel (u, v) coords directly (no clip-space intermediate). Differentiable
    in `points` — used by the GB keypoint-reprojection stage.

    points : (N, 3) or (1, N, 3) float world space
    cam    : dict with K (3x3), D (N,), R (3x3), T (3,)
    Returns ((N, 2) float pixel coords [u, v], (N,) bool valid mask). `valid` is
    False where the point is at/behind the camera; those points are neutralized so
    the output and its gradient are always finite — drop them via the mask.
    """
    p = points.reshape(-1, 3)
    K, D, R, T = cam['K'], cam['D'], cam['R'], cam['T']

    v_cam = p @ R.T + T
    valid = v_cam[:, 2] > z_min
    # Neutralize behind/at-camera points BEFORE the distortion polynomial: xy->0, z->1
    # so they land on the principal point — finite and zero-gradient (via where) —
    # instead of exploding through r**6 to inf/nan. norm_clamp bounds far-but-in-front
    # points so r**6 can't overflow fp32 for them either.
    z  = torch.where(valid, v_cam[:, 2], torch.ones_like(v_cam[:, 2]))
    xy = torch.where(valid.unsqueeze(-1), v_cam[:, :2], torch.zeros_like(v_cam[:, :2]))
    x_n = (xy[:, 0] / z).clamp(-norm_clamp, norm_clamp)
    y_n = (xy[:, 1] / z).clamp(-norm_clamp, norm_clamp)

    k1 = D[0]; k2 = D[1]
    p1 = D[2]; p2 = D[3]
    k3 = D[4] if D.shape[0] > 4 else torch.zeros((), device=D.device, dtype=D.dtype)

    r2 = x_n ** 2 + y_n ** 2
    radial = 1.0 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    x_d = x_n * radial + 2.0 * p1 * x_n * y_n + p2 * (r2 + 2.0 * x_n ** 2)
    y_d = y_n * radial + p1 * (r2 + 2.0 * y_n ** 2) + 2.0 * p2 * x_n * y_n

    u = K[0, 0] * x_d + K[0, 2]
    v = K[1, 1] * y_d + K[1, 2]
    return torch.stack([u, v], dim=-1), valid        # (N, 2)
