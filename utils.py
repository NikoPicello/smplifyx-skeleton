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