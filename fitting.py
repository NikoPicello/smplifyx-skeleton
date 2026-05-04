# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

# import sys
# import os

# import time

import numpy as np

import torch
import torch.nn as nn

# from mesh_viewer import MeshViewer
import utils

import nvdiffrast.torch as dr


def build_camera_tensors(camera_params, device):
    """
    Convert OpenCV camera parameters to tensors for nvdiffrast projection.

    camera_params keys:
        K         : (3, 3) OpenCV intrinsics
        R         : (3, 3) world-to-cam rotation (OpenCV column-vector convention)
        T         : (3,)   world-to-cam translation (OpenCV)
        image_size: (H, W)
    """
    K = torch.from_numpy(np.asarray(camera_params['K'], dtype=np.float32)).to(device)
    R = torch.from_numpy(np.asarray(camera_params['R'], dtype=np.float32)).to(device)
    T = torch.from_numpy(np.asarray(camera_params['T'], dtype=np.float32).ravel()).to(device)
    H, W = camera_params['image_size']
    return {'K': K, 'R': R, 'T': T, 'H': H, 'W': W}


def _project_to_clip(verts, cam):
    """
    Project world-space vertices to nvdiffrast clip space.

    OpenCV convention: x_cam = R @ x_world + T  (column vectors)
    nvdiffrast clip space: (x_clip, y_clip, z_clip, w) where NDC = clip / w,
    y-up (OpenGL convention).

    verts : (1, V, 3) float32 world space
    cam   : dict with K (3x3), R (3x3), T (3,), H (int), W (int)
    Returns (1, V, 4) float32 clip space
    """
    v = verts[0]                              # (V, 3)
    K, R, T = cam['K'], cam['R'], cam['T']
    H, W = cam['H'], cam['W']

    # Camera space  (row-vector form: v_cam = v @ R.T + T)
    v_cam = v @ R.T + T                       # (V, 3)

    z = v_cam[:, 2].clamp(min=1e-4)          # depth, avoid divide-by-zero
    u   = v_cam[:, 0] / z * K[0, 0] + K[0, 2]   # pixel x
    v_p = v_cam[:, 1] / z * K[1, 1] + K[1, 2]   # pixel y

    # Convert to NDC then to clip space (clip = ndc * w)
    # nvdiffrast NDC: x ∈ [-1,1] left→right, y ∈ [-1,1] bottom→top (OpenGL y-up)
    x_clip = (2.0 * u / W - 1.0) * z
    y_clip = (1.0 - 2.0 * v_p / H) * z      # flip y for OpenGL
    z_clip = z                                # monotone depth (not true NDC z, but fine for silhouette)
    w      = z

    clip = torch.stack([x_clip, y_clip, z_clip, w], dim=-1)  # (V, 4)
    return clip.unsqueeze(0)                  # (1, V, 4)




class FittingMonitor(object):
    def __init__(self, summary_steps=1, visualize=False,
                 maxiters=100, ftol=2e-09, gtol=1e-05,
                 body_color=(1.0, 1.0, 0.9, 1.0),
                 model_type='smpl',
                 **kwargs):
        super(FittingMonitor, self).__init__()

        self.maxiters = maxiters
        self.ftol = ftol
        self.gtol = gtol

        self.visualize = visualize
        self.summary_steps = summary_steps
        self.body_color = body_color
        self.model_type = model_type

    def __enter__(self):
        self.steps = 0
        if self.visualize:
            self.mv = MeshViewer(body_color=self.body_color)
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        if self.visualize:
            self.mv.close_viewer()

    def set_colors(self, vertex_color):
        batch_size = self.colors.shape[0]

        self.colors = np.tile(
            np.array(vertex_color).reshape(1, 3),
            [batch_size, 1])

    def run_fitting(self, optimizer, closure, params, body_model,
                    use_vposer=True, pose_embedding=None, vposer=None,
                    **kwargs):
        ''' Helper function for running an optimization process
            Parameters
            ----------
                optimizer: torch.optim.Optimizer
                    The PyTorch optimizer object
                closure: function
                    The function used to calculate the gradients
                params: list
                    List containing the parameters that will be optimized
                body_model: nn.Module
                    The body model PyTorch module
                use_vposer: bool
                    Flag on whether to use VPoser (default=True).
                pose_embedding: torch.tensor, BxN
                    The tensor that contains the latent pose variable.
                vposer: nn.Module
                    The VPoser module
            Returns
            -------
                loss: float
                The final loss value
        '''
        append_wrists = self.model_type == 'smpl' and use_vposer
        prev_loss = None
        # flag = False
        for n in range(self.maxiters):
            loss = optimizer.step(closure)

            if torch.isnan(loss).sum() > 0:
                print('NaN loss value, stopping!')
                break

            if torch.isinf(loss).sum() > 0:
                print('Infinite loss value, stopping!')
                break

            if n > 0 and prev_loss is not None and self.ftol > 0:
                loss_rel_change = utils.rel_change(prev_loss, loss.item())

                if loss_rel_change <= self.ftol:
                    break

            if all([torch.abs(var.grad.view(-1).max()).item() < self.gtol
                    for var in params if var.grad is not None]):
                break

            if self.visualize and n % self.summary_steps == 0:
                body_pose = vposer.decode(
                    pose_embedding, output_type='aa').view(
                        1, -1) if use_vposer else None

                if append_wrists:
                    wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                             dtype=body_pose.dtype,
                                             device=body_pose.device)
                    body_pose = torch.cat([body_pose, wrist_pose], dim=1)
                model_output = body_model(
                    return_verts=True, body_pose=body_pose)
                vertices = model_output.vertices.detach().cpu().numpy()

                self.mv.update_mesh(vertices.squeeze(),
                                    body_model.faces)

            prev_loss = loss.item()

        return prev_loss

    def create_fitting_closure(self,
                               optimizer, body_model, camera=None,
                               gt_joints=None, loss=None,
                               joints_conf=None,
                               joint_weights=None,
                               return_verts=True, return_full_pose=False,
                               use_vposer=False, vposer=None,
                               pose_embedding=None,
                               create_graph=False,
                               gt_silhouettes=None,
                               **kwargs):
        faces_tensor = body_model.faces_tensor.view(-1)
        append_wrists = self.model_type == 'smpl' and use_vposer

        def fitting_func(backward=True):
            if backward:
                optimizer.zero_grad()

            body_pose = vposer.decode(
                pose_embedding, output_type='aa').view(
                    1, -1) if use_vposer else None

            if append_wrists:
                wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                         dtype=body_pose.dtype,
                                         device=body_pose.device)
                body_pose = torch.cat([body_pose, wrist_pose], dim=1)

            body_model_output = body_model(return_verts=return_verts,
                                           body_pose=body_pose,
                                           return_full_pose=return_full_pose)
            total_loss = loss(body_model_output, camera=camera,
                              gt_joints=gt_joints,
                              body_model_faces=faces_tensor,
                              joints_conf=joints_conf,
                              joint_weights=joint_weights,
                              pose_embedding=pose_embedding,
                              use_vposer=use_vposer,
                              gt_silhouettes=gt_silhouettes,
                              **kwargs)

            if backward:
                total_loss.backward(create_graph=create_graph)

            self.steps += 1
            if self.visualize and self.steps % self.summary_steps == 0:
                model_output = body_model(return_verts=True,
                                          body_pose=body_pose)
                vertices = model_output.vertices.detach().cpu().numpy()

                self.mv.update_mesh(vertices.squeeze(),
                                    body_model.faces)

            return total_loss

        return fitting_func



class SMPLifyLoss(nn.Module):

    def __init__(self, search_tree=None,
                 pen_distance=None, tri_filtering_module=None,
                 rho=100,
                 body_pose_prior=None,
                 shape_prior=None,
                 expr_prior=None,
                 angle_prior=None,
                 jaw_prior=None,
                #  use_joints_conf=True,
                 use_face=True, use_hands=True,
                 left_hand_prior=None, right_hand_prior=None,
                 interpenetration=True, dtype=torch.float32,
                 data_weight=1.0,
                 body_pose_weight=0.0,
                 shape_weight=0.0,
                 bending_prior_weight=0.0,
                 hand_prior_weight=0.0,
                 expr_prior_weight=0.0, jaw_prior_weight=0.0,
                 coll_loss_weight=0.0,
                 silhouette_weight=0.0,
                 cameras=None,
                 body_faces=None,
                 reduction='sum',
                 **kwargs):

        super(SMPLifyLoss, self).__init__()

        # self.use_joints_conf = use_joints_conf
        self.angle_prior = angle_prior

        self.robustifier = utils.GMoF(rho=rho)
        self.rho = rho

        self.body_pose_prior = body_pose_prior

        self.shape_prior = shape_prior

        self.interpenetration = interpenetration
        if self.interpenetration:
            self.search_tree = search_tree
            self.tri_filtering_module = tri_filtering_module
            self.pen_distance = pen_distance

        self.use_hands = use_hands
        if self.use_hands:
            self.left_hand_prior = left_hand_prior
            self.right_hand_prior = right_hand_prior

        self.use_face = use_face
        if self.use_face:
            self.expr_prior = expr_prior
            self.jaw_prior = jaw_prior

        self.register_buffer('data_weight',
                             torch.tensor(data_weight, dtype=dtype))
        self.register_buffer('body_pose_weight',
                             torch.tensor(body_pose_weight, dtype=dtype))
        self.register_buffer('shape_weight',
                             torch.tensor(shape_weight, dtype=dtype))
        self.register_buffer('bending_prior_weight',
                             torch.tensor(bending_prior_weight, dtype=dtype))
        if self.use_hands:
            self.register_buffer('hand_prior_weight',
                                 torch.tensor(hand_prior_weight, dtype=dtype))
        if self.use_face:
            self.register_buffer('expr_prior_weight',
                                 torch.tensor(expr_prior_weight, dtype=dtype))
            self.register_buffer('jaw_prior_weight',
                                 torch.tensor(jaw_prior_weight, dtype=dtype))
        if self.interpenetration:
            self.register_buffer('coll_loss_weight',
                                 torch.tensor(coll_loss_weight, dtype=dtype))

        self.use_silhouette = (cameras is not None and len(cameras) > 0 and body_faces is not None)
        if self.use_silhouette:
            self.glctx = dr.RasterizeCudaContext()
            self.cameras = cameras            # list of dicts {K, R, T, H, W} (tensors on device)
            # (F, 3) int32 — nvdiffrast requires int32 faces, no batch dim
            self.body_faces_sil = body_faces.view(-1, 3).int()
        self.register_buffer('silhouette_weight',
                             torch.tensor(silhouette_weight, dtype=dtype))

    def reset_loss_weights(self, loss_weight_dict):
        for key in loss_weight_dict:
            if hasattr(self, key):
                weight_tensor = getattr(self, key)
                if 'torch.Tensor' in str(type(loss_weight_dict[key])):
                    weight_tensor = loss_weight_dict[key].clone().detach()
                else:
                    weight_tensor = torch.tensor(loss_weight_dict[key],
                                                 dtype=weight_tensor.dtype,
                                                 device=weight_tensor.device)
                setattr(self, key, weight_tensor)

    def forward(self, body_model_output, gt_joints,
                body_model_faces, joint_weights,
                use_vposer=False, pose_embedding=None,
                gt_silhouettes=None,
                **kwargs):
        projected_joints = body_model_output.joints
        # Calculate the weights for each joints
        weights = joint_weights.unsqueeze(dim=-1)
        print(f'gt silhouette : {gt_silhouettes}')

        # Calculate the distance of the projected joints from
        # the ground truth 2D detections
        joint_diff = self.robustifier(gt_joints - projected_joints)
        joint_loss = (torch.sum(weights ** 2 * joint_diff) *
                      self.data_weight ** 2)

        # Calculate the loss from the Pose prior
        if use_vposer:
            pprior_loss = (pose_embedding.pow(2).sum() *
                           self.body_pose_weight ** 2)
        else:
            pprior_loss = torch.sum(self.body_pose_prior(
                body_model_output.body_pose,
                body_model_output.betas)) * self.body_pose_weight ** 2

        shape_loss = torch.sum(self.shape_prior(
            body_model_output.betas)) * self.shape_weight ** 2
        # Calculate the prior over the joint rotations. This a heuristic used
        # to prevent extreme rotation of the elbows and knees
        body_pose = body_model_output.full_pose[:, 3:66]
        angle_prior_loss = torch.sum(
            self.angle_prior(body_pose)) * self.bending_prior_weight

        # Apply the prior on the pose space of the hand
        left_hand_prior_loss, right_hand_prior_loss = 0.0, 0.0
        if self.use_hands and self.left_hand_prior is not None:
            left_hand_prior_loss = torch.sum(
                self.left_hand_prior(
                    body_model_output.left_hand_pose)) * \
                self.hand_prior_weight ** 2

        if self.use_hands and self.right_hand_prior is not None:
            right_hand_prior_loss = torch.sum(
                self.right_hand_prior(
                    body_model_output.right_hand_pose)) * \
                self.hand_prior_weight ** 2

        expression_loss = 0.0
        jaw_prior_loss = 0.0
        if self.use_face:
            expression_loss = torch.sum(self.expr_prior(
                body_model_output.expression)) * \
                self.expr_prior_weight ** 2

            if hasattr(self, 'jaw_prior'):
                jaw_prior_loss = torch.sum(
                    self.jaw_prior(
                        body_model_output.jaw_pose.mul(
                            self.jaw_prior_weight)))

        pen_loss = 0.0
        # Calculate the loss due to interpenetration
        if (self.interpenetration and self.coll_loss_weight.item() > 0):
            batch_size = projected_joints.shape[0]
            triangles = torch.index_select(
                body_model_output.vertices, 1,
                body_model_faces).view(batch_size, -1, 3, 3).contiguous()

            with torch.no_grad():
                collision_idxs = self.search_tree(triangles)

            # Remove unwanted collisions
            if self.tri_filtering_module is not None:
                collision_idxs = self.tri_filtering_module(collision_idxs)

            if collision_idxs.ge(0).sum().item() > 0:
                pen_loss = torch.sum(
                    self.coll_loss_weight *
                    self.pen_distance(triangles, collision_idxs))


        sil_loss = 0.0
        if (self.use_silhouette and gt_silhouettes is not None
                and self.silhouette_weight.item() > 0):
            verts = body_model_output.vertices.float()  # (1, V, 3)
            faces = self.body_faces_sil.to(verts.device)  # (F, 3) int32
            V = verts.shape[1]
            # Per-vertex alpha = 1 everywhere (used to get a solid silhouette)
            alpha_vtx = torch.ones(1, V, 1, device=verts.device, dtype=torch.float32)
            n_views = min(len(self.cameras), len(gt_silhouettes))
            for v_idx in range(n_views):
                gt = gt_silhouettes[v_idx]
                if gt is None:
                    continue
                cam = self.cameras[v_idx]
                H, W = cam['H'], cam['W']
                clip = _project_to_clip(verts, cam)          # (1, V, 4)
                rast, _ = dr.rasterize(self.glctx, clip, faces, resolution=[H, W])
                alpha, _ = dr.interpolate(alpha_vtx, rast, faces)  # (1, H, W, 1)
                rendered_sil = dr.antialias(alpha, rast, clip, faces)[..., 0]  # (1, H, W)
                gt_f = gt.to(rendered_sil.device).float()
                if gt_f.dim() == 2:
                    gt_f = gt_f.unsqueeze(0)
                intersection = (rendered_sil * gt_f).sum()
                union = (rendered_sil + gt_f - rendered_sil * gt_f).sum()
                sil_loss = sil_loss + (1.0 - intersection / (union + 1e-6))
            sil_loss = sil_loss * self.silhouette_weight ** 2

        total_loss = (joint_loss + pprior_loss + shape_loss +
                      angle_prior_loss + pen_loss +
                      jaw_prior_loss + expression_loss +
                      left_hand_prior_loss + right_hand_prior_loss +
                      sil_loss)
        return total_loss


