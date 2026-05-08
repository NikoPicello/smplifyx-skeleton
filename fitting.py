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
import os
import cv2

# import time

import numpy as np

import torch
import torch.nn as nn

# from mesh_viewer import MeshViewer
import utils

import nvdiffrast.torch as dr


def _reset_lbfgs_history(optimizer):
    """Clear the L-BFGS curvature history without removing the state entry.

    optimizer.state = {} crashes lbfgs_ls because it keys state on the first
    parameter tensor and accesses it unconditionally at the top of step().
    Resetting history fields inside the existing entry is safe.
    """
    for state in optimizer.state.values():
        state['n_iter'] = 0
        state['old_dirs'] = []
        state['old_stps'] = []
        state['H_diag'] = 1
        for k in ('d', 't', 'ro', 'prev_flat_grad', 'prev_loss', 'al'):
            state.pop(k, None)


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
                    stage_idx=0, **kwargs):
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
        stuck_count = 0
        # Stage 0 is joint-fitting dominant: react faster and allow more restarts.
        stuck_patience = 3 if stage_idx == 0 else 5
        max_restarts   = 5 if stage_idx == 0 else 3
        n_restarts     = 0

        for n in range(self.maxiters):
            loss = optimizer.step(closure)

            if torch.isnan(loss).sum() > 0:
                print('NaN loss value, stopping!')
                break

            if torch.isinf(loss).sum() > 0:
                print('Infinite loss value, stopping!')
                break

            # If the loss spiked (Hessian estimate is now corrupted), wipe the
            # L-BFGS history so the next step starts fresh from current params.
            if prev_loss is not None and loss.item() > 50 * prev_loss:
                print(f'  [optimizer reset] loss spike {prev_loss:.1f} → {loss.item():.1f}')
                _reset_lbfgs_history(optimizer)
                stuck_count = 0

            if n > 0 and prev_loss is not None and self.ftol > 0:
                loss_rel_change = utils.rel_change(prev_loss, loss.item())

                if loss_rel_change <= self.ftol:
                    # Stuck in a local minimum — perturb and keep going.
                    if n_restarts < max_restarts:
                        n_restarts += 1
                        stuck_count += 1
                        if stuck_count >= stuck_patience:
                            stuck_count = 0
                            print(f'  [perturb] stage={stage_idx} stuck at {loss.item():.2f}, restart {n_restarts}/{max_restarts}')
                            dev = pose_embedding.device if pose_embedding is not None \
                                  else body_model.global_orient.device
                            gen = torch.Generator(device=dev)
                            gen.manual_seed(n + n_restarts * 1000 + stage_idx * 100000)
                            # Scale up with each restart; stage 0 uses larger kicks
                            # because only joint positions matter there.
                            base = 0.003 if stage_idx == 0 else 0.002
                            noise_scale = base * n_restarts
                            with torch.no_grad():
                                if pose_embedding is not None:
                                    pose_embedding.data += torch.randn(
                                        pose_embedding.shape, generator=gen,
                                        device=dev, dtype=pose_embedding.dtype
                                    ) * noise_scale
                                    pose_embedding.data.clamp_(-5.0, 5.0)
                                # global_orient is the main driver of where joints
                                # land — perturbing it is the primary escape lever
                                # when the joint loss is stuck.
                                if body_model.global_orient is not None:
                                    orient_noise = torch.randn(
                                        body_model.global_orient.shape, generator=gen,
                                        device=dev, dtype=body_model.global_orient.dtype
                                    ) * (noise_scale * 0.4)
                                    body_model.global_orient.data += orient_noise
                                    go = body_model.global_orient.data
                                    norm = go.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                                    body_model.global_orient.data = torch.where(
                                        norm > torch.pi, go / norm * torch.pi, go)
                                if body_model.transl is not None:
                                    body_model.transl.data += torch.randn(
                                        body_model.transl.shape, generator=gen,
                                        device=dev, dtype=body_model.transl.dtype
                                    ) * (noise_scale * 0.05)
                            _reset_lbfgs_history(optimizer)
                    else:
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

            # Project parameters to safe ranges before every closure call
            # (including L-BFGS internal line-search trials).  This is
            # box-constrained L-BFGS: trial steps beyond the bounds are
            # projected back so the line search never evaluates an exploded state.
            with torch.no_grad():
                if pose_embedding is not None:
                    pose_embedding.data.clamp_(-5.0, 5.0)
                body_model.betas.data.clamp_(-10.0, 10.0)
                if body_model.transl is not None:
                    body_model.transl.data.clamp_(-50.0, 50.0)
                # global_orient is axis-angle; norm > π is a sign of explosion
                if body_model.global_orient is not None:
                    go = body_model.global_orient.data
                    norm = go.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                    body_model.global_orient.data = torch.where(
                        norm > torch.pi, go / norm * torch.pi, go)

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
                params_to_clip = [p for g in optimizer.param_groups for p in g['params']]
                torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=10e5)

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
                 face_weight=0.0,
                 lmk_faces_idx=None,
                 lmk_bary_coords=None,
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

        # Face landmark loss: 51 static landmarks (dlib 17-67) via barycentric
        # interpolation on the SMPLX mesh. face_weight is shared with the
        # face_joints_weights schedule so no new config key is needed.
        self.use_face_landmarks = (
            lmk_faces_idx is not None and lmk_bary_coords is not None
            and body_faces is not None)
        if self.use_face_landmarks:
            self.register_buffer('lmk_faces_idx',
                                 torch.tensor(lmk_faces_idx, dtype=torch.long))
            self.register_buffer('lmk_bary_coords',
                                 torch.tensor(lmk_bary_coords, dtype=dtype))
            self.body_faces_lmk = body_faces.view(-1, 3).long()
        self.register_buffer('face_weight',
                             torch.tensor(face_weight, dtype=dtype))

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
                gt_face_landmarks=None,
                **kwargs):
        projected_joints = body_model_output.joints
        # Calculate the weights for each joints
        weights = joint_weights.unsqueeze(dim=-1)

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


        face_lmk_loss = 0.0
        if (self.use_face_landmarks and gt_face_landmarks is not None
                and self.face_weight.item() > 0):
            verts = body_model_output.vertices[0]           # (V, 3)
            tri_verts = verts[self.body_faces_lmk[self.lmk_faces_idx]]  # (51, 3, 3)
            lmk_pos = (tri_verts * self.lmk_bary_coords.unsqueeze(-1)).sum(dim=1)  # (51, 3)
            # Per-landmark validity mask: NaN means the landmark was not visible
            valid = ~torch.isnan(gt_face_landmarks).any(dim=-1)  # (51,)
            gt_lmks = torch.nan_to_num(gt_face_landmarks, nan=0.0)
            # Plain L2 (no GMoF robustifier): the GMoF with rho=150 shrinks gradients
            # by ~22500x for meter-scale residuals, killing the orientation signal entirely.
            # 3D triangulated landmarks are reliable enough to not need heavy robustification.
            diff = (gt_lmks - lmk_pos).pow(2) * valid.unsqueeze(-1)
            face_lmk_loss = diff.sum() * self.face_weight ** 2

        sil_loss = 0.0
        if (self.use_silhouette and gt_silhouettes is not None
                and self.silhouette_weight.item() > 0):
            verts = body_model_output.vertices.float()  # (1, V, 3)
            faces = self.body_faces_sil.to(verts.device)
            V = verts.shape[1]
            alpha_vtx = torch.ones(1, V, 1, device=verts.device, dtype=torch.float32)
            for v_idx in range(min(len(self.cameras), len(gt_silhouettes))):
                gt = gt_silhouettes[v_idx]
                if gt is None:
                    continue
                cam = self.cameras[v_idx]
                H, W = cam['H'], cam['W']
                clip = _project_to_clip(verts, cam)
                rast, _ = dr.rasterize(self.glctx, clip, faces, resolution=[H, W])
                alpha, _ = dr.interpolate(alpha_vtx, rast, faces)
                rendered_sil = dr.antialias(alpha, rast, clip, faces)[..., 0].clamp(0.0, 1.0)
                rendered_sil = rendered_sil.flip(dims=[1])  # OpenGL row-0=bottom → cv2 row-0=top
                gt_f = gt.to(rendered_sil.device).float()
                if gt_f.dim() == 2:
                    gt_f = gt_f.unsqueeze(0)
                if gt_f.sum() < 1.0:
                    continue
                intersection = (rendered_sil * gt_f).sum()
                union = (rendered_sil + gt_f - rendered_sil * gt_f).sum()
                sil_loss = sil_loss + (1.0 - intersection / (union + 1e-6))
            sil_loss = sil_loss * self.silhouette_weight ** 2

        def _clamp_term(x, cap=1e5, name=''):
            v = x.item() if isinstance(x, torch.Tensor) else float(x)
            if v > cap:
                print(f"  [loss clamp] {name} {v:.2e} → {cap:.2e}")
                return x * (cap / v) if isinstance(x, torch.Tensor) else cap
            return x

        joint_loss            = _clamp_term(joint_loss,            1e8, 'joint')
        pprior_loss           = _clamp_term(pprior_loss,           1e5, 'pose')
        shape_loss            = _clamp_term(shape_loss,            1e5, 'shape')
        angle_prior_loss      = _clamp_term(angle_prior_loss,      1e5, 'angle')
        pen_loss              = _clamp_term(pen_loss,              1e5, 'pen')
        jaw_prior_loss        = _clamp_term(jaw_prior_loss,        1e5, 'jaw')
        expression_loss       = _clamp_term(expression_loss,       1e5, 'expr')
        left_hand_prior_loss  = _clamp_term(left_hand_prior_loss,  1e5, 'lhand')
        right_hand_prior_loss = _clamp_term(right_hand_prior_loss, 1e5, 'rhand')
        sil_loss              = _clamp_term(sil_loss,              1e5, 'sil')
        face_lmk_loss         = _clamp_term(face_lmk_loss,         1e5, 'face_lmk')

        total_loss = (joint_loss + pprior_loss + shape_loss +
                      angle_prior_loss + pen_loss +
                      jaw_prior_loss + expression_loss +
                      left_hand_prior_loss + right_hand_prior_loss +
                      sil_loss + face_lmk_loss)

        def _v(x):
            return x.item() if isinstance(x, torch.Tensor) else float(x)
        parts = {
            'joint':    _v(joint_loss),
            'pose':     _v(pprior_loss),
            'shape':    _v(shape_loss),
            'angle':    _v(angle_prior_loss),
            'pen':      _v(pen_loss),
            'jaw':      _v(jaw_prior_loss),
            'expr':     _v(expression_loss),
            'lhand':    _v(left_hand_prior_loss),
            'rhand':    _v(right_hand_prior_loss),
            'sil':      _v(sil_loss),
            'face_lmk': _v(face_lmk_loss),
        }
        parts_sum = sum(parts.values())
        parts_str = '  '.join(f'{k}={v:>8.2f}' for k, v in parts.items())
        print(f"  {parts_str}  sum={parts_sum:>8.2f}  total={_v(total_loss):>8.2f}")
        return total_loss

    # ------------------------------------------------------------------
    # Per-stage visualisation (call from fit_single_frame after each stage)
    # ------------------------------------------------------------------
    def visualize_stage(self, verts, gt_silhouettes, stage_idx, frame_idx,
                        out_dir='./tmp/sil_vis', cam_names=None):
        """
        Render the current mesh silhouette for every camera and save overlay
        PNGs to out_dir/f{frame_idx:04d}/stage{stage_idx:02d}_v{i:02d}.png.

        Colour key in the saved image (BGR):
            Green  : GT mask only   (model missing)
            Red    : rendered only  (model too big / wrong place)
            Yellow : both           (correct overlap)
            Black  : neither
        """
        if not self.use_silhouette or gt_silhouettes is None:
            return
        save_dir = os.path.join(out_dir, f'f{frame_idx:04d}')
        os.makedirs(save_dir, exist_ok=True)

        with torch.no_grad():
            verts_f = verts.float()
            faces   = self.body_faces_sil.to(verts_f.device)
            V       = verts_f.shape[1]
            alpha_vtx = torch.ones(1, V, 1, device=verts_f.device, dtype=torch.float32)

            for v_idx in range(min(len(self.cameras), len(gt_silhouettes))):
                gt = gt_silhouettes[v_idx]
                if gt is None:
                    continue
                cam  = self.cameras[v_idx]
                H, W = cam['H'], cam['W']

                clip = _project_to_clip(verts_f, cam)
                rast, _  = dr.rasterize(self.glctx, clip, faces, resolution=[H, W])
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


