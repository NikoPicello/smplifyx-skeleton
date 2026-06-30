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

# import time

import numpy as np

import torch
import torch.nn as nn

# from mesh_viewer import MeshViewer
import utils


_LOWER_BODY_POSE_DOFS = [
    0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14,
    18, 19, 20, 21, 22, 23, 27, 28, 29, 30, 31, 32,
]
_UPPER_BODY_POSE_DOFS = [d for d in range(63) if d not in set(_LOWER_BODY_POSE_DOFS)]


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


def _project_to_pixels(points, cam, z_min=0.05, norm_clamp=20.0):
    """
    Project world-space points to distorted pixel coords, matching cv.projectPoints.

    Same OpenCV radial+tangential model as utils._project_to_clip, but returns pixel
    (u, v) instead of clip space. Differentiable in `points` — used by the GB
    keypoint-reprojection stage.

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
        self.stol = 1e-6

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
                    stage_idx=0, frame_idx=0, **kwargs):
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
        #   stuck_patience = consecutive no-progress iters before a perturbation kick
        #   max_restarts   = total kicks allowed before we give up and stop
        # NOTE: max_restarts must be >= stuck_patience or the kick never fires.
        stuck_patience = 2 if stage_idx == 0 else 3
        # The perturbation kicks the upper-body DOFs (incl. arms) whenever the *total*
        # loss plateaus. In refinement stages that throws already-placed arms off their
        # keypoints, and keep-best (judged on total loss) won't restore a small term
        # like a lone wrist — so confine the kick to the cold-start global fit (stage 0).
        max_restarts   = 6 if stage_idx == 0 else 0
        n_restarts     = 0
        # Keep-best: a perturbation can land us in a *worse* basin, so track the
        # lowest-loss state seen and restore it at the end. `snapshot` is the exact
        # param set optimizer.step evaluates its returned loss at (L-BFGS reports
        # the pre-step loss), so (loss, snapshot) is a consistent pair.
        best_loss  = float('inf')
        best_state = None

        for n in range(self.maxiters):
            snapshot = {p: p.data.clone() for p in params}
            loss = optimizer.step(closure)

            if torch.isnan(loss).sum() > 0:
                print('NaN loss value — restoring last good params and stopping')
                with torch.no_grad():
                    for p in params:
                        p.data.copy_(snapshot[p])
                break

            if torch.isinf(loss).sum() > 0:
                print('Infinite loss value, stopping!')
                with torch.no_grad():
                    for p in params:
                        p.data.copy_(snapshot[p])
                break

            # Keep-best: snapshot is the param set this loss was evaluated at, so
            # remember it whenever the loss improves (restored after the loop).
            if loss.item() < best_loss:
                best_loss  = loss.item()
                best_state = snapshot

            # If the loss spiked (Hessian estimate is now corrupted), wipe the
            # L-BFGS history so the next step starts fresh from current params.
            if prev_loss is not None and loss.item() > 50 * prev_loss:
                print(f'  [optimizer reset] loss spike {prev_loss:.1f} → {loss.item():.1f}')
                _reset_lbfgs_history(optimizer)
                stuck_count = 0

            if n > 0 and prev_loss is not None and self.stol > 0:
                loss_rel_change = utils.rel_change(prev_loss, loss.item())

                if loss_rel_change <= self.stol:
                    # No measurable progress this iter. Count it; once we've been
                    # stuck for `stuck_patience` iters, kick the pose to escape the
                    # local minimum — or, if out of kicks, stop.
                    stuck_count += 1
                    if stuck_count >= stuck_patience:
                        if n_restarts >= max_restarts:
                            break
                        n_restarts += 1
                        stuck_count = 0
                        print(f'  [perturb] stage={stage_idx} stuck at {loss.item():.2f}, '
                              f'restart {n_restarts}/{max_restarts}')
                        dev = (pose_embedding.device if pose_embedding is not None
                               else body_model.global_orient.device)
                        gen = torch.Generator(device=dev)
                        gen.manual_seed(n + n_restarts * 1000 + stage_idx * 100000)
                        # Scale the kick up with each restart; stage 0 kicks harder
                        # because only joint positions matter there.
                        base = 0.05 if stage_idx == 0 else 0.03
                        noise_scale = base * n_restarts
                        with torch.no_grad():
                            if use_vposer and pose_embedding is not None:
                                # VPoser: perturb the latent pose code.
                                pose_embedding.data += torch.randn(
                                    pose_embedding.shape, generator=gen,
                                    device=dev, dtype=pose_embedding.dtype) * noise_scale
                                pose_embedding.data.clamp_(-5.0, 5.0)
                            elif body_model.body_pose is not None:
                                # Direct param mode: perturb the pose we actually
                                # solve for, but only the upper-body DOFs (the lower
                                # body has a small data weight on noisy GT and is
                                # overridden at frame end, so kicking it just adds
                                # noise the legs would then chase).
                                bp_noise = torch.randn(
                                    body_model.body_pose.shape, generator=gen,
                                    device=dev, dtype=body_model.body_pose.dtype) * noise_scale
                                mask = torch.zeros_like(body_model.body_pose)
                                mask[..., _UPPER_BODY_POSE_DOFS] = 1.0
                                body_model.body_pose.data += bp_noise * mask
                                # Keep each joint axis-angle within ±π (the closure
                                # re-clamps too, but stay safe before the next step).
                                bp = body_model.body_pose.data.view(-1, 3)
                                bn = bp.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                                body_model.body_pose.data = torch.where(
                                    bn > torch.pi, bp / bn * torch.pi, bp
                                ).view(body_model.body_pose.data.shape)
                            # global_orient kick only on frame 0 (cold start). On
                            # later frames the carried-over orientation is already
                            # close; perturbing risks flipping rotation basins.
                            if frame_idx == 0 and body_model.global_orient is not None:
                                orient_noise = torch.randn(
                                    body_model.global_orient.shape, generator=gen,
                                    device=dev, dtype=body_model.global_orient.dtype
                                ) * (noise_scale * 0.4)
                                body_model.global_orient.data += orient_noise
                                go = body_model.global_orient.data
                                norm = go.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                                body_model.global_orient.data = torch.where(
                                    norm > torch.pi, go / norm * torch.pi, go)
                            # transl only when it's actually being optimized (frame 0);
                            # on later frames it's frozen, so a kick would just shift
                            # the body without being corrected.
                            if body_model.transl is not None and body_model.transl.requires_grad:
                                body_model.transl.data += torch.randn(
                                    body_model.transl.shape, generator=gen,
                                    device=dev, dtype=body_model.transl.dtype
                                ) * (noise_scale * 0.05)
                        _reset_lbfgs_history(optimizer)

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

        # The loop exits with the params in their final (post-step) state, which
        # was never scored on its own. Measure it once (no backward) and keep it
        # only if it's finite and beats the best snapshot; otherwise restore the
        # best state — so a perturbation can never leave us worse than before.
        final_loss = float(closure(backward=False))
        if best_state is not None and (not np.isfinite(final_loss) or best_loss < final_loss):
            with torch.no_grad():
                for p in params:
                    p.data.copy_(best_state[p])
            return best_loss
        return final_loss

    def create_fitting_closure(self,
                               optimizer, body_model, camera=None,
                               gt_joints=None, loss=None,
                               joints_conf=None,
                               joint_weights=None,
                               return_verts=True, return_full_pose=False,
                               use_vposer=False, vposer=None,
                               pose_embedding=None,
                               create_graph=False,
                               smpler_body_pose=None,
                               prev_body_pose=None,
                               prev_global_orient=None,
                               prev_translation=None,
                               **kwargs):
        faces_tensor = body_model.faces_tensor.view(-1)
        append_wrists = self.model_type == 'smpl' and use_vposer

        def fitting_func(backward=True):
            if backward:
                optimizer.zero_grad()

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
                # Without VPoser, body_pose is 63 raw axis-angles; clamp each joint to ≤ π
                if not use_vposer and hasattr(body_model, 'body_pose') and body_model.body_pose is not None:
                    bp = body_model.body_pose.data.view(-1, 3)
                    bp_norms = bp.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                    body_model.body_pose.data = torch.where(
                        bp_norms > torch.pi, bp / bp_norms * torch.pi, bp
                    ).view(body_model.body_pose.data.shape)

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
                                           return_full_pose=return_full_pose,
                                           create_global_orient=True,
                                           create_transl=True)
            total_loss = loss(body_model_output, camera=camera,
                              gt_joints=gt_joints,
                              body_model_faces=faces_tensor,
                              joints_conf=joints_conf,
                              joint_weights=joint_weights,
                              pose_embedding=pose_embedding,
                              use_vposer=use_vposer,
                              prev_body_pose=prev_body_pose,
                              prev_global_orient=prev_global_orient,
                              prev_translation=prev_translation,
                              smpler_body_pose=smpler_body_pose,
                              **kwargs)

            if backward:
                total_loss.backward(create_graph=create_graph)
                params_to_clip = [p for g in optimizer.param_groups for p in g['params']]
                torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=10.0)

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
                 pen_distance=None,
                 tri_filtering_module=None,
                 rho=100,
                 body_pose_prior=None,
                 shape_prior=None,
                 expr_prior=None,
                 angle_prior=None,
                 jaw_prior=None,
                 use_face=True,
                 use_hands=True,
                 left_hand_prior=None,
                 right_hand_prior=None,
                 interpenetration=True,
                 data_weight=1.0,
                 body_pose_weight=0.0,
                 shape_weight=0.0,
                 translation_weight=0.0,
                 global_orient_weight=0.0,
                 bending_prior_weight=0.0,
                 hand_prior_weight=0.0,
                 expr_prior_weight=0.0,
                 jaw_prior_weight=0.0,
                 coll_loss_weight=0.0,
                 face_weight=0.0,
                 lmk_faces_idx=None,
                 lmk_bary_coords=None,
                 body_faces=None,
                 dtype=torch.float32,
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
        self.register_buffer('translation_weight',
                             torch.tensor(translation_weight, dtype=dtype))
        self.register_buffer('global_orient_weight',
                             torch.tensor(global_orient_weight, dtype=dtype))
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

        self.register_buffer('face_weight',
                             torch.tensor(face_weight, dtype=dtype))
        self.register_buffer('temporal_weight',
                             torch.tensor(0.0, dtype=dtype))
        self.register_buffer('smpler_pose_weight',
                             torch.tensor(0.0, dtype=dtype))

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

        # ── SMPLer-X pose anchor (declarative) ───────────────────────────────
        # ONE place that says which body_pose joints are pinned to the per-frame SMPLer-X
        # init, and how strongly (relative weight, scaled by the per-stage smpler_pose_weight).
        # Anchor ONLY the under-observed, task-specific seated DOFs: the legs (occluded /
        # single-view → never triangulated, and the GMM prior won't know they're seated) and
        # the spine (only the hip→shoulder endpoints constrain it). Plausibility of every
        # OTHER joint (collars, shoulders, elbows, …) is the GMM body-pose prior's job — not
        # an anchor's. To change what's anchored, edit this dict (0 / absent = free).
        _SMPLX_BODY_JOINTS = [
            'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee', 'spine2',
            'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot', 'neck',
            'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist']
        _ANCHOR_JOINT_W = {
            'left_hip':   1.0, 'right_hip':   1.0,
            'left_knee':  1.0, 'right_knee':  1.0,
            'left_ankle': 1.0, 'right_ankle': 1.0,
            'left_foot':  1.0, 'right_foot':  1.0,
            'spine1':     1.0, 'spine2':      1.0, 'spine3': 1.0,
            # Keep the neck anchored to per-frame SMPLer-X for NATURALNESS + an absolute
            # reference. Without it the neck is under-constrained — unnatural at frame 0 (no
            # previous frame for the temporal prior yet) and the optimiser diverges for a few
            # frames. The ×4 temporal boost (fit_single_frame _NECK_COLLAR_TEMPORAL_BOOST)
            # dominates 4:1 and removes the jitter; this per-frame pull just keeps it plausible.
            # Lower toward 0.3 if it still jitters, but do NOT set 0 (that caused the explosions).
            'neck':       1.0,
        }
        _anchor_w = torch.zeros(1, 63, dtype=dtype)
        for _ji, _jn in enumerate(_SMPLX_BODY_JOINTS):
            _anchor_w[0, 3 * _ji: 3 * _ji + 3] = _ANCHOR_JOINT_W.get(_jn, 0.0)
        self.register_buffer('smpler_anchor_dof_w', _anchor_w)

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
                gt_face_landmarks=None,
                prev_body_pose=None,
                prev_global_orient=None,
                prev_translation=None,
                smpler_body_pose=None,
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


        def _clamp_term(x, cap=1e5, name=''):
            v = x.item() if isinstance(x, torch.Tensor) else float(x)
            if v > cap:
                print(f"  [loss clamp] {name} {v:.2e} → {cap:.2e}")
                return x * (cap / v) if isinstance(x, torch.Tensor) else cap
            return x

        temporal_loss = 0.0
        if prev_body_pose is not None:
            _td = (body_model_output.body_pose - prev_body_pose).pow(2)  # (1, 63)
            temporal_dof_w = kwargs.get('temporal_dof_weights', None)
            if temporal_dof_w is not None:
                _td = _td * temporal_dof_w
            temporal_loss = _td.sum() * self.temporal_weight ** 2


        global_orient_loss = 0.0
        if prev_global_orient is not None and self.global_orient_weight.item() > 0:
            _go = body_model_output.global_orient
            # Resolve the axis-angle pi-wrap so the anchor measures the true rotation
            # difference, not a spurious ~2*pi vector jump (see utils.aa_nearest).
            with torch.no_grad():
                _go_ref = utils.aa_nearest(prev_global_orient, _go)
            global_orient_loss = ((_go - _go_ref).pow(2).sum()
                * (self.global_orient_weight.item() ** 2))

        translation_loss = 0.0
        if prev_translation is not None and self.translation_weight.item() > 0:
            translation_loss = ((body_model_output.transl - prev_translation).pow(2).sum()
                * (self.translation_weight.item() ** 2))


        smpler_pose_loss = 0.0
        if smpler_body_pose is not None and self.smpler_pose_weight.item() > 0:
            # Single declarative anchor (per-joint weights in smpler_anchor_dof_w, built in
            # __init__): pin only the under-observed seated DOFs (legs + spine) to the
            # per-frame SMPLer-X init. Plausibility of everything else is the GMM prior's job.
            _d = (body_model_output.body_pose - smpler_body_pose).pow(2)
            smpler_pose_loss = (_d * self.smpler_anchor_dof_w).sum() * self.smpler_pose_weight ** 2

        face_lmk_loss = 0.0
        if (self.use_face_landmarks and gt_face_landmarks is not None
                and self.face_weight.item() > 0):
            verts = body_model_output.vertices[0]           # (V, 3)
            tri_verts = verts[self.body_faces_lmk[self.lmk_faces_idx]]  # (51, 3, 3)
            lmk_pos = (tri_verts * self.lmk_bary_coords.unsqueeze(-1)).sum(dim=1)  # (51, 3)
            valid = ~torch.isnan(gt_face_landmarks).any(dim=-1)  # (51,)
            gt_lmks = torch.nan_to_num(gt_face_landmarks, nan=0.0)
            diff = (gt_lmks - lmk_pos).pow(2) * valid.unsqueeze(-1)
            face_lmk_loss = diff.sum() * self.face_weight ** 2


        joint_loss            = _clamp_term(joint_loss,            1e8, 'joint')
        pprior_loss           = _clamp_term(pprior_loss,           1e5, 'pose')
        shape_loss            = _clamp_term(shape_loss,            1e5, 'shape')
        angle_prior_loss      = _clamp_term(angle_prior_loss,      1e5, 'angle')
        pen_loss              = _clamp_term(pen_loss,              1e5, 'pen')
        jaw_prior_loss        = _clamp_term(jaw_prior_loss,        1e5, 'jaw')
        expression_loss       = _clamp_term(expression_loss,       1e5, 'expr')
        left_hand_prior_loss  = _clamp_term(left_hand_prior_loss,  1e5, 'lhand')
        right_hand_prior_loss = _clamp_term(right_hand_prior_loss, 1e5, 'rhand')
        face_lmk_loss         = _clamp_term(face_lmk_loss,         1e5, 'face_lmk')
        temporal_loss         = _clamp_term(temporal_loss,         1e5, 'temporal')
        global_orient_loss    = _clamp_term(global_orient_loss,    1e5, 'go')
        translation_loss      = _clamp_term(translation_loss,      1e5, 'tr')
        smpler_pose_loss      = _clamp_term(smpler_pose_loss,      1e5, 'smpler')

        total_loss = (joint_loss + pprior_loss + shape_loss +
                      angle_prior_loss + pen_loss +
                      jaw_prior_loss + expression_loss +
                      left_hand_prior_loss + right_hand_prior_loss +
                      face_lmk_loss + temporal_loss +
                      global_orient_loss + translation_loss +
                      smpler_pose_loss)
        def _v(x):
            return x.item() if isinstance(x, torch.Tensor) else float(x)
        parts = {
            'joint'  : _v(joint_loss),
            'pose'   : _v(pprior_loss),
            'shape'  : _v(shape_loss),
            'angle'  : _v(angle_prior_loss),
            'pen'    : _v(pen_loss),
            'jaw'    : _v(jaw_prior_loss),
            'expr'   : _v(expression_loss),
            'lhand'  : _v(left_hand_prior_loss),
            'rhand'  : _v(right_hand_prior_loss),
            'face'   : _v(face_lmk_loss),
            'temp'   : _v(temporal_loss),
            'smpler' : _v(smpler_pose_loss),
            'tr'     : _v(translation_loss),
            'go'     : _v(global_orient_loss),
        }
        parts_str = '  '.join(f'{k}={v:>6.3f}' for k, v in parts.items())
        print(f"  {parts_str}  tot={_v(total_loss):>6.3f}")
        return total_loss


