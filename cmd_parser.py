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
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

# This parser was rewritten to match the CURRENT fitting pipeline (temporal_window.py's
# windowed Stage A/B/C, driven from main.py) instead of the original per-frame SMPLify-X
# optimizer (fitting.py's run_fitting/fit_single_frame, no longer called from main.py).
# Every arg below is either read directly by main.py/data_parser.py/prior.py/smplx's own
# model constructor, or overrides one of temporal_window.py's / cvars.py's tuning
# constants via their configure(args) hooks -- see temporal_window.py's own "declarative
# config, no cross-module arg plumbing" note. Nothing here is consumed by fitting.py.

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

# import sys
import os

import configargparse


def parse_config(argv=None):
    arg_formatter = configargparse.ArgumentDefaultsHelpFormatter

    cfg_parser = configargparse.YAMLConfigFileParser
    description = 'PyTorch implementation of SMPLifyX with temporal windowing'
    parser = configargparse.ArgParser(formatter_class=arg_formatter,
                                      config_file_parser_class=cfg_parser,
                                      description=description,
                                      prog='SMPLifyX')

    # ── core I/O / run setup ─────────────────────────────────────────────────
    parser.add_argument('-c', '--config',
                        required=True, is_config_file=True,
                        help='config file path')
    parser.add_argument('--data_folder',
                        default=os.getcwd(),
                        help='The directory that contains the data.')
    parser.add_argument('--output_folder',
                        default='output',
                        type=str,
                        help='The folder where the output is stored')
    parser.add_argument('--max_persons', type=int, default=2,
                        help='The maximum number of persons to process')
    parser.add_argument('--save_mesh',
                        required=False,
                        type=lambda arg: arg.lower() == 'true',
                        default=True,
                        help='Whether to export a mesh (.obj) per frame in addition '
                             'to body_smplx.json')
    parser.add_argument('--use_cuda',
                        type=lambda arg: arg.lower() == 'true',
                        default=True,
                        help='Use CUDA for the computations')
    parser.add_argument('--gpu_id', default=0, type=int,
                        help='The ID of the GPU to use')
    parser.add_argument('--dataset', default='custom', type=str,
                        help='The name of the dataset loader to use (custom / ADT)')
    parser.add_argument('--float_dtype', type=str, default='float32',
                        help='The type of floats used (float32 / float64)')

    # ── SMPL-X model config ──────────────────────────────────────────────────
    parser.add_argument('--model_folder',
                        default='models',
                        type=str,
                        help='The directory where the models are stored.')
    parser.add_argument('--model_type', default='smplx', type=str,
                        choices=['smpl', 'smplh', 'smplx'],
                        help='The type of the body model to fit.')
    parser.add_argument('--gender', type=str,
                        default='neutral',
                        choices=['neutral', 'male', 'female'],
                        help='Use gender neutral or gender specific SMPL model')
    parser.add_argument('--num_betas', default=10, type=int,
                        help='The number of shape (betas) components for the body model.')
    parser.add_argument('--use_hands', default=True,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='Use the hand keypoints (Stage B hand refinement)')
    parser.add_argument('--use_face', default=True,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='Use the facial landmarks (Stage B head refinement)')
    parser.add_argument('--use_face_contour', default=True,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='Use the dynamic contours of the face')
    parser.add_argument('--use_pca', default=False,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='Use the low dimensional PCA space for the hands '
                             '(False = full 45-DoF axis-angle hand pose)')
    parser.add_argument('--num_pca_comps', default=45, type=int,
                        help='The number of PCA components for the hand (only used '
                             'when use_pca is True)')
    parser.add_argument('--flat_hand_mean', default=True,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Use the flat hand as the mean pose')
    parser.add_argument('--use_vposer', default=False,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Use the VAE pose embedding instead of direct body_pose '
                             '(currently unused end-to-end -- kept for compatibility)')

    # ── priors ────────────────────────────────────────────────────────────────
    parser.add_argument('--prior_folder', type=str, default='priors',
                        help='The folder where the pose priors (e.g. gmm_08.pkl) are stored')
    parser.add_argument('--body_prior_type', default='gmm', type=str,
                        choices=['gmm', 'l2', 'none'],
                        help='Prior regularizing the whole-body pose')
    parser.add_argument('--num_gaussians',
                        default=8,
                        type=int,
                        help='The number of Gaussians for the GMM body pose prior')
    parser.add_argument('--left_hand_prior_type', default='l2', type=str,
                        choices=['gmm', 'l2', 'none'],
                        help='Prior regularizing the left hand pose')
    parser.add_argument('--right_hand_prior_type', default='l2', type=str,
                        choices=['gmm', 'l2', 'none'],
                        help='Prior regularizing the right hand pose')
    parser.add_argument('--jaw_prior_type', default='l2', type=str,
                        choices=['l2', 'none'],
                        help='Prior regularizing the jaw pose')
    parser.add_argument('--expr_prior_type', default='l2', type=str,
                        choices=['l2', 'none'],
                        help='Prior regularizing the facial expression')
    parser.add_argument('--shape_prior_type', default='l2', type=str,
                        choices=['l2', 'none'],
                        help='Prior regularizing the shape (betas)')

    # ── joint / keypoint handling ────────────────────────────────────────────
    parser.add_argument('--joints_to_ign', default=[], type=int,
                        nargs='*',
                        help='Indices of mapped joints to always ignore (e.g. knees/ankles)')
    parser.add_argument('--joint_conf_threshold', type=float, default=0.0,
                        help='Per-frame keypoint confidence gate: joints with conf '
                             'below this are dropped for that frame (0.0 = use any conf>0)')
    parser.add_argument('--hip_weight', type=float, default=1.0,
                        help='Multiplier on the hip keypoints (11,12) in the Stage A data term')

    # ── interpenetration (currently INERT: built by main.py but not wired into the
    #    windowed Stage A/B/C loss -- see the "why isn't it in the loss" discussion.
    #    Kept configurable in case it gets re-integrated; leave False otherwise, it
    #    only costs setup time for nothing right now) ────────────────────────────
    parser.add_argument('--interpenetration',
                        default=False,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='Build the self-intersection BVH (currently not consumed '
                             'by the windowed fit -- see module note)')
    parser.add_argument('--df_cone_height', default=0.0001, type=float,
                        help='Height of the cone used for the penetration distance field')
    parser.add_argument('--max_collisions', default=1280, type=int,
                        help='The maximum number of bounding box collisions')
    parser.add_argument('--point2plane', default=True,
                        type=lambda arg: arg.lower() in ['true', '1'],
                        help='Use point to plane distance for the penetration term')
    parser.add_argument('--penalize_outside', default=True,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='Penalize the exterior term too, not just interior')
    parser.add_argument('--part_segm_fn', default='', type=str,
                        help='The file with the part segmentation for the faces of the model')
    parser.add_argument('--ign_part_pairs', default=None,
                        nargs='*', type=str,
                        help='Pairs of parts whose collisions will be ignored')

    # ═══════════════════════════════════════════════════════════════════════════
    # Everything below overrides temporal_window.py's / cvars.py's module constants
    # via their own configure(args) hook (called once at startup, see main.py) --
    # NOT threaded through any function signature. Defaults below match each
    # module's current hardcoded value, so an unmodified cfg reproduces today's
    # behaviour exactly.
    # ═══════════════════════════════════════════════════════════════════════════

    # ── keypoint preprocessing (cvars.py) ───────────────────────────────────────
    parser.add_argument('--kp_conf_power', type=float, default=None,
                        help='Data weight = joint_weight * valid * conf**this. Higher '
                             'suppresses flickering low-confidence detections harder.')
    parser.add_argument('--kp_fill_max_gap', type=int, default=None,
                        help='Linearly interpolate a keypoint across missing runs up to '
                             'this many frames (0 disables gap-fill)')
    parser.add_argument('--kp_fill_conf', type=float, default=None,
                        help='Confidence stamped on gap-filled keypoints')
    parser.add_argument('--seated_hip_x', type=float, default=None,
                        help='Seated-template hip flexion (body_pose DOFs 0/3), used as '
                             'the SEATED_LEGS fallback when no per-camera leg export exists')
    parser.add_argument('--seated_knee_x', type=float, default=None,
                        help='Seated-template knee flexion (body_pose DOFs 9/12)')

    # ── window geometry ──────────────────────────────────────────────────────
    parser.add_argument('--win_size', type=int, default=None,
                        help='Frames optimised jointly per window (== batched model '
                             'batch_size). Larger amortises LBFGS/line-search overhead '
                             'over more frames at the cost of GPU memory.')
    parser.add_argument('--win_overlap', type=int, default=None,
                        help='Boundary frames pinned to the previous window\'s committed '
                             'solve (>=2 for a C1 seam)')

    # ── temporal smoothness (Stage A) ───────────────────────────────────────────
    parser.add_argument('--lambda_vel_bp', type=float, default=None,
                        help='Stage A body_pose velocity penalty')
    parser.add_argument('--lambda_acc_bp', type=float, default=None,
                        help='Stage A body_pose acceleration (jerk) penalty')
    parser.add_argument('--lambda_vel_tr', type=float, default=None,
                        help='Stage A translation velocity penalty')
    parser.add_argument('--lambda_acc_tr', type=float, default=None,
                        help='Stage A translation acceleration penalty')
    parser.add_argument('--lambda_vel_go', type=float, default=None,
                        help='Stage A global_orient (6D) velocity penalty')
    parser.add_argument('--lambda_acc_go', type=float, default=None,
                        help='Stage A global_orient (6D) acceleration penalty')

    # ── anchors (Stage A) ────────────────────────────────────────────────────
    parser.add_argument('--lambda_root', type=float, default=None,
                        help='Quadratic spring holding (global_orient, transl) at the '
                             'static-root solve. Free within ~a couple cm, stiff beyond '
                             '~5cm. Sweep down for a more data-driven pelvis (lets it '
                             'follow larger genuine motion, e.g. leaning), up toward '
                             'frozen/anti-jitter behaviour.')
    parser.add_argument('--lambda_bnd', type=float, default=None,
                        help='Pins overlap frames to the previous window\'s committed solve')
    parser.add_argument('--lambda_go_anchor', type=float, default=None,
                        help='Observability-gated pull of global_orient toward the '
                             'window\'s well-observed consensus orientation')
    parser.add_argument('--lambda_bp_still', type=float, default=None,
                        help='Absolute "keep in place" anchor pulling body_pose toward '
                             'a reference pose (window mean, or mamma\'s pose if given)')
    parser.add_argument('--lambda_cerv', type=float, default=None,
                        help='Neck<->head bend-sharing penalty')

    # ── data term / priors (Stage A) ────────────────────────────────────────────
    parser.add_argument('--data_rho', type=float, default=None,
                        help='GMoF robustifier scale (metres) for the Stage A 3D data term')
    parser.add_argument('--lambda_pose', type=float, default=None,
                        help='GMM body-pose prior weight in Stage A')
    parser.add_argument('--lambda_angle', type=float, default=None,
                        help='Knee/elbow hyper-extension prior weight in Stage A')

    # ── coarse-to-fine stage schedule (Stage A) ─────────────────────────────────
    # Three parallel lists, one entry per stage (same convention as the old per-frame
    # config's weight schedules) -- rebuilt into temporal_window.STAGE_SCHEDULE by
    # its configure(args) hook. Must all be the same length.
    parser.add_argument('--stage_data_weights', type=float, nargs='*',
                        default=None,
                        help='Per-stage data-term weight (coarse -> fine)')
    parser.add_argument('--stage_temporal_weights', type=float, nargs='*',
                        default=None,
                        help='Per-stage temporal (vel/acc) weight multiplier (coarse -> fine)')
    parser.add_argument('--stage_lbfgs_steps', type=int, nargs='*',
                        default=None,
                        help='Per-stage number of LBFGS opt.step() calls')

    # ── Stage 0: betas refinement (bone-length fit) ─────────────────────────────
    parser.add_argument('--betas_steps', type=int, default=None,
                        help='LBFGS steps for the shared betas bone-length refinement')
    parser.add_argument('--betas_nsat', type=int, default=None,
                        help='Samples at which a bone-length segment reaches full fit weight')
    parser.add_argument('--betas_len_w', type=float, default=None,
                        help='Bone-length data term weight')
    parser.add_argument('--betas_rho0', type=float, default=None,
                        help='GMoF scale (m) on bone-length residuals, coarse end of the '
                             'BETAS_STEPS anneal (wide enough that a real-but-large segment '
                             'discrepancy can still pull the fit instead of saturating)')
    parser.add_argument('--betas_rho1', type=float, default=None,
                        help='GMoF scale (m) on bone-length residuals, fine end of the '
                             'BETAS_STEPS anneal (also the threshold used for the "likely '
                             'corrupt" saturation warning)')
    parser.add_argument('--betas_anchor_w', type=float, default=None,
                        help='Anchor toward the SMPLer-X init along the bone-length directions')
    parser.add_argument('--betas_null_w', type=float, default=None,
                        help='Anchor toward the SMPLer-X init orthogonal to the bone-length '
                             'directions (girth etc., no data to constrain it)')
    parser.add_argument('--betas_conf_thr', type=float, default=None,
                        help='Minimum confidence for a bone-length segment endpoint to '
                             'count as observed')

    # ── static root solve ────────────────────────────────────────────────────
    parser.add_argument('--solve_static_root', type=lambda x: x.lower() in ['true', '1'],
                        default=None,
                        help='Solve one (global_orient, transl) for the whole sequence '
                             'from 3D + multi-view 2D evidence before Stage A')
    parser.add_argument('--freeze_root', type=lambda x: x.lower() in ['true', '1'],
                        default=None,
                        help='Hold the static root solve FIXED for every frame (True), or '
                             'let Stage A refine it per frame around lambda_root\'s anchor '
                             '(False). Free lets the pelvis answer to genuine motion '
                             '(e.g. leaning) instead of forcing it into the spine, at the '
                             'cost of re-opening root wander/jitter -- see lambda_root, '
                             'the root velocity/acceleration lambdas, and lambda_bp_still.')
    parser.add_argument('--root_stride', type=int, default=None,
                        help='Fit every k-th frame in the static root solve '
                             '(auto-lowered on short clips)')
    parser.add_argument('--root_frame_start', type=int, default=None,
                        help='First frame (of the full sequence) eligible for the static root '
                             'solve -- raise this to skip an unstable/settling-in lead-in')
    parser.add_argument('--root_frame_count', type=int, default=None,
                        help='If set, only this many frames starting at root_frame_start are '
                             'eligible for the static root solve, instead of the whole sequence '
                             '(stride auto-drops to 1 for a pool this small)')
    parser.add_argument('--root_data_w', type=float, default=None,
                        help='3D trunk data weight in the static root solve')
    parser.add_argument('--root_conf_floor', type=float, default=None,
                        help='Ignore 2D detections below this confidence in the root solve')
    parser.add_argument('--root_steps', type=int, default=None,
                        help='LBFGS steps (annealing stages) for the static root solve')
    parser.add_argument('--root_go_anchor_w', type=float, default=None,
                        help='Anchor weight pulling the root solve\'s global_orient toward init')
    parser.add_argument('--root_tr_anchor_w', type=float, default=None,
                        help='Anchor weight pulling the root solve\'s translation toward init')
    parser.add_argument('--root_refit', type=lambda x: x.lower() in ['true', '1'],
                        default=None,
                        help='Re-solve the static root once on the Stage-A-fitted pose '
                             '(only meaningful with freeze_root=True)')
    parser.add_argument('--root_refit_thr_mm', type=float, default=None,
                        help='Apply the root refit only if translation moved more than this')
    parser.add_argument('--root_refit_thr_deg', type=float, default=None,
                        help='Apply the root refit only if orientation moved more than this')

    # ── Stage B: hands ───────────────────────────────────────────────────────
    parser.add_argument('--hand_data_w', type=float, default=None,
                        help='3D hand-keypoint data weight')
    parser.add_argument('--hand_wilor_w', type=float, default=None,
                        help='Pull toward the WiLoR hand-pose init')
    parser.add_argument('--hand_prior_w', type=float, default=None,
                        help='L2 hand-pose prior weight')
    parser.add_argument('--hand_arm_anchor', type=float, default=None,
                        help='Keep the arm body_pose cols near the Stage-A reach')
    parser.add_argument('--hand_steps', type=int, default=None,
                        help='LBFGS steps for Stage B hand refinement')
    parser.add_argument('--hand_place_steps', type=int, default=None,
                        help='LBFGS steps for the arm-placement pre-phase (0 disables it)')

    # ── Stage B: head ────────────────────────────────────────────────────────
    parser.add_argument('--head_face_w', type=float, default=None,
                        help='Face-landmark data weight')
    parser.add_argument('--head_jaw_w', type=float, default=None,
                        help='Jaw L2 prior weight')
    parser.add_argument('--head_pose_w', type=float, default=None,
                        help='Keep neck/head near neutral')
    parser.add_argument('--head_anchor', type=float, default=None,
                        help='Keep neck/head near the Stage-A value')
    parser.add_argument('--head_expr_w', type=float, default=None,
                        help='L2 expression prior weight')
    parser.add_argument('--head_eye_w', type=float, default=None,
                        help='L2 eye-pose regularizer toward neutral')
    parser.add_argument('--head_ear_rho', type=float, default=None,
                        help='Fixed (not annealed) GMoF scale (m) for the ear landmarks')
    parser.add_argument('--head_steps', type=int, default=None,
                        help='LBFGS steps for Stage B head refinement')

    # ── legs ─────────────────────────────────────────────────────────────────
    parser.add_argument('--freeze_legs', type=lambda x: x.lower() in ['true', '1'],
                        default=None,
                        help='Hold the leg body_pose cols at their init through Stage A '
                             '(no 3D leg data to fit them to)')
    parser.add_argument('--leg_pose_cam', type=str, default=None,
                        help='The only camera view used for the per-camera seated-leg export')

    # ── Stage C: whole-sequence smoothing ────────────────────────────────────
    parser.add_argument('--smooth_lam_bp', type=float, default=None,
                        help='Smoothing strength for body_pose')
    parser.add_argument('--smooth_lam_leg', type=float, default=None,
                        help='Smoothing strength for the leg body_pose cols (near-static)')
    parser.add_argument('--smooth_lam_go', type=float, default=None,
                        help='Smoothing strength for global_orient (unwrapped)')
    parser.add_argument('--smooth_lam_tr', type=float, default=None,
                        help='Smoothing strength for translation')
    parser.add_argument('--smooth_lam_hand', type=float, default=None,
                        help='Smoothing strength for hand poses (kept light -- fingers move fast)')
    parser.add_argument('--smooth_lam_head', type=float, default=None,
                        help='Smoothing strength for jaw + expression + eyes')

    # ── misc ─────────────────────────────────────────────────────────────────
    parser.add_argument('--term_cap', type=float, default=None,
                        help='Per-loss-term clamp so one spike stays finite for the line search')
    parser.add_argument('--log_every', type=int, default=None,
                        help='Print the compact per-iteration loss line every N closure calls')

    args = parser.parse_args(argv)
    return vars(args)
