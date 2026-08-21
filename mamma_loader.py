# -*- coding: utf-8 -*-
"""
Loads reference SMPLX fits from the external `mamma` pipeline (resources/mamma_results/)
for use as priors/anchors in Stage A — see temporal_window.py for where these plug in
(Stage 0 betas, the static-root warm start, the window stillness anchor).

mamma runs an independent, multi-view, occlusion-gated, VPoser-regularised SMPLX fit over
the SAME 6-camera rig this pipeline uses (its run_args.json cam_names match fitter_pipeline
.cam_map's logical names exactly) and — verified empirically by comparing the pipeline's own
triangulated 'nose' keypoint against mamma's triangulated_3d_pts over ~9300 frames of one
session (median distance 1.6-2.3cm, best time-lag 0) — lands in the SAME world coordinate
frame as this pipeline's triangulation_results. No rigid transform, no rescaling needed.

mamma's body_id-NN track numbering is NOT the same convention as this pipeline's person_id,
and is not even STABLE across one video: mamma reassigns which body_id is which physical
person at every ~500-frame processing chunk boundary (verified on 005013/lego: body_id-00
and body_id-01 swap identity at frame 500, 3500, 5000, 6000, ... — not always the same
direction). find_segments()/load_mamma() below detect these boundaries once per video and
stitch a single, whole-video-consistent trajectory per person, so the rest of the pipeline
never has to think about it again.
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import glob
import os.path as osp

import numpy as np
import torch

# 'nose' is index 0 in this pipeline's mapped 17-joint (COCO-style) keypoint layout.
NOSE_KP_IDX = 0
SCAN_STRIDE = 50   # frames between coarse segment-boundary scan samples


def _to_np(a):
    return a.detach().cpu().numpy() if torch.is_tensor(a) else np.asarray(a)


def _list_body_ids(mamma_dir):
    paths = sorted(glob.glob(osp.join(mamma_dir, 'smplx_params_body_id-*.npz')))
    return [osp.basename(p).split('body_id-')[1].split('.npz')[0] for p in paths]


def _nearest_dist(tri_frame, point):
    return float(np.linalg.norm(tri_frame - point[None, :], axis=1).min())


def _winner_at(tri_by_id, body_ids, frame, point):
    dists = [_nearest_dist(tri_by_id[bid][frame], point) for bid in body_ids]
    return body_ids[int(np.argmin(dists))]


def find_segments(tri_by_id, body_ids, nose_traj, nose_valid, T):
    """Partition [0, T) into (start, end, body_id) runs by which body_id's triangulated
    points are nearest to our own nose trajectory at each point, refining every detected
    transition down to the exact frame. Handles mamma reassigning body_id per processing
    chunk (verified: chunk boundaries at every 500 frames, not always the same direction).
    """
    valid_idx = np.where(nose_valid[:T])[0]
    sample_idx = valid_idx[::SCAN_STRIDE]
    if len(sample_idx) == 0:
        return [(0, T, body_ids[0])]
    winners = [_winner_at(tri_by_id, body_ids, f, nose_traj[f]) for f in sample_idx]

    segments = []
    seg_start, seg_id = 0, winners[0]
    for i in range(1, len(sample_idx)):
        if winners[i] == seg_id:
            continue
        lo, hi = sample_idx[i - 1], sample_idx[i]
        boundary = hi
        for f in range(lo + 1, hi + 1):
            if not nose_valid[f]:
                continue
            if _winner_at(tri_by_id, body_ids, f, nose_traj[f]) != seg_id:
                boundary = f
                break
        segments.append((seg_start, boundary, seg_id))
        seg_start, seg_id = boundary, winners[i]
    segments.append((seg_start, T, seg_id))
    return segments


def peek_gender(mamma_dir):
    """Cheap peek at mamma's exported gender convention (e.g. 'neutral') from any one body_id's
    npz, without needing the person-ID resolution — mamma exports one consistent gender for the
    whole scene/session, so any track will do. Returns None if unavailable.

    Must be checked BEFORE the body model is built: mamma's betas/global_orient/translation are
    only mutually consistent under the SAME shapedirs basis mamma used to produce them (gender-
    specific in SMPLX — the rest-pose pelvis for the same betas can sit 15+cm apart between
    genders). Building this pipeline's body model under a different hardcoded gender breaks that
    consistency before mamma's data is ever loaded.
    """
    if mamma_dir is None:
        return None
    body_ids = _list_body_ids(mamma_dir)
    if not body_ids:
        return None
    data = np.load(osp.join(mamma_dir, f'smplx_params_body_id-{body_ids[0]}.npz'), allow_pickle=True)
    return str(data['smplx_export_gender'])


def load_mamma(mamma_dir, nose_traj, nose_valid, device, dtype,
                min_frames=None):
    """Load this person's mamma reference fit for the WHOLE video, stitched from whichever
    body_id matches our own triangulated nose at each point (see module docstring — mamma's
    body_id assignment isn't stable across the video). Returns None (never raises) when mamma
    has nothing usable, so callers can treat it like the existing optional folder args.

        nose_traj/nose_valid : (N,3)/(N,) this pipeline's own triangulated nose + validity
        min_frames : reject any mamma track shorter than this (partial/spurious detections).
                      Defaults to len(nose_traj).

    Returns a dict of torch tensors (T = shortest usable track length), or None:
        body_id (str, majority track)         global_orient (T,3)   body_pose (T,63)
        transl (T,3)   betas (1,16)           contact (T,512)       floor_contact (T,512)
        triangulated_3d_pts (T,512,3)
    """
    nose_traj  = _to_np(nose_traj)
    nose_valid = _to_np(nose_valid).astype(bool)
    if min_frames is None:
        min_frames = len(nose_traj)

    if mamma_dir is None:
        print("[mamma] no mamma_dir given -> skipping")
        return None

    body_ids = _list_body_ids(mamma_dir)
    if not body_ids:
        print(f"[mamma] no smplx_params_body_id-*.npz under {mamma_dir} -> skipping")
        return None

    data_by_id = {}
    for bid in body_ids:
        d = np.load(osp.join(mamma_dir, f'smplx_params_body_id-{bid}.npz'), allow_pickle=True)
        if d['smplx_pose'].shape[0] < min_frames:
            print(f"[mamma] body_id-{bid}: {d['smplx_pose'].shape[0]} frames < {min_frames} "
                  f"required -> partial/spurious track, skipping")
            continue
        data_by_id[bid] = d
    if not data_by_id:
        print("[mamma] no body_id track long enough -> skipping")
        return None

    tri_by_id = {bid: d['triangulated_3d_pts'] for bid, d in data_by_id.items()}
    T = min(d['smplx_pose'].shape[0] for d in data_by_id.values())
    T = min(T, len(nose_traj))
    segments = find_segments(tri_by_id, list(data_by_id.keys()), nose_traj, nose_valid, T)
    print(f"[mamma] segments: {segments}")

    pose          = np.zeros((T, 165), dtype=np.float32)
    transl        = np.zeros((T, 3),   dtype=np.float32)
    contact       = np.zeros((T, 512), dtype=np.float32)
    floor_contact = np.zeros((T, 512), dtype=np.float32)
    tri           = np.zeros((T, 512, 3), dtype=np.float32)
    seg_frames = {}
    for start, end, bid in segments:
        d = data_by_id[bid]
        pose[start:end]          = d['smplx_pose'][start:end]
        transl[start:end]        = d['smplx_translation'][start:end]
        contact[start:end]       = d['smplx_contact'][start:end]
        floor_contact[start:end] = d['smplx_floor_contact'][start:end]
        tri[start:end]           = d['triangulated_3d_pts'][start:end]
        seg_frames[bid] = seg_frames.get(bid, 0) + (end - start)

    # betas per body_id file IS internally consistent with that same file's pose (verified:
    # ~2.5cm median vs our own triangulation) — the bug was pairing pose from one file with
    # betas from another. Use the file governing the FIRST segment (every run starts at frame
    # 0); betas is still one fixed value for the whole returned trajectory, so a run spanning
    # a later segment boundary would still see this same mismatch there.
    first_bid = segments[0][2]
    betas = data_by_id[first_bid]['smplx_betas']
    print(f"[mamma] resolved {len(segments)} segment(s), betas from body_id-{first_bid} "
          f"(governs frame 0; seg totals {seg_frames})")

    def to_t(a):
        return torch.as_tensor(np.asarray(a, dtype=np.float32), dtype=dtype, device=device)

    return {
        'body_id':             first_bid,
        'global_orient':       to_t(pose[:, 0:3]),
        'body_pose':           to_t(pose[:, 3:66]),
        'transl':              to_t(transl),
        'betas':               to_t(betas),
        'contact':             to_t(contact),
        'floor_contact':       to_t(floor_contact),
        'triangulated_3d_pts': to_t(tri),
    }
