#!/usr/bin/env python3
"""Batch version of hip_lift_prototype.py: run the same lift + holdout validation over
every (session, activity, person) that has both triangulation_results and rtmo_results,
and print one compact summary row each -- so a bug or a rig-specific quirk (like the
GB miscalibration / near-vs-far root flip found on 001080) shows up as a pattern across
sessions instead of needing a manual re-run per session.

Usage:
    python hip_lift_eval_all.py                      # every session under resources/
    python hip_lift_eval_all.py --sid 005013 006007   # just these
"""
import argparse
import glob
import os.path as osp

import numpy as np

from hip_lift_prototype import (
    TRIG_ROOT, RTMO_ROOT, HIP_PAIRS, CROSS_PAIRS, CONF_FLOOR_2D, CONF_FLOOR_3D, VALID_CONF_3D,
    load_calib, load_body_npy, load_rtmo, bone_length_stats, cross_length_stats,
    camera_ray_batch, ray_sphere_lift_batch, dual_anchor_lift_batch,
)


def find_pairs(sid_filter=None):
    pairs = []
    for sid_path in sorted(glob.glob(osp.join(TRIG_ROOT, '*'))):
        sid = osp.basename(sid_path)
        if sid_filter and sid not in sid_filter:
            continue
        for act_path in sorted(glob.glob(osp.join(sid_path, '*'))):
            act = osp.basename(act_path)
            if osp.isdir(osp.join(RTMO_ROOT, sid, act)):
                people = sorted(int(osp.basename(p)[1:]) for p in glob.glob(osp.join(act_path, 'p*'))
                                 if osp.isfile(osp.join(p, 'body.npy')))
                if people:
                    pairs.append((sid, act, people))
    return pairs


def shoulder_flags(cams, rtmo_by_cam, body_frames, person_id, bad_px=15.0):
    """Returns {cam_name: median_px_error}; caller decides the bad-camera cutoff."""
    out = {}
    for cam_name, arr in rtmo_by_cam.items():
        cam = cams[cam_name]
        import cv2 as cv
        rvec, _ = cv.Rodrigues(cam['R'])
        errs = []
        for fidx, fr in body_frames.items():
            det = arr[fidx].get(person_id) if fidx < len(arr) and isinstance(arr[fidx], dict) else None
            if det is None:
                continue
            for sho in (5, 6):
                if fr['confidence'][sho] <= CONF_FLOOR_3D or det['keypoint_scores'][sho] < CONF_FLOOR_2D:
                    continue
                S = fr['kpts_3d'][sho]
                if not np.isfinite(S).all():
                    continue
                proj, _ = cv.projectPoints(S.reshape(1, 3), rvec, cam['T'].reshape(3, 1), cam['K'], cam['D'])
                errs.append(np.linalg.norm(proj.reshape(2) - det['keypoints'][sho]))
        out[cam_name] = float(np.median(errs)) if errs else None
    return out


def eval_one(sid, act, person_id, bad_px=15.0, method='same_side'):
    cams = load_calib(sid)
    body_frames = load_body_npy(sid, act, person_id)
    bone = bone_length_stats(body_frames)
    row = dict(sid=sid, act=act, pid=person_id, n_tri=len(body_frames))
    if bone['L'][0] is None or bone['R'][0] is None:
        row['error'] = 'no trunk-length samples'
        return row
    trunk = {'L': bone['L'][0], 'R': bone['R'][0]}
    row['trunk_L_cm'], row['trunk_R_cm'] = trunk['L'] * 100, trunk['R'] * 100
    cross = None
    if method == 'dual_anchor':
        cross_stats = cross_length_stats(body_frames)
        if cross_stats['L'][0] is None or cross_stats['R'][0] is None:
            row['error'] = 'no cross-length samples for dual-anchor'
            return row
        cross = {'L': cross_stats['L'][0], 'R': cross_stats['R'][0]}

    rtmo_by_cam = {c: load_rtmo(sid, act, c) for c in cams}
    rtmo_by_cam = {c: v for c, v in rtmo_by_cam.items() if v is not None}
    flags = shoulder_flags(cams, rtmo_by_cam, body_frames, person_id, bad_px)
    good_cams = {c for c, e in flags.items() if e is not None and e <= bad_px}
    bad_cams = {c: e for c, e in flags.items() if e is not None and e > bad_px}
    row['bad_cams'] = ','.join(f'{c}:{e:.0f}px' for c, e in sorted(bad_cams.items())) or '-'

    for side, (sho, hip) in HIP_PAIRS.items():
        cross_sho = CROSS_PAIRS[side][0]
        n_gt_confident = sum(1 for fr in body_frames.values() if fr['confidence'][hip] >= VALID_CONF_3D)
        best = {}   # fidx -> (score, P)

        for cam_name in good_cams:
            arr = rtmo_by_cam[cam_name]
            cam = cams[cam_name]

            # gather every eligible frame for this (camera, side) up front, then lift the
            # whole batch in one shot -- a per-frame scipy call for dual_anchor was too
            # slow to run over a real video (didn't finish one session in minutes)
            fidxs, pxpy, S_same, S_cross, scores = [], [], [], [], []
            for fidx, fr in body_frames.items():
                if fidx >= len(arr):
                    continue
                det = arr[fidx].get(person_id) if isinstance(arr[fidx], dict) else None
                if det is None or det['keypoint_scores'][hip] < CONF_FLOOR_2D:
                    continue
                cs, S = fr['confidence'][sho], fr['kpts_3d'][sho]
                if cs <= CONF_FLOOR_3D or not np.isfinite(S).all():
                    continue
                if method == 'dual_anchor':
                    cc, Sc = fr['confidence'][cross_sho], fr['kpts_3d'][cross_sho]
                    if cc <= CONF_FLOOR_3D or not np.isfinite(Sc).all():
                        continue
                    S_cross.append(Sc)
                fidxs.append(fidx); pxpy.append(det['keypoints'][hip])
                S_same.append(S); scores.append(float(det['keypoint_scores'][hip]))
            if not fidxs:
                continue
            pxpy, S_same = np.asarray(pxpy), np.asarray(S_same)
            C, U = camera_ray_batch(pxpy, cam['K'], cam['D'], cam['R'], cam['T'])

            if method == 'dual_anchor':
                S_cross = np.asarray(S_cross)
                t0 = ray_sphere_lift_batch(C, U, S_same, trunk[side])
                closest = np.einsum('ij,ij->i', S_same - C[None, :], U)   # closest-approach fallback
                t0 = np.where(np.isfinite(t0), t0, closest)
                t = dual_anchor_lift_batch(C, U, S_same, trunk[side], S_cross, cross[side], t0)
                valid = t > 0
                P_all = C[None, :] + t[:, None] * U
            else:
                t = ray_sphere_lift_batch(C, U, S_same, trunk[side])
                valid = np.isfinite(t)
                P_all = C[None, :] + t[:, None] * U

            for i, fidx in enumerate(fidxs):
                if not valid[i]:
                    continue
                if fidx not in best or scores[i] > best[fidx][0]:
                    best[fidx] = (scores[i], P_all[i])

        errs = []
        for fidx, fr in body_frames.items():
            if fr['confidence'][hip] < VALID_CONF_3D or not np.isfinite(fr['kpts_3d'][hip]).all():
                continue
            if fidx in best:
                errs.append(np.linalg.norm(best[fidx][1] - fr['kpts_3d'][hip]) * 1000.0)
        row[f'{side}_n_lifted']   = len(best)
        row[f'{side}_n_gt_conf']  = n_gt_confident
        row[f'{side}_med_err_mm'] = round(float(np.median(errs)), 0) if errs else None
        row[f'{side}_p90_err_mm'] = round(float(np.percentile(errs, 90)), 0) if errs else None
        row[f'{side}_n_val']      = len(errs)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sid', nargs='+', default=None)
    ap.add_argument('--bad-px', type=float, default=15.0)
    ap.add_argument('--method', choices=['same_side', 'dual_anchor'], default='same_side')
    args = ap.parse_args()

    pairs = find_pairs(set(args.sid) if args.sid else None)
    print(f"{len(pairs)} session/activity pairs with both data sources\n")

    hdr = (f"{'session':8s} {'activity':10s} p {'n_tri':>6s}  {'bad_cams':22s}  "
           f"{'L n_lift/gt_conf':>18s} {'L med/p90 mm':>13s} {'L n_val':>7s}  "
           f"{'R n_lift/gt_conf':>18s} {'R med/p90 mm':>13s} {'R n_val':>7s}")
    print(hdr)
    print('-' * len(hdr))
    all_errs = []
    for sid, act, people in pairs:
        for pid in people:
            row = eval_one(sid, act, pid, args.bad_px, args.method)
            if 'error' in row:
                print(f"{sid:8s} {act:10s} {pid} {row['n_tri']:6d}  {row['error']}")
                continue
            L = row; R = row
            l_lg = f"{row['L_n_lifted']}/{row['L_n_gt_conf']}"
            r_lg = f"{row['R_n_lifted']}/{row['R_n_gt_conf']}"
            l_err = f"{row['L_med_err_mm']:.0f}/{row['L_p90_err_mm']:.0f}" if row['L_med_err_mm'] is not None else "n/a"
            r_err = f"{row['R_med_err_mm']:.0f}/{row['R_p90_err_mm']:.0f}" if row['R_med_err_mm'] is not None else "n/a"
            print(f"{sid:8s} {act:10s} {pid} {row['n_tri']:6d}  {row['bad_cams']:22s}  "
                  f"{l_lg:>18s} {l_err:>13s} {row['L_n_val']:7d}  "
                  f"{r_lg:>18s} {r_err:>13s} {row['R_n_val']:7d}")
            for side in ('L', 'R'):
                if row[f'{side}_med_err_mm'] is not None:
                    all_errs.append(row[f'{side}_med_err_mm'])
    if all_errs:
        all_errs = np.array(all_errs)
        print(f"\noverall: {len(all_errs)} (session,person,side) rows had >=1 holdout sample; "
              f"median-of-medians={np.median(all_errs):.0f}mm  worst={all_errs.max():.0f}mm")


if __name__ == '__main__':
    main()
