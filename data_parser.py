from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
# from utils import smpl_to_adt



#########################
###### ADT Dataset ######
#########################
class ADT(Dataset):

    NUM_BODY_JOINTS = 21
    NUM_HAND_JOINTS = 15

    def __init__(self, sequence_path,
                 use_hands=False,
                 dtype=torch.float32,
                 model_type='smplx',
                 joints_to_ign=None,
                 adt_format='adt51',
                 **kwargs):
        super(ADT, self).__init__()

        self.use_hands = use_hands
        self.model_type = model_type
        self.dtype = dtype
        self.joints_to_ign = joints_to_ign
        self.adt_format = adt_format

        self.num_joints = (self.NUM_BODY_JOINTS +
                           2 * self.NUM_HAND_JOINTS * use_hands)  # 21 + 2*15

        self.skeleton_data = self.read_adt_skeleton_sequence(sequence_path)
        self.cnt = 0

    def read_adt_skeleton_sequence(self, sequence_path):
        # load ADT data path
        if sequence_path.split("_")[-1][0] == "M":
            selected_skeleton_file = "Skeleton_T.json"
        else:
            selected_skeleton_file = "Skeleton_C.json"
        skeleton_path = os.path.join(sequence_path, selected_skeleton_file)
        with open(skeleton_path) as f:
            json_data = json.load(f)
        raw_skeleton_data = json_data["frames"]
        num_frames = len(raw_skeleton_data)
        print(f"Load {num_frames} frames from {skeleton_path}")
        # read each frame's raw 3D keypoints
        skeleton_data = []
        for frame_idx in range(len(raw_skeleton_data)):
            raw_keypoints_3d = np.array(raw_skeleton_data[frame_idx]["joints"])
            skeleton_data.append(raw_keypoints_3d)
        skeleton_data = np.array(skeleton_data)
        return skeleton_data

    def get_model2data(self):
        if self.adt_format.lower() == 'adt51':
            if self.model_type == 'smplx':
                # ['Skeleton', 'Ab', 'Chest', 'Neck', 'Head', 'LShoulder', 'LUArm', 'LFArm', 'LHand',  'RShoulder',
                # 'RUArm', 'RFArm', 'RHand',  'LThigh', 'LShin', 'LFoot', 'LToe', 'RThigh', 'RShin', 'RFoot', 'RToe']
                body_mapping = np.array([0, 3, 9, 12, 15, 13, 16, 18, 20, 14,
                                        17, 19, 21, 1, 4, 7, 60, 2, 5, 8, 63], dtype=np.int32)  # 21
                mapping = [body_mapping]
                if self.use_hands:
                    # 'LHand', 'LThumb1', 'LThumb2', 'LThumb3', 'LIndex1', 'LIndex2', 'LIndex3', 'LMiddle1',
                    #  'LMiddle2', 'LMiddle3', 'LRing1', 'LRing2', 'LRing3', LPinky1', 'LPinky2', 'LPinky3',
                    lhand_mapping = np.array([20, 37, 38, 39, 25, 26, 27,
                                            28, 29, 30, 34, 35, 36,
                                            31, 32, 33], dtype=np.int32)  # 16
                    # 'RHand', 'RThumb1', 'RThumb2', 'RThumb3', 'RIndex1', 'RIndex2', 'RIndex3', 'RMiddle1',
                    #  'RMiddle2', 'RMiddle3', 'RRing1', 'RRing2', 'RRing3', 'RPinky1', 'RPinky2', 'RPinky3',
                    rhand_mapping = np.array([21, 52, 53, 54, 40, 41, 42,
                                            43, 44, 45, 49, 50, 51,
                                            46, 47, 48], dtype=np.int32)  # 16

                    mapping += [lhand_mapping, rhand_mapping]
                return np.concatenate(mapping)
        else:
            raise ValueError('Unknown joint format: {}'.format(self.adt_format))

    def get_joint_weights(self):
        # The weights for the joint terms in the optimization
        optim_weights = np.ones(self.num_joints + 2 * self.use_hands,
                                dtype=np.float32)

        # Neck, Left and right hip
        # These joints are ignored because SMPL has no neck joint and the
        # annotation of the hips is ambiguous.
        if self.joints_to_ign is not None and -1 not in self.joints_to_ign:
            optim_weights[self.joints_to_ign] = 0.
        return torch.tensor(optim_weights, dtype=self.dtype)

    def __len__(self):
        return self.skeleton_data.shape[0]

    def __getitem__(self, idx):
        keypoints_frame = self.skeleton_data[idx]
        return self.read_item(keypoints_frame)

    def read_item(self, keypoints_frame):
        body_keypoints = np.vstack((keypoints_frame[0:9], keypoints_frame[24:28], keypoints_frame[43:]), dtype=np.float32)
        if self.use_hands:
            left_hand_keyp = keypoints_frame[8:24].astype(np.float32)
            right_hand_keyp = keypoints_frame[27:43].astype(np.float32)
            body_keypoints = np.concatenate(
                [body_keypoints, left_hand_keyp, right_hand_keyp], axis=0)
        body_keypoints = np.expand_dims(body_keypoints, axis=0)  # 1 x num_joints x 3
        return body_keypoints

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.cnt >= self.skeleton_data.shape[0]:
            raise StopIteration

        keypoints_frame = self.skeleton_data[self.cnt]
        self.cnt += 1

        return self.read_item(keypoints_frame)




class CustomDataset(Dataset):

    NUM_BODY_JOINTS = 17
    NUM_HAND_JOINTS = 20
    NUM_FACE_JOINTS = 51
    NUM_CONT_JOINTS = 17


    def __init__(self,
                 data_path,
                 person_id,
                 use_hands=False,
                 use_face=False,
                 dtype=torch.float32,
                 model_type='smplx',
                 use_face_contour=False,
                 joints_to_ign=None,
                 body_format='coco17',
                 **kwargs):
        super(CustomDataset, self).__init__()

        self.use_hands = use_hands
        self.use_face = use_face
        self.use_face_contour = use_face_contour
        self.model_type = model_type
        self.dtype = dtype
        self.joints_to_ign = joints_to_ign
        self.body_format = body_format

        self.num_joints = (self.NUM_BODY_JOINTS +
                           2 * self.NUM_HAND_JOINTS * use_hands)

        self.body_file  = os.path.join(data_path, 'body.npy')
        self.lhand_file = os.path.join(data_path, 'left_hand.npy')
        self.rhand_file = os.path.join(data_path, 'right_hand.npy')
        self.face_file  = os.path.join(data_path, 'face.npy')
        self.smpl_file  = os.path.join(data_path, 'smpl.npy')

        self.body_data = self.read_data_file(self.body_file)
        if self.use_hands:
          self.lhand_data = self.read_data_file(self.lhand_file)
          self.rhand_data = self.read_data_file(self.rhand_file)

          # check which one between wilor and rtmo have the best confidence for the wrist position
          for frame_idx in range(self.body_data.shape[0]):
            # if self.body_data[frame_idx, 9, 3] > self.lhand_data[frame_idx, 0, 3]:
            if self.lhand_data[frame_idx, 0, 3] > 0.:
              self.body_data[frame_idx, 9, :] = self.lhand_data[frame_idx, 0, :]
            else:
              self.lhand_data[frame_idx, 0, :] = self.body_data[frame_idx, 9, :]

            # if self.body_data[frame_idx, 10, 3] > self.rhand_data[frame_idx, 0, 3]:
            if self.rhand_data[frame_idx, 0, 3] > 0.:
              self.body_data[frame_idx, 10, :] = self.rhand_data[frame_idx, 0, :]
            else:
              self.rhand_data[frame_idx, 0, :] = self.body_data[frame_idx, 10, :]


        if self.use_face:
          self.face_data = self.read_data_file(self.face_file)
        else:
          self.face_data = None
        self.cnt = 0

    def read_data_file(self, data_file):
        data_raw = np.load(data_file, allow_pickle=True).item()
        data = []

        for frame_idx, frame_data in data_raw.items():
          if isinstance(frame_idx, int):
            data.append(np.hstack((frame_data['kpts_3d'], frame_data['confidence'].reshape(-1, 1))))
        data = np.array(data, dtype=np.float32)
        invalid = np.isnan(data[..., :3]).any(-1) | (data[..., 3] <= 0)
        data[invalid] = 0.

        return data

    def get_model2data(self):
        if self.body_format.lower() == 'coco17':
            if self.model_type == 'smplx':
                # [nose, Leye, Reye, Lear, Rear, Lshoulder, Rshoulder, Lelbow, Relbow, Lwrist, Rwrist,
                #  Lhip, Rhip, Lknee, Rknee, Lankle, Rankle]
                body_mapping = np.array([55, 57, 56, 59, 58, 16, 17, 18, 19, 20, 21,
                                         1, 2, 4, 5, 7, 8], dtype=np.int32)  # 17
                mapping = [body_mapping]

                if self.use_hands:
                    # [Lwrist, LThumb1, LThumb2, LThumb3, LThumb4, LIndex1', LIndex2, LIndex3, LIndex4,
                    #  LMiddle1, LMiddle2, LMiddle3, LMiddle4, LRing1, LRing2, LRing3, LRing4,
                    #  LPinky1, LPinky2, LPinky3, LPinky4]
                    lhand_mapping = np.array([20, 37, 38, 39, 66, 25, 26, 27, 67,
                                              28, 29, 30, 68, 34, 35, 36, 69,
                                              31, 32, 33, 70], dtype=np.int32)  # 21
                    # [Rwrist, RThumb1, RThumb2, RThumb3, RThumb4, RIndex1', RIndex2, RIndex3, RIndex4,
                    #  RMiddle1, RMiddle2, RMiddle3, RMiddle4, RRing1, RRing2, RRing3, RRing4,
                    #  RPinky1, RPinky2, RPinky3, RPinky4]
                    rhand_mapping = np.array([21, 52, 53, 54, 71, 40, 41, 42, 72,
                                              43, 44, 45, 73, 49, 50, 51, 74,
                                              46, 47, 48, 75], dtype=np.int32)  # 21

                    mapping += [lhand_mapping, rhand_mapping]

                if self.use_face:
                    #  end_idx = 127 + 17 * use_face_contour
                    face_mapping = np.arange(76, 127 + 17 * self.use_face_contour,
                                             dtype=np.int32)
                    mapping += [face_mapping]
                return np.concatenate(mapping)
        else:
            raise ValueError('Unknown joint format: {}'.format(self.body_format))

    def get_joint_weights(self):
        # The weights for the joint terms in the optimization
        optim_weights = np.ones(self.num_joints + 2 * self.use_hands +
                                self.NUM_FACE_JOINTS * self.use_face +
                                self.NUM_CONT_JOINTS * self.use_face_contour,  # 21 + 2*15
                                dtype=np.float32)

        # Neck, Left and right hip
        # These joints are ignored because SMPL has no neck joint and the
        # annotation of the hips is ambiguous.
        if self.joints_to_ign is not None and -1 not in self.joints_to_ign:
            optim_weights[self.joints_to_ign] = 0.

        # body_weights = self.body_data[idx][:,:,3]
        # if self.use_hands:
        #   body_weights = np.hstack(body_weights, self.lhand_data[idx][:,:,3], self.rhand_data[idx][:,:,3])
        # if self.use_face:
        #   body_weights = np.hstack(body_weights, self.face_data[idx][:,:,3])

        # optim_weights

        return torch.tensor(optim_weights, dtype=self.dtype)

    def get_init_hand_poses(self, hand_side):
        if hand_side == 'left':
          hand_file = self.lhand_file
        elif hand_side == 'right':
          hand_file = self.rhand_file
        else:
          raise ValueError('Unknown hand side: {}'.format(hand_side))

        hand_data_raw = np.load(hand_file, allow_pickle=True).item()

        # Per-frame MANO warm-start pose. A hand can be fully occluded in a frame,
        # so 'hand_pose' may be None (or the key absent). Keep such frames as None
        # instead of stacking into an array: np.array() over a mix of (45,) vectors
        # and None raises "inhomogeneous shape", and — more importantly —
        # fit_single_frame treats a None init as "carry the previous frame's
        # optimized hand pose" rather than snapping the hand to a flat/zero pose.
        # Dict order is preserved so indices stay aligned with read_data_file()/
        # lhand_data (both iterate the same file in the same order).
        hand_poses = []
        for frame_idx, frame_data in hand_data_raw.items():
          if not isinstance(frame_idx, int):
            continue
          hp = frame_data.get('hand_pose') if isinstance(frame_data, dict) else None
          if hp is None:
            hand_poses.append(None)
          else:
            hand_poses.append(
                np.nan_to_num(np.asarray(hp, dtype=np.float32).reshape(-1), nan=0.0))

        return hand_poses

    def get_init_body(self, init_poses=True):
      body_data_raw = np.load(self.smpl_file, allow_pickle=True).item()

      # Same occlusion handling as get_init_hand_poses: a missing/None per-frame
      # 'body_pose' is kept as None (not stacked) so the caller falls back to the
      # previous frame / rest pose instead of crashing on an inhomogeneous
      # np.array or silently injecting a zero body pose.
      bps, gos, trs = [], [], []
      for frame_idx, frame_data in body_data_raw.items():
        if not isinstance(frame_idx, int):
          continue
        bp = frame_data.get('body_pose')     if isinstance(frame_data, dict) else None
        go = frame_data.get('global_orient') if isinstance(frame_data, dict) else None
        tr = frame_data.get('transl')        if isinstance(frame_data, dict) else None

        if bp is None and init_poses:
          bps.append(None)
        else:
          bps.append(np.nan_to_num(np.asarray(bp, dtype=np.float32).reshape(-1), nan=0.0))

        gos.append(None if go is None else
                   np.nan_to_num(np.asarray(go, dtype=np.float32).reshape(-1), nan=0.0))
        trs.append(None if tr is None else
                   np.nan_to_num(np.asarray(tr, dtype=np.float32).reshape(-1), nan=0.0))

      if not init_poses:
        bps = None

      betas = body_data_raw['betas']
      return bps, gos, trs, betas


    def __len__(self):
        return self.body_data.shape[0]

    def __getitem__(self, idx):
        return self.read_item(idx)

    def read_item(self, idx):
        body = self.body_data[idx]
        if self.use_hands:
          body = np.vstack([body, self.lhand_data[idx], self.rhand_data[idx]])
        if self.use_face:
          body = np.vstack((body, self.face_data[idx]))
        return body[None] # return (1, J, 4)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.cnt >= self.body_data.shape[0]:
            raise StopIteration
        self.cnt += 1
        return self.read_item(self.cnt)

