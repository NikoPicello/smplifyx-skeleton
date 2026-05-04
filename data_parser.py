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

    NUM_BODY_JOINTS = 21
    NUM_HAND_JOINTS = 15

    def __init__(self, sequence_path,
                 use_hands=False,
                 dtype=torch.float32,
                 model_type='smplx',
                 joints_to_ign=None,
                 adt_format='adt51',
                 **kwargs):
        super(CustomDataset, self).__init__()

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
        # load custom data path: take ADT format for convenience
        selected_skeleton_file = "skeletons.json"  # "skeletons.json"
        # selected_skeleton_file = "filtered_skeletons.json"   # "skeletons.json"
        skeleton_path = os.path.join(sequence_path, selected_skeleton_file)
        json_data = []
        with open(skeleton_path, 'r') as f:
            for line in f:
                json_data.append(json.loads(line))
        num_frames = len(json_data)
        print(f"Load {num_frames} frames from {skeleton_path}")
        # read each frame's raw 3D keypoints
        skeleton_data = []
        for frame_idx in range(num_frames):
            raw_keypoints_3d = np.array(json_data[frame_idx]["joints"]).reshape(-1, 3)
            skeleton_data.append(raw_keypoints_3d)
        skeleton_data = np.array(skeleton_data)
        return skeleton_data

    def get_model2data(self):
        if self.adt_format.lower() == 'adt51':
            if self.model_type == 'smplx':
                # ['Skeleton', 'Ab', 'Chest', 'Neck', 'Head', 'LShoulder', 'LUArm', 'LFArm', 'LHand',  'RShoulder',
                # 'RUArm', 'RFArm', 'RHand',  'LThigh', 'LShin', 'LFoot', 'LToe', 'RThigh', 'RShin', 'RFoot', 'RToe']
                # in smplx names:
                # ['pelvis', 'spine1', 'spine3', 'neck', 'head', 'left collar', 'left shoulder', 'left elbow', 'left wrist', 'right collar', 'right shoulder',
                # 'right elbow', 'right wrist', 'left hip', 'left knee', 'left ankle', left big toe', 'right hip', 'right knee', 'right ankle', 'right big toe']
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

