"""
@author: Wen Li
@file: oxford.py
@time: 2023/9/18 19:27
"""
import os
import h5py
import torch
import numpy as np
import os.path as osp
# import torchvision.transforms as T
from copy import deepcopy
from torch.utils import data
from datasets.projection import RangeProjection
from datasets.augmentor import Augmentor, AugmentParams
from utils.pose_util import process_poses, filter_overflow_ts, poses_foraugmentaion
from datasets.robotcar_sdk.python.interpolate_poses import interpolate_ins_poses
from datasets.robotcar_sdk.python.transform import build_se3_transform, euler_to_so3

BASE_DIR = osp.dirname(osp.abspath(__file__))

class Oxford(data.Dataset):
    def __init__(self, config, split='train'):
        # directories
        if split == 'train':
            self.is_train = True
        else:
            self.is_train = False

        lidar = 'velodyne_left'
        data_path = config.train.dataroot

        data_dir = osp.join(data_path, 'Oxford')
        extrinsics_dir = osp.join(BASE_DIR, 'robotcar_sdk', 'extrinsics')

        # decide which sequences to use
        if split == 'train':
            split_filename = osp.join(data_dir, 'train_split.txt')
        else:
            split_filename = osp.join(data_dir, 'valid_split.txt')
        with open(split_filename, 'r') as f:
            seqs = [l.rstrip() for l in f if not l.startswith('#')]

        ps = {}
        ts = {}
        vo_stats = {}
        self.pcs = []

        # extrinsic reading
        with open(os.path.join(extrinsics_dir, lidar + '.txt')) as extrinsics_file:
            extrinsics = next(extrinsics_file)
        G_posesource_laser = build_se3_transform([float(x) for x in extrinsics.split(' ')])
        with open(os.path.join(extrinsics_dir, 'ins.txt')) as extrinsics_file:
            extrinsics = next(extrinsics_file)
            G_posesource_laser = np.linalg.solve(build_se3_transform([float(x) for x in extrinsics.split(' ')]),
                                                 G_posesource_laser)  # (4, 4)
        for seq in seqs:
            seq_dir = osp.join(data_dir, seq + '-radar-oxford-10k')
            # read the image timestamps
            h5_path = osp.join(seq_dir, lidar + '_' + 'False.h5')

            if not os.path.isfile(h5_path):
                print('interpolate ' + seq)
                ts_filename = osp.join(seq_dir, lidar + '.timestamps')
                with open(ts_filename, 'r') as f:
                    ts_raw = [int(l.rstrip().split(' ')[0]) for l in f]
                # GT poses
                ins_filename = osp.join(seq_dir, 'gps', 'ins.csv')
                ts[seq] = filter_overflow_ts(ins_filename, ts_raw)
                p = np.asarray(interpolate_ins_poses(ins_filename, deepcopy(ts[seq]), ts[seq][0]))  # (n, 4, 4)
                p = np.asarray([np.dot(pose, G_posesource_laser) for pose in p])  # (n, 4, 4)
                ps[seq] = np.reshape(p[:, :3, :], (len(p), -1))  # (n, 12)
                # write to h5 file
                print('write interpolate pose to ' + h5_path)
                h5_file = h5py.File(h5_path, 'w')
                h5_file.create_dataset('valid_timestamps', data=np.asarray(ts[seq], dtype=np.int64))
                h5_file.create_dataset('poses', data=ps[seq])
            else:
                # load h5 file, save pose interpolating time
                print("load " + seq + ' pose from ' + h5_path)
                h5_file = h5py.File(h5_path, 'r')
                ts[seq] = h5_file['valid_timestamps'][...]
                ps[seq] = h5_file['poses'][...]

            vo_stats[seq] = {'R': np.eye(3), 't': np.zeros(3), 's': 1}

            if self.is_train:
                self.pcs.extend(
                [osp.join(seq_dir, 'SPVNAS_velodyne_left_plane_segmented', '{:d}.bin'.format(t)) for t in ts[seq]])
            else:
                self.pcs.extend(
                    [osp.join(seq_dir, 'velodyne_left', '{:d}.bin'.format(t)) for t in ts[seq]])

        # read / save pose normalization information
        poses = np.empty((0, 12))
        for p in ps.values():
            poses = np.vstack((poses, p))
        pose_stats_filename = osp.join(data_dir, 'Oxford_pose_stats.txt')
        if split == 'train':
            mean_t = np.mean(poses[:, [3, 7, 11]], axis=0)  # (3,)
            std_t = np.std(poses[:, [3, 7, 11]], axis=0)  # (3,)
            np.savetxt(pose_stats_filename, np.vstack((mean_t, std_t)), fmt='%8.7f')
        else:
            mean_t, std_t = np.loadtxt(pose_stats_filename)

        self.poses = np.empty((0, 6))
        self.rots = np.empty((0, 3, 3))
        for seq in seqs:
            pss, rotation, pss_max, pss_min = process_poses(poses_in=ps[seq], mean_t=mean_t, std_t=std_t,
                                                            align_R=vo_stats[seq]['R'], align_t=vo_stats[seq]['t'],
                                                            align_s=vo_stats[seq]['s'])
            self.poses = np.vstack((self.poses, pss))
            self.rots = np.vstack((self.rots, rotation))

        self.proj_img_mean = torch.tensor(config.sensors.image_mean, dtype=torch.float)
        self.proj_img_stds = torch.tensor(config.sensors.image_stds, dtype=torch.float)

        if split == 'train':
            print("train data num:" + str(len(self.poses)))
        else:
            print("valid data num:" + str(len(self.poses)))

        self.projection = RangeProjection(
            fov_up=config.sensors.fov_up, fov_down=config.sensors.fov_down,
            fov_left=config.sensors.fov_left, fov_right=config.sensors.fov_right,
            proj_h=config.sensors.proj_h, proj_w=config.sensors.proj_w,
        )

        augment_params = AugmentParams()
        augment_config = config.augmentation

        # Point cloud augmentations
        if self.is_train:
            augment_params.setTranslationParams(
                p_transx=augment_config['p_transx'], trans_xmin=augment_config[
                    'trans_xmin'], trans_xmax=augment_config['trans_xmax'],
                p_transy=augment_config['p_transy'], trans_ymin=augment_config[
                    'trans_ymin'], trans_ymax=augment_config['trans_ymax'],
                p_transz=augment_config['p_transz'], trans_zmin=augment_config[
                    'trans_zmin'], trans_zmax=augment_config['trans_zmax'])
            augment_params.setRotationParams(
                p_rot_roll=augment_config['p_rot_roll'], rot_rollmin=augment_config[
                    'rot_rollmin'], rot_rollmax=augment_config['rot_rollmax'],
                p_rot_pitch=augment_config['p_rot_pitch'], rot_pitchmin=augment_config[
                    'rot_pitchmin'], rot_pitchmax=augment_config['rot_pitchmax'],
                p_rot_yaw=augment_config['p_rot_yaw'], rot_yawmin=augment_config[
                    'rot_yawmin'], rot_yawmax=augment_config['rot_yawmax'])
            if 'p_scale' in augment_config:
                augment_params.sefScaleParams(
                    p_scale=augment_config['p_scale'],
                    scale_min=augment_config['scale_min'],
                    scale_max=augment_config['scale_max'])
                print(
                    f'Adding scaling augmentation with range [{augment_params.scale_min}, {augment_params.scale_max}] and probability {augment_params.p_scale}')
            self.augmentor = Augmentor(augment_params)
        else:
            self.augmentor = None

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx_N):
        scan_path = self.pcs[idx_N]
        if self.is_train:
            # generate by SPVCNN, (x, y, z, intensity, static objects mask)
            pointcloud = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 5)
        else:
            #fill with zeros
            pointcloud = np.fromfile(scan_path, dtype=np.float32).reshape(4, -1).transpose()
            pointcloud = np.concatenate((pointcloud, np.zeros(len(pointcloud, 1))), axis=1)
        # flip z
        T = euler_to_so3([np.pi, 0, np.pi / 2])
        pointcloud[:, :3] = (T[:3, :3] @ pointcloud[:, :3].transpose()).transpose()
        if self.is_train:
            pointcloud, rotation = self.augmentor.doAugmentation(pointcloud)  # n, 5
            original_rots = self.rots[idx_N]  # [3, 3]
            present_rots = rotation @ original_rots
            poses = poses_foraugmentaion(present_rots, self.poses[idx_N])
        else:
            poses = self.poses[idx_N]
        # Generate RangeImage
        proj_pointcloud, proj_range, proj_idx, proj_mask = self.projection.doProjection(pointcloud)
        proj_mask_tensor = torch.from_numpy(proj_mask)
        proj_range_tensor = torch.from_numpy(proj_range)  # [32, 512]
        proj_xyz_tensor = torch.from_numpy(proj_pointcloud[..., :3])  # [32, 512, 3]
        proj_intensity_tensor = torch.from_numpy(proj_pointcloud[..., 3])
        proj_label_tensor = torch.from_numpy(proj_pointcloud[..., 4])
        proj_feature_tensor = torch.cat(
            [proj_range_tensor.unsqueeze(0), proj_xyz_tensor.permute(2, 0, 1), proj_intensity_tensor.unsqueeze(0),
             proj_label_tensor.unsqueeze(0)], 0)
        pose_tensor = torch.from_numpy(poses.astype(np.float32))
        # normalization
        proj_feature_tensor = (proj_feature_tensor - self.proj_img_mean[:, None, None]) / self.proj_img_stds[:, None
        , None]
        proj_feature_tensor = proj_feature_tensor * proj_mask_tensor.unsqueeze(0).float()
        proj_tensor = torch.cat(
            (proj_feature_tensor,
             proj_mask_tensor.unsqueeze(0)), dim=0)

        return proj_tensor[:5], pose_tensor, proj_tensor[5]

