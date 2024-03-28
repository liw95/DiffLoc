"""Visualization code for point clouds and 3D bounding boxes with mayavi.

Modified by Charles R. Qi
Date: September 2017
"""

import argparse
import os

import numpy as np
import torch
from torchsparse import SparseTensor
# from torchsparse.utils.quantize import sparse_quantize
from torchsparse.utils.quantize import sparse_quantize

from model_zoo import minkunet, spvcnn, spvnas_specialized

import open3d as o3d

cpu_num = 2 # cpu core
os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)


def process_point_cloud(input_point_cloud, input_labels=None, voxel_size=0.15):
    input_point_cloud[:, 3] = input_point_cloud[:, 3]
    pc_ = np.round(input_point_cloud[:, :3] / voxel_size)
    pc_ -= pc_.min(0, keepdims=1)


    label_map = create_label_map()
    if input_labels is not None:
        labels_ = label_map[input_labels].astype(
            np.int64)  # semantic labels
    else:
        labels_ = np.zeros(pc_.shape[0], dtype=np.int64)

    feat_ = input_point_cloud

    if input_labels is not None:
        out_pc = input_point_cloud[labels_ != labels_.max(), :3]
        pc_ = pc_[labels_ != labels_.max()]
        feat_ = feat_[labels_ != labels_.max()]
        labels_ = labels_[labels_ != labels_.max()]
    else:
        out_pc = input_point_cloud
        pc_ = pc_


    coords_, inds, inverse_map = sparse_quantize(pc_,
                                                 return_index=True,
                                                 return_inverse=True)

    pc = np.zeros((inds.shape[0], 4))
    pc[:, :3] = pc_[inds]

    feat = feat_[inds]
    labels = labels_[inds]
    lidar = SparseTensor(
        torch.from_numpy(feat).float(),
        torch.from_numpy(pc).int())
    return {
        'pc': out_pc,
        'lidar': lidar,
        'targets': labels,
        'targets_mapped': labels_,
        'inverse_map': inverse_map
    }


def create_label_map(num_classes=3):
    name_label_mapping = {
    # 不需要的类
    'unlabeled': 0,
    'outlier': 1,
    'other-ground': 49,
    'other-structure': 52,
    # ground
    'ground': 10,
    # plane
    'plane': 20,
    # other
    'other': 30
    }

    # for k in name_label_mapping:
    #     name_label_mapping[k] = name_label_mapping[k.replace('moving-', '')]
    train_label_name_mapping = {
        0: 'ground',
        1: 'plane',
        2: 'other',
    }

    label_map = np.zeros(260) + num_classes
    for i in range(num_classes):
        cls_name = train_label_name_mapping[i]
        # print(cls_name)
        label_map[name_label_mapping[cls_name]] = min(num_classes, i)
    return label_map.astype(np.int64)


cmap = np.array([
    [245, 150, 100, 255],
    [245, 230, 100, 255],
    [150, 60, 30, 255],
    [180, 30, 80, 255],
    [255, 0, 0, 255],
    [30, 30, 255, 255],
    [200, 40, 255, 255],
    [90, 30, 150, 255],
    [255, 0, 255, 255],
    [255, 150, 255, 255],
    [75, 0, 75, 255],
    [75, 0, 175, 255],
    [0, 200, 255, 255],
    [50, 120, 255, 255],
    [0, 175, 0, 255],
    [0, 60, 135, 255],
    [80, 240, 150, 255],
    [150, 240, 255, 255],
    [0, 0, 255, 255],
])
cmap = cmap[:, [2, 1, 0, 3]]  # convert bgra to rgba


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 数据集位置
    parser.add_argument('--velodyne-dir', type=str, default='Oxford')
    parser.add_argument('--model',
                        type=str,
                        default='SemanticKITTI_val_SPVCNN@119GMACs')
    args = parser.parse_args()
    output_dir = os.path.join(args.velodyne_dir, 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    if 'MinkUNet' in args.model:
        model = minkunet(args.model, pretrained=True)
    elif 'SPVCNN' in args.model:
        model = spvcnn(args.model, pretrained=True)
    elif 'SPVNAS' in args.model:
        model = spvnas_specialized(args.model, pretrained=True)
    else:
        raise NotImplementedError

    model = model.to(device)
    file_list = ['2019-01-11-14-02-26-radar-oxford-10k', '2019-01-14-12-05-52-radar-oxford-10k',\
                 '2019-01-14-14-48-55-radar-oxford-10k', '2019-01-18-15-20-12-radar-oxford-10k',\
                 '2019-01-15-13-06-37-radar-oxford-10k', '2019-01-17-14-03-00-radar-oxford-10k',\
                 '2019-01-18-14-14-42-radar-oxford-10k', '2019-01-17-13-26-39-radar-oxford-10k']
    # file_list = ['2012-02-18', '2012-05-11', '2012-02-12', '2012-02-19']
    for file in file_list:
        # mkdir
        if os.path.exists(os.path.join(args.velodyne_dir, file, 'SPVNAS_velodyne_left_plane_segmented')) == 0:
            os.mkdir(os.path.join(args.velodyne_dir, file, 'SPVNAS_velodyne_left_plane_segmented'))
        input_path = os.path.join(args.velodyne_dir, file, 'velodyne_left')
        input_point_clouds = sorted(os.listdir(input_path))
        for point_cloud_name in input_point_clouds:
            print(input_path+ '/'+ point_cloud_name)
            if not point_cloud_name.endswith('.bin'):
                continue
            label_file_name = point_cloud_name.replace('.bin', '.label')
            vis_file_name = point_cloud_name.replace('.bin', '.png')
            gt_file_name = point_cloud_name.replace('.bin', '_GT.png')

            # Oxford
            pc = np.fromfile(f'{input_path}/{point_cloud_name}',
                             dtype=np.float32).reshape(4, -1).transpose()
            pc[:, 2] = -1 * pc[:, 2]

            # nclt
            # pc = np.fromfile(f'{input_path}/{point_cloud_name}',
            #                  dtype=np.float32).reshape(-1, 4)


            if os.path.exists(label_file_name):
                label = np.fromfile(f'{args.velodyne_dir}/{label_file_name}',
                                    dtype=np.int32)
            else:
                label = None
            feed_dict = process_point_cloud(pc, label)
            inputs = feed_dict['lidar'].to(device)
            outputs = model(inputs)
            predictions = outputs.argmax(1).cpu().numpy()
            # print(predictions.shape)
            predictions = predictions[feed_dict['inverse_map']]
            predictions = predictions.astype(np.int32)
            results     = np.concatenate((feed_dict['pc'][:,:4], predictions.reshape(-1,1)), axis=1)
            results[:, 2] = -1 * results[:, 2]

            # print(results.shape)
            # np.savetxt(point_cloud_name[:-4]+'.txt', results, fmt='%.6f')
            plane_list = []
            ground_list = []
            other_list = []
            # 对于提供好参数的模型执行下列操作
            for i in range(results.shape[0]):
                if results[i, 4] == 12:
                    plane_list.append(results[i, :4])
                elif results[i, 4] == 13:
                    plane_list.append(results[i, :4])
                # ground
                elif results[i, 4] == 8:
                    ground_list.append(results[i, :4])
                elif results[i, 4] == 9:
                    ground_list.append(results[i, :4])
                elif results[i, 4] == 10:
                    ground_list.append(results[i, :4])
                elif results[i, 4] == 11:
                    ground_list.append(results[i, :4])
                # other
                else:
                    other_list.append(results[i, :4])


            plane_list = np.array(plane_list).reshape(-1,4)
            plane_list = np.concatenate((plane_list, np.ones((len(plane_list), 1))), axis=1)
            planes     = plane_list.astype(np.float32)
            ground_list = np.array(ground_list).reshape(-1,4)
            ground_list = np.concatenate((ground_list, np.ones((len(ground_list), 1))), axis=1)
            grounds = ground_list.astype(np.float32)
            other_list = np.array(other_list).reshape(-1, 4)
            other_list = np.concatenate((other_list, np.zeros((len(other_list), 1))), axis=1)
            results = np.concatenate((ground_list, plane_list, other_list), axis=0)
            results = results.astype(np.float32)
            results.tofile(os.path.join(input_path[:-13]+'SPVNAS_velodyne_left_plane_segmented', point_cloud_name))