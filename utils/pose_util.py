import numpy as np
import pandas as pd
import transforms3d.quaternions as txq
import struct
import open3d
import torch
import math
import scipy.interpolate
import scipy.linalg as slin
from os import path as osp
from datasets.robotcar_sdk.python.transform import build_se3_transform

def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part last, as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    i, j, k, r = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )

    return o.reshape(quaternions.shape[:-1] + (3, 3))


def vdot(v1, v2):
  """
  Dot product along the dim=1
  :param v1: N x d
  :param v2: N x d
  :return: N x 1
  """
  out = torch.mul(v1, v2)
  out = torch.sum(out, 1)
  return out


def normalize(x, p=2, dim=0):
  """
  Divides a tensor along a certain dim by the Lp norm
  :param x:
  :param p: Lp norm
  :param dim: Dimension to normalize along
  :return:
  """
  xn = x.norm(p=p, dim=dim)
  x = x / xn.unsqueeze(dim=dim)
  return x


def qmult(q1, q2):
  """
  Multiply 2 quaternions
  :param q1: Tensor N x 4
  :param q2: Tensor N x 4
  :return: quaternion product, Tensor N x 4
  """
  q1s, q1v = q1[:, :1], q1[:, 1:]
  q2s, q2v = q2[:, :1], q2[:, 1:]

  qs = q1s*q2s - vdot(q1v, q2v)
  qv = q1v.mul(q2s.expand_as(q1v)) + q2v.mul(q1s.expand_as(q2v)) +\
       torch.cross(q1v, q2v, dim=1)
  q  = torch.cat((qs, qv), dim=1)

  # normalize
  q = normalize(q, dim=1)

  return q


def qinv(q):
  """
  Inverts quaternions
  :param q: N x 4
  :return: q*: N x 4
  """
  q_inv = torch.cat((q[:, :1], -q[:, 1:]), dim=1)
  return q_inv


def qexp_t_safe(q):
  """
  Applies exponential map to log quaternion (safe implementation that does not
  maintain gradient flow)
  :param q: N x 3
  :return: N x 4
  """
  q = torch.from_numpy(np.asarray([qexp(qq) for qq in q.numpy()],
                                  dtype=np.float32))
  return q


def qlog_t_safe(q):
  """
  Applies the log map to a quaternion (safe implementation that does not
  maintain gradient flow)
  :param q: N x 4
  :return: N x 3
  """
  q = torch.from_numpy(np.asarray([qlog(qq) for qq in q.numpy()],
                                  dtype=np.float32))
  return q


def rotate_vec_by_q(t, q):
  """
  rotates vector t by quaternion q
  :param t: vector, Tensor N x 3
  :param q: quaternion, Tensor N x 4
  :return: t rotated by q: t' = t + 2*qs*(qv x t) + 2*qv x (qv x r)
  """
  qs, qv = q[:, :1], q[:, 1:]
  b  = torch.cross(qv, t, dim=1)
  c  = 2 * torch.cross(qv, b, dim=1)
  b  = 2 * b.mul(qs.expand_as(b))
  tq = t + b + c
  return tq


def calc_vo_logq_safe(p0, p1):
  """
  VO in the p0 frame using numpy fns
  :param p0:
  :param p1:
  :return:
  """
  vos_t = p1[:, :3] - p0[:, :3]
  q0 = qexp_t_safe(p0[:, 3:])
  q1 = qexp_t_safe(p1[:, 3:])
  vos_t = rotate_vec_by_q(vos_t, qinv(q0))
  vos_q = qmult(qinv(q0), q1)
  vos_q = qlog_t_safe(vos_q)
  return torch.cat((vos_t, vos_q), dim=1)


def calc_vos_safe_fc(poses):
    """
  calculate the VOs, from a list of consecutive poses (fully connected)
  :param poses: N x T x 7
  :return: N x TC2 x 7
  """
    vos = []
    for p in poses:
        pvos = []
        for i in range(p.size(0)):
            for j in range(i + 1, p.size(0)):
                pvos.append(calc_vo_logq_safe(p[i].unsqueeze(0), p[j].unsqueeze(0)))
        vos.append(torch.cat(pvos, dim=0))
    vos = torch.stack(vos, dim=0)
    return vos


def qlog(q):
    """
    Applies logarithm map to q
    :param q: (4,)
    :return: (3,)
    """
    if all(q[1:] == 0):
        q = np.zeros(3)
    else:
        q = np.arccos(q[0]) * q[1:] / np.linalg.norm(q[1:])

    return q


def qexp(q):
    """
    Applies the exponential map to q
    :param q: (3,)
    :return: (4,)
    """
    n = np.linalg.norm(q)
    q = np.hstack((np.cos(n), np.sinc(n / np.pi) * q))

    return q


def qexp_t(q):
    """
    Applies exponential map to log quaternion
    :param q: N x 3
    :return: N x 4
    """
    n = torch.norm(q, p=2, dim=1, keepdim=True)
    n = torch.clamp(n, min=1e-8)
    q = q * torch.sin(n)
    q = q / n
    q = torch.cat((torch.cos(n), q), dim=1)

    return q


def process_poses(poses_in, mean_t, std_t, align_R, align_t, align_s):
    poses_out = np.zeros((len(poses_in), 6))
    rot_out = np.zeros((len(poses_in), 3, 3))
    poses_out[:, 0:3] = poses_in[:, [3, 7, 11]]

    # align
    for i in range(len(poses_out)):
        R = poses_in[i].reshape((3, 4))[:3, :3]
        rot_out[i, :, :] = R
        q = txq.mat2quat(np.dot(align_R, R))
        q *= np.sign(q[0])  # constrain to hemisphere
        q = qlog(q)
        poses_out[i, 3:] = q
        t = poses_out[i, :3] - align_t
        poses_out[i, :3] = align_s * np.dot(align_R, t[:, np.newaxis]).squeeze()

    # normalize translation
    poses_out[:, :3] -= mean_t
    poses_out[:, :3] /= std_t

    # max and min
    pose_max = np.max(poses_out[:, :2], axis=0)  # (2,)
    pose_min = np.min(poses_out[:, :2], axis=0)  # (2,)

    return poses_out, rot_out, pose_max, pose_min


def calibrate_process_poses(poses_in, mean_t, std_t, align_R, align_t, align_s):
    poses_out = np.zeros((len(poses_in), 6))
    rot_out = np.zeros((len(poses_in), 3, 3))
    poses_out[:, 0:3] = poses_in[:, 9:]

    # align
    for i in range(len(poses_out)):
        R = poses_in[i, :9].reshape((3, 3))
        rot_out[i, :, :] = R
        q = txq.mat2quat(np.dot(align_R, R))
        q *= np.sign(q[0])  # constrain to hemisphere
        q = qlog(q)
        poses_out[i, 3:] = q
        t = poses_out[i, :3] - align_t
        poses_out[i, :3] = align_s * np.dot(align_R, t[:, np.newaxis]).squeeze()

    # normalize translation
    poses_out[:, :3] -= mean_t
    poses_out[:, :3] /= std_t

    # max and min
    pose_max = np.max(poses_out[:, :2], axis=0)  # (2,)
    pose_min = np.min(poses_out[:, :2], axis=0)  # (2,)

    return poses_out, rot_out, pose_max, pose_min


def poses_foraugmentaion(poses_r, pose_t):
    poses_out = np.zeros(6)
    poses_out[:3] = pose_t[:3]
    align_R = np.eye(3)
    align_t = np.zeros(3)
    align_s = 1
    # align
    R = poses_r
    q = txq.mat2quat(np.dot(align_R, R))
    q *= np.sign(q[0])  # constrain to hemisphere
    q = qlog(q)
    poses_out[3:] = q
    t = poses_out[:3] - align_t
    poses_out[:3] = align_s * np.dot(align_R, t[:, np.newaxis]).squeeze()

    return poses_out


def val_translation(pred_p, gt_p):
    """
    test model, compute error (numpy)
    input:
        pred_p: [3,]
        gt_p: [3,]
    returns:
        translation error (m):
    """
    if isinstance(pred_p, np.ndarray):
        predicted = pred_p
        groundtruth = gt_p
    else:
        predicted = pred_p.cpu().numpy()
        groundtruth = gt_p.cpu().numpy()
    error = np.linalg.norm(groundtruth - predicted)

    return error


def val_rotation(pred_q, gt_q):
    """
    test model, compute error (numpy)
    input:
        pred_q: [4,]
        gt_q: [4,]
    returns:
        rotation error (degrees):
    """
    if isinstance(pred_q, np.ndarray):
        predicted = pred_q
        groundtruth = gt_q
    else:
        predicted = pred_q.cpu().numpy()
        groundtruth = gt_q.cpu().numpy()

    # d = abs(np.sum(np.multiply(groundtruth, predicted)))
    # if d != d:
    #     print("d is nan")
    #     raise ValueError
    # if d > 1:
    #     d = 1
    # error = 2 * np.arccos(d) * 180 / np.pi0
    # d     = abs(np.dot(groundtruth, predicted))
    # d     = min(1.0, max(-1.0, d))

    d = np.abs(np.dot(groundtruth, predicted))
    d = np.minimum(1.0, np.maximum(-1.0, d))
    error = 2 * np.arccos(d) * 180 / np.pi

    return error


def poses2mats(poses_in):
    poses_out = np.zeros((len(poses_in), 3, 3))  # (B, 3, 3)
    poses_qua = np.asarray([qexp(q) for q in poses_in.cpu().detach().numpy()])

    # align
    for i in range(len(poses_out)):
        R = txq.quat2mat(poses_qua[i])
        poses_out[i, ...] = R

    return poses_out


def transform(pts, trans):
    """
    Applies the SE3 transformations, support torch.Tensor and np.ndarry.  Equation: trans_pts = R @ pts + t
    Input
        - pts: [num_pts, 3] or [bs, num_pts, 3], pts to be transformed
        - trans: [4, 4] or [bs, 4, 4], SE3 transformation matrix
    Output
        - pts: [num_pts, 3] or [bs, num_pts, 3] transformed pts
    """
    if len(pts.shape) == 3:
        trans_pts = trans[:, :3, :3] @ pts.permute(0,2,1) + trans[:, :3, 3:4]
        return trans_pts.permute(0,2,1)
    else:
        trans_pts = trans[:3, :3] @ pts.T + trans[:3, 3:4]
        return trans_pts.T


def calc_vos_simple(poses):
    """
    calculate the VOs, from a list of consecutive poses
    :param poses: N x T x 7
    :return: N x (T-1) x 7
    """
    vos = []
    for p in poses:
        pvos = [p[i + 1].unsqueeze(0) - p[i].unsqueeze(0) for i in range(len(p) - 1)]
        vos.append(torch.cat(pvos, dim=0))
    vos = torch.stack(vos, dim=0)

    return vos


def filter_overflow_ts(filename, ts_raw):
    #
    file_data = pd.read_csv(filename)
    base_name = osp.basename(filename)

    if base_name.find('vo') > -1:
        ts_key = 'source_timestamp'
    else:
        ts_key = 'timestamp'

    pose_timestamps = file_data[ts_key].values
    min_pose_timestamps = min(pose_timestamps)
    max_pose_timestamps = max(pose_timestamps)
    ts_filted = [t for t in ts_raw if min_pose_timestamps < t < max_pose_timestamps]
    abandon_num = len(ts_raw) - len(ts_filted)
    print('abandom %d pointclouds that exceed the range of %s' % (abandon_num, filename))

    return ts_filted


def filter_overflow_nclt(gt_filename, ts_raw):  # 滤波函数
    # gt_filename: GT对应的文件名
    # ts_raw: 原始数据集提供的点云时间戳
    ground_truth = np.loadtxt(gt_filename, delimiter=",")[1:, 0]
    min_pose_timestamps = min(ground_truth)
    max_pose_timestamps = max(ground_truth)
    ts_filted = [t for t in ts_raw if min_pose_timestamps < t < max_pose_timestamps]
    abandon_num = len(ts_raw) - len(ts_filted)
    print('abandom %d pointclouds that exceed the range of %s' % (abandon_num, gt_filename))

    return ts_filted


def interpolate_pose_nclt(gt_filename, ts_raw):  # 插值函数
    # gt_filename: GT对应文件名
    # ts_raw: 滤波后的点云时间戳
    ground_truth = np.loadtxt(gt_filename, delimiter=",")
    ground_truth = ground_truth[np.logical_not(np.any(np.isnan(ground_truth), 1))]
    interp = scipy.interpolate.interp1d(ground_truth[:, 0], ground_truth[:, 1:], kind='nearest', axis=0)
    pose_gt = interp(ts_raw)

    return pose_gt


def so3_to_euler_nclt(poses_in):
    N = len(poses_in)
    poses_out = np.zeros((N, 4, 4))
    for i in range(N):
        poses_out[i, :, :] = build_se3_transform([poses_in[i, 0], poses_in[i, 1], poses_in[i, 2],
                                                  poses_in[i, 3], poses_in[i, 4], poses_in[i, 5]])

    return poses_out


def convert_nclt(x_s, y_s, z_s):  # 输入点云转换函数
    # 文档种提供的转换函数
    # 原文档返回为x, y, z，但在绘制可视化图时z取负，此处先取负
    scaling = 0.005  # 5 mm
    offset = -100.0

    x = x_s * scaling + offset
    y = y_s * scaling + offset
    z = z_s * scaling + offset

    return x, y, -z


def load_velodyne_binary_nclt(filename):  # 读入二进制点云
    f_bin = open(filename, "rb")
    hits = []
    while True:
        x_str = f_bin.read(2)
        if x_str == b'':  # eof
            break
        x = struct.unpack('<H', x_str)[0]
        y = struct.unpack('<H', f_bin.read(2))[0]
        z = struct.unpack('<H', f_bin.read(2))[0]
        i = struct.unpack('B', f_bin.read(1))[0]
        l = struct.unpack('B', f_bin.read(1))[0]

        x, y, z = convert_nclt(x, y, z)

        hits += [[x, y, z]]

    f_bin.close()

    hits = np.array(hits)

    return hits