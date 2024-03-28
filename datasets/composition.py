import torch
import numpy as np
import sys
sys.path.insert(0, '../')

from .oxford import Oxford
from .nclt import NCLT
from torch.utils import data
from utils.pose_util import calc_vos_safe_fc


class MF(data.Dataset):
    def __init__(self, dataset, config, split='train', include_vos=False):

        self.steps = config.train.steps
        self.skip = config.train.skip
        self.train = split

        if dataset == 'Oxford':
            self.dset = Oxford(config, split)
        elif dataset == 'NCLT':
            self.dset = NCLT(config, split)
        else:
            raise NotImplementedError('{:s} dataset is not implemented!')

        self.L = self.steps * self.skip
        # GCS
        self.include_vos = include_vos
        self.vo_func = calc_vos_safe_fc


    def get_indices(self, index):
        skips = self.skip * np.ones(self.steps-1)
        offsets = np.insert(skips, 0, 0).cumsum()  # (self.steps,)
        offsets -= offsets[len(offsets) // 2]
        offsets = offsets.astype(np.int_)
        idx = index + offsets
        idx = np.minimum(np.maximum(idx, 0), len(self.dset)-1)
        assert np.all(idx >= 0), '{:d}'.format(index)
        assert np.all(idx < len(self.dset))
        return idx

    def __getitem__(self, index):
        idx   = self.get_indices(index)
        clip  = [self.dset[i] for i in idx]
        pcs   = torch.stack([c[0] for c in clip], dim=0)  # (self.steps, N, 3)
        poses = torch.stack([c[1] for c in clip], dim=0)  # (self.steps, 6)
        mask = torch.stack([c[2] for c in clip], dim=0)   # (self.steps, 512)
        if self.include_vos:
            vos = self.vo_func(poses.unsqueeze(0))[0]
            # 前3维是真值，后3维是相对位姿
            poses = torch.cat((poses, vos), dim=0)

        batch = {
            "image": pcs,
            "pose": poses,
            "mask": mask
        }
        return batch

    def __len__(self):
        L = len(self.dset)
        return L