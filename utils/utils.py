import random

import numpy as np
import torch
import tempfile


def seed_all_random_engines(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def calc_vos_simple(poses):
    vos = []
    for p in poses:
        pvos = [p[i+1].unsqueeze(0) - p[i].unsqueeze(0) for i in range(len(p)-1)]
        vos.append(torch.cat(pvos, dim=0))
    vos = torch.stack(vos, dim=0)
    return vos