import os
import glob
import math
import torch
import logging
import warnings
import torch.optim
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import cycle
from matplotlib import colors as mcolors
from torch.utils.data import BatchSampler
from pytorch3d.transforms import so3_relative_angle
from pytorch3d.implicitron.tools.stats import Stats
from pytorch3d.implicitron.tools.vis_utils import get_visdom_connection
from accelerate.utils import set_seed as accelerate_set_seed
import torch.optim.lr_scheduler as toptim

logger = logging.getLogger(__name__)


def set_seed_and_print(seed):
    accelerate_set_seed(seed, device_specific=True)
    print(f"----------Seed is set to {np.random.get_state()[1][0]} now----------")


class VizStats(Stats):
    def plot_stats(
            self, viz=None, visdom_env=None, plot_file=None, visdom_server=None, visdom_port=None
    ):
        # use the cached visdom env if none supplied
        if visdom_env is None:
            visdom_env = self.visdom_env
        if visdom_server is None:
            visdom_server = self.visdom_server
        if visdom_port is None:
            visdom_port = self.visdom_port
        if plot_file is None:
            plot_file = self.plot_file

        stat_sets = list(self.stats.keys())

        logger.debug(
            f"printing charts to visdom env '{visdom_env}' ({visdom_server}:{visdom_port})"
        )

        novisdom = False

        if viz is None:
            viz = get_visdom_connection(server=visdom_server, port=visdom_port)

        if viz is None or not viz.check_connection():
            logger.info("no visdom server! -> skipping visdom plots")
            novisdom = True

        lines = []

        # plot metrics
        if not novisdom:
            viz.close(env=visdom_env, win=None)

        for stat in self.log_vars:
            vals = []
            stat_sets_now = []
            for stat_set in stat_sets:
                val = self.stats[stat_set][stat].get_epoch_averages()
                if val is None:
                    continue
                else:
                    val = np.array(val).reshape(-1)
                    stat_sets_now.append(stat_set)
                vals.append(val)

            if len(vals) == 0:
                continue

            lines.append((stat_sets_now, stat, vals))

        if not novisdom:
            for tmodes, stat, vals in lines:
                title = "%s" % stat
                opts = {"title": title, "legend": list(tmodes)}
                for i, (tmode, val) in enumerate(zip(tmodes, vals)):
                    update = "append" if i > 0 else None
                    valid = np.where(np.isfinite(val))[0]
                    if len(valid) == 0:
                        continue
                    x = np.arange(len(val))
                    viz.line(
                        Y=val[valid],
                        X=x[valid],
                        env=visdom_env,
                        opts=opts,
                        win=f"stat_plot_{title}",
                        name=tmode,
                        update=update,
                    )

        if plot_file:
            logger.info(f"plotting stats to {plot_file}")
            ncol = 3
            nrow = int(np.ceil(float(len(lines)) / ncol))
            matplotlib.rcParams.update({"font.size": 5})
            color = cycle(plt.cm.tab10(np.linspace(0, 1, 10)))
            fig = plt.figure(1)
            plt.clf()
            for idx, (tmodes, stat, vals) in enumerate(lines):
                c = next(color)
                plt.subplot(nrow, ncol, idx + 1)
                plt.gca()
                for vali, vals_ in enumerate(vals):
                    c_ = c * (1.0 - float(vali) * 0.3)
                    valid = np.where(np.isfinite(vals_))[0]
                    if len(valid) == 0:
                        continue
                    x = np.arange(len(vals_))
                    plt.plot(x[valid], vals_[valid], c=c_, linewidth=1)
                plt.ylabel(stat)
                plt.xlabel("epoch")
                plt.gca().yaxis.label.set_color(c[0:3] * 0.75)
                plt.legend(tmodes)
                gcolor = np.array(mcolors.to_rgba("lightgray"))
                grid_params = {"visible": True, "color": gcolor}
                plt.grid(**grid_params, which="major", linestyle="-", linewidth=0.4)
                plt.grid(**grid_params, which="minor", linestyle="--", linewidth=0.2)
                plt.minorticks_on()

            plt.tight_layout()
            plt.show()
            try:
                fig.savefig(plot_file)
            except PermissionError:
                warnings.warn("Cant dump stats due to insufficient permissions!")


def rotation_angle(rot_gt, rot_pred, batch_size=None):
    # rot_gt, rot_pred (B, 3, 3)
    # masks_flat: B, 1
    rel_angle_cos = so3_relative_angle(rot_gt, rot_pred, eps=1e-4)
    rel_rangle_deg = rel_angle_cos * 180 / np.pi

    if batch_size is not None:
        rel_rangle_deg = rel_rangle_deg.reshape(batch_size, -1)

    return rel_rangle_deg


def translation_angle(tvec_gt, tvec_pred, batch_size=None):
    rel_tangle_deg = evaluate_translation_batch(tvec_gt, tvec_pred)
    rel_tangle_deg = rel_tangle_deg * 180.0 / np.pi

    if batch_size is not None:
        rel_tangle_deg = rel_tangle_deg.reshape(batch_size, -1)

    return rel_tangle_deg


def evaluate_translation_batch(t_gt, t, eps=1e-15, default_err=1e6):
    """Normalize the translation vectors and compute the angle between them."""
    t_norm = torch.norm(t, dim=1, keepdim=True)
    t = t / (t_norm + eps)

    t_gt_norm = torch.norm(t_gt, dim=1, keepdim=True)
    t_gt = t_gt / (t_gt_norm + eps)

    loss_t = torch.clamp_min(1.0 - torch.sum(t * t_gt, dim=1) ** 2, eps)
    err_t = torch.acos(torch.sqrt(1 - loss_t))

    err_t[torch.isnan(err_t) | torch.isinf(err_t)] = default_err
    return err_t


def batched_all_pairs(B, N):
    # B, N = se3.shape[:2]
    i1_, i2_ = torch.combinations(
        torch.arange(N), 2, with_replacement=False
    ).unbind(-1)
    i1, i2 = [
        (i[None] + torch.arange(B)[:, None] * N).reshape(-1)
        for i in [i1_, i2_]
    ]

    return i1, i2


class WarmupCosineRestarts(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_0, iters_per_epoch, T_mult=1, eta_min=0, warmup_ratio=0.1, warmup_lr_init=1e-7,
                 last_epoch=-1):
        self.T_0 = T_0 * iters_per_epoch  # 50 * 156988
        self.T_mult = T_mult  # 1
        self.eta_min = eta_min  # 0
        self.warmup_iters = int(T_0 * warmup_ratio * iters_per_epoch)  # int(50 * 156988 * 0.1 * 156988)
        self.warmup_lr_init = warmup_lr_init  # 1e-7
        super(WarmupCosineRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_mult == 1:
            i_restart = self.last_epoch // self.T_0  # 算出是否需要restart
            T_cur = self.last_epoch - i_restart * self.T_0  #
        else:
            n = int(math.log((self.last_epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
            T_cur = self.last_epoch - self.T_0 * (self.T_mult ** n - 1) // (self.T_mult - 1)

        if T_cur < self.warmup_iters:
            warmup_ratio = T_cur / self.warmup_iters
            return [self.warmup_lr_init + (base_lr - self.warmup_lr_init) * warmup_ratio for base_lr in self.base_lrs]
        else:
            T_cur_adjusted = T_cur - self.warmup_iters
            T_i = self.T_0 - self.warmup_iters
            return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * T_cur_adjusted / T_i)) / 2
                    for base_lr in self.base_lrs]


class WarmupCosineLR(toptim._LRScheduler):
    ''' Warmup learning rate scheduler.
        Initially, increases the learning rate from 0 to the final value, in a
        certain number of steps. After this number of steps, each step decreases
        LR exponentially.
    '''

    def __init__(self, optimizer, lr, warmup_steps, momentum, max_steps):
        # cyclic params
        self.optimizer = optimizer
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.momentum = momentum

        # cap to one
        if self.warmup_steps < 1:
            self.warmup_steps = 1

        # cyclic lr
        self.cosine_scheduler = toptim.CosineAnnealingLR(
            self.optimizer, T_max=max_steps)

        self.initial_scheduler = toptim.CyclicLR(self.optimizer,
                                                 base_lr=0,
                                                 max_lr=self.lr,
                                                 step_size_up=self.warmup_steps,
                                                 step_size_down=self.warmup_steps,
                                                 cycle_momentum=False,
                                                 base_momentum=self.momentum,
                                                 max_momentum=self.momentum)

        self.last_epoch = -1
        self.finished = False
        super().__init__(optimizer)

    def step(self, epoch=None):
        if self.finished or self.initial_scheduler.last_epoch >= self.warmup_steps:
            if not self.finished:
                self.base_lrs = [self.lr for lr in self.base_lrs]
                self.finished = True
            return self.cosine_scheduler.step(epoch)
        else:
            return self.initial_scheduler.step(epoch)

# 训练策略
POWER = 0.9


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** power)


def lr_warmup(base_lr, iter, max_iter, warmup_iter):
    return base_lr * (float(iter) / warmup_iter)


def adjust_learning_rate(lr, i_iter, max_iter, PREHEAT_STEPS):
    if i_iter < PREHEAT_STEPS:
        lr = lr_warmup(lr, i_iter, max_iter, PREHEAT_STEPS)
    else:
        lr = lr_poly(lr, i_iter, max_iter, POWER)

    return lr


class DynamicBatchSampler(BatchSampler):
    def __init__(self, num_sequences, dataset_len=1024, max_images=128, images_per_seq=(3, 20)):
        # len(dataset): 32; cfg.train.len_train: 16384; max_image: 512; images_per_seq: [3, 51]
        # self.dataset = dataset
        self.max_images = max_images
        self.images_per_seq = list(range(images_per_seq[0], images_per_seq[1]))
        self.num_sequences = num_sequences
        self.dataset_len = dataset_len

    def _capped_random_choice(self, x, size, replace: bool = True):
        len_x = x if isinstance(x, int) else len(x)
        if replace:
            return np.random.choice(x, size=size, replace=len_x < size)
        else:
            return np.random.choice(x, size=min(size, len_x), replace=False)

    def __iter__(self):
        for batch_idx in range(self.dataset_len):
            # NOTE batch_idx is never used later
            # print(f"process {batch_idx}")
            n_per_seq = np.random.choice(self.images_per_seq)
            n_seqs = (self.max_images // n_per_seq)

            chosen_seq = self._capped_random_choice(self.num_sequences, n_seqs)
            # print(f"get the chosen_seq for {batch_idx}")

            batches = [(bidx, n_per_seq) for bidx in chosen_seq]
            # print(f"yield the batches for {batch_idx}")
            yield batches

    def __len__(self):
        return self.dataset_len


class FixBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, dataset_len=1024, batch_size=64, max_images=128, images_per_seq=(3, 20)):
        # dataset; len_train: 16384; max_images: 512; images_per_seq: [3, 51]
        self.dataset = dataset
        self.max_images = max_images  # 32
        self.images_per_seq = list(range(images_per_seq[0], images_per_seq[1]))  # [3, ..., 21]
        self.num_sequences = len(self.dataset)  # 4
        self.dataset_len = dataset_len  # 156988
        self.batch_size = 48 # bath_size
        self.fix_images_per_seq = True

    def _capped_random_choice(self, x, size, replace: bool = True):
        len_x = x if isinstance(x, int) else len(x)
        if replace:
            return np.random.choice(x, size=size, replace=len_x < size)
        else:
            return np.random.choice(x, size=min(size, len_x), replace=False)

    def __iter__(self):
        for batch_idx in range(self.dataset_len):
            # NOTE batch_idx is never used later
            # print(f"process {batch_idx}")
            if self.fix_images_per_seq:
                # n_per_seq = 12
                n_per_seq = 1
            else:
                n_per_seq = np.random.choice(self.images_per_seq)

            n_seqs = self.batch_size

            chosen_seq = self._capped_random_choice(self.num_sequences, n_seqs)
            # print(f"get the chosen_seq for {batch_idx}")

            batches = [(bidx, n_per_seq) for bidx in chosen_seq]
            # print(f"yield the batches for {batch_idx}")
            yield batches

    def __len__(self):
        return self.dataset_len


def find_last_checkpoint(
        exp_dir, any_path: bool = False, all_checkpoints: bool = False
):
    if any_path:
        exts = [".pth", "_stats.jgz", "_opt.pth"]
    else:
        exts = [".pth"]

    for ext in exts:
        fls = sorted(
            glob.glob(
                os.path.join(glob.escape(exp_dir), "model_epoch_" + "[0-9]" * 8 + ext)
            )
        )
        if len(fls) > 0:
            break
    # pyre-fixme[61]: `fls` is undefined, or not always defined.
    if len(fls) == 0:
        fl = None
    else:
        if all_checkpoints:
            # pyre-fixme[61]: `fls` is undefined, or not always defined.
            fl = [f[0: -len(ext)] + ".pth" for f in fls]
        else:
            # pyre-fixme[61]: `ext` is undefined, or not always defined.
            fl = fls[-1][0: -len(ext)] + ".pth"

    return fl
