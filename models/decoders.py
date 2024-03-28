# Copyright 2023 - Valeo Comfort and Driving Assistance
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

from models.model_utils import get_grid_size_1d, get_grid_size_2d, init_weights


class DecoderLinear(nn.Module):
    # From R. Strudel et al.
    # https://github.com/rstrudel/segmenter
    def __init__(self, n_cls, patch_size, d_encoder, patch_stride=None):
        super().__init__()

        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.n_cls = n_cls

        self.head = nn.Linear(self.d_encoder, n_cls)
        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(self, x, im_size, skip=None):
        H, W = im_size
        GS_H, GS_W = get_grid_size_2d(H, W, self.patch_size, self.patch_stride)
        x1 = self.head(x)
        x2 = rearrange(x1, 'b (h w) c -> b c h w', h=GS_H)
        return x1, x2
