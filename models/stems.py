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

from .model_utils import get_grid_size_1d, get_grid_size_2d


class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, patch_stride, embed_dim, channels, resize_emb=True):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        if patch_stride is None:
            patch_stride = patch_size
        else:
            if isinstance(patch_stride, int):
                patch_stride = (patch_stride, patch_stride)
        assert isinstance(patch_size, (list, tuple))
        assert isinstance(patch_stride, (list, tuple))
        assert len(patch_stride) == 2
        assert len(patch_size) == 2
        patch_size = tuple(patch_size)
        patch_stride = tuple(patch_stride)

        self.image_size = image_size
        if image_size[0] % patch_size[0] != 0 or image_size[1] % patch_size[1] != 0:
            raise ValueError('image dimensions must be divisible by the patch size')
        self.grid_size = (
            get_grid_size_1d(image_size[0], patch_size[0], patch_stride[0]),
            get_grid_size_1d(image_size[1], patch_size[1], patch_stride[1]))

        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.proj = nn.Conv2d(
            channels, embed_dim, kernel_size=patch_size, stride=patch_stride
        )
        self.resize_emb = resize_emb

    def get_grid_size(self, H, W):
        return get_grid_size_2d(H, W, self.patch_size, self.patch_stride)

    def forward(self, im):
        B, C, H, W = im.shape
        if self.resize_emb:
            x = self.proj(im).flatten(2).transpose(1, 2) # shape: B, N, D
        else:
            x = self.proj(im) # shape: B, D, new_H, new_W
        return x, None


class ConvStem(nn.Module):
    def __init__(self,
                 in_channels=5,
                 base_channels=32,
                 img_size=(32, 384),
                 patch_stride=(2, 8),
                 embed_dim=384,
                 flatten=True,
                 hidden_dim=None):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = 2 * base_channels

        self.base_channels = base_channels
        self.dropout_ratio = 0.2

        # Build stem, similar to the design in https://github.com/TiagoCortinhal/SalsaNext
        self.conv_block = nn.Sequential(
            ResContextBlock(in_channels, base_channels),
            ResContextBlock(base_channels, base_channels),
            ResContextBlock(base_channels, base_channels),
            ResBlock(base_channels, hidden_dim, self.dropout_ratio, pooling=False, drop_out=False))

        assert patch_stride[0] % 2 == 0
        assert patch_stride[1] % 2 == 0

        kernel_size = (patch_stride[0] + 1, patch_stride[1] + 1)
        padding = (patch_stride[0] // 2, patch_stride[1] // 2)
        patch_stride = (patch_stride[0], patch_stride[1])
        self.proj_block = nn.Sequential(
             nn.AvgPool2d(kernel_size=kernel_size, stride=patch_stride, padding=padding),
             nn.Conv2d(hidden_dim, embed_dim, kernel_size=1))

        self.patch_stride = patch_stride
        self.patch_size = patch_stride
        self.grid_size = (img_size[0] // patch_stride[0], img_size[1] // patch_stride[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

    def get_grid_size(self, H, W):
        return get_grid_size_2d(H, W, self.patch_size, self.patch_stride)

    def forward(self, x):
        B, C, H, W = x.shape  # B, in_channels, image_size[0], image_size[1]
        x_base = self.conv_block(x) # B, hidden_dim, image_size[0], image_size[1]
        x = self.proj_block(x_base)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC

        return x, x_base


class ResContextBlock(nn.Module):
    # From T. Cortinhal et al.
    # https://github.com/TiagoCortinhal/SalsaNext
    def __init__(self, in_filters, out_filters):
        super(ResContextBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=1)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(out_filters, out_filters, (3, 3), padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, (3, 3), dilation=2, padding=2)
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)

        resA = self.conv2(shortcut)
        resA = self.act2(resA)
        resA1 = self.bn1(resA)

        resA = self.conv3(resA1)
        resA = self.act3(resA)
        resA2 = self.bn2(resA)

        output = shortcut + resA2
        return output


class ResBlock(nn.Module):
    # From T. Cortinhal et al.
    # https://github.com/TiagoCortinhal/SalsaNext
    def __init__(self, in_filters, out_filters, dropout_rate, kernel_size=(3, 3), stride=1,
                 pooling=True, drop_out=True):
        super(ResBlock, self).__init__()
        self.pooling = pooling
        self.drop_out = drop_out
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=stride)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(in_filters, out_filters, kernel_size=(3, 3), padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, kernel_size=(3, 3), dilation=2, padding=2)
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

        self.conv4 = nn.Conv2d(out_filters, out_filters, kernel_size=(2, 2), dilation=2, padding=1)
        self.act4 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm2d(out_filters)

        self.conv5 = nn.Conv2d(out_filters * 3, out_filters, kernel_size=(1, 1))
        self.act5 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm2d(out_filters)

        if pooling:
            self.dropout = nn.Dropout2d(p=dropout_rate)
            self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=2, padding=1)
        else:
            self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)

        resA = self.conv2(x)
        resA = self.act2(resA)
        resA1 = self.bn1(resA)

        resA = self.conv3(resA1)
        resA = self.act3(resA)
        resA2 = self.bn2(resA)

        resA = self.conv4(resA2)
        resA = self.act4(resA)
        resA3 = self.bn3(resA)

        concat = torch.cat((resA1, resA2, resA3), dim=1)
        resA = self.conv5(concat)
        resA = self.act5(resA)
        resA = self.bn4(resA)
        resA = shortcut + resA

        if self.pooling:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            resB = self.pool(resB)

            return resB, resA
        else:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            return resB
