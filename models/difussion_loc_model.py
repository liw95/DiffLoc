"""
@author: Wen Li
@file: diffusion_loc_model.py
@time: 2023/9/18 20:44
"""
import torch
import torch.nn as nn
import logging
from typing import Dict, Optional
from hydra.utils import instantiate

logger = logging.getLogger(__name__)


class DiffusionLocModel(nn.Module):
    def __init__(
            self,
            IMAGE_FEATURE_EXTRACTOR: Dict,
            DIFFUSER: Dict,
            DENOISER: Dict,
    ):
        """ Initializes a DiffusionLoc model
        Args:
            image_feature_extractor_cfg (Dict):
                Configuration for the image feature extractor.
            diffuser_cfg (Dict):
                Configuration for the diffuser.
            denoiser_cfg (Dict):
                Configuration for the denoiser.
        """

        super().__init__()

        self.image_feature_extractor = instantiate(
            IMAGE_FEATURE_EXTRACTOR, _recursive_=False
        )

        self.diffuser = instantiate(DIFFUSER, _recursive_=False)

        denoiser = instantiate(DENOISER, _recursive_=False)
        self.diffuser.model = denoiser

        self.target_dim = denoiser.target_dim

    def forward(
            self,
            image: torch.Tensor,
            pose: Optional[torch.Tensor] = None,
            sampling_timesteps = 10,
            training=True,
    ):
        """
        Forward pass of the PoseDiffusionModel.

        Args:
            image (torch.Tensor):
                Input image tensor, BxNx5xHxW.
            pose (Optional[CamerasBase], optional):
                Camera object. Defaults to None.
            training train or eval

        Return:
            Prected poses: BxNx6

        """
        shapelist = list(image.shape)
        batch_size = len(image)
        if training:
            reshaped_image = image.reshape(shapelist[0] * shapelist[1], *shapelist[2:])
            z, pred_mask = self.image_feature_extractor(reshaped_image)  # [B, N, 384] [B, N, 256, 1]
            z = z.reshape(batch_size, shapelist[1], -1)  # [B*N, C]   [B, N, C]
            pred_mask = pred_mask.squeeze(1).reshape(shapelist[0] * shapelist[1], shapelist[-2],
                                                     shapelist[-1])  # [B*N, 32, 512]
            diffusion_results = self.diffuser(pose, z=z)
            diffusion_results['pred_pose'] = diffusion_results["x_0_pred"]
            # SOAP mask
            diffusion_results['pred_mask'] = pred_mask

            return diffusion_results

        else:
            reshaped_image = image.reshape(shapelist[0] * shapelist[1], *shapelist[2:])
            z, pred_mask = self.image_feature_extractor(reshaped_image)
            z = z.reshape(batch_size, shapelist[1], -1)
            pred_mask = pred_mask.squeeze(1).reshape(shapelist[0] * shapelist[1], shapelist[-2], shapelist[-1])
            B, N, _ = z.shape

            target_shape = [B, N, self.target_dim]

            # sampling
            # ddpm
            # pred_pose, pred_pose_diffusion_samples = self.diffuser.sample(shape=target_shape, z=z)
            # ddim
            pred_pose, _ = self.diffuser.ddim_sample(shape=target_shape, z=z, sampling_timesteps=sampling_timesteps)
            diffusion_results = {
                "pred_pose": pred_pose,  # [B, N, 6]
                'pred_mask': pred_mask,  # [B, N, 256, 1]
                "z": z
            }

            return diffusion_results
