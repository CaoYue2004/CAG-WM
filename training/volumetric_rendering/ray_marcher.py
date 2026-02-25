# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
The ray marcher takes the raw output of the implicit representation and uses the volume rendering equation to produce composited colors and depths.
Based off of the implementation in MipNeRF (this one doesn't do any cone tracing though!)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MipRayMarcher2(nn.Module):
    def __init__(self):
        super().__init__()


    def run_forward(self, colors, densities, depths, rendering_options):
        # print('colors',colors.shape) # [2,4096,48,32]
        # print('densities',densities.shape)# [2,4096,48,1]
        # print('depths',depths.shape)# [2,4096,48,1]
        # depths 沿射线近似均匀分布的样本的深度
        deltas = depths[:, :, 1:] - depths[:, :, :-1]   # 光线在每个深度采样点之间的步�?        # 相邻采样点的平均�?        
        colors_mid = (colors[:, :, :-1] + colors[:, :, 1:]) / 2
        densities_mid = (densities[:, :, :-1] + densities[:, :, 1:]) / 2
        depths_mid = (depths[:, :, :-1] + depths[:, :, 1:]) / 2
        # print('deltas',deltas.shape) # [2,4096,95,1]
        # print('colors_mid',colors_mid.shape)# [2,4096,95,3]
        # print('densities_mid',densities_mid.shape)# [2,4096,95,1]
        # print('depths_mid', depths_mid.shape)  # [2,4096,95,1]

        # print('colors.shape{}'.format(colors.shape))
        # print('densities.shape{}'.format(densities.shape))

        if rendering_options['clamp_mode'] == 'softplus':
            densities_mid = F.softplus(densities_mid - 1) # activation bias of -1 makes things initialize better
        else:
            assert False, "MipRayMarcher only supports `clamp_mode`=`softplus`!"

        density_delta = densities_mid * deltas
        # print('density_delta', density_delta.shape)  # [2,4096,95,1] or [2,4096,47,1]
        alpha = 1 - torch.exp(-density_delta)   # 不透明�?
        alpha_shifted = torch.cat([torch.ones_like(alpha[:, :, :1]), 1-alpha + 1e-10], -2)
        weights = alpha * torch.cumprod(alpha_shifted, -2)[:, :, :-1]   # 延光线方向利用透明度计算权�?        
        # print('weights', weights.shape)# [2,4096,95,1] or [2,4096,47,1]
        
        # print( torch.mean(colors_mid))
        # print(torch.mean(colors_mid).shape)
        
        
        # TODO: Remove Color
        colors_mid = torch.mean(colors_mid)*torch.ones(colors_mid.shape,device=colors_mid.device)
        
        
        
        composite_rgb = torch.sum(weights * colors_mid, -2)
        #composite_rgb = torch.sum(weights, -2)
        # print('composite_rgb', composite_rgb.shape)# [2,4096,32]
        # composite_density = torch.sum(weights * densities_mid, -2).repeat(1,1,32)
        # print('composite_density', composite_density.shape)# [2,4096,1]
        # composite_rgb = composite_density
        weight_total = weights.sum(2)
        composite_depth = torch.sum(weights * depths_mid, -2) / weight_total
        # print('colors_mid.shape{}'.format(colors_mid.shape))
        # print('depths_mid.shape{}'.format(depths_mid.shape))      

        # clip the composite to min/max range of depths
        composite_depth = torch.nan_to_num(composite_depth, float('inf'))
        composite_depth = torch.clamp(composite_depth, torch.min(depths), torch.max(depths))
        
        rgb_composite_depth = composite_depth.repeat(1,1,32)
        # print('rgb_composite_depth:{}'.format(rgb_composite_depth.shape))
        # print('composite_depth:{}'.format(composite_depth.shape))
        # print('composite_rgb:{}'.format(composite_rgb.shape))

        if rendering_options.get('white_back', False):
            composite_rgb = composite_rgb + 1 - weight_total

        composite_rgb = composite_rgb * 2 - 1 # Scale to (-1, 1)
        # print(f'composite_rgb={composite_rgb}')
        # print(f'composite_rgb.shape={composite_rgb.shape}')     # [1, 262144, 32]
        # print('composite_rgb')# [2, 4096, 32]
        # print(composite_rgb.shape)
        # TODO: Depth
        # composite_rgb = rgb_composite_depth
        
        return composite_rgb, composite_depth, weights


    def forward(self, colors, densities, depths, rendering_options):
        composite_rgb, composite_depth, weights = self.run_forward(colors, densities, depths, rendering_options)

        return composite_rgb, composite_depth, weights
