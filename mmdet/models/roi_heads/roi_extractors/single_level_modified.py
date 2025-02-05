from __future__ import division

import torch
import torch.nn as nn
from typing import List, Optional, Tuple

from mmcv import ops

# from mmdet.core import force_fp32
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptMultiConfig

from .base_roi_extractor import BaseRoIExtractor
import torchvision
from mmcv.ops import roi_pool as roipool_mm

# from ps_roi_align_ori.ps_roi_align import PSROIAlign_ori
# roi_size=7, sampling_ratio=2, pooled_dim=5
torchvision.ops.PSRoIAlign
# torchvision.ops.ps_roi_align(input: Tensor, boxes: Tensor, output_size: int, spatial_scale: float = 1.0, sampling_ratio: int = - 1) 

@MODELS.register_module()
class SingleRoIExtractorModified(BaseRoIExtractor):
    """Extract RoI features from a single level feature map.

    If there are mulitple input feature levels, each RoI is mapped to a level
    according to its scale.

    Args:
        roi_layer (dict): Specify RoI layer type and arguments.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (int): Strides of input feature maps.
        finest_scale (int): Scale threshold of mapping to level 0.
    """

    def __init__(self,
                 roi_layer: ConfigType,
                 out_channels: int,
                 featmap_strides: List[int],
                 finest_scale: int = 56,

                 cfg_roi_scale_factor=None,
                 out_size=(7, 7),####
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            roi_layer=roi_layer,
            out_channels=out_channels,
            featmap_strides=featmap_strides,
            init_cfg=init_cfg)
        
        self.finest_scale = finest_scale
        self.cfg_roi_scale_factor = cfg_roi_scale_factor
        self.out_size = out_size####
        self.roi_layers = self.build_roi_layers(roi_layer, featmap_strides)
        self.out_channels = out_channels
        self.featmap_strides = featmap_strides
        self.finest_scale = finest_scale
        self.fp16_enabled = False
        
        self.cfg_roi_scale_factor = cfg_roi_scale_factor
        self.out_size = out_size####
    
    # def __init__(self,
    #              roi_layer,
    #              out_channels,
    #              featmap_strides,
    #              finest_scale=56,
    #              cfg_roi_scale_factor=None,
    #              out_size=(7, 7),####
    #             ):
    #     super(SingleRoIExtractorModified, self).__init__()
    #     self.roi_layers = self.build_roi_layers(roi_layer, featmap_strides)
    #     self.out_channels = out_channels
    #     self.featmap_strides = featmap_strides
    #     self.finest_scale = finest_scale
    #     self.fp16_enabled = False
        
    #     self.cfg_roi_scale_factor = cfg_roi_scale_factor
    #     self.out_size = out_size####
    @property
    def num_inputs(self):
        """int: Input feature map levels."""
        return len(self.featmap_strides)

    def init_weights(self):
        pass

    def build_roi_layers(self, layer_cfg, featmap_strides):
        print("--------------------------")
        print(layer_cfg)
        cfg = layer_cfg.copy()
        layer_type = cfg.pop('type')

        print(layer_type)
        # assert hasattr(ops, layer_type)
        # layer_cls = getattr(ops, layer_type)

        # if isinstance(layer_type, str):
        #     print("STRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRrr")
        #     assert hasattr(ops, layer_type)
        #     layer_cls = getattr(ops, layer_type)
        # else:
        #     layer_cls = layer_type

        # self,
        # output_size: int,
        # spatial_scale: float,
        # sampling_ratio: int,

        roi_layers = nn.ModuleList([
                torchvision.ops.PSRoIAlign(output_size=layer_cfg['roi_size'], 
                                           spatial_scale=1 / s,
                                           sampling_ratio=layer_cfg['sampling_ratio']) for s in featmap_strides
                                           ])
        return roi_layers

    def map_roi_levels(self, rois, num_levels):
        """Map rois to corresponding feature levels by scales.

        - scale < finest_scale * 2: level 0
        - finest_scale * 2 <= scale < finest_scale * 4: level 1
        - finest_scale * 4 <= scale < finest_scale * 8: level 2
        - scale >= finest_scale * 8: level 3

        Args:
            rois (Tensor): Input RoIs, shape (k, 5).
            num_levels (int): Total level number.

        Returns:
            Tensor: Level index (0-based) of each RoI, shape (k, )
        """
        scale = torch.sqrt(
            (rois[:, 3] - rois[:, 1] + 1) * (rois[:, 4] - rois[:, 2] + 1))
        target_lvls = torch.floor(torch.log2(scale / self.finest_scale + 1e-6))
        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
        return target_lvls

    def roi_rescale(self, rois, scale_factor):
        cx = (rois[:, 1] + rois[:, 3]) * 0.5
        cy = (rois[:, 2] + rois[:, 4]) * 0.5
        w = rois[:, 3] - rois[:, 1] + 1
        h = rois[:, 4] - rois[:, 2] + 1
        new_w = w * scale_factor
        new_h = h * scale_factor
        x1 = cx - new_w * 0.5 + 0.5
        x2 = cx + new_w * 0.5 - 0.5
        y1 = cy - new_h * 0.5 + 0.5
        y2 = cy + new_h * 0.5 - 0.5
        new_rois = torch.stack((rois[:, 0], x1, y1, x2, y2), dim=-1)
        return new_rois

    # @force_fp32(apply_to=('feats', ), out_fp16=True)
    def forward(self, feats, rois, roi_scale_factor=None):

        if len(feats) == 1:
            return self.roi_layers[0](feats[0], rois)


        num_levels = len(feats)
        target_lvls = self.map_roi_levels(rois, num_levels)
        roi_feats = feats[0].new_zeros(
            rois.size(0), self.out_channels, *self.out_size)

        if self.cfg_roi_scale_factor is not None:
            rois = self.roi_rescale(rois, self.cfg_roi_scale_factor)


        if roi_scale_factor is not None:
            rois = self.roi_rescale(rois, roi_scale_factor)
        for i in range(num_levels):
            inds = target_lvls == i
            if inds.any():
                rois_ = rois[inds, :]

                roi_feats_t = self.roi_layers[i](feats[i], rois_)
                roi_feats[inds] = roi_feats_t
        return roi_feats






