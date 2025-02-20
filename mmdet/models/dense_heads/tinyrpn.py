import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import normal_init

from mmdet.models.task_modules.coders.delta_xywh_bbox_coder import delta2bbox
from mmcv.ops import nms


from mmdet.registry import MODELS
from .anchor_head import AnchorHead
from mmdet.utils import InstanceList, MultiConfig, OptInstanceList

@MODELS.register_module()
class TinyRPN(AnchorHead):

    # def __init__(self, in_channels, **kwargs):
    def __init__(self,
                 in_channels: int,
                 num_classes: int = 1,
                 init_cfg: MultiConfig = dict(
                     type='Normal', layer='Conv2d', std=0.01),
                 num_convs: int = 1,
                 **kwargs):

        self.num_convs = num_convs

        if 'rpn_conv_groups' in kwargs:
            self.rpn_conv_groups = kwargs.pop('rpn_conv_groups')
        else:
            self.rpn_conv_groups = 1

        # assert num_classes == 1

        # super(TinyRPN, self).__init__(""" 8, in_channels,  """ **kwargs)
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            init_cfg=init_cfg,
            **kwargs)
        
        # super(TinyRPN, self).__init__(
        # num_classes: int,
        # in_channels: int,
        # feat_channels: int = 256,
        # anchor_generator: ConfigType = dict(
        #     type='AnchorGenerator',
        #     scales=[8, 16, 32],
        #     ratios=[0.5, 1.0, 2.0],
        #     strides=[4, 8, 16, 32, 64]),
        # bbox_coder: ConfigType = dict(
        #     type='DeltaXYWHBBoxCoder',
        #     clip_border=True,
        #     target_means=(.0, .0, .0, .0),
        #     target_stds=(1.0, 1.0, 1.0, 1.0)),
        # reg_decoded_bbox: bool = False,
        # loss_cls: ConfigType = dict(
        #     type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        # loss_bbox: ConfigType = dict(
        #     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        # train_cfg: OptConfigType = None,
        # test_cfg: OptConfigType = None,
        # init_cfg: OptMultiConfig = dict(
        #     type='Normal', layer='Conv2d', std=0.01)
   

    def _init_layers(self):


        self.rpn_conv = nn.Sequential(
            nn.Conv2d(
                self.in_channels, self.in_channels, 3, 
                padding=1, groups=self.in_channels),
            nn.Conv2d(
                self.in_channels, self.feat_channels, 1, groups=self.rpn_conv_groups))


        self.rpn_cls = nn.Conv2d(self.feat_channels,
                                 self.num_anchors * self.cls_out_channels, 1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        # normal_init(self.rpn_conv, std=0.01)
        # normal_init(self.rpn_cls, std=0.01)
        # normal_init(self.rpn_reg, std=0.01)

    def forward_single(self, x):

        x = self.rpn_conv(x)
        x = F.relu(x, inplace=True)
        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)
        return rpn_cls_score, rpn_bbox_pred

    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        losses = super(TinyRPN, self).loss(
            cls_scores,
            bbox_preds,
            gt_bboxes,
            None,
            img_metas,
            cfg,
            gt_bboxes_ignore=gt_bboxes_ignore)
        return dict(
            loss_rpn_cls=losses['loss_cls'], loss_rpn_bbox=losses['loss_bbox'])

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          mlvl_anchors,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        mlvl_proposals = []
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            anchors = mlvl_anchors[idx]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                scores = rpn_cls_score.softmax(dim=1)[:, 1]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
                _, topk_inds = scores.topk(cfg.nms_pre)
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]
                scores = scores[topk_inds]
            proposals = delta2bbox(anchors, rpn_bbox_pred, self.target_means,
                                   self.target_stds, img_shape)
            if cfg.min_bbox_size > 0:
                w = proposals[:, 2] - proposals[:, 0] + 1
                h = proposals[:, 3] - proposals[:, 1] + 1
                valid_inds = torch.nonzero((w >= cfg.min_bbox_size) &
                                           (h >= cfg.min_bbox_size)).squeeze()
                proposals = proposals[valid_inds, :]
                scores = scores[valid_inds]
            proposals = torch.cat([proposals, scores.unsqueeze(-1)], dim=-1)
            proposals, _ = nms(proposals, cfg.nms_thr)
            proposals = proposals[:cfg.nms_post, :]
            mlvl_proposals.append(proposals)
        proposals = torch.cat(mlvl_proposals, 0)
        if cfg.nms_across_levels:
            proposals, _ = nms(proposals, cfg.nms_thr)
            proposals = proposals[:cfg.max_num, :]
        else:
            scores = proposals[:, 4]
            num = min(cfg.max_num, proposals.shape[0])
            _, topk_inds = scores.topk(num)
            proposals = proposals[topk_inds, :]
        return proposals
