_base_ = [
    '../_base_/datasets/aitod_detection.py',
    # '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
]
# optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)
# , lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, 
# optimizer = dict(type='Adam', lr=0.001, weight_decay=0.0001 )

# checkpoint = 'https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b3_3rdparty_8xb32-aa_in1k_20220119-5b4887a0.pth'  # noqa
# optimizer = dict(type='Adam', lr=0.001, weight_decay=0.0001 )
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    # optimizer=optimizer,
    optimizer=dict(type='SGD', lr=0.02/4, momentum=0.9, weight_decay=0.0001)
    )

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
# auto_scale_lr = dict(enable=False, base_batch_size=16)



checkpoint = 'https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b0_3rdparty_8xb32-aa-advprop_in1k_20220119-26434485.pth' 

norm_cfg = dict(type='SyncBN', requires_grad=True, eps=1e-3, momentum=0.01)


model = dict(
    type='FasterRCNN',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),

    backbone=dict(
        type='EfficientNet',
        arch='b0',
        drop_path_rate=0.2,
        out_indices=(3, 4, 5),
        frozen_stages=0,
        conv_cfg=dict(type='Conv2dSamePadding'),
        norm_cfg=norm_cfg,
        norm_eval=True,
        init_cfg=dict(
            type='Pretrained', prefix='backbone', checkpoint=checkpoint)),

    neck=dict(
        type='BiFPN',
        num_stages=3,
        in_channels=[40, 112, 320],
        out_channels=64,
        start_level=0,
        norm_cfg=norm_cfg),

    # neck=dict(
    #     type='FPN',
    #     # in_channels=[256, 512, 1024, 2048],
    #     in_channels=[40, 112, 320],
    #     out_channels=64,
    #     num_outs=5),

    rpn_head=dict(
        type='RPNHead',
        in_channels=64,
        feat_channels=64,
        anchor_generator=dict(
            type='RFGenerator', # Effective Receptive Field as prior
            fpn_layer='p2', # bottom FPN layer P2
            fraction=0.5, # the fraction of ERF to TRF
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=1.5,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='HuberLoss', beta=0.1, loss_weight=50)),
        # loss_cls=dict(
        #     type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        # loss_bbox=dict(type='L1Loss', loss_weight=1.0)),

        
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=64,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=64,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=8,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='HieAssigner'        , # Hierarchical Label Assigner (HLA)
                ignore_iof_thr=-1,
                gpu_assign_thr=512,
                iou_calculator=dict(type='BboxDistanceMetric'),
                assign_metric='kl', # KLD as RFD for label assignment
                topk=[3,1],
                ratio=0.9), # decay factor
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=3000,
            max_per_img=3000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1,
                gpu_assign_thr=512),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=3000,
            max_per_img=3000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=3000)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ))

#fp16 = dict(loss_scale=512.)


# learning policy
checkpoint_config = dict(interval=4)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=5000,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=100)
# evaluation = dict(interval=12, metric='bbox')