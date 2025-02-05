_base_ = [
    # '../_base_/datasets/aitod_detection.py',
    # '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
]

total_epochs = 600


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


lr_config = dict(
    policy='cosine',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    target_lr=1e-6)
# optimizer
# optimizer=dict(
#         type='AdamW',
#         lr=0.00001,
#         betas=(0.9, 0.999),
#         weight_decay=0.05)

optimizer = dict(type='SGD',lr=0.05, momentum=0.9, weight_decay=1e-5)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer)

auto_scale_lr = dict(enable=False, base_batch_size=32)

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True, eps=1e-3, momentum=0.01)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=total_epochs, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

model = dict(
    type='TinyHRDet',
    init_cfg=dict(type='Pretrained', checkpoint='pth_file/tinydet_L.pth'),
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='MobileNetV3BC',
        with_fc=False,
        out_indices=[5, 8, 14, 17, 19],
        norm_eval=False,
        norm_eps=1e-5,
        # norm_cfg=norm_cfg,
        num_classes=1000,
        add_extra_stages=True,
        pretrained= 'pth_file/mobilenetv3_bc.pt',
        ), 

    neck=dict(
        type='TinyFPN',
        in_channels=[36, 60, 112, 160, 160],
        out_channels=245,
        num_outs=5,
        # act_cfg=None,
        fpn_conv_groups=[49, 7, 5, 1, 1],
        with_shuffle=False),
        
        
    rpn_head=dict(
        type='TinyRPN',
        in_channels=245,
        feat_channels=245,
        # num_classes=8,
        # anchor_scales=[2.25],
        # anchor_ratios=[0.5, 1.0, 2.0],
        # anchor_strides=[8, 16, 32, 64],
        anchor_generator=dict(
            type='RFGenerator', # Effective Receptive Field as prior
            fpn_layer='p2', # bottom FPN layer P2
            fraction=0.5, # the fraction of ERF to TRF
            scales=[5.12],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64],
            ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        rpn_conv_groups=49,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        # loss_bbox=dict(type='L1Loss', loss_weight=8.0)
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=8.0)
        ),
    
    roi_head=dict(
        type='StandardRoIHead',

        # bbox_roi_extractor=dict(
        #     type='SingleRoIExtractor',
        #     roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
        #     out_channels=5,
        #     featmap_strides=[8, 16],
        #     finest_scale=30
        #     ),
            
        
        bbox_roi_extractor=dict(
            type='SingleRoIExtractorModified',
            roi_layer=dict(type='PSROIAlign_ori', roi_size=7, sampling_ratio=2, pooled_dim=5),
            out_channels=5,             
            featmap_strides=[4, 8, 16],
            finest_scale=35.8),

        bbox_head=dict(
            type='SharedFCBBoxHeadModified',
            num_fcs=1,
            in_channels=5,
            fc_out_channels=1024,
            roi_feat_size=7,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=True,
            num_classes=81,
            # target_means=[0., 0., 0., 0.],
            # target_stds=[0.1, 0.1, 0.2, 0.2],
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            # loss_bbox=dict(type='L1Loss', loss_weight=8.0)
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=8.0)
            ),
    ),
        
    train_cfg = dict(
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
            allowed_border=0,
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
                ignore_iof_thr=-1),
            sampler=dict(
                type='OHEMSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg = dict(

        rpn=dict(
            nms_pre=3000,
            max_per_img=3000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05, 
            # nms=dict(type='nms', iou_threshold=0.5),
            nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05),
            max_per_img=100,
        )
    )

)

image_size = (512, 512)


dataset_type = 'CocoDataset'
data_root = 'data/coco/'

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection/coco/'

# Method 2: Use `backend_args`, `file_client_args` in versions before 3.0.0rc6
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))
backend_args = None

# train_pipeline = [
#     dict(type='LoadImageFromFile', backend_args=backend_args),
#     dict(type='LoadAnnotations', with_bbox=True),

#     dict(
#         type='PhotoMetricDistortion',
#         brightness_delta=32,
#         contrast_range=(0.5, 1.5),
#         saturation_range=(0.5, 1.5),
#         hue_delta=18),
#     dict(
#         type='Expand',
#         mean=[123.675, 116.28, 103.53],
#         to_rgb=True,
#         ratio_range=(1, 4)),
#     dict(
#         type='MinIoURandomCrop',
#         min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
#         min_crop_size=0.3),

#     dict(type='Resize', scale=image_size, keep_ratio=False),
#     dict(type='RandomFlip', prob=0.5),
#     dict(type='PackDetInputs')
# ]
# test_pipeline = [
#     dict(type='LoadImageFromFile', backend_args=backend_args),
#     dict(type='Resize', scale=image_size, keep_ratio=False),
#     # If you don't have a gt annotation, delete the pipeline
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(
#         type='PackDetInputs',
#         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
#                    'scale_factor'))
# ]


backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),

    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='Expand',
        mean=[123.675, 116.28, 103.53],
        to_rgb=True,
        ratio_range=(1, 4)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),

    dict(type='Resize', scale=image_size, keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=image_size, keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]


train_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val2017.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator


val_cfg = dict(type='ValLoop')



optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy


runner = dict(type='EpochBasedRunner', max_epochs=100)
# evaluation = dict(interval=12, metric='bbox')
                     
checkpoint_config = dict(interval=12)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TensorboardLoggerHook')
        # dict(type='TextLoggerHook'),
    ])
# yapf:enable
evaluation = dict(interval=12)
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'

load_from = None
resume_from = None
workflow = [ ('train', 1)]
  
 