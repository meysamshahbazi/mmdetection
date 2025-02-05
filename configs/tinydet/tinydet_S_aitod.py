_base_ = [
    # '../_base_/datasets/aitod_detection.py',
    # '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
]

# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=200, val_interval=5)
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
# optimizer=dict(
#         type='AdamW',
#         lr=0.0002,
#         betas=(0.9, 0.999),
#         weight_decay=0.05)

optimizer = dict(type='SGD', lr=0.35, momentum=0.9, weight_decay=1e-5)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer)

auto_scale_lr = dict(enable=False, base_batch_size=16)

total_epochs = 240


# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True, eps=1e-3, momentum=0.01)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=total_epochs, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

model = dict(
    type='TinyHRDet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='MobileNetV3D',
        with_fc=False,
        out_indices=[6, 12, 15, 16],
        norm_eval=False,
        norm_eps=1e-5,
        # norm_cfg=norm_cfg,
        num_classes=1000,
        add_extra_stages=True,
        pretrained= 'pth_file/mobilenetv3_d.pt',
        ), 

    neck=dict(
        type='TinyFPN',
        in_channels=[40, 112, 160, 160],
        out_channels=245,
        num_outs=4,
        # act_cfg=None,
        fpn_conv_groups=[7, 5, 1, 1],
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
            scales=[2.25],
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64],
            ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        # target_means=[.0, .0, .0, .0],
        # target_stds=[1.0, 1.0, 1.0, 1.0],
        rpn_conv_groups=49,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        # loss_bbox=dict(type='L1Loss', loss_weight=1.0)
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)
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
            featmap_strides=[8, 16],
            finest_scale=30),

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
            num_classes=9,
            # target_means=[0., 0., 0., 0.],
            # target_stds=[0.1, 0.1, 0.2, 0.2],
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            # loss_bbox=dict(type='L1Loss', loss_weight=1.0)
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)
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
            nms_pre=2000,
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
            nms_pre=2000,
            max_per_img=3000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.01, nms=dict(type='soft_nms', iou_thr=0.5), max_per_img=100) ),

)

    
dataset_type = 'AITODDataset'
data_root = 'data/AI-TOD/aitod/'

image_size = (320, 320)

## version 3
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomCrop',
        crop_type='absolute',
        crop_size=image_size,
        allow_negative_crop=False,
        recompute_bbox = True),

    dict(type='RandomFlip', prob=0.5),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor')),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=image_size, keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_type='absolute',
        crop_size=image_size,
        allow_negative_crop=False,
        recompute_bbox = True),

    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor')),]

## version 3
train_dataloader = dict(
    batch_size=160,
    num_workers=1,
    persistent_workers=True,  # Avoid recreating subprocesses after each iteration
    sampler=dict(type='DefaultSampler', shuffle=True),  # Default sampler, supports both distributed and non-distributed training
    batch_sampler=dict(type='AspectRatioBatchSampler'),  # Default batch_sampler, used to ensure that images in the batch have similar aspect ratios, so as to better utilize graphics memory
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/small_trainval_v1_1.0.json',
        data_prefix=dict(img='images/trainval/'),
        # filter_cfg=dict(filter_empty_gt=True, min_size=0),
        pipeline=train_pipeline))
# In version 3.x, validation and test dataloaders can be configured independently



val_dataloader = dict(
    batch_size=200,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/small_test_v1_1.0.json',
        data_prefix=dict(img='images/test/'),
        test_mode=True,
        pipeline=test_pipeline))

test_dataloader = dict(
    batch_size=200,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/small_test_v1_1.0.json',
        data_prefix=dict(img='images/test/'),
        test_mode=True,
        pipeline=test_pipeline))

val_cfg = dict(type='ValLoop')


val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/small_test_v1_1.0.json',
    metric=['bbox'],
    format_only=False)

test_evaluator = val_evaluator






optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='cosine',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    target_lr=1e-5)

runner = dict(type='EpochBasedRunner', max_epochs=100)
# evaluation = dict(interval=12, metric='bbox')
                     
checkpoint_config = dict(interval=12)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
# yapf:enable
evaluation = dict(interval=12)
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'

load_from = None
resume_from = None
workflow = [ ('train', 1)]
  
 