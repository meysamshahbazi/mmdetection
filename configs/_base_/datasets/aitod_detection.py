dataset_type = 'AITODDataset'
data_root = 'data/AI-TOD/aitod/'

image_size = (320, 320)

## version 3
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='Resize', scale=(800, 800), keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=image_size,
        allow_negative_crop=True),

    dict(type='RandomFlip', prob=0.5),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor')),
]
""" 
## version 2 
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(800, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg), # have merged to model.data_preprocessor
    dict(type='Pad', size_divisor=32), # have merged to model.data_preprocessor
    dict(type='DefaultFormatBundle'), 
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']), # merged to PackDetInputs
]
 """

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=image_size, keep_ratio=True),
    # dict(
    #     type='RandomCrop',
    #     crop_type='absolute_range',
    #     crop_size=image_size,
    #     allow_negative_crop=True),
    # dict(type='RandomFlip', prob=0.5),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor')),
    # dict(type='ImageToTensor', keys=['img']),
]
""" 
## version 2
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]  
"""

## version 3
train_dataloader = dict(
    batch_size=64,
    num_workers=12,
    persistent_workers=True,  # Avoid recreating subprocesses after each iteration
    sampler=dict(type='DefaultSampler', shuffle=True),  # Default sampler, supports both distributed and non-distributed training
    batch_sampler=dict(type='AspectRatioBatchSampler'),  # Default batch_sampler, used to ensure that images in the batch have similar aspect ratios, so as to better utilize graphics memory
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/small_trainval_v1_1.0.json',
        data_prefix=dict(img='images/trainval/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=0),
        pipeline=train_pipeline))
# In version 3.x, validation and test dataloaders can be configured independently



val_dataloader = dict(
    batch_size=96,
    num_workers=20,
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
    batch_size=96,
    num_workers=20,
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

## version 2
# data = dict(
#     samples_per_gpu=2,
#     workers_per_gpu=2,
#     train=dict(
#         type=dataset_type,
#         ann_file=data_root + 'annotations/small_trainval_v1_1.0.json',
#         img_prefix=data_root + 'images/trainval/',
#         pipeline=train_pipeline),
#     val=dict(
#         type=dataset_type,
#         ann_file=data_root + 'annotations/small_test_v1_1.0.json',
#         img_prefix=data_root + 'images/test/',
#         pipeline=test_pipeline),
#     test=dict(
#         type=dataset_type,
#         ann_file=data_root + 'annotations/small_test_v1_1.0.json',
#         img_prefix=data_root + 'images/test/',
#         pipeline=test_pipeline))

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/small_test_v1_1.0.json',
    metric=['bbox'],
    format_only=False)
test_evaluator = val_evaluator

""" 
# version 2
evaluation = dict(interval=12, metric='bbox')
 """