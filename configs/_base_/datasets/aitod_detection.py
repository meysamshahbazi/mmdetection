dataset_type = 'AITODDataset'
data_root = 'data/AI-TOD/aitod/'

image_size = (800, 800)

## version 3
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=image_size, keep_ratio=True),
    # dict(
    #     type='RandomCrop',
    #     crop_type='absolute_range',
    #     crop_size=image_size,
    #     allow_negative_crop=True),

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
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor')),]

## version 3
train_dataloader = dict(
    train_dataloader=12,
    num_workers=2,
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
    batch_size=32,
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
    batch_size=32,
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

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/small_test_v1_1.0.json',
    metric=['bbox'],
    format_only=False)
test_evaluator = val_evaluator
