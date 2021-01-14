import random
from pathlib import Path

import albumentations as A

args = {
    # data args
    'data_path': Path('../data/'),
    'train_path': Path('../data/train'),
    'test_path': Path('../data/test'),
    'csv_path': Path('../data/train.csv'),
    'kfold_csv_path': Path('../data/usr/train_folds.csv'),
    'num_class': 11,

    # train val split
    'n_fold': 5,
    'fold_id': 0,

    # data loader
    'batch_size': 32,

    # training
    'gpus': [0, 1, 2, 3],
    'distributed_strategy': 'ddp',
    'debug': False,
    'num_workers': 8,
    'learning_rate': 1e-3,
    'gradient_accumulation_steps': 2,
    'precision': 16,
    'calc_macro_auc': False,

    # model checkpointing
    'save_checkpoint': True,
    'model_dir': Path('../data/usr/models'),
    'save_top_k': 1,
    'save_weights_only': False,
    'log_artifact': False,

    # model
    'backbone': 'tf_efficientnet_b5',
    'num_epochs': 20,
    'pretrained_path': None,

    # scheduler
    'scheduler': {
        "method": "cosine",
        "warmup_epochs": 3
    },
    'monitor_lr': True,

    # lightning
    'logger': 'neptune',
    'project_name': 'amaarora/RANZCR',
    'distributed_backend': 'ddp',
    'seed': 42,
    'deterministic': True,

    # augmentation
    'img_size': 512,
    'smallest_side': 576,
    'random_scale': 0.05,
    'rotate': 50,
    'random_brightness': 0.15,
    'random_contrast': 0.1,
    'shear': 0.1,
    'cutout': 16
}

args['train_aug'] = A.Compose([
    A.RandomResizedCrop(args['img_size'], args['img_size'], scale=(0.9, 1), p=1),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.7),
    A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.7),
    A.CLAHE(clip_limit=(1, 4), p=0.5),
    A.OneOf([
        A.OpticalDistortion(distort_limit=1.0),
        A.GridDistortion(num_steps=5, distort_limit=1.),
        A.ElasticTransform(alpha=3),
    ],
            p=0.2),
    A.OneOf([
        A.GaussNoise(var_limit=[10, 50]),
        A.GaussianBlur(),
        A.MotionBlur(),
        A.MedianBlur(),
    ], p=0.2),
    A.Resize(args['img_size'], args['img_size']),
    A.OneOf([
        A.JpegCompression(),
        A.Downscale(scale_min=0.1, scale_max=0.15),
    ], p=0.2),
    A.IAAPiecewiseAffine(p=0.2),
    A.IAASharpen(p=0.2),
    A.Cutout(max_h_size=int(args['img_size'] * 0.1), max_w_size=int(args['img_size'] * 0.1), num_holes=5, p=0.5),
    A.Normalize(),
])

args['val_aug'] = A.Compose([
    A.SmallestMaxSize(args['smallest_side']),
    A.CenterCrop(args['img_size'], args['img_size']),
    A.Normalize(always_apply=True)
])
