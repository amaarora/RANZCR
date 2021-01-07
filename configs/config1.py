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
    'batch_size': 64,

    # training
    'gpus': [0],
    'distributed_strategy': 'ddp',
    'debug': False,
    'num_workers': 8,
    'learning_rate': 1e-3,
    'gradient_accumulation_steps': 1,
    'precision': 16,

    # model checkpointing
    'save_checkpoint': True,
    'model_dir': Path('../data/usr/models'),
    'save_top_k': 1,
    'save_weights_only': False,
    'log_artifact': False,

    # model
    'backbone': 'efficientnet_b0',
    'num_epochs': 15,
    'pretrained_path': None,

    # scheduler
    'scheduler': {
        "method": "cosine",
        "warmup_epochs": 2
    },
    'monitor_lr': True,

    # lightning
    'logger': 'neptune',
    'project_name': 'amaarora/RANZCR',
    'distributed_backend': 'ddp',
    'seed': 42,
    'deterministic': True,

    # augmentation
    'img_size': 256,
    'smallest_side': 288,
    'random_scale': 0.05,
    'rotate': 50,
    'random_brightness': 0.15,
    'random_contrast': 0.1,
    'shear': 0.1,
    'cutout': 16
}

args['train_aug'] = A.Compose([
    A.RandomScale(args['random_scale']),
    A.Rotate(args['rotate']),
    A.RandomBrightnessContrast(args['random_brightness'], args['random_contrast']),
    A.Flip(),
    A.IAAAffine(shear=args['shear']),
    A.SmallestMaxSize(args['smallest_side']),
    A.RandomCrop(args['img_size'], args['img_size']),
    A.OneOf([
        A.Cutout(random.randint(1, 8), args['cutout'], args['cutout']),
        A.CoarseDropout(random.randint(1, 8), args['cutout'], args['cutout'])
    ]),
    A.Normalize(always_apply=True)
])

args['val_aug'] = A.Compose([
    A.SmallestMaxSize(args['smallest_side']),
    A.CenterCrop(args['img_size'], args['img_size']),
    A.Normalize(always_apply=True)
])
