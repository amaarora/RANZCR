import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers.neptune import NeptuneLogger
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from torch.autograd import Variable
from transformers import (get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup)

from config import *
from dataset import Ranzcr_Dataset


def setup(args):
    # important for `ddp`
    seed_everything(args.seed)

    if args.kfold_csv_path:
        df = pd.read_csv(args.kfold_csv_path)
        df['file_path'] = str(args.train_path) + '/' + df.StudyInstanceUID + '.jpg'
    else:
        raise FileNotFoundError("Please provide file path for Kfolds csv.")

    train_df = df.query(f"fold!={args.fold_id}").reset_index(drop=True)
    val_df = df.query(f"fold=={args.fold_id}").reset_index(drop=True)

    return train_df, val_df


class Model(pl.LightningModule):

    def __init__(self, backbone, n_class, pretrained_path=None, num_train_steps=None, target_cols=None):
        super().__init__()
        self.num_train_steps = num_train_steps
        self.backbone = timm.create_model(model_name=backbone,
                                          pretrained=True if not pretrained_path else False,
                                          num_classes=n_class)
        if pretrained_path:
            self.load_state_dict(torch.load(pretrained_path))
            logging.info(f"pretrained weights loaded successfully from {pretrained_path}")
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.target_cols = target_cols

    def forward(self, x):
        return self.backbone(x)

    def step(self, batch):
        x, y = batch['image'], batch['label']
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        return loss, y, y_hat

    def training_step(self, batch, batch_idx):
        loss, y, y_hat = self.step(batch)
        self.log('train_loss', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        loss, y, y_hat = self.step(batch)
        self.log('val_loss', loss)
        return {'loss': loss, 'y': y.detach(), 'y_hat': y_hat.detach()}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        macro_auc, aucs = self.get_score(outputs)

        # log individual aucs
        for i, (auc, target_col) in enumerate(zip(aucs, target_cols)):
            self.log(f"auc_{target_col}", auc)

        print(f"Epoch {self.current_epoch} | Macro AUC :{macro_auc}")
        self.log('val_epoch_loss', avg_loss)
        self.log('macro_auc', macro_auc)

    def get_score(self, outputs):
        y = torch.cat([x['y'] for x in outputs])
        y_hat = torch.cat([x['y_hat'] for x in outputs])
        aucs = []
        for i in range(11):
            aucs.append(roc_auc_score(y[:, i].cpu().numpy(), y_hat[:, i].cpu().numpy()))
        return np.mean(aucs), aucs

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.learning_rate)
        if args.scheduler['method'] == 'cosine':
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.num_train_steps * args.scheduler["warmup_epochs"],
                num_training_steps=int(self.num_train_steps * (args.num_epochs)))
            return [self.optimizer], [{'scheduler': self.scheduler, 'interval': 'step'}]

        return [self.optimizer]


if __name__ == '__main__':
    train_df, val_df = setup(args)

    target_cols = train_df.columns[1:12]

    train_dataset = Ranzcr_Dataset(train_df, args.train_path, aug=args.train_aug)
    val_dataset = Ranzcr_Dataset(val_df, args.train_path, aug=args.val_aug)

    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   num_workers=args.num_workers,
                                                   shuffle=True,
                                                   batch_size=args.batch_size)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 num_workers=args.num_workers,
                                                 shuffle=False,
                                                 batch_size=args.batch_size)

    logging.info(f"train dataloader shape: {next(iter(train_dataloader))['image'].shape}")

    if args.distributed_strategy == "ddp":
        num_train_steps = math.ceil(len(train_dataloader) / (len(args.gpus) * args.gradient_accumulation_steps))
    else:
        num_train_steps = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)

    model = Model(backbone=args.backbone,
                  n_class=args.num_class,
                  pretrained_path=args.pretrained_path,
                  num_train_steps=num_train_steps,
                  target_cols=target_cols)

    if args.save_checkpoint:
        experiment_path = args.model_dir / args.experiment_name
        ckpt_save_path = str(experiment_path) + f'/checkpoint_fold_{args.fold_id}/'
        if not os.path.exists(ckpt_save_path):
            Path(ckpt_save_path).mkdir(parents=True, exist_ok=True)
        logging.info(f"Model checkpoint will be saved at {ckpt_save_path}")
        ckpt = ModelCheckpoint(dirpath=ckpt_save_path,
                               monitor='macro_auc',
                               filename=f'{args.backbone}_fold_{args.fold_id}_{args.img_size}_{args.img_size}' +
                               '_{epoch:02d}_{macro_auc:.4f}',
                               verbose=True,
                               mode='max',
                               period=1,
                               save_top_k=args.save_top_k,
                               save_last=True,
                               save_weights_only=args.save_weights_only)

    callbacks = None
    if args.monitor_lr:
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks = [lr_monitor]

    if args.logger == 'neptune':
        neptune_logger = NeptuneLogger(
            api_key=
            'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiOWQxZTIzMGUtYzJlYi00NTllLTkyMTEtMDA5MWQ1ODQ3ZmRlIn0=',
            project_name=args.project_name,
            experiment_name=args.experiment_name,
            params=args.__dict__,
            upload_source_files='*.py')

    trainer = pl.Trainer(default_root_dir=ckpt_save_path,
                         gpus=args.gpus,
                         max_epochs=args.num_epochs,
                         num_sanity_val_steps=1 if args.debug else 0,
                         logger=neptune_logger,
                         distributed_backend=args.distributed_backend,
                         checkpoint_callback=ckpt if args.save_checkpoint else None,
                         accumulate_grad_batches=args.gradient_accumulation_steps,
                         callbacks=callbacks,
                         deterministic=args.deterministic,
                         precision=args.precision)

    trainer.fit(model, train_dataloader, val_dataloader)

    if args.log_artifact:
        # Log best model checkpoints to Neptune
        neptune_logger.experiment.set_property('best_model_score', ckpt.best_model_score.tolist())
        for k in ckpt.best_k_models.keys():
            model_name = 'checkpoint/' + k.split('/')[-1]
            neptune_logger.experiment.log_artifact(k, model_name)
