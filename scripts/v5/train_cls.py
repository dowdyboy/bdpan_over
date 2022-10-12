import paddle
from paddle.io import DataLoader

import os
import argparse
from PIL import Image
import numpy as np

from dowdyboy_lib.paddle.trainer import Trainer, TrainerConfig

from bdpan_over.v4.dataset import OverDataset
from bdpan_over.v4.model import OverClassifyModelV4
from bdpan_over.v4.loss import ClassifyLoss


parser = argparse.ArgumentParser(description='over net train v4')
# data config
parser.add_argument('--train-data-dir', type=str, required=True, help='train data dir')
parser.add_argument('--val-data-dir', type=str, required=True, help='val data dir')
parser.add_argument('--img-size', type=int, default=1024, help='input img size')
parser.add_argument('--num-workers', type=int, default=4, help='num workers')
# optimizer config
parser.add_argument('--lr', type=float, default=1e-2, help='lr')
parser.add_argument('--use-scheduler', default=False, action='store_true', help='use schedule')
parser.add_argument('--use-warmup', default=False, action='store_true', help='use warmup')
parser.add_argument('--weight-decay', type=float, default=5e-5, help='model weight decay')
# train config
parser.add_argument('--epoch', type=int, default=10, help='epoch num')
parser.add_argument('--batch-size', type=int, default=2, help='batch size')
parser.add_argument('--out-dir', type=str, default='./output', help='out dir')
parser.add_argument('--resume', type=str, default=None, help='resume checkpoint')
parser.add_argument('--last-epoch', type=int, default=-1, help='last epoch')
parser.add_argument('--seed', type=int, default=2022, help='random seed')
parser.add_argument('--log-interval', type=int, default=500, help='log process')
parser.add_argument('--sync-bn', default=False, action='store_true', help='sync_bn')
parser.add_argument('--device', default=None, type=str, help='device')
args = parser.parse_args()


def build_data():
    train_dataset = OverDataset(
        data_dir=args.train_data_dir,
        data_size=5000,
        img_size=args.img_size,
        dense_crop_p=0.8,
        dense_crop_max_count=256,
        dense_crop_rate=0.05,
        flip_p=0.5,
        scale_p=0.2,
        use_hsv=True,
        cache_img_period=5,
        cache_max_size=128,
        is_val=False,
        is_to_tensor=True,
        no_limit_patch_pos_p=0.5,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, drop_last=False)
    val_dataset = OverDataset(
        data_dir=args.val_data_dir,
        data_size=500,
        img_size=args.img_size,
        dense_crop_p=0.5,
        dense_crop_max_count=128,
        dense_crop_rate=0.05,
        flip_p=None,
        scale_p=None,
        use_hsv=False,
        cache_img_period=2,
        cache_max_size=64,
        is_val=False,
        is_to_tensor=True,
        no_limit_patch_pos_p=None,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, drop_last=False)
    return train_loader, train_dataset, val_loader, val_dataset


def build_model():
    cls_model = OverClassifyModelV4()
    return cls_model


def build_optimizer(model: paddle.nn.Layer):
    lr = args.lr
    lr_scheduler = None
    if args.use_scheduler:
        lr = paddle.optimizer.lr.CosineAnnealingDecay(lr, args.epoch, last_epoch=args.last_epoch, verbose=True)
        lr_scheduler = lr
    if args.use_warmup:
        lr = paddle.optimizer.lr.LinearWarmup(lr, 10, args.lr * 0.1, args.lr, last_epoch=args.last_epoch, verbose=True)
        lr_scheduler = lr
    optimizer = paddle.optimizer.Adam(lr, parameters=model.parameters(), weight_decay=args.weight_decay)
    return optimizer, lr_scheduler


def train_step(trainer: Trainer, bat, bat_idx, global_step):
    [cls_model] = trainer.get_models()
    [cls_loss_func] = trainer.get_components()

    bat_im, bat_gt, bat_mask, bat_cls = bat

    pred_cls = cls_model(bat_im)
    pred_label = paddle.argmax(pred_cls, axis=1)
    acc_count = paddle.sum(pred_label == bat_cls).item()
    bat_count = bat_im.shape[0]
    loss = cls_loss_func(pred_cls, bat_cls)

    trainer.set_records({
        'train_loss': loss.item(),
        'train_acc_count': acc_count,
        'train_bat_count': bat_count,
    })
    trainer.log({
        'train_loss': loss.item(),
    }, global_step)

    if global_step % args.log_interval == 0:
        trainer.print(f'global step: {global_step}, loss_cls: {loss.item()} ')

    return loss


def val_step(trainer: Trainer, bat, bat_idx, global_step):
    [cls_model] = trainer.get_models()
    [cls_loss_func] = trainer.get_components()

    bat_im, bat_gt, bat_mask, bat_cls = bat

    pred_cls = cls_model(bat_im)
    pred_label = paddle.argmax(pred_cls, axis=1)
    acc_count = paddle.sum(pred_label == bat_cls).item()
    bat_count = bat_im.shape[0]
    loss = cls_loss_func(pred_cls, bat_cls)

    trainer.set_records({
        'val_loss': loss.item(),
        'val_acc_count': acc_count,
        'val_bat_count': bat_count,
    })
    trainer.log({
        'val_loss': loss.item(),
    }, global_step)

    return loss


def on_epoch_end(trainer: Trainer, ep):
    [cls_optimizer], \
    [cls_lr_scheduler] = trainer.get_optimizers()
    rec = trainer.get_records()
    val_acc_count = paddle.sum(rec['val_acc_count']).item()
    val_bat_count = paddle.sum(rec['val_bat_count']).item()
    val_acc = float(val_acc_count) / val_bat_count
    train_acc_count = paddle.sum(rec['train_acc_count']).item()
    train_bat_count = paddle.sum(rec['train_bat_count']).item()
    train_acc = float(train_acc_count) / train_bat_count
    ep_val_loss = paddle.mean(rec['val_loss']).item()
    ep_train_loss = paddle.mean(rec['train_loss']).item()
    lr = cls_optimizer.get_lr()
    trainer.log({
        'ep_val_loss': ep_val_loss,
        'ep_train_loss': ep_train_loss,
        'val_acc': val_acc,
        'train_acc': train_acc,
        'lr': lr,
    }, ep)
    trainer.print(f'ep_train_loss: {ep_train_loss} , ep_val_loss: {ep_val_loss}')
    trainer.print(f'train_acc: {train_acc} , val_acc: {val_acc}')
    trainer.print(f'lr: {lr}')


def save_best_calc_func(trainer: Trainer):
    rec = trainer.get_records()
    val_acc_count = paddle.sum(rec['val_acc_count']).item()
    val_bat_count = paddle.sum(rec['val_bat_count']).item()
    return float(val_acc_count) / val_bat_count


def main():
    cfg = TrainerConfig(
        epoch=args.epoch,
        out_dir=args.out_dir,
        mixed_precision='fp16',
        multi_gpu=False,
        device=args.device,
        save_interval=5,
        save_best=True,
        save_best_type='max',
        seed=args.seed,
        auto_optimize=True,
        auto_schedule=True,
        auto_free=True,
        sync_bn=args.sync_bn,
    )
    trainer = Trainer(cfg)
    trainer.print(args)

    train_loader, train_dataset, val_loader, val_dataset = build_data()
    trainer.print(f'train size: {len(train_dataset)}, val size: {len(val_dataset)}')

    cls_model = build_model()
    cls_loss_func = ClassifyLoss()

    cls_optimizer, cls_lr_scheduler = build_optimizer(cls_model)

    trainer.set_train_dataloader(train_loader)
    trainer.set_val_dataloader(val_loader)
    trainer.set_models([cls_model])
    trainer.set_components([cls_loss_func])
    trainer.set_optimizers(
        [cls_optimizer],
        [cls_lr_scheduler]
    )
    trainer.set_save_best_calc_func(save_best_calc_func)

    if args.resume is not None:
        trainer.load_checkpoint(args.resume)
        trainer.print(f'load checkpoint from {args.resume}')

    trainer.fit(
        train_step=train_step,
        val_step=val_step,
        on_epoch_end=on_epoch_end,
    )

    return


if __name__ == '__main__':
    main()
