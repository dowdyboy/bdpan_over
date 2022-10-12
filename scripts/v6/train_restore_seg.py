import paddle
from paddle.io import DataLoader

import os
import argparse
from PIL import Image
import numpy as np

from dowdyboy_lib.paddle.trainer import Trainer, TrainerConfig

from bdpan_over.v6.dataset import OverDataset
from bdpan_over.v6.model import over_model_b0, over_model_b2, over_model_b4
from bdpan_over.v6.loss import RestoreSegmentLoss


parser = argparse.ArgumentParser(description='over net train v6')
# model config
parser.add_argument('--use-sigmoid', default=False, action='store_true', help='use-sigmoid')
# data config
parser.add_argument('--train-data-dir', type=str, required=True, help='train data dir')
parser.add_argument('--val-data-dir', type=str, required=True, help='val data dir')
parser.add_argument('--train-data-size', type=int, default=5000, help='train data size')
parser.add_argument('--val-data-size', type=int, default=50, help='val data size')
parser.add_argument('--img-size', type=int, default=1024, help='input img size')
parser.add_argument('--num-workers', type=int, default=4, help='num workers')
# optimizer config
parser.add_argument('--lr', type=float, default=1e-3, help='lr')
parser.add_argument('--use-scheduler', default=False, action='store_true', help='use schedule')
parser.add_argument('--use-warmup', default=False, action='store_true', help='use warmup')
parser.add_argument('--weight-decay', type=float, default=5e-6, help='model weight decay')
# train config
parser.add_argument('--epoch', type=int, default=10, help='epoch num')
parser.add_argument('--batch-size', type=int, default=2, help='batch size')
parser.add_argument('--out-dir', type=str, default='./output', help='out dir')
parser.add_argument('--resume', type=str, default=None, help='resume checkpoint')
parser.add_argument('--last-epoch', type=int, default=-1, help='last epoch')
parser.add_argument('--seed', type=int, default=2022, help='random seed')
parser.add_argument('--log-interval', type=int, default=500, help='log process')
parser.add_argument('--save-val-count', type=int, default=50, help='log process')
parser.add_argument('--sync-bn', default=False, action='store_true', help='sync_bn')
parser.add_argument('--device', default=None, type=str, help='device')
args = parser.parse_args()


def build_data():
    train_dataset = OverDataset(
        data_dir=args.train_data_dir,
        data_size=args.train_data_size,
        img_size=args.img_size,
        dense_crop_p=0.8,
        dense_crop_max_count=256,
        dense_crop_rate=0.05,
        flip_p=0.5,
        scale_p=0.2,
        use_hsv=True,
        cache_img_period=5,
        cache_max_size=256,
        is_val=False,
        is_to_tensor=True,
        no_limit_patch_pos_p=0.5,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, drop_last=False)
    val_dataset = OverDataset(
        data_dir=args.val_data_dir,
        data_size=args.val_data_size,
        img_size=args.img_size,
        is_val=True,
        is_to_tensor=True,
        cache_img_period=None,
    )
    val_loader = DataLoader(val_dataset, batch_size=1,
                            shuffle=False, num_workers=args.num_workers, drop_last=False)
    return train_loader, train_dataset, val_loader, val_dataset


def build_model():
    restore_model = over_model_b2(
        img_size=args.img_size,
        encoder_pretrain='checkpoint/pretrain/pvt_v2_b2.pdparams',
        use_sigmoid=args.use_sigmoid,
    )
    # restore_model = over_model_b0(
    #     img_size=args.img_size,
    #     encoder_pretrain='checkpoint/pretrain/pvt_v2_b0.pdparams'
    # )
    return restore_model


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
    [restore_model] = trainer.get_models()
    [restore_loss_func] = trainer.get_components()

    bat_im, bat_gt, bat_mask, bat_cls = bat

    pred_im, pred_mask = restore_model(bat_im)
    loss = restore_loss_func(global_step, pred_im, bat_gt, pred_mask, bat_mask)

    trainer.log({
        'train_loss': loss.item(),
    }, global_step)

    if global_step % args.log_interval == 0:
        trainer.print(f'global step: {global_step}, loss: {loss.item()}')

    return loss


def val_step(trainer: Trainer, bat, bat_idx, global_step):
    from bdpan_over.v2.utils import pd_tensor2img, compute_psnr
    [restore_model] = trainer.get_models()
    [restore_loss_func] = trainer.get_components()

    bat_im, bat_gt, bat_mask, bat_cls = bat

    _, _, h, w = bat_im.shape
    rh, rw = h, w
    step = args.img_size
    pad_h = step - h if h < step else 0
    pad_w = step - w if w < step else 0
    m = paddle.nn.Pad2D((0, pad_w, 0, pad_h))
    bat_im = m(bat_im)
    _, _, h, w = bat_im.shape
    res = paddle.zeros_like(bat_im)
    res_seg = paddle.zeros((bat_im.shape[0], bat_im.shape[-2], bat_im.shape[-1]))
    restore_loss_list = []
    for i in range(0, h, step):
        for j in range(0, w, step):
            if h - i < step:
                i = h - step
            if w - j < step:
                j = w - step
            clip_im = bat_im[:, :, i:i+step, j:j+step]
            clip_gt = bat_gt[:, :, i:i+step, j:j+step]
            clip_mask = bat_mask[:, i:i+step, j:j+step]
            # ##
            pred_im, pred_mask = restore_model(clip_im)
            seg_im = paddle.argmax(pred_mask, axis=1)
            seg_im = seg_im.astype('float32')
            loss_restore = restore_loss_func(0, pred_im, clip_gt, pred_mask, clip_mask)
            restore_loss_list.append(loss_restore.item())
            # ##
            res[:, :, i:i+step, j:j+step] = pred_im
            res_seg[:, i:i+step, j:j+step] = seg_im

    loss_restore = sum(restore_loss_list) / len(restore_loss_list)
    res = res[:, :, :rh, :rw]
    res_seg = res_seg[:, :rh, :rw]
    res = res * res_seg + bat_im * (1. - res_seg)
    output = pd_tensor2img(res)
    target = pd_tensor2img(bat_gt)
    output_mask = pd_tensor2img(res_seg)

    if int(np.sum(output != target)) > 0 and int(np.sum(output_mask)) > 0:
        psnr = compute_psnr(target, output)
        if psnr != float("inf"):
            trainer.set_records({
                'psnr': psnr,
                'loss_restore': loss_restore,
            })
            trainer.set_bar_state({
                'psnr': psnr,
            })
            trainer.log({
                'psnr': psnr,
                'val_loss_restore': loss_restore,
            }, global_step)

    Image.fromarray(output).save(os.path.join(args.out_dir, f'{global_step % args.save_val_count}_pred.jpg'))
    Image.fromarray(target).save(os.path.join(args.out_dir, f'{global_step % args.save_val_count}_gt.jpg'))
    Image.fromarray(output_mask).save(os.path.join(args.out_dir, f'{global_step % args.save_val_count}_mask.jpg'))

    return loss_restore


def on_epoch_end(trainer: Trainer, ep):
    [restore_optimizer], \
    [restore_lr_scheduler] = trainer.get_optimizers()
    rec = trainer.get_records()
    psnr = paddle.mean(rec['psnr']).item()
    loss_restore = paddle.mean(rec['loss_restore']).item()
    trainer.log({
        'ep_psnr': psnr,
        'ep_loss_restore': loss_restore,
        'ep_lr': restore_optimizer.get_lr(),
    }, ep)
    trainer.print(f'loss_restore: {loss_restore}, '
                  f'psnr: {psnr}, lr: {restore_optimizer.get_lr()}')


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
        save_best_rec='psnr',
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

    restore_model = build_model()
    restore_loss_func = RestoreSegmentLoss(
        (args.train_data_size // args.batch_size) * 10
    )

    restore_optimizer, restore_lr_scheduler = build_optimizer(restore_model)

    trainer.set_train_dataloader(train_loader)
    trainer.set_val_dataloader(val_loader)
    trainer.set_models([restore_model])
    trainer.set_components([restore_loss_func])
    trainer.set_optimizers(
        [restore_optimizer],
        [restore_lr_scheduler]
    )

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
