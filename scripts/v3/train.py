import paddle
from paddle.io import DataLoader

import os
import argparse
from PIL import Image
import numpy as np

from dowdyboy_lib.paddle.trainer import Trainer, TrainerConfig

from bdpan_over.v3.dataset import OverDataset
from bdpan_over.v3.model import OverRestoreModelV3, OverSegmentModelV3, OverClassifyModelV3
from bdpan_over.v3.loss import RestoreLoss, ClassifyLoss, SegLoss


parser = argparse.ArgumentParser(description='over net train v2')
# data config
parser.add_argument('--train-data-dir', type=str, required=True, help='train data dir')
parser.add_argument('--val-data-dir', type=str, required=True, help='val data dir')
parser.add_argument('--img-size', type=int, default=512, help='input img size')
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
args = parser.parse_args()


def build_data():
    train_dataset = OverDataset(
        data_dir=args.train_data_dir,
        data_size=10,
        img_size=args.img_size,
        dense_crop_p=0.8,
        dense_crop_max_count=256,
        flip_p=0.5,
        scale_p=0.2,
        use_hsv=True,
        cache_img_period=5,
        cache_max_size=64,
        is_val=False,
        is_to_tensor=True,
        resize=0.5,
        no_limit_patch_pos_p=0.5,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, drop_last=False)
    val_dataset = OverDataset(
        data_dir=args.val_data_dir,
        data_size=2,
        img_size=args.img_size,
        is_val=True,
        is_to_tensor=True,
        cache_img_period=None,
        resize=0.5,
    )
    val_loader = DataLoader(val_dataset, batch_size=1,
                            shuffle=False, num_workers=args.num_workers, drop_last=False)
    return train_loader, train_dataset, val_loader, val_dataset


def build_model():
    restore_model = OverRestoreModelV3()
    seg_model = OverSegmentModelV3()
    cls_model = OverClassifyModelV3()
    return restore_model, seg_model, cls_model


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
    [restore_model, seg_model, cls_model] = trainer.get_models()
    [cls_loss_func, seg_loss_func, restore_loss_func] = trainer.get_components()
    [restore_optimizer, seg_optimizer, cls_optimizer], \
    [restore_lr_scheduler, seg_lr_scheduler, cls_lr_scheduler] = trainer.get_optimizers()

    bat_im, bat_gt, bat_mask, bat_cls = bat

    trainer.zero_grad(restore_optimizer)
    pred_im = restore_model(bat_im)
    loss_restore = restore_loss_func(pred_im, bat_gt)
    trainer.backward(loss_restore)
    trainer.step(optimizer=restore_optimizer)

    trainer.zero_grad(seg_optimizer)
    pred_seg = seg_model(bat_im)
    pred_seg = paddle.transpose(pred_seg, [0, 2, 3, 1])
    loss_seg = seg_loss_func(pred_seg, bat_mask)
    trainer.backward(loss_seg)
    trainer.step(optimizer=seg_optimizer)

    trainer.zero_grad(cls_optimizer)
    pred_cls = cls_model(bat_im)
    loss_cls = cls_loss_func(pred_cls, bat_cls)
    trainer.backward(loss_cls)
    trainer.step(optimizer=cls_optimizer)

    loss = loss_cls + loss_seg + loss_restore

    trainer.set_bar_state({
        'loss_cls': loss_cls.item(),
        'loss_seg': loss_seg.item(),
        'loss_restore': loss_restore.item(),
    })
    trainer.log({
        'train_loss_cls': loss_cls.item(),
        'train_loss_seg': loss_seg.item(),
        'train_loss_restore': loss_restore.item(),
    }, global_step)

    if global_step % args.log_interval == 0:
        trainer.print(f'global step: {global_step}, loss_cls: {loss_cls.item()} '
                      f'loss_seg: {loss_seg.item()} loss_restore: {loss_restore.item()}')

    return loss


def val_step(trainer: Trainer, bat, bat_idx, global_step):
    from bdpan_over.v2.utils import pd_tensor2img, compute_psnr
    [restore_model, seg_model, cls_model] = trainer.get_models()
    [cls_loss_func, seg_loss_func, restore_loss_func] = trainer.get_components()

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
    seg_loss_list = []
    cls_loss_list = []
    seg_acc_count = 0
    seg_total_count = 0
    cls_acc_count = 0
    cls_total_count = 0
    for i in range(0, h, step):
        for j in range(0, w, step):
            if h - i < step:
                i = h - step
            if w - j < step:
                j = w - step
            clip_im = bat_im[:, :, i:i+step, j:j+step]
            clip_gt = bat_gt[:, :, i:i+step, j:j+step]
            clip_mask = bat_mask[:, i:i+step, j:j+step]
            mask = clip_mask.numpy()
            clip_cls = 1 if np.sum(mask) >= 1 else 0
            clip_cls = paddle.unsqueeze(paddle.to_tensor(np.array(clip_cls, dtype=np.long)), axis=0)
            # ##
            pred_im = restore_model(clip_im)
            loss_restore = restore_loss_func(pred_im, clip_gt)
            restore_loss_list.append(loss_restore.item())
            # ##
            pred_seg = seg_model(clip_im)
            seg_im = paddle.argmax(pred_seg, axis=1)
            seg_acc_count += paddle.sum(seg_im == clip_mask).item()
            seg_total_count += seg_im.shape[-1] * seg_im.shape[-2]
            res_seg[:, i:i+step, j:j+step] = seg_im.astype('float32')
            pred_seg = paddle.transpose(pred_seg, [0, 2, 3, 1])
            loss_seg = seg_loss_func(pred_seg, clip_mask)
            seg_loss_list.append(loss_seg.item())
            # ##
            pred_cls = cls_model(clip_im)
            res_cls = paddle.argmax(pred_cls, axis=1)
            cls_acc_count += paddle.sum(res_cls == clip_cls).item()
            cls_total_count += pred_cls.shape[0]
            loss_cls = cls_loss_func(pred_cls, clip_cls)
            cls_loss_list.append(loss_cls.item())
            # ##
            res[:, :, i:i+step, j:j+step] = pred_im
            # print(1)
    loss_restore = sum(restore_loss_list) / len(restore_loss_list)
    loss_seg = sum(seg_loss_list) / len(seg_loss_list)
    loss_cls = sum(cls_loss_list) / len(cls_loss_list)
    res = res[:, :, :rh, :rw]
    res_seg = res_seg[:, :rh, :rw]
    output = pd_tensor2img(res)
    target = pd_tensor2img(bat_gt)
    output_mask = pd_tensor2img(res_seg)
    psnr = compute_psnr(target, output)
    seg_acc = seg_acc_count / float(seg_total_count)
    cls_acc = cls_acc_count / float(cls_total_count)

    trainer.set_records({
        'psnr': psnr,
        'loss_cls': loss_cls,
        'loss_seg': loss_seg,
        'loss_restore': loss_restore,
        'seg_acc': seg_acc,
        'cls_acc': cls_acc,
    })
    trainer.set_bar_state({
        'psnr': psnr,
        'loss_restore': loss_restore,
        'loss_seg': loss_seg,
        'loss_cls': loss_cls,
    })
    trainer.log({
        'psnr': psnr,
        'val_loss_restore': loss_restore,
        'val_loss_seg': loss_seg,
        'val_loss_cls': loss_cls,
        'val_seg_acc': seg_acc,
        'val_cls_acc': cls_acc,
    }, global_step)

    Image.fromarray(output).save(os.path.join(args.out_dir, f'{global_step % args.save_val_count}_pred.jpg'))
    Image.fromarray(target).save(os.path.join(args.out_dir, f'{global_step % args.save_val_count}_gt.jpg'))
    Image.fromarray(output_mask).save(os.path.join(args.out_dir, f'{global_step % args.save_val_count}_mask.jpg'))

    return loss_restore + loss_seg + loss_cls


def on_epoch_end(trainer: Trainer, ep):
    [restore_optimizer, seg_optimizer, cls_optimizer], \
    [restore_lr_scheduler, seg_lr_scheduler, cls_lr_scheduler] = trainer.get_optimizers()
    rec = trainer.get_records()
    psnr = paddle.mean(rec['psnr']).item()
    loss_restore = paddle.mean(rec['loss_restore']).item()
    loss_seg = paddle.mean(rec['loss_seg']).item()
    loss_cls = paddle.mean(rec['loss_cls']).item()
    seg_acc = paddle.mean(rec['seg_acc']).item()
    cls_acc = paddle.mean(rec['cls_acc']).item()
    trainer.log({
        'ep_psnr': psnr,
        'ep_loss_restore': loss_restore,
        'ep_loss_seg': loss_seg,
        'ep_loss_cls': loss_cls,
        'ep_lr': restore_optimizer.get_lr(),
        'ep_seg_acc': seg_acc,
        'ep_cls_acc': cls_acc,
    }, ep)
    trainer.print(f'loss_restore: {loss_restore}, loss_seg: {loss_seg}, '
                  f'loss_cls: {loss_cls}, psnr: {psnr}, lr: {restore_optimizer.get_lr()}, '
                  f'seg_acc: {seg_acc}, cls_acc: {cls_acc}')


def main():
    cfg = TrainerConfig(
        epoch=args.epoch,
        out_dir=args.out_dir,
        mixed_precision='no',
        multi_gpu=False,
        save_interval=1,
        save_best=True,
        save_best_type='max',
        save_best_rec='psnr',
        seed=args.seed,
        auto_optimize=False,
        auto_schedule=True,
        auto_free=True,
    )
    trainer = Trainer(cfg)
    trainer.print(args)

    train_loader, train_dataset, val_loader, val_dataset = build_data()
    trainer.print(f'train size: {len(train_dataset)}, val size: {len(val_dataset)}')

    restore_model, seg_model, cls_model = build_model()
    cls_loss_func = ClassifyLoss()
    seg_loss_func = SegLoss()
    restore_loss_func = RestoreLoss()

    restore_optimizer, restore_lr_scheduler = build_optimizer(restore_model)
    seg_optimizer, seg_lr_scheduler = build_optimizer(seg_model)
    cls_optimizer, cls_lr_scheduler = build_optimizer(cls_model)

    trainer.set_train_dataloader(train_loader)
    trainer.set_val_dataloader(val_loader)
    trainer.set_models([restore_model, seg_model, cls_model])
    trainer.set_components([cls_loss_func, seg_loss_func, restore_loss_func])
    trainer.set_optimizers(
        [restore_optimizer, seg_optimizer, cls_optimizer],
        [restore_lr_scheduler, seg_lr_scheduler, cls_lr_scheduler]
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
