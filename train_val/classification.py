"""Callable Trainer and Validationer using the project's flat YAML config.

    trainer = Trainer(conf, train_loader, optimizer, scheduler)
    validationer = Validationer(conf, optimizer=optimizer, scheduler=scheduler)
    trainer(model, epoch)
    validationer(model, epoch, valid_loader)

Epochs are numbered from 1. Trainer advances the epoch-based scheduler once
after training. Validationer aggregates DDP metrics and saves resumable
checkpoints on rank zero; evaluate: true disables checkpoint saving.
"""

import logging
from pathlib import Path
import shutil
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel

from utils.train_utils import AverageMeter


def get_criterion(conf):
    return nn.CrossEntropyLoss()


def _get_device(conf):
    if conf['no_accel'] or not torch.cuda.is_available():
        return torch.device('cpu')
    return torch.device('cuda', conf['gpu'] if conf['gpu'] is not None else 0)


def unwrap_model(model):
    return model.module if isinstance(model, (DataParallel, DistributedDataParallel)) else model


@torch.no_grad()
def accuracy(output, target):
    predictions = output.topk(min(5, output.size(1)), dim=1).indices
    correct = predictions.eq(target[:, None])
    return (correct[:, :1].any(dim=1).float().mean().mul(100).item(),
            correct.any(dim=1).float().mean().mul(100).item())


class Trainer:
    def __init__(self, conf, dataloader, optimizer, scheduler):
        self.conf = conf
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = _get_device(conf)
        self.criterion = get_criterion(conf).to(self.device)
        self.report_freq = conf['print_freq']
        self.train_step = conf['start_epoch'] * len(dataloader)
        self.epoch = conf['start_epoch']
        self.batch_mix = None
        if conf['dataset'] == 'image_net_1K' and not conf['dummy']:
            from torchvision.transforms import v2
            # Apply one of MixUp/CutMix to 10% of training batches.
            self.batch_mix = v2.RandomChoice([
                v2.MixUp(alpha=0.8, num_classes=1000),
                v2.CutMix(alpha=1.0, num_classes=1000),
            ])

    def __call__(self, model, epoch):
        self.epoch = epoch
        model.train()
        if self.conf.get('distributed', False):
            self.dataloader.sampler.set_epoch(epoch - 1)
        losses = AverageMeter('Loss')
        top1, top5 = AverageMeter('Acc@1'), AverageMeter('Acc@5')
        batch_time, data_time = AverageMeter('Time'), AverageMeter('Data')
        end = time.perf_counter()
        for index, (images, target) in enumerate(self.dataloader):
            data_time.update(time.perf_counter() - end)
            images = images.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            loss_target = target
            mixed = self.batch_mix is not None and torch.rand(()).item() < 0.1
            if mixed:
                images, loss_target = self.batch_mix(images, target)
            output = model(images)
            loss = self.criterion(output, loss_target)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            # Soft-label weighted accuracy for mixed batches.
            if mixed:
                predictions = output.detach().topk(min(5, output.size(1)), dim=1).indices
                scores = loss_target.gather(1, predictions)
                acc1 = scores[:, 0].mean().mul(100).item()
                acc5 = scores.sum(dim=1).mean().mul(100).item()
            else:
                acc1, acc5 = accuracy(output.detach(), target)
            losses.update(loss.item(), images.size(0))
            top1.update(acc1, images.size(0))
            top5.update(acc5, images.size(0))
            batch_time.update(time.perf_counter() - end)
            if index % self.report_freq == 0:
                self.report(index, losses, top1, top5, data_time, batch_time)
            self.train_step += 1
            end = time.perf_counter()
        if self.scheduler is not None:
            self.scheduler.step()

    def report(self, index, losses, top1, top5, data_time, batch_time):
        logging.info('Epoch [%d] [%d/%d] Step %d Data %.3f Time %.3f '
                     'Loss %.4f Acc@1 %.3f Acc@5 %.3f LR %.6g',
                     self.epoch, index + 1, len(self.dataloader), self.train_step,
                     data_time.val, batch_time.val, losses.avg, top1.avg, top5.avg,
                     self.optimizer.param_groups[0]['lr'])


class Validationer:
    def __init__(self, conf, *, optimizer=None, scheduler=None, best_acc=0.0):
        self.conf = conf
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = _get_device(conf)
        self.criterion = get_criterion(conf).to(self.device)
        self.report_freq = conf['print_freq']
        self.distributed = conf.get('distributed', False)
        self.is_main = not self.distributed or conf['rank'] == 0
        dataset_tag = 'imgn' if conf['dataset'] == 'image_net_1K' else conf['dataset']
        suffix = f'-{dataset_tag}' + ('-dummy' if conf['dummy'] else '')
        self.local_path = Path(conf['save']) / (conf['arch'] + suffix)
        self.best_acc = float(best_acc)
        self.epoch = conf['start_epoch']
        self.is_best = False

    @torch.no_grad()
    def __call__(self, model, epoch, dataloader):
        self.epoch = epoch
        model.eval()
        eval_model = unwrap_model(model)
        if self.distributed:
            # Sync BN buffers once, then avoid forward collectives on uneven shards.
            for buffer in eval_model.buffers():
                dist.broadcast(buffer, src=0)
        # Reset for each call: validation accuracy is for this epoch only.
        totals = torch.zeros(4, dtype=torch.float64, device=self.device)
        for index, (images, target) in enumerate(dataloader):
            images = images.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            output = eval_model(images)
            loss = self.criterion(output, target)
            acc1, acc5 = accuracy(output, target)
            count = images.size(0)
            totals += totals.new_tensor([loss.item() * count, acc1 * count, acc5 * count, count])
            if index % self.report_freq == 0:
                logging.info('Test [%d/%d] Loss %.4f Acc@1 %.3f Acc@5 %.3f',
                             index + 1, len(dataloader), *(totals[:3] / totals[3]).tolist())
        if self.distributed:
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        if totals[3].item() == 0:
            raise ValueError('Validation dataset is empty')
        loss, acc1, acc5 = (totals[:3] / totals[3]).tolist()
        self.is_best = acc1 > self.best_acc
        if not self.conf['evaluate']:
            self.best_acc = max(self.best_acc, acc1)
            if self.is_main:
                self.save(model)
        self.report(loss, acc1, acc5)
        return acc1

    def report(self, loss, acc1, acc5):
        logging.info(' * Epoch %d Loss %.4f Acc@1 %.3f Acc@5 %.3f',
                     self.epoch, loss, acc1, acc5)

    def save(self, model):
        self.local_path.mkdir(parents=True, exist_ok=True)
        state = {
            'epoch': self.epoch,
            'arch': self.conf['arch'],
            'dataset': self.conf['dataset'],
            'state_dict': unwrap_model(model).state_dict(),
            'best_acc1': self.best_acc,
        }
        if self.optimizer is not None:
            state['optimizer'] = self.optimizer.state_dict()
        if self.scheduler is not None:
            state['scheduler'] = self.scheduler.state_dict()
        path = self.local_path / f'{self.epoch:03d}.pt'
        torch.save(state, path)
        if self.is_best:
            shutil.copyfile(path, self.local_path / 'model_best.pth.tar')
