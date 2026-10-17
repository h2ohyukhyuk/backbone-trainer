"""Callable Trainer and Validationer using the project's flat YAML config.

    trainer = Trainer(conf, train_loader, optimizer, scheduler)
    validationer = Validationer(conf, optimizer=optimizer, scheduler=scheduler)
    trainer(model, epoch)
    validationer(model, epoch, valid_loader)

Epochs are numbered from 1. Trainer steps batch schedulers after each optimizer
update and epoch schedulers after training. Validationer aggregates DDP metrics and saves resumable
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

from dataset.augmentation import build_batch_mix
from utils.train_utils import AverageMeter
from utils.run_output import resolved_config


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


def amp_settings(conf, device):
    enabled = conf.get('amp', False) and device.type == 'cuda'
    dtype = torch.bfloat16 if conf.get('amp_dtype', 'float16') == 'bfloat16' else torch.float16
    if enabled and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise ValueError('This GPU does not support bfloat16; select amp_dtype: float16')
    return enabled, dtype


class Trainer:
    def __init__(self, conf, dataloader, optimizer, scheduler, *, logger=None, scaler=None):
        self.conf = conf
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger
        self.device = _get_device(conf)
        self.criterion = get_criterion(conf).to(self.device)
        self.amp_enabled, self.amp_dtype = amp_settings(conf, self.device)
        self.scaler = scaler if scaler is not None else torch.amp.GradScaler(
            'cuda', enabled=self.amp_enabled and self.amp_dtype == torch.float16)
        self.report_freq = conf['print_freq']
        self.train_step = conf['start_epoch'] * len(dataloader)
        self.epoch = conf['start_epoch']
        self.batch_mix = build_batch_mix(conf)
        self.batch_mix_p = conf['AUGMENTATION']['batch']['p']

    def __call__(self, model, epoch):
        self.epoch = epoch
        model.train()
        if self.conf.get('distributed', False):
            self.dataloader.sampler.set_epoch(epoch - 1)
        losses = AverageMeter('Loss')
        batch_time, data_time = AverageMeter('Time'), AverageMeter('Data')
        totals = torch.zeros(2, dtype=torch.float64, device=self.device)
        end = time.perf_counter()
        for index, (images, target) in enumerate(self.dataloader):
            data_time.update(time.perf_counter() - end)
            images = images.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            loss_target = target
            mixed = self.batch_mix is not None and torch.rand(()).item() < self.batch_mix_p
            if mixed:
                images, loss_target = self.batch_mix(images, target)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled):
                output = model(images)
                loss = self.criterion(output, loss_target)
            self.scaler.scale(loss).backward()
            previous_scale = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            updated = self.scaler.get_scale() >= previous_scale
            if updated and self.scheduler is not None and getattr(self.scheduler, "step_per_batch", False):
                self.scheduler.step()
            count = images.size(0)
            totals[0] += loss.detach().double() * count
            totals[1] += count
            batch_time.update(time.perf_counter() - end)
            if index % self.report_freq == 0:
                losses.avg = (totals[0] / totals[1]).item()
                self.report(index, losses, data_time, batch_time)
            self.train_step += 1
            end = time.perf_counter()
        if self.scheduler is not None and not getattr(self.scheduler, "step_per_batch", False):
            self.scheduler.step()

        if self.conf.get('distributed', False):
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        if totals[1].item() == 0:
            raise ValueError('Training dataset is empty')
        train_loss = (totals[0] / totals[1]).item()
        logging.info(' * Epoch %d Train loss %.4f', epoch, train_loss)
        if self.logger is not None:
            self.logger.report_scalar('loss', 'train', train_loss, iteration=epoch)
            self.logger.report_scalar('learning rate', 'train', self.optimizer.param_groups[0]['lr'], iteration=epoch)
        return train_loss

    def report(self, index, losses, data_time, batch_time):
        logging.info('Epoch [%d] [%d/%d] Step %d Data %.3f Time %.3f '
                     'Train loss %.4f LR %.6g',
                     self.epoch, index + 1, len(self.dataloader), self.train_step,
                     data_time.val, batch_time.val, losses.avg,
                     self.optimizer.param_groups[0]['lr'])


class Validationer:
    def __init__(self, conf, *, optimizer=None, scheduler=None, best_acc=0.0, logger=None, scaler=None, registry=None):
        self.registry = registry
        self.logger = logger
        self.scaler = scaler
        self.conf = conf
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = _get_device(conf)
        self.criterion = get_criterion(conf).to(self.device)
        self.amp_enabled, self.amp_dtype = amp_settings(conf, self.device)
        self.report_freq = conf['print_freq']
        self.distributed = conf.get('distributed', False)
        self.is_main = not self.distributed or conf['global_rank'] == 0
        dataset_tag = 'imgn' if conf['dataset'] == 'image_net_1K' else conf['dataset']
        suffix = f'-{dataset_tag}' + ('-dummy' if conf['dummy'] else '')
        self.local_path = Path(conf['output_dir']) if conf.get('output_dir') else Path(conf['save']) / (conf['arch'] + suffix)
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
            with torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled):
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
                self.save(model, valid_loss=loss, acc1=acc1, acc5=acc5)
        if self.logger is not None:
            self.logger.report_scalar('loss', 'valid', loss, iteration=epoch)
            self.logger.report_scalar('accuracy', 'valid_top1', acc1, iteration=epoch)
            self.logger.report_scalar('accuracy', 'valid_top5', acc5, iteration=epoch)
        self.report(loss, acc1, acc5)
        return acc1

    def report(self, loss, acc1, acc5):
        logging.info(' * Epoch %d Loss %.4f Acc@1 %.3f Acc@5 %.3f',
                     self.epoch, loss, acc1, acc5)

    def save(self, model, *, valid_loss, acc1, acc5):
        self.local_path.mkdir(parents=True, exist_ok=True)
        state = {
            'epoch': self.epoch,
            'arch': self.conf['arch'],
            'dataset': self.conf['dataset'],
            'state_dict': unwrap_model(model).state_dict(),
            'best_acc1': self.best_acc,
            'config': resolved_config(self.conf),
            'metrics': {'valid_loss': valid_loss, 'top1': acc1, 'top5': acc5},
        }
        if self.optimizer is not None:
            state['optimizer'] = self.optimizer.state_dict()
        if self.scheduler is not None:
            state['scheduler'] = self.scheduler.state_dict()
        if self.scaler is not None:
            state['scaler'] = self.scaler.state_dict()
        path = self.local_path / f'{self.epoch:03d}.pt'
        torch.save(state, path)
        if self.is_best:
            best_path = self.local_path / 'model_best.pt'
            shutil.copyfile(path, best_path)
            if self.is_main and self.registry is not None:
                uri = self.registry.update(best_path, epoch=self.epoch, accuracy=self.best_acc, accuracy_top5=acc5)
                logging.info('ClearML best model uploaded: %s', uri)
