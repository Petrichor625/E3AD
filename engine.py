import math
import os
import sys
from typing import Iterable

import torch

import util.misc as utils
from util import box_ops

import logging
import torch.distributed as dist
import time
import datetime
from tqdm import tqdm
import json

import numpy as np
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as pic_transforms
import matplotlib.patches as patches

import matplotlib.pyplot as plt


class data_prefetcher():
    def __init__(self, loader, device):
        self.length = len(loader)

        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.device = device
        self.preload()

    def preload(self):
        try:
            samples, targets = next(self.loader)
            self.next_img, self.next_mask = samples.decompose()
            self.next_target = targets
        except StopIteration:
            self.next_img = self.next_mask = self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_img = self.next_img.to(self.device, non_blocking=True)
            self.next_mask = self.next_mask.to(self.device, non_blocking=True)
            tensor_dict = self.next_target.tensor_dict
            dict_out = ['command_token',"Command Keywords","Command Analysis","Scenario Analysis",'Command','image_path']

            self.next_target.tensor_dict = {
                k: tensor_dict[k].to(self.device, non_blocking=True)  if k not in dict_out else tensor_dict[k]
                                             for k in tensor_dict}
            #             self.next_target.tensor_dict = {
            #    k: tensor_dict[k].to(self.device, non_blocking=True) if k != 'phrase' and k !='command_token'and k !='Command Keywords'and k !='Command Analysis'and k !='Scenario Analysis' else tensor_dict["command_token"]
            #                                 for k in tensor_dict}

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        img, mask, target = self.next_img, self.next_mask, self.next_target
        self.preload()
        return img, mask, target

    def __next__(self):
        img, mask, target = self.next()
        if img == None:
            raise StopIteration
        return img, mask, target

    def __iter__(self):
        return self

    def __len__(self):
        return self.length

def move_dict_tensors_to_device(tensor_dict, device):

    for key, value in tensor_dict.items():
        if isinstance(value, torch.Tensor):  # 确保只处理张量类型
            tensor_dict[key] = value.to(device)
        elif isinstance(value, dict):  # 对于嵌套字典，递归处理
            tensor_dict[key] = move_dict_tensors_to_device(value, device)
    return tensor_dict


def train_one_epoch_origin(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, epochs: int, max_norm: float = 0):
    
    model.train()
    criterion.train()
    logger = logging.getLogger("train")
    metric_logger = utils.MetricLogger(delimiter="  ")

    iter_time = utils.SmoothedValue(fmt='{avg:.3f}')
    data_time = utils.SmoothedValue(fmt='{avg:.3f}')
    header = 'Epoch [{epoch}][{iter}/{max_iter}]'

    max_iter = len(data_loader)
    end = time.time()

    prefetcher = data_prefetcher(data_loader, device)
    img, mask, target = prefetcher.next()
    iteration = 0
    while img is not None:
        target_dict = target.tensor_dict
        word_id, word_mask = target_dict['word_id'], target_dict['word_mask']
        iteration = iteration + 1
        data_time.update(time.time() - end)

        outputs = model(img, mask, word_id, word_mask,target)
        
        # print(outputs.keys()) dict_keys(['pred_boxes', 'aux_outputs'])
        # print(target_dict.keys()) dict_keys(['bbox', 'dxdy', 'size', 'word_id', 'word_mask', 'id'])
        #print("pred",outputs['pred_boxes'][0])
        # print("gt",target_dict['bbox'][0])
        loss_dict = criterion(outputs, target_dict)

        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        iter_time.update(time.time() - end)
        end = time.time()
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)

        if iteration % 100 == 0 or iteration == max_iter:
            eta_seconds = iter_time.global_avg * (max_iter - iteration + max_iter * (epochs-epoch-1))
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            logger.info(
                metric_logger.delimiter.join(
                    [header,
                     "lr: {lr}",
                     "eta: {eta}",
                     "time: {time}",
                     "data: {data}",
                     "memory: {memory:.0f}",
                     "{meters}"
                     ]
                ).format(
                    epoch=epoch+1, iter=iteration, max_iter=max_iter,
                    lr=optimizer.param_groups[0]["lr"],
                    eta=eta_string,
                    time=str(iter_time),
                    data=str(data_time),
                    memory=torch.cuda.max_memory_allocated() / (1024. * 1024),
                    meters=str(metric_logger)
                ))

        img, mask, target = prefetcher.next()

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def train_CAVG_one_epoch_v0(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, epochs: int, max_norm: float = 0):
    
    model.train()
    criterion.train()
    logger = logging.getLogger("train")
    metric_logger = utils.MetricLogger(delimiter="  ")

    iter_time = utils.SmoothedValue(fmt='{avg:.3f}')
    data_time = utils.SmoothedValue(fmt='{avg:.3f}')
    header = 'Epoch [{epoch}][{iter}/{max_iter}]'

    max_iter = len(data_loader)
    end = time.time()
    iteration = 0
    # prefetcher = data_prefetcher(data_loader, device)
    for batch in data_loader:

        target=batch
        target = move_dict_tensors_to_device(target, device)

        iteration = iteration + 1
        data_time.update(time.time() - end)
        outputs = model(target)
        

        loss_dict = criterion(outputs, target)

        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())


        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()

        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        iter_time.update(time.time() - end)
        end = time.time()
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)

        if iteration % 10 == 0 or iteration == max_iter:
            eta_seconds = iter_time.global_avg * (max_iter - iteration + max_iter * (epochs-epoch-1))
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            logger.info(
                metric_logger.delimiter.join(
                    [header,
                     "lr: {lr}",
                     "eta: {eta}",
                     "time: {time}",
                     "data: {data}",
                     "memory: {memory:.0f}",
                     "{meters}"
                     ]
                ).format(
                    epoch=epoch+1, iter=iteration, max_iter=max_iter,
                    lr=optimizer.param_groups[0]["lr"],
                    eta=eta_string,
                    time=str(iter_time),
                    data=str(data_time),
                    memory=torch.cuda.max_memory_allocated() / (1024. * 1024),
                    meters=str(metric_logger)
                ))
 



    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_CAVG_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, epochs: int, max_norm: float = 0):

    model.train()
    criterion.train()
    logger = logging.getLogger("train")
    metric_logger = utils.MetricLogger(delimiter="  ")

    iter_time = utils.SmoothedValue(fmt='{avg:.3f}')
    data_time = utils.SmoothedValue(fmt='{avg:.3f}')
    header = 'Epoch [{epoch}][{iter}/{max_iter}]'

    max_iter = len(data_loader)
    end = time.time()

    prefetcher = data_prefetcher(data_loader, device)
    img,mask,target = prefetcher.next()
    iteration = 0

    while img is not None:
        target_dict = move_dict_tensors_to_device(target.tensor_dict, device)

        iteration = iteration + 1
        data_time.update(time.time() - end)


        outputs = model(img,target_dict)
        loss_dict = criterion(outputs, target_dict)

        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()

        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        iter_time.update(time.time() - end)
        end = time.time()
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)

        if iteration==1 or iteration % 50 == 0 or iteration == max_iter:
            eta_seconds = iter_time.global_avg * (max_iter - iteration + max_iter * (epochs-epoch-1))
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            logger.info(
                metric_logger.delimiter.join(
                    [header,
                     "lr: {lr}",
                     "eta: {eta}",
                     "time: {time}",
                     "data: {data}",
                     "memory: {memory:.0f}",
                     "{meters}"
                     ]
                ).format(
                    epoch=epoch+1, iter=iteration, max_iter=max_iter,
                    lr=optimizer.param_groups[0]["lr"],
                    eta=eta_string,
                    time=str(iter_time),
                    data=str(data_time),
                    memory=torch.cuda.max_memory_allocated() / (1024. * 1024),
                    meters=str(metric_logger)
                ))
        if iteration % 100 == 0 or iteration==1:
            #torch.save(outputs,'./RegionProposal/utils')

            index = 0
            with open(target_dict['image_path'][index], "rb") as f:
                print(target_dict['image_path'][index])
                original_image = Image.open(f).convert("RGB")


            import matplotlib.pyplot as plt
            import numpy as np


            image= pic_transforms.Compose([
                                            pic_transforms.ToTensor(),  
                                            ])

            layer_attn_scores, bbox_pred = outputs['attn_tensor'],outputs['pred_bbox']
            upsampled_attention_map = F.interpolate(layer_attn_scores, size=(900, 1600), mode='bilinear', align_corners=False)

            pred_scores= upsampled_attention_map.squeeze()

            src_boxes = bbox_pred.detach() #box_ops.box_cxcywh_to_xyxy(bbox_pred.detach())
            x_1,y_1,x_2,y_2 = src_boxes[index].cpu()

            x1,y1,x2,y2 = target_dict['gt_bbox'][index].cpu()

            use_dataset = False
            if use_dataset== True:
                downsampled_mask =target['downsampled_mask'][index]
                downsampled_mask = downsampled_mask.unsqueeze(0).unsqueeze(0)

                upsampled_attention_map = F.interpolate(downsampled_mask, size=(900, 1600), mode='bilinear', align_corners=False)
                attention_map_np = upsampled_attention_map.cpu().squeeze().numpy()
                attention_map_np = (attention_map_np - attention_map_np.min()) / (attention_map_np.max() - attention_map_np.min())
            else:
                attention_map_np = pred_scores[index].cpu().detach().numpy()


            fig, ax = plt.subplots(1)

            ax.imshow(original_image)

            cax=ax.imshow(attention_map_np, cmap='jet', alpha=0.5)
            fig.colorbar(cax, ax=ax)

            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='black', facecolor='none')
            ax.add_patch(rect)
            rect = patches.Rectangle((x_1, y_1), x_2-x_1, y_2-y_1, linewidth=1, edgecolor='red', facecolor='none')
            plt.savefig("./utils/example_with_bbox.jpg")
            ax.add_patch(rect)

            plt.savefig("./utils/example.jpg")

        img,mask ,target = prefetcher.next()

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_w_accum(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, epochs: int, max_norm: float = 0):
    model.train()
    criterion.train()
    logger = logging.getLogger("train")
    metric_logger = utils.MetricLogger(delimiter="  ")

    iter_time = utils.SmoothedValue(fmt='{avg:.3f}')
    data_time = utils.SmoothedValue(fmt='{avg:.3f}')
    header = 'Epoch [{epoch}][{iter}/{max_iter}]'

    max_iter = len(data_loader)
    end = time.time()

    prefetcher = data_prefetcher(data_loader, device)
    img, mask, target = prefetcher.next()
    iteration = 0
    while img is not None:
        target_dict = target.tensor_dict
        iteration = iteration + 1
        data_time.update(time.time() - end)

        B = img.shape[0]
        b = B // 2
        loss_dicts = list()
        weight_dict = criterion.weight_dict
        for i in range(2):
            b_img = img[i*b:(i+1)*b]
            b_mask = mask[i*b:(i+1)*b]
            b_target = {k: target_dict[k][i*b:(i+1)*b] for k in target_dict}
            b_word_id, b_word_mask = b_target['word_id'], b_target['word_mask']

            outputs = model(b_img, b_mask, b_word_id, b_word_mask)

            loss_dict = criterion(outputs, b_target)
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict) / 2
            losses.backward()
            loss_dicts.append(loss_dict)

        loss_dict_accum_scaled = {k: (loss_dicts[0][k] + loss_dicts[1][k]) * weight_dict[k] / 2
                                    for k in loss_dicts[0].keys() if k in weight_dict}

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced_scaled = utils.reduce_dict(loss_dict_accum_scaled)
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced_scaled)
            sys.exit(1)

        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        optimizer.zero_grad()

        iter_time.update(time.time() - end)
        end = time.time()
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)

        if iteration % 100 == 0 or iteration == max_iter:
            eta_seconds = iter_time.global_avg * (max_iter - iteration + max_iter * (epochs-epoch-1))
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            logger.info(
                metric_logger.delimiter.join(
                    [header,
                     "lr: {lr}",
                     "eta: {eta}",
                     "time: {time}",
                     "data: {data}",
                     "memory: {memory:.0f}",
                     "{meters}"
                     ]
                ).format(
                    epoch=epoch+1, iter=iteration, max_iter=max_iter,
                    lr=optimizer.param_groups[0]["lr"],
                    eta=eta_string,
                    time=str(iter_time),
                    data=str(data_time),
                    memory=torch.cuda.max_memory_allocated() / (1024. * 1024),
                    meters=str(metric_logger)
                ))

        img, mask, target = prefetcher.next()

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate_CAVG_v0(model, criterion, postprocessor, data_loader, device, save_path=''):
    model.eval()
    if criterion:
        criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    iter_time = utils.SmoothedValue(fmt='{avg:.3f}')
    data_time = utils.SmoothedValue(fmt='{avg:.3f}')

    accum_acc = 0
    accum_iou = 0
    accum_sample = 0
    iou_thrs = torch.as_tensor([0.5 + 0.05 * i for i in range(0,9)], device=device)

    end = time.time()

    all_pred_ious = []
    all_pred_boxes = []
    for batch in data_loader:
        
        target=batch

        target = move_dict_tensors_to_device(target, device)

       # word_id, word_mask = target_dict['word_id'], target_dict['word_mask']
        gt_bbox = target ['origin_boundingbox_xyxy'] # xyxy

        data_time.update(time.time() - end)
        outputs = model(target)

        if criterion:
            loss_dict = criterion(outputs, target)
            weight_dict = criterion.weight_dict

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                        for k, v in loss_dict_reduced.items() if k in weight_dict}
            loss_value = sum(loss_dict_reduced_scaled.values()).item()
            metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)


        ious = box_ops.box_pair_iou(gt_bbox, gt_bbox)[0]
        sum_iou = ious.sum()
        num_acc = (ious[:, None] > iou_thrs[None]).sum(dim=0)
        num_sample = torch.as_tensor(gt_bbox.size(0), device=gt_bbox.device)

        accum_acc += num_acc
        accum_iou += sum_iou
        accum_sample += num_sample

        iter_time.update(time.time() - end)
        end = time.time()


    if utils.get_world_size() > 1:
        dist.all_reduce(accum_acc)
        dist.all_reduce(accum_iou)
        dist.all_reduce(accum_sample)

    acc = accum_acc / accum_sample.float().item()
    miou = accum_iou.item() / accum_sample.float().item()

    val_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    val_acc = {f'Acc@{t:.2f}': a.item() for t, a in zip(iou_thrs, acc)}
    val_acc.update({'Mean_iou': miou})
    val_time = {'data_time': data_time.global_avg, 'time': iter_time.global_avg}
    return val_stats, val_acc, val_time


@torch.no_grad()
def evaluate_CAVG(model, criterion, postprocessor, data_loader, device, save_path=''):
    model.eval()
    if criterion:
        criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    iter_time = utils.SmoothedValue(fmt='{avg:.3f}')
    data_time = utils.SmoothedValue(fmt='{avg:.3f}')

    accum_acc = 0
    accum_iou = 0
    accum_sample = 0
    iou_thrs = torch.as_tensor([0.5 + 0.05 * i for i in range(0,9)], device=device)

    end = time.time()

    all_pred_ious = []
    all_pred_boxes = []
    prefetcher = data_prefetcher(data_loader, device)
    for iteration, (img, mask, target) in enumerate(tqdm(prefetcher)):
        target_dict = move_dict_tensors_to_device(target.tensor_dict, device)
       # word_id, word_mask = target_dict['word_id'], target_dict['word_mask']
        gt_bbox = target_dict['gt_bbox'] # xyxy

        data_time.update(time.time() - end)
        outputs = model(img,target_dict)

        if criterion:
            loss_dict = criterion(outputs, target_dict)
            weight_dict = criterion.weight_dict

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                        for k, v in loss_dict_reduced.items() if k in weight_dict}
            loss_value = sum(loss_dict_reduced_scaled.values()).item()
            metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)


        src_boxes = outputs['pred_bbox'].detach() #box_ops.box_cxcywh_to_xyxy(outputs['pred_bbox'].detach())

        ious = box_ops.box_pair_iou(gt_bbox, src_boxes)[0]

        sum_iou = ious.sum()
        num_acc = (ious[:, None] > iou_thrs[None]).sum(dim=0)
        
        num_sample = torch.as_tensor(img.size(0), device=img.device)

        
        accum_acc += num_acc
        accum_iou += sum_iou
        accum_sample += num_sample

        iter_time.update(time.time() - end)
        end = time.time()

        all_pred_ious.append(ious.view(-1, 1))
        all_pred_boxes.append(src_boxes)

    if save_path:
        torch.save({'pred_boxes': torch.cat(all_pred_boxes, dim=0),
                    'pred_ious': torch.cat(all_pred_ious, dim=0)},
                   save_path + 'pred_boxes')
    # accumulate predictions from all images
    if utils.get_world_size() > 1:
        dist.all_reduce(accum_acc)
        dist.all_reduce(accum_iou)
        dist.all_reduce(accum_sample)

    acc = accum_acc / accum_sample.float().item()
    miou = accum_iou.item() / accum_sample.float().item()

    val_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    val_acc = {f'Acc@{t:.2f}': a.item() for t, a in zip(iou_thrs, acc)}
    val_acc.update({'Mean_iou': miou})
    val_time = {'data_time': data_time.global_avg, 'time': iter_time.global_avg}
    return val_stats, val_acc, val_time


@torch.no_grad()
def evaluate(model, criterion, postprocessor, data_loader, device, save_path=''):
    model.eval()
    if criterion:
        criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    iter_time = utils.SmoothedValue(fmt='{avg:.3f}')
    data_time = utils.SmoothedValue(fmt='{avg:.3f}')

    accum_acc = 0
    accum_iou = 0
    accum_sample = 0
    iou_thrs = torch.as_tensor([0.5 + 0.05 * i for i in range(0,9)], device=device)

    end = time.time()

    all_pred_ious = []
    all_pred_boxes = []
    prefetcher = data_prefetcher(data_loader, device)
    for iteration, (img, mask, target) in enumerate(tqdm(prefetcher)):
        target_dict = target.tensor_dict
        word_id, word_mask = target_dict['word_id'], target_dict['word_mask']
        gt_bbox = target_dict['orig_bbox'] # xyxy

        data_time.update(time.time() - end)
        outputs = model(img, mask, word_id, word_mask)

        if criterion:
            loss_dict = criterion(outputs, target_dict)
            weight_dict = criterion.weight_dict

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                        for k, v in loss_dict_reduced.items() if k in weight_dict}
            loss_value = sum(loss_dict_reduced_scaled.values()).item()
            metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)


        pred_boxes = postprocessor(outputs, target_dict)

        ious = box_ops.box_pair_iou(gt_bbox, pred_boxes)[0]
        sum_iou = ious.sum()
        num_acc = (ious[:, None] > iou_thrs[None]).sum(dim=0)
        num_sample = torch.as_tensor(img.size(0), device=img.device)

        accum_acc += num_acc
        accum_iou += sum_iou
        accum_sample += num_sample

        iter_time.update(time.time() - end)
        end = time.time()

        all_pred_ious.append(ious.view(-1, 1))
        all_pred_boxes.append(pred_boxes)

    if save_path:
        torch.save({'pred_boxes': torch.cat(all_pred_boxes, dim=0),
                    'pred_ious': torch.cat(all_pred_ious, dim=0)},
                   save_path + 'pred_boxes')
    # accumulate predictions from all images
    if utils.get_world_size() > 1:
        dist.all_reduce(accum_acc)
        dist.all_reduce(accum_iou)
        dist.all_reduce(accum_sample)

    acc = accum_acc / accum_sample.float().item()
    miou = accum_iou.item() / accum_sample.float().item()

    val_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    val_acc = {f'Acc@{t:.2f}': a.item() for t, a in zip(iou_thrs, acc)}
    val_acc.update({'Mean_iou': miou})
    val_time = {'data_time': data_time.global_avg, 'time': iter_time.global_avg}
    return val_stats, val_acc, val_time

@torch.no_grad()
def evaluate_talk2car(model, criterion, postprocessor, data_loader_val,data_loader_test, device, save_path='',split='train'):
    model.eval()
    if criterion:
        criterion.eval()
    
    path_val_token = "./Models/utils/dataloader/Mytoken_val.json"
    path_test_token = "./Models/utils/dataloader/Mytoken_test.json"

    metric_logger = utils.MetricLogger(delimiter="  ")
    iter_time = utils.SmoothedValue(fmt='{avg:.3f}')
    data_time = utils.SmoothedValue(fmt='{avg:.3f}')

    accum_acc = 0
    accum_iou = 0
    accum_sample = 0
    iou_thrs = torch.as_tensor([0.5 + 0.05 * i for i in range(0,9)], device=device)

    end = time.time()

    all_pred_ious = []
    all_pred_boxes = []
    prefetcher_val = data_prefetcher(data_loader_val, device)
    prefetcher_test = data_prefetcher(data_loader_test, device)
    import json 
    with open(path_val_token,'r') as f1:
        dict_token_val = json.load(f1)


    with open(path_test_token,'r') as f2:
        dict_token_test = json.load(f2)

    output_dic = {}

    for iteration, (img, mask, target) in enumerate(tqdm(prefetcher_val)):
        target_dict = target.tensor_dict
        word_id, word_mask = target_dict['word_id'], target_dict['word_mask']
        gt_bbox = target_dict['orig_bbox']
        data_time.update(time.time() - end)

        outputs = model(img, mask, word_id, word_mask)
        pred_boxes = postprocessor(outputs, target_dict)
        

        for i in range(pred_boxes.shape[0]):
            id = target_dict['id'][i].tolist()
            id = str(id)
            # dx,dy = target_dict['dxdy'][i]
            # ratiox,ratioy =  target_dict['ratio'][i]
            
            token = target_dict['command_token'][i]
            
            x1,y1,x2,y2 = pred_boxes[i].tolist()

            output_dic[token] =[x1,y1,x2-x1,y2-y1]



        # for i in range(gt_bbox.shape[0]):
            
            # id = target_dict['id'][i].tolist()
            # id = str(id)
            
            # token = dict_token_val[id]
            
        #     # output_dic[token] = [x1*1800,y1*900,(x1-x2)*1800,(y1-y2)*900]
        #     x1,y1,x2,y2 = box_ops.box_cxcywh_to_xyxy(pre_bbox[i]).tolist()[0]
        #     # print([x1*1800,y1*900,(x1-x2)*1800,(y1-y2)*900],gt_bbox[i])
        #     output_dic[token] = [x1*1800,y1*900,x2*1800,y2*900]

    # for iteration, (img, mask, target) in enumerate(tqdm(prefetcher_test)):
    #     target_dict = target.tensor_dict
    #     word_id, word_mask = target_dict['word_id'], target_dict['word_mask']
    #     gt_bbox = target_dict['orig_bbox']
    #     data_time.update(time.time() - end)
    #     outputs = model(img, mask, word_id, word_mask)
    #     pre_bbox = outputs['pred_boxes']
        
    #     for i in range(gt_bbox.shape[0]):
            
            
            
    #         id = target_dict['id'][i].tolist()
    #         id = str(id)
            
    #         token = dict_token_test[id]
    #         # output_dic[token] = [x1*1800,y1*900,(x1-x2)*1800,(y1-y2)*900]
    #         x1,y1,x2,y2 = box_ops.box_cxcywh_to_xyxy(pre_bbox[i]).tolist()[0]
    #         # print([x1*1800,y1*900,(x1-x2)*1800,(y1-y2)*900],gt_bbox[i])
    #         output_dic[token] = [x1*1800,y1*900,x2*1800,y2*900]


    # for iteration, (img, mask, target) in enumerate(tqdm(prefetcher_test)):
    #     target_dict = target.tensor_dict
    #     word_id, word_mask = target_dict['word_id'], target_dict['word_mask']
    #     gt_bbox = target_dict['orig_bbox']
    #     data_time.update(time.time() - end)
    #     outputs = model(img, mask, word_id, word_mask)
    #     pre_bbox = outputs['pred_boxes']
    #     out_w, out_h = 640,640#target_dict['size']
        
    #     pre_bbox = pre_bbox * torch.tensor([out_w, out_h, out_w, out_h], dtype=torch.float32).to("cuda:0")
    #     # dx,dy = target_dict['dxdy']
    #     # ratiox,ratioy =  target_dict['redio']  
    #     pre_bbox=box_ops.box_cxcywh_to_xyxy(pre_bbox)
    #     for i in range(pre_bbox.shape[0]):
    #         id = target_dict['id'][i].tolist()
    #         id = str(id)
    #         dx,dy = target_dict['dxdy'][i]
    #         ratiox,ratioy =  target_dict['ratio'][i]
    #         token = dict_token_test[id]
    #         #print(pre_bbox[i])
    #         x1,y1,x2,y2 = pre_bbox[i][0]
    #         x1,y1,x2,y2 = x1-dx,y1-dy,x2-dx,y2-dy
    #         x1,y1,x2,y2 = x1/ratiox,y1/ratioy,x2/ratiox,y2/ratioy
    #         x1,y1,x2,y2 = x1.tolist(),y1.tolist(),x2.tolist(),y2.tolist()
            
      #      output_dic[token] = [x1,y1,x2,y2]

    for iteration, (img, mask, target) in enumerate(tqdm(prefetcher_test)):
        target_dict = target.tensor_dict
        word_id, word_mask = target_dict['word_id'], target_dict['word_mask']
        gt_bbox = target_dict['orig_bbox']
        data_time.update(time.time() - end)

        outputs = model(img, mask, word_id, word_mask)
        pred_boxes = postprocessor(outputs, target_dict)
        

        for i in range(pred_boxes.shape[0]):
            id = target_dict['id'][i].tolist()
            id = str(id)
            # dx,dy = target_dict['dxdy'][i]
            # ratiox,ratioy =  target_dict['ratio'][i]
            token = target_dict['command_token'][i]
            
            x1,y1,x2,y2 = pred_boxes[i].tolist()

            output_dic[token] =[x1,y1,x2-x1,y2-y1]
    
    with open("./resultes/prediction.json",'w+') as f3:
        json.dump(output_dic,f3)
        return

def val_output(model, criterion, postprocessor, data_loader_val, device, save_path=''):
    model.eval()
    if criterion:
        criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    iter_time = utils.SmoothedValue(fmt='{avg:.3f}')
    data_time = utils.SmoothedValue(fmt='{avg:.3f}')

    accum_acc = 0
    accum_iou = 0
    accum_sample = 0
    iou_thrs = torch.as_tensor([0.5 + 0.05 * i for i in range(0,9)], device=device)

    end = time.time()

    all_pred_ious = []
    all_pred_boxes = []
    prefetcher = data_prefetcher(data_loader_val, device)
    for iteration, (img, mask, target) in enumerate(tqdm(prefetcher)):
        target_dict = target.tensor_dict
        word_id, word_mask = target_dict['word_id'], target_dict['word_mask']

        gt_bbox = target_dict['orig_bbox']

        data_time.update(time.time() - end)

        outputs = model(img, mask, word_id, word_mask)
        
        if criterion:
            loss_dict = criterion(outputs, target_dict)
            weight_dict = criterion.weight_dict

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                        for k, v in loss_dict_reduced.items() if k in weight_dict}
            loss_value = sum(loss_dict_reduced_scaled.values()).item()
            metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)


        pred_boxes = postprocessor(outputs, target_dict)

        ious = box_ops.box_pair_iou(gt_bbox, pred_boxes)[0]
        sum_iou = ious.sum()
        num_acc = (ious[:, None] > iou_thrs[None]).sum(dim=0)
        num_sample = torch.as_tensor(img.size(0), device=img.device)

        accum_acc += num_acc
        accum_iou += sum_iou
        accum_sample += num_sample

        iter_time.update(time.time() - end)
        end = time.time()

        all_pred_ious.append(ious.view(-1, 1))
        all_pred_boxes.append(pred_boxes)

    if save_path:
        torch.save({'pred_boxes': torch.cat(all_pred_boxes, dim=0),
                    'pred_ious': torch.cat(all_pred_ious, dim=0)},
                   save_path + 'pred_boxes')
    # accumulate predictions from all images
    if utils.get_world_size() > 1:
        dist.all_reduce(accum_acc)
        dist.all_reduce(accum_iou)
        dist.all_reduce(accum_sample)

    acc = accum_acc / accum_sample.float().item()
    miou = accum_iou.item() / accum_sample.float().item()

    val_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    val_acc = {f'Val Acc@{t:.2f}': a.item() for t, a in zip(iou_thrs, acc)}
    print(val_acc)
    val_acc.update({'Val Mean_iou': miou})
    print('Val Mean_iou',miou)
    val_time = {'Val data_time': data_time.global_avg, 'Val time': iter_time.global_avg}
    return val_stats, val_acc, val_time

def test_output(model, criterion, postprocessor, data_loader_test, device, save_path=''):
    model.eval()
    if criterion:
        criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    iter_time = utils.SmoothedValue(fmt='{avg:.3f}')
    data_time = utils.SmoothedValue(fmt='{avg:.3f}')

    accum_acc = 0
    accum_iou = 0
    accum_sample = 0
    iou_thrs = torch.as_tensor([0.5 + 0.05 * i for i in range(0,9)], device=device)

    end = time.time()

    all_pred_ious = []
    all_pred_boxes = []
    prefetcher = data_prefetcher(data_loader_test, device)
    for iteration, (img, mask, target) in enumerate(tqdm(prefetcher)):
        target_dict = target.tensor_dict
        word_id, word_mask = target_dict['word_id'], target_dict['word_mask']

        gt_bbox = target_dict['orig_bbox']

        data_time.update(time.time() - end)

        outputs = model(img, mask, word_id, word_mask)
        
        if criterion:
            loss_dict = criterion(outputs, target_dict)
            weight_dict = criterion.weight_dict

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                        for k, v in loss_dict_reduced.items() if k in weight_dict}
            loss_value = sum(loss_dict_reduced_scaled.values()).item()
            metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)


        pred_boxes = postprocessor(outputs, target_dict)

        ious = box_ops.box_pair_iou(gt_bbox, pred_boxes)[0]
        sum_iou = ious.sum()
        num_acc = (ious[:, None] > iou_thrs[None]).sum(dim=0)
        num_sample = torch.as_tensor(img.size(0), device=img.device)

        accum_acc += num_acc
        accum_iou += sum_iou
        accum_sample += num_sample

        iter_time.update(time.time() - end)
        end = time.time()

        all_pred_ious.append(ious.view(-1, 1))
        all_pred_boxes.append(pred_boxes)

    if save_path:
        torch.save({'pred_boxes': torch.cat(all_pred_boxes, dim=0),
                    'pred_ious': torch.cat(all_pred_ious, dim=0)},
                   save_path + 'pred_boxes')
    # accumulate predictions from all images
    if utils.get_world_size() > 1:
        dist.all_reduce(accum_acc)
        dist.all_reduce(accum_iou)
        dist.all_reduce(accum_sample)

    acc = accum_acc / accum_sample.float().item()
    miou = accum_iou.item() / accum_sample.float().item()

    val_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    val_acc = {f'Test Acc@{t:.2f}': a.item() for t, a in zip(iou_thrs, acc)}
    
    val_acc.update({'Test Mean_iou': miou})
    print('Test Mean_iou',miou)
    val_time = {'Test data_time': data_time.global_avg, 'Test time': iter_time.global_avg}
    return val_stats, val_acc, val_time
        
@torch.no_grad()
def plot_talk2car(model, criterion, postprocessor, data_loader, device, save_path='',split='train'):
    train_data_path ="./Talk2Car/data/commands/train_commands.json"
    val_data_path = "./Talk2Car/data/commands/val_commands.json"
    test_data_path = "./Talk2Car/data/commands/test_commands.json"
    img_data_path = "./Talk2Car/data/image"


    with open(train_data_path) as f2:
        train_data = json.load(f2)

    with open(val_data_path) as f3:
        val_data = json.load(f3)

    with open(test_data_path) as f4:
        test_data = json.load(f4)   
    
    model.eval()
    if criterion:
        criterion.eval()

    for iteration, (img, mask, target) in enumerate(tqdm(data_loader)):
        target_dict = target.tensor_dict
        word_id, word_mask = target_dict['word_id'], target_dict['word_mask']
        gt_bbox = target_dict['orig_bbox']

        outputs = model(img, mask, word_id, word_mask)
        pred_boxes = postprocessor(outputs, target_dict)
        
        for i in range(pred_boxes.shape[0]):
            id = target_dict['id'][i].tolist()
            id = str(id)
            token = target_dict['command_token'][i]
            x1,y1,x2,y2 = pred_boxes[i].tolist()
        

            bbox = [x1,y1,x2-x1,y2-y1]

            for i,data in enumerate(val_data['commands'][:]):
                if data['command_token'] == token:
                    img_path = os.path.join(img_data_path,data['t2c_img'])
                    with open(img_path, 'rb') as f_img:
                        img = Image.open(f_img).convert('RGB')
                    gt_bbox = data['2d_box']
                    x1,y1,w1,h1 = gt_bbox[0],gt_bbox[1],gt_bbox[2],gt_bbox[3]# gt_bbox[0],gt_bbox[1],gt_bbox[0]+gt_bbox[2],gt_bbox[1]+gt_bbox[3]
                    fig, ax = plt.subplots(1)
                    ax.imshow(img)
                    rect = patches.Rectangle((x1, y1), w1, h1, fill=False, edgecolor='r')
                    ax.add_patch(rect)


                    x2,y2,w2,h2 = bbox[0],bbox[1],bbox[2],bbox[3]

                    rect = patches.Rectangle((x2, y2), w2, h2, fill=False, edgecolor='g')
                    ax.add_patch(rect)
                    
                    output_dir = "./resultes/VLTVG_output_test"
                    plt.axis('off')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, 'vltvg_val_{}_bboxes.png'.format(i)), bbox_inches='tight')
                    plt.close()   
                    print("successful")
                    break
    

 