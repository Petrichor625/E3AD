import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import util.misc as utils
from util.misc import collate_fn_with_mask as collate_fn
from engine import train_CAVG_one_epoch, train_one_epoch_w_accum,evaluate_CAVG
from models import *

from datasets import build_Talk2car_dataset, train_transforms, test_transforms

from util.logger import get_logger
from util.config import Config


def get_args_parser():
    parser = argparse.ArgumentParser('Transformer-based visual grounding', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-4, type=float)
    parser.add_argument('--lr_vis_enc', default=1e-5, type=float)
    parser.add_argument('--lr_bert', default=1e-5, type=float)

    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument('--lr_drop', default=60, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--checkpoint_step', default=1, type=int)
    parser.add_argument('--checkpoint_latest', action='store_true')
    parser.add_argument('--checkpoint_best', action='store_true')

    # Model parameters
    parser.add_argument('--load_weights_path', type=str, default=None,
                        help="Path to the pretrained model.")
    parser.add_argument('--freeze_modules', type=list, default=[])
    parser.add_argument('--freeze_param_names', type=list, default=[])
    parser.add_argument('--freeze_epochs', type=int, default=1)
    parser.add_argument('--freeze_losses', type=list, default=[])

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=1, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Bert
    parser.add_argument('--bert_model', default='bert-base-uncased', type=str,
                        help='Bert model')
    parser.add_argument('--bert_token_mode', default='bert-base-uncased', type=str, help='Bert tokenizer mode')
    parser.add_argument('--bert_output_dim', default=768, type=int,
                        help='Size of the output of Bert')
    parser.add_argument('--bert_output_layers', default=4, type=int,
                        help='the output layers of Bert')
    parser.add_argument('--max_query_len', default=40, type=int,
                        help='The maximum total input sequence length after WordPiece tokenization.')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    parser.add_argument('--loss_loc', default='loss_boxes', type=str,
                        help="The loss function for the predicted boxes")
    parser.add_argument('--box_xyxy', default=False,help='Use xyxy format to encode bounding boxes')
    # * Loss coefficients
    parser.add_argument('--bbox_loss_coef', default=0.2, type=float)
    parser.add_argument('--giou_loss_coef', default=0.2, type=float)
    parser.add_argument('--BCEWithLogitLoss', default=2, type=float)
    parser.add_argument('--other_loss_coefs', default={}, type=float)

    # dataset parameters
    parser.add_argument('--data_root', default='./data/')
    parser.add_argument('--split_root', default='./split/data/')
    parser.add_argument('--dataset', default='gref')
    parser.add_argument('--test_split', default='val')
    parser.add_argument('--img_size', default=900)
    parser.add_argument('--cache_images', action='store_true')
    parser.add_argument('--output_dir', default='work_dirs/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--save_pred_path', default='')
    parser.add_argument('--device', default='cuda:1',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_memory', default=True, type=boolean_string)
    parser.add_argument('--collate_fn', default='collate_fn')
    parser.add_argument('--batch_size_val', default=8, type=int)
    parser.add_argument('--batch_size_test', default=1, type=int)
    parser.add_argument('--train_transforms', default=train_transforms)
    parser.add_argument('--test_transforms', default=test_transforms)
    parser.add_argument('--enable_batch_accum', action='store_true')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # configure file
    parser.add_argument('--config', default="/home/tam/Documents/RSDLayerAttn/VLTVG_RegionProposal/configs/VLTVG_R101_talk2car_Region_proposal.py")
    parser.add_argument('--model_name', default='CAVG')
    parser.add_argument('--model_config')
    parser.add_argument('--checkpoint',default='None')
    
    return parser


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


class ParamFreezer(object):
    def __init__(self, module_names, param_names=[]):
        self.module_names = module_names
        self.freeze_params = dict()
        self.global_param_names = param_names

    def freeze(self, model):
        for name in self.module_names:
            module = getattr(model, name)
            self.freeze_params[name] = list()
            for k, v in module.named_parameters():
                if v.requires_grad:
                    v.requires_grad_(False)
                    self.freeze_params[name].append(k)

        if len(self.global_param_names) == 0:
            return
        for k, v in model.named_parameters():
            if k in self.global_param_names and v.requires_grad:
                v.requires_grad_(False)

    def unfreeze(self, model):
        for name in self.module_names:
            module = getattr(model, name)
            keys = self.freeze_params[name]
            for k, v in module.named_parameters():
                if k in keys:
                    v.requires_grad_(True)

        if len(self.global_param_names) == 0:
            return
        for k, v in model.named_parameters():
            if k in self.global_param_names:
                v.requires_grad_(True)


def main(args):
    utils.init_distributed_mode(args)

    logger = get_logger("train", args.output_dir, utils.get_rank(), filename='iter.log')
    epoch_logger = get_logger("train_epoch", args.output_dir, utils.get_rank(), filename='epoch.log')
    logger.info(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessor = build_model_with_region_proposal(args)
    model.to(device)

    if args.checkpoint != 'None':
        # 使用 torch.load 加载 checkpoint 文件
        checkpoint = torch.load(args.checkpoint)
        # 从 checkpoint 中提取权重参数 state_dict
        print(checkpoint['epoch'])
        model.load_state_dict(checkpoint['model'])
        print(f"successful load checkpint{args.checkpoint}")    
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
        
    dataset_train = build_Talk2car_dataset(args=args,split_talk2car='train')
    dataset_val = build_Talk2car_dataset(args=args,split_talk2car='val')

    if args.model_name == 'vltvg':
        backbone_param = [p for p in model_without_ddp.backbone.parameters() if p.requires_grad]
        
        vis_enc_param = [p for n, p in model_without_ddp.named_parameters() if p.requires_grad and
                        (n.startswith('trans_encoder') or n.startswith('input_proj'))]
        
        bert_param = [p for p in model_without_ddp.bert.parameters() if p.requires_grad]
        
        rest_param = [p for n, p in model_without_ddp.named_parameters() if p.requires_grad and
                    (n.startswith('trans_decoder') or n.startswith('bbox_embed') or n.startswith('bert_proj'))]

        cnt_backbone = sum([p.numel() for p in backbone_param])
        cnt_vis_enc = sum([p.numel() for p in vis_enc_param])
        cnt_bert = sum([p.numel() for p in bert_param])
        cnt_rest = sum([p.numel() for p in rest_param])
        cnt_whole = sum([p.numel() for p in model_without_ddp.parameters() if p.requires_grad])

        logger.info(f'The num of learnable parameters: backbone({cnt_backbone}), vis_enc({cnt_vis_enc}), '
                    f'bert({cnt_bert}), rest({cnt_rest})')
        logger.info(f'Check the whole parameters: {cnt_whole} = {cnt_backbone + cnt_vis_enc + cnt_bert + cnt_rest}')

        param_dicts = [{'params': rest_param}, # base_lr
                    {'params': backbone_param, 'lr': args.lr_backbone}, # base_lr/10.
                    {'params': vis_enc_param, 'lr': args.lr_vis_enc},
                    {'params': bert_param, 'lr': args.lr_bert},] # base_lr/10.

        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,weight_decay=args.weight_decay)
        
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
        
    elif args.model_name == 'CAVG_with_GPT_Score':
        
        backbone_param = [p for p in model_without_ddp.backbone.parameters() if p.requires_grad]
        vis_enc_param = [p for n, p in model_without_ddp.named_parameters() if p.requires_grad and (n.startswith('trans_encoder') or n.startswith('input_proj'))]
        bert_param = [p for p in model_without_ddp.bert.parameters() if p.requires_grad]
        blip_parm = [p for p in model_without_ddp.Extra_Information_Embedding.parameters() if p.requires_grad]
        
        rest_param = [p for p in model.parameters() if p.requires_grad and not (p in backbone_param or p in vis_enc_param or p in bert_param or p in blip_parm)]
        
        param_dicts = [{'params': rest_param}, # base_lr
                    {'params': backbone_param, 'lr': args.lr_backbone}, # base_lr/10.
                    {'params': vis_enc_param, 'lr': args.lr_vis_enc},
                    {'params': bert_param, 'lr': args.lr_bert},
                    {'params': blip_parm, 'lr': args.lr_bert},] 
        
        # optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr,
        #                               weight_decay=args.weight_decay)
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
        
        optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr,
                                      weight_decay=args.weight_decay)
        
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
        
    elif args.model_name == 'CAVG_without_score':
        # backbone_param = [p for p in model_without_ddp.backbone.parameters() if p.requires_grad]
        # vis_enc_param = [p for n, p in model_without_ddp.named_parameters() if p.requires_grad and (n.startswith('trans_encoder') or n.startswith('input_proj'))]
        #bert_param = [p for p in model_without_ddp.bert.parameters() if p.requires_grad]
        blip_parm = [p for p in model_without_ddp.Extra_Information_Embedding.parameters() if p.requires_grad]
        
        rest_param = [p for p in model.parameters() if p.requires_grad and not (p in blip_parm)]
        
        param_dicts = [{'params': rest_param}, # base_lr
                        {'params': blip_parm, 'lr': args.lr_bert},] 
        
        optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr,
                                      weight_decay=args.weight_decay)
        # scheduler对优化器的学习率进行调整
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)        
    
    else:
        optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr,
                                      weight_decay=args.weight_decay)
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
        
        from pytorch_transformers.optimization import AdamW, WarmupConstantSchedule, WarmupLinearSchedule
        
        epoch = 50
        batch_size = 32
        total_steps = int((len(dataset_train) // batch_size) * epoch) if len(dataset_train) % batch_size == 0 else (len(dataset_train) // batch_size + 1) * epoch 
        # 每一个epoch中有多少个step可以根据len(DataLoader)计算：total_steps = len(DataLoader) * epoch
        warm_up_ratio = 0.2

        lr_scheduler = WarmupLinearSchedule(optimizer, warmup_steps= warm_up_ratio * total_steps, t_total= total_steps)
    



    logger.info(f'The size of dataset: train({len(dataset_train)})  val({len(dataset_val)})')

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if dataset_val.cache_images==True:
        for i in sampler_val: dataset_val.cache(i)


 # batch_sampler_train = torch.utils.data.BatchSampler(
    #     sampler_train, args.batch_size, drop_last=True)
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)
    batch_sampler_val = torch.utils.data.BatchSampler(
        sampler_val, args.batch_size, drop_last=True)
    # batch_sampler_val = torch.utils.data.BatchSampler(
    #     sampler_val, args.batch_size, drop_last=True)

    # data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
    #                                pin_memory=args.pin_memory, collate_fn=collate_fn,
    #                                num_workers=args.num_workers)
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   pin_memory=args.pin_memory, collate_fn=collate_fn,
                                   )
    
    data_loader_val = DataLoader(dataset_val, args.batch_size_val, sampler=sampler_val,
                                 pin_memory=args.pin_memory, drop_last=False,
                                 collate_fn=collate_fn)
    # 111梦开始的地方
    epoch_trainer = train_CAVG_one_epoch

    if args.enable_batch_accum:
        epoch_trainer = train_one_epoch_w_accum
    epoch_eval = evaluate_CAVG

    output_dir = Path(args.output_dir)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
    # elif args.load_weights_path:
    #     model_without_ddp.load_pretrained_weights(args.load_weights_path)

    if args.eval:
        print(epoch_eval)
        test_stats, test_acc, test_time = epoch_eval(model, criterion, postprocessor, data_loader_val, device, args.save_pred_path)
        logger.info('  '.join(['[Test accuracy]',
                         *[f'{k}: {v:.4f}' for k, v in test_acc.items()],
                         '\n[Test time]',
                         *[f'{k}: {v:.6f}' for k, v in test_time.items()]]))
        return
    outlist = ['CAVG_with_GPT_Score','CAVG_V2']
    
    if args.model_name =='CAVG':
        if args.start_epoch < args.freeze_epochs and args.freeze_modules:
            logger.info(f'Freeze weights: {args.freeze_modules} and {args.freeze_param_names}')
            param_freezer = ParamFreezer(args.freeze_modules, args.freeze_param_names)
            param_freezer.freeze(model_without_ddp)
            if args.distributed: # re-wrap the model to avoid error: 'parameters that were not used in producing loss'
                model = torch.nn.parallel.DistributedDataParallel(model_without_ddp, device_ids=[args.gpu])
                model_without_ddp = model.module


    logger.info("Start training")
    start_time = time.time()
    best_acc = 0
    for epoch in range(args.start_epoch, args.epochs):
        torch.cuda.empty_cache()
        if args.distributed:
            sampler_train.set_epoch(epoch)

        # 2222
        train_stats = epoch_trainer(
            model, criterion, data_loader_train, optimizer, device, epoch, args.epochs, args.clip_max_norm
        )
        if args.model_name =='CAVG':
            if (epoch + 1) == args.freeze_epochs and args.freeze_modules:
                logger.info(f'Unfreeze weights: {args.freeze_modules}')
                param_freezer.unfreeze(model_without_ddp)
                if args.distributed: # re-wrap the model to ensure the same gradients for unfrozen weights
                    model = torch.nn.parallel.DistributedDataParallel(model_without_ddp, device_ids=[args.gpu])
                    model_without_ddp = model.module

        lr_scheduler.step()

        val_stats, val_acc, _ = epoch_eval(
            model, criterion, postprocessor, data_loader_val, device, args.save_pred_path
        )

        if args.output_dir:
            print(f'Save model to {args.output_dir}')
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            if args.checkpoint_best:
                if val_acc['Acc@0.50'] > best_acc:
                    checkpoint_paths.append(output_dir / 'checkpoint_best_acc.pth')
                    best_acc = val_acc['Acc@0.50']
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.checkpoint_step == 0:
                if args.checkpoint_latest:
                    checkpoint_paths.append(output_dir / 'checkpoint_latest.pth')
                else:
                    checkpoint_paths.append(output_dir / f'checkpoint{epoch+1:04}.pth')

            for checkpoint_path in checkpoint_paths:
                if checkpoint_path.name == 'checkpoint.pth':
                    if (epoch + 1) == args.epochs:
                        utils.save_on_master({
                            'model': model_without_ddp.state_dict(),
                            'epoch': epoch,
                            'args': args,
                        }, checkpoint_path)
                elif checkpoint_path.name in ['checkpoint_best_acc.pth']:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }, checkpoint_path)
                else:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }, checkpoint_path)


        if args.output_dir and utils.is_main_process():
            epoch_logger.info('  '.join(
                [f'Epoch [{epoch + 1}](train stats)',
                 *[f'train_{k}: {v:.4f}' for k, v in train_stats.items()]]))
            epoch_logger.info('  '.join(
                [f'Epoch [{epoch + 1}](val stats)',
                 *[f'  val_{k}: {v:.4f}' for k, v in val_stats.items()]]))
            epoch_logger.info('  '.join(
                [f'Epoch [{epoch + 1}](val acc)',
                 *[f'{k}: {v:.4f}' for k, v in val_acc.items()]]))
            epoch_logger.info('\n')


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    import multiprocessing

# 在主程序中设置启动方法为'spawn'
    multiprocessing.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser('VLTVG training script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.config:
        cfg = Config(args.config)
        cfg.merge_to_args(args)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
# /home/HaichengLiao/VLTVG_region_proposal/train_talk2car.py --config /home/HaichengLiao/VLTVG_region_proposal/configs/VLTVG_R101_talk2car_Region_proposal.py
# /home/HaichengLiao/VLTVG_region_proposal/train_talk2car.py --config /home/HaichengLiao/VLTVG_region_proposal/configs/VLTVG_R101_talk2car_Region_proposal_without_gpt.py
# /home/HaichengLiao/VLTVG_region_proposal/train_talk2car.py --config /home/HaichengLiao/VLTVG_region_proposal/configs/VLTVG_R101_talk2car_Region_proposal_without_gpt.py --model_name CAVG_with_score_gpt


# pip install torch==1.8.0 torchvision==0.9.0
# pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
# # python -m torch.distributed.launch --nproc_per_node=1 --use_env train.py --config /home/tam/Documents/RSDLayerAttn/VLTVG/configs/VLTVG_R50_gref.py 