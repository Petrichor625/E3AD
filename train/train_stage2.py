# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2020, Emanuele Bugliarello (@e-bug).

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import json
import yaml
import random
import logging
import argparse
from io import open
from tqdm import tqdm
#from easydict import EasyDict as edict
# from sklearn.metrics import f1_score
import numpy as np
from easydict import EasyDict as edict
import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts

from pytorch_transformers.optimization import AdamW, WarmupConstantSchedule, WarmupLinearSchedule
from torch.optim import Adam

from volta.config import BertConfig
from volta.optimization import RAdam
from volta.encoders import MyBertForVLTasks


from Mymodels.blip_nlvr import blip_nlvr

from volta.train_utils import freeze_layers, tbLogger, summary_parameters, save, resume

from volta.task_utils import LoadDataset, LoadLoss, ForwardModelsTrain, ForwardModelsVal

# /home/tam/Documents/RSDLayerAttn/BLIP
# /home/tam/Documents/RSDLayerAttn/RSDLayerAttn

from Mymodels.blip import load_checkpoint
from Mymodels.blip_itm import blip_itm



logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument("--from_pretrained", default="bert-base-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--config_file", default="config/vilbert_base.json", type=str,
                        help="The config file which specified the model details.")
    parser.add_argument("--MEconfig", default="/home/tam/Documents/RSDLayerAttn/RSDLayerAttn/Mymodels/med_config.json", type=str,
                    help="The config file which specified the model details.")
    parser.add_argument("--resume_file", default="", type=str,
                        help="Resume from checkpoint")
    # Output
    parser.add_argument("--output_dir", default="save", type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--logdir", default="logs", type=str,
                        help="The logging directory where the training logs will be written.")
    parser.add_argument("--save_name", default="", type=str,
                        help="save name for training.")
    # Task
    parser.add_argument("--tasks_config_file", default="config_tasks/vilbert_trainval_tasks.yml", type=str,
                        help="The config file which specified the tasks details.")
    parser.add_argument("--task", default="", type=str,
                        help="training task number")
    parser.add_argument("--probe_layer_idx", default=None, type=int,
                        help="The layer to probe for layer probing")
    parser.add_argument("--weighted_sampling", default=False, action='store_true',
                        help="Use weighted random sampler for imbalanced learning.")
    # Training
    parser.add_argument("--num_train_epochs", default=20, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--gradient_accumulation_steps", dest="grad_acc_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--drop_last", action="store_true",
                        help="whether to drop last incomplete batch")
    # Scheduler
    parser.add_argument("--lr_scheduler", default="warmup_linear", type=str,
                        help="whether use learning rate scheduler.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--warmup_steps", default=None, type=float,
                        help="Number of training steps to perform linear learning rate warmup for. "
                             "It overwrites --warmup_proportion.")
    # Seed
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed for initialization")
    # Distributed
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--num_workers", type=int, default=16,
                        help="Number of workers in the dataloader.")
    parser.add_argument("--in_memory", default=False, type=bool,
                        help="whether use chunck for parallel training.")
    # Optimization
    parser.add_argument("--optim", default="AdamW", type=str,
                        help="what to use for the optimization.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--adam_betas", default=(0.9, 0.999), nargs="+", type=float,
                        help="Betas for Adam optimizer.")
    parser.add_argument("--adam_correct_bias", default=False, action='store_true',
                        help="Correct bias for Adam optimizer.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay for Adam optimizer.")
    parser.add_argument("--clip_grad_norm", default=0.0, type=float,
                        help="Clip gradients within the specified range.")
                        
    parser.add_argument("--blip_pretrained", default='https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_14M.pth', type=str,
                        )

    return parser.parse_args()

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")
    print(f"Non-Trainable Parameters: {non_trainable_params}")

def main():
    args = parse_args()
    
    # Devices 此处为分布式训练
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend="nccl")
    default_gpu = False
    if dist.is_available() and args.local_rank != -1:
        rank = dist.get_rank()
        if rank == 0:
            default_gpu = True
    else:
        default_gpu = True
    logger.info(f"device: {device} n_gpu: {n_gpu}, distributed training: {bool(args.local_rank != -1)}")

    # Load config 
    config = BertConfig.from_json_file(args.config_file)
    Meconfig = yaml.load(open(args.MEconfig, 'r'), Loader=yaml.Loader)

    # Load task config 超参数加载
    with open(args.tasks_config_file, "r") as f:
        task_cfg = edict(yaml.safe_load(f))
    task_id = args.task.strip()
    task = "TASK" + task_id
    
    task_name = task_cfg[task]["name"]
    base_lr = task_cfg[task]["lr"]
    if task_cfg[task].get("fusion_method", None):
        # VL-BERT pooling for VQA
        config.fusion_method = task_cfg[task]["fusion_method"]
    # Output dirs
    if args.save_name:
        prefix = "-" + args.save_name
    else:
        prefix = ""
    timestamp = (task_name + "_" + args.config_file.split("/")[1].split(".")[0] + prefix)
    save_path = os.path.join(args.output_dir, timestamp)
    if default_gpu:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # save all the hidden parameters.
        with open(os.path.join(save_path, "command.txt"), "w") as f:
            print(args, file=f)  # Python 3.x
            print("\n", file=f)
           # print(config, file=f)

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    torch.manual_seed(3407) # 3407
    # Dataset
    '''
    batch_size: 批次大小，表示每次训练迭代中使用的样本数。
    task2num_iters: 一个映射（字典或类似的数据结构），将任务映射到迭代次数。这可能是用于不同任务的训练迭代次数。
    dset_train: 训练数据集对象。
    dset_val: 验证数据集对象。
    dl_train: dataloder_train训练数据加载器,用于获取训练数据的迭代器。
    dl_val: dataloder_val 验证数据加载器，用于获取验证数据的迭代器。
    '''
    # batch_size, task2num_iters, dset_train, dset_val, dl_train, dl_val
    batch_size, task2num_iters, dset_train, dset_val, dl_train, dl_val,dl_test = LoadDataset(args, config, task_cfg,args.task,data_imgs_root=None)

    # Logging
    logdir = os.path.join(args.logdir, timestamp)
    tb_logger = tbLogger(logdir, save_path, [task_name], [task], task2num_iters, args.grad_acc_steps)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    

    # Model
    model = MyBertForVLTasks.from_pretrained(args.from_pretrained,config=config, task_cfg=task_cfg, task_ids=[task], probe_layer_idx=args.probe_layer_idx,Meconfig=Meconfig)

    # model_val = blip_nlvr(pretrained=config['Val_pretrained'], image_size=config['image_size'], 
    count_parameters(model)

    '''
    vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'])
    model_dict = model.state_dict()

    blip_model = blip_itm()
    blip_model_dict = blip_model.state_dict()

    pretrained_dict = {k: v for k, v in blip_model_dict.items() if k in model_dict}
    print("-----------------------------------------------------blip pretrained----------------------------------------------------------------------")
    print(pretrained_dict)
    print("---------------------------------------------------------------------------------------------------------------------------------------")
    '''

    model_blip = blip_itm(pretrained=args.blip_pretrained)

    #'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_14M.pth')
    # model = blip_nlvr(model=model,pretrained=config.Val_pretrained) 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_nlvr.pth' # /home/tam/Documents/RSDLayerAttn/BLIP/Pretrained_Model/BIPw_VIPB_CFL/model_large.pth 
    count_parameters(model_blip)

    if task_cfg[task].get("embed_clf", None):
        logger.info('Initializing classifier weight for %s from pretrained word embeddings...' % task)
        answers_word_embed = []
        for k, v in model.state_dict().items():
            if 'bert.embeddings.word_embeddings.weight' in k:
                word_embeddings = v.detach().clone()
                break

        for answer, label in sorted(dset_train.ans2label.items()):
            a_tokens = dset_train._tokenizer.tokenize(answer)
            a_ids = dset_train._tokenizer.convert_tokens_to_ids(a_tokens)
            if len(a_ids):
                a_word_embed = (torch.stack([word_embeddings[a_id] for a_id in a_ids], dim=0)).mean(dim=0)
            else:
                a_tokens = dset_train._tokenizer.tokenize("<unk>")
                a_id = dset_train._tokenizer.convert_tokens_to_ids(a_tokens)[0]
                a_word_embed = word_embeddings[a_id]
            answers_word_embed.append(a_word_embed)
        answers_word_embed_tensor = torch.stack(answers_word_embed, dim=0)
        for name, module in model.named_modules():
            if name.endswith('clfs_dict.%s.logit_fc.3' % task):
                module.weight.data = answers_word_embed_tensor.to(device=module.weight.data.device)

    # Optimization details
    freeze_layers(model)
    
    print("-----------------------------------------------------Frozen Layer----------------------------------------------------------------------")
    #print(model.config.fixed_layers)
    print("--------------------------------------------------------------------------------------------------------------------------------------")



    criterion = LoadLoss(task_cfg, args.task)
    print("Loss:")
    print(criterion)
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = []

    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if "vil_" in key:
                print(key)
                lr = 1e-43
            else:
                lr = base_lr
            if any(nd in key for nd in no_decay):
                optimizer_grouped_parameters += [{"params": [value], "lr": lr, "weight_decay": 0.0}]
            if not any(nd in key for nd in no_decay):
                optimizer_grouped_parameters += [{"params": [value], "lr": lr, "weight_decay": args.weight_decay}]
    
    # 

    if default_gpu:
        print(len(list(model.named_parameters())), len(optimizer_grouped_parameters))

    # 选择优化器
    if args.optim == "AdamW":
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=base_lr,
                          eps=args.adam_epsilon,
                          betas=args.adam_betas,
                          correct_bias=args.adam_correct_bias)
    elif args.optim == "RAdam":
        optimizer = RAdam(optimizer_grouped_parameters, lr=base_lr)
    elif args.optim == "Adam":
        optimizer = Adam(optimizer_grouped_parameters, lr=base_lr)

    num_train_optim_steps = (task2num_iters[task] * args.num_train_epochs // args.grad_acc_steps)

    warmup_steps = args.warmup_steps or args.warmup_proportion * num_train_optim_steps
    if args.lr_scheduler == "warmup_linear":
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=num_train_optim_steps)
    else:
        scheduler = WarmupConstantSchedule(optimizer, warmup_steps=warmup_steps)


    start_iter_id, global_step, start_epoch, tb_logger, max_score = \
        resume(args.resume_file, model, optimizer, scheduler, tb_logger)

    # Move to GPU(s)
    model.to(device)
    model_blip.to(device)

    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()

    # 不使用分布式训练，就单卡训练就oK
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
            )
        model = DDP(model, delay_allreduce=True)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)


    # Save starting model 
    # 在模型最开始运行的时候(epcho=1)奖励存档
    
    if start_epoch == 0:
        save(save_path, logger, -1, model, model_blip,optimizer, scheduler, global_step, tb_logger, default_gpu, max_score)

    # Print summary
    if default_gpu:
        summary_parameters(model, logger)
        print("***** Running training *****")
        print("  Num Iters: ", task2num_iters[task])
        print("  Batch size: ", batch_size)
        print("  Num steps: %d" % num_train_optim_steps)

    # Train
    for epoch_id in tqdm(range(start_epoch, args.num_train_epochs), desc="Epoch"):
        model.train()

        # feature_data_dict = {}
        # flag = 1
        # for step, batch in enumerate(dl_train):
            
        #     batch = tuple(t.cuda(device=device, non_blocking=True) if not isinstance(t, list) else t for t in batch)
        #     features, spatials, spatials_ori, image_mask, question, target, input_mask, segment_ids, question_id, imgs, txt_raw, name = batch

        #     if flag == 1:
        #         print(spatials_ori[0])
        #         flag = 0
        #     for j in range(features.shape[0]):
        #         dict1 = {}
        #         dict1["all_region_feature"] = features[j]
        #         dict1["raw_text"] = txt_raw[j]
        #         dict1["all_region_IOU"] = target[j]

        #         dict1["all_region_bbox_ori"] = spatials_ori[j,:, :].cpu().detach().tolist()
        #         dict1["all_region_bbox"] = spatials[j,:,:].cpu().detach().tolist()

        #         dict1["image_mask"] = image_mask[j]
        #         dict1["input_mask"] = input_mask[j]
        #         dict1["question_id"] = question_id[j]
        #         dict1["question"]=question[j]
        #         dict1["segment_ids"] = segment_ids[j]
                

              
        #         raw_text_key = name[j]

        #         # 检查key是否已存在
        #         if raw_text_key in feature_data_dict:
        #             print(f"Key {raw_text_key} already exists. Taking some action.")
        #             # 如果需要，可以在这里执行一些操作，比如更新dict1或合并数据
        #         else:
        #             feature_data_dict[raw_text_key] = dict1

        #     print(len(feature_data_dict.keys()))
        # # file_path = "/home/tam/Documents/RSDLayerAttn/MyTrain/Talk2Car/data/train_feature_data.pt"
        # # torch.save(feature_data_dict,file_path)

        # for step, batch in enumerate(dl_val):
            
        #     batch = tuple(t.cuda(device=device, non_blocking=True) if not isinstance(t, list) else t for t in batch)
        #     features, spatials, spatials_ori, image_mask, question, target, input_mask, segment_ids, question_id, imgs, txt_raw, name = batch

        #     for j in range(features.shape[0]):
        #         dict1 = {}
        #         dict1["all_region_feature"] = features[j]
        #         dict1["raw_text"] = txt_raw[j]
        #         dict1["all_region_IOU"] = target[j]
        #         dict1["all_region_bbox_ori"] = spatials_ori[j,:, :].cpu().detach().tolist()
        #         dict1["all_region_bbox"] = spatials[j,:,:].cpu().detach().tolist()
        #         dict1["image_mask"] = image_mask[j]
        #         dict1["input_mask"] = input_mask[j]
        #         dict1["question_id"] = question_id[j]

        #         dict1["question"]=question[j]
        #         dict1["segment_ids"] = segment_ids[j]
        #         raw_text_key = name[j]
                
        #         print(raw_text_key)
        #         # 检查key是否已存在
        #         if raw_text_key in feature_data_dict:
        #             print(f"Key {raw_text_key} already exists. Taking some action.")
        #             # 如果需要，可以在这里执行一些操作，比如更新dict1或合并数据
        #         else:
        #             feature_data_dict[raw_text_key] = dict1

        # print(len(feature_data_dict.keys()))


        for step, batch in enumerate(dl_test):
            
            batch = tuple(t.cuda(device=device, non_blocking=True) if not isinstance(t, list) else t for t in batch)
            features, spatials, spatials_ori, image_mask, question, target, input_mask, segment_ids, question_id, imgs, txt_raw, name = batch

        #     for j in range(features.shape[0]):
        #         dict1 = {}
        #         dict1["all_region_feature"] = features[j]
        #         dict1["raw_text"] = txt_raw[j]
        #         dict1["all_region_IOU"] = target[j]
        #         dict1["all_region_bbox_ori"] = spatials_ori[j,:, :].cpu().detach().tolist()
        #         dict1["all_region_bbox"] = spatials[j,:,:].cpu().detach().tolist()
        #         dict1["image_mask"] = image_mask[j]
        #         dict1["input_mask"] = input_mask[j]
        #         dict1["question_id"] = question_id[j]
        #         dict1["question"]=question[j]
        #         dict1["segment_ids"] = segment_ids[j]
        #         raw_text_key = name[j]

                
        #         raw_text_key = name[j]
        #         print(raw_text_key)
        #         # 检查key是否已存在
        #         if raw_text_key in feature_data_dict:
        #             print(f"Key {raw_text_key} already exists. Taking some action.")
        #             # 如果需要，可以在这里执行一些操作，比如更新dict1或合并数据
        #         else:
        #             feature_data_dict[raw_text_key] = dict1

        # print(len(feature_data_dict.keys()))
        # file_path = "/home/tam/Documents/RSDLayerAttn/MyTrain/Talk2Car/data/region_feature_data.pt"
        # torch.save(feature_data_dict,file_path)
        
        
        for step, batch in enumerate(dl_test):

            iter_id = start_iter_id + step + (epoch_id * len(dl_train))
            # 在volta.task_utils中指定了不同种类情况下的传播方式{from volta.task_utils import LoadDataset, LoadLoss, ForwardModelsTrain, ForwardModelsVal}，结果返回loss和score
            # config args.config中的参数，由输入的时候指定config/${MODEL_CONFIG}.json
            # task_cfg为args.task_config指定的地址下保存的文件  config_tasks/${TASKS_CONFIG}.yml 里面存放了调用预训练模型路径和基本信息
            #   dataroot
            #   features_h5path1: data/talk2car/resnet101_faster_rcnn_genome_imgfeats_centernet/volta/refcoco+

            # task
            # batch
            # criterion 损失函数
            #############33

            # V-logit-fuse-self-attention
 
            loss, score = ForwardModelsTrain(config, task_cfg, device, task, batch, model, criterion,model_blip)

            if args.grad_acc_steps > 1:
                loss = loss / args.grad_acc_steps
            loss.backward()

            if (step + 1) % args.grad_acc_steps == 0:
                # Clip gradient
                if args.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)

                optimizer.step()

                if global_step < warmup_steps or args.lr_scheduler == "warmup_linear":
                    scheduler.step()

                model.zero_grad()
                global_step += 1

                if default_gpu:
                    tb_logger.step_train(epoch_id, iter_id, float(loss), float(score),
                                         optimizer.param_groups[0]["lr"], task, "train")

            if (step % (20 * args.grad_acc_steps) == 0) and step != 0 and default_gpu:
                tb_logger.showLossTrain()
            
            # Decide whether to evaluate task
            if iter_id != 0 and iter_id % task2num_iters[task] == 0:
                score = evaluate(config, dl_val, task_cfg, device, task, model,model_blip,criterion, epoch_id, default_gpu, tb_logger)
                if score > max_score:
                    max_score = score
                    save(save_path, logger, epoch_id, model,model_blip,optimizer, scheduler,
                         global_step, tb_logger, default_gpu, max_score, is_best=True)
                    


        logger.info('max validation score: {:.2f}'.format(max_score * 100))
        

        # score = evaluate(config, dl_val, task_cfg, device, task, model, criterion, epoch_id, default_gpu, tb_logger)

        save(save_path, logger, epoch_id, model,model_blip,optimizer, scheduler, global_step, tb_logger, default_gpu, max_score)


    tb_logger.txt_close()



def evaluate(config, dataloader_val, task_cfg, device, task_id, model,model_blip ,criterion, epoch_id, default_gpu, tb_logger):
    model.eval()
    print("task_id")
    print(task_id)
    if task_id == "TASK91" or task_id == "TASK95":  # for computing micro f1 score for probing task
        pred_all = []
        ref_all = []
        print("init holder for f1 score")
    for i, batch in enumerate(dataloader_val):
 
        loss, score, batch_size = ForwardModelsVal(config, task_cfg, device, task_id, batch, model,model_blip,criterion)

        if task_id == "TASK91" or task_id == "TASK95":  # for probing task
            acc_score, pred_list, ref_list = score
            pred_all += pred_list
            ref_all += ref_list
            score = acc_score
        tb_logger.step_val(epoch_id, float(loss), float(score), task_id, batch_size, "val")
        if default_gpu:
            sys.stdout.write("%d/%d\r" % (i, len(dataloader_val)))
            sys.stdout.flush()
    if task_id == "TASK91" or task_id == "TASK95":
        score = f1_score(ref_all, pred_all, average='macro')
        logger.info('validation score: {:.2f}'.format(score * 100))
    else:
        score = tb_logger.showLossVal(task_id)
    model.train()
    return score


    
if __name__ == "__main__":
    main()

