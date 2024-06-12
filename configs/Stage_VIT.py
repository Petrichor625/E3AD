
# python -m torch.distributed.launch --nproc_per_node=2 --use_env train_talk2car.py --config /home/zhenningli/lhc/Talk2car/VLTVG_RegionProposal/configs/Stage_VIT.py --model_name='Stage1'
dataset='gref'
model_name = 'Stage1'
checkpoint_best=True
lr = 5e-5

lr_bert = 5e-7
lr_backbone = 5e-7

output_dir='/home/HN/Stage1/work_dirs/Stage1_VIT_huge'
#resume = '/home/HN/Stage1/work_dirs/Stage1_without_vltvg/checkpoint0007.pth'
blip_checkpoint_path = "/home/HN/download/model_large_retrieval_coco.pth"
#resume ='/home/HN/Stage1/work_dirs/Stage1_without_vltvg/checkpoint_best_acc.pth'
use_checkpoint_optimizer=False

batch_size=16
epochs=30

freeze_epochs=10
freeze_modules=['visual_encoder', 'bert',]
load_weights_path='pretrained_checkpoints/detr-r101-gref.pth'

warm_up_ratio=0
adam_epsilon=1e-6
adam_betas=[0.9,0.99]

bbox_loss_coef=0
giou=0
FocalLoss=10
binary_tversky=1
TverskyLoss=1
BCEWithLogitLoss_up=0
BCEWithLogitLoss_down=0
DiceLoss = 4
TverskyLoss_a=0.3
TverskyLoss_b=0.7

backbone='resnet50'
model_config = dict()
