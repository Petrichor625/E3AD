
from .dataset import *

def build_dataset(test,args,Talk2car=False,split_talk2car='train'):
    if test:
        return VGDataset(data_root=args.data_root,
                         split_root=args.split_root,
                         dataset=args.dataset,
                         split=args.test_split,
                         test=True,
                         transforms=args.test_transforms,
                         max_query_len=args.max_query_len,
                         bert_mode=args.bert_token_mode)

    else:
        return VGDataset(data_root=args.data_root,
                          split_root=args.split_root,
                          dataset=args.dataset,
                          split='train',
                          transforms=args.train_transforms,
                          max_query_len=args.max_query_len,
                          bert_mode=args.bert_token_mode)

def build_Talk2car_dataset(args,split_talk2car='train'):

        if split_talk2car == 'val':
            print("talk2car val_dataset load")
            return Talk2Car_base(root=args.data_root,
                            split_root=args.split_root,
                            dataset=args.dataset,
                            split='val',
                            test=True,
                            transforms=test_transforms,
                            max_query_len=args.max_query_len,
                            bert_mode=args.bert_token_mode)
        elif split_talk2car == 'test':
            print("talk2car test_dataset load")
            return Talk2Car_base(root=args.data_root,
                            split_root=args.split_root,
                            dataset=args.dataset,
                            split='test',
                            test=True,
                            transforms=test_transforms,
                            max_query_len=args.max_query_len,
                            bert_mode=args.bert_token_mode)
        elif split_talk2car == 'train':
            print("talk2car train_dataset load")
            return Talk2Car_base(root=args.data_root,
                            split_root=args.split_root,
                            dataset=args.dataset,
                            split='train',
                            #transforms=args.test_transforms,
                            transforms=train_transforms,
                            max_query_len=args.max_query_len,
                            bert_mode=args.bert_token_mode)
        



train_transforms = [
    dict(
         # sui
        type='RandomSelect',
        transforms1=dict(type='RandomResize', sizes=[448, 480, 512, 544, 576, 608, 640]),
        transforms2=dict(
            type='Compose',
            transforms=[
                dict(type='RandomResize', sizes=[400, 500, 600], resize_long_side=False),
                dict(type='RandomSizeCrop', min_size=384, max_size=600, check_method=dict(func='iou', iou_thres=0.5)),
                dict(type='RandomResize', sizes=[448, 480, 512, 544, 576, 608, 640])
            ],
        ),
        p=0.5
    ),
    dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4),
    dict(type='RandomHorizontalFlip'),
    dict(type='ToTensor', keys=[]),
    dict(type='NormalizeAndPad', size=640, aug_translate=True)
]
# test_transforms = [
#     dict(type='RandomResize', sizes=[640], record_resize_info=True)
#     dict(type='ToTensor', keys=[]),
#     dict(type='NormalizeAndPad', size=, center_place=True)
# ]
test_transforms = [
    dict(type='RandomResize', sizes=[640], record_resize_info=True),
    dict(type='ToTensor', keys=[]),
    dict(type='NormalizeAndPad', size=640, center_place=True)
]


train_Talk2car_transforms = [
    dict(
         # sui
        type='RandomSelect',
        transforms1=dict(type='RandomResize', sizes=[1120, 1200, 1280, 1360, 1440, 1520, 1600]),
        transforms2=dict(
            type='Compose',
            transforms=[
                dict(type='RandomResize', sizes=[1000, 1250, 1500], resize_long_side=False),
                dict(type='RandomSizeCrop', min_size=960, max_size=1600, check_method=dict(func='iou', iou_thres=0.5)),
                dict(type='RandomResize', sizes=[1120, 1200, 1280, 1360, 1440, 1520, 1600])
            ],
        ),
        p=0.5
    ),
    dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4),
    dict(type='RandomHorizontalFlip'),
    dict(type='ToTensor', keys=[]),
    dict(type='NormalizeAndPad', size=1600, aug_translate=True)
]

test_Talk2car_transforms = [
    dict(type='RandomResize', sizes=[1600], record_resize_info=True),
    dict(type='ToTensor', keys=[]),
    dict(type='NormalizeAndPad', size=1600, center_place=True)
]


transforms = [
    dict(type='MyRandomResize', sizes=[1600], record_resize_info=True),
    dict(type='ToTensor', keys=[]),
    dict(type='MyNormalizeAndPad', size=1600, center_place=True)
]

mytrain_transforms = [
    dict(
         # sui
        type='RandomSelect',
        transforms1=dict(type='RandomResize', sizes=[448, 480, 512, 544, 576, 608, 640]),
        transforms2=dict(
            type='Compose',
            transforms=[
                dict(type='RandomResize', sizes=[400, 500, 600], resize_long_side=False),
                dict(type='RandomSizeCrop', min_size=384, max_size=600, check_method=dict(func='iou', iou_thres=0.5)),
                dict(type='RandomResize', sizes=[448, 480, 512, 544, 576, 608, 640])
            ],
        ),
        p=0.5
    ),
    dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4),
    dict(type='RandomHorizontalFlip'),
    dict(type='ToTensor', keys=[]),
    dict(type='NormalizeAndPad', size=640, aug_translate=True)
]

mytest_transforms = [
    dict(type='MyRandomResize', sizes=[1600], record_resize_info=True),
    dict(type='ToTensor', keys=[]),
    dict(type='MyNormalizeAndPad', size=1600, center_place=True)
]

if '__name__'  == '__main':
    val =  Talk2Car_base(root='/home/ming/Documents/download/data/images',
                            split='val')
    train =  Talk2Car_base(root='/home/ming/Documents/download/data/images',
                            split='train')