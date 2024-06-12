import os
import os.path as osp
import sys
import random
import math
import numpy as np
import torch
import pickle
import PIL
from PIL import Image
import io

from torch.utils.data import Dataset
import sys

from .utils import convert_examples_to_features, read_examples

from util.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy
from pytorch_pretrained_bert.tokenization import BertTokenizer
# from transforms import PIL_TRANSFORMS
import json
import torchvision.transforms as pic_transforms
# Meta Information
import torch.nn.functional as F
SUPPORTED_DATASETS = {
    'referit': {'splits': ('train', 'val', 'trainval', 'test')},
    'unc': {
        'splits': ('train', 'val', 'trainval', 'testA', 'testB'),
        'params': {'dataset': 'refcoco', 'split_by': 'unc'}
    },
    'unc+': {
        'splits': ('train', 'val', 'trainval', 'testA', 'testB'),
        'params': {'dataset': 'refcoco+', 'split_by': 'unc'}
    },
    'gref': {
        'splits': ('train', 'val'),
        'params': {'dataset': 'refcocog', 'split_by': 'google'}
    },
    'gref_umd': {
            'splits': ('train', 'val', 'test'),
            'params': {'dataset': 'refcocog', 'split_by': 'umd'}
    },
    'flickr': {
        'splits': ('train', 'val', 'test')}
}


class VGDataset(Dataset):
    def __init__(self, data_root, split_root='data', dataset='referit', transforms=[],
                 debug=False, test=False, split='train', max_query_len=128,
                 bert_mode='bert-base-uncased', cache_images=False):
        super(VGDataset, self).__init__()

        self.data_root = data_root
        self.split_root = split_root
        self.dataset = dataset
        self.test = test
        self.transforms = []

        self.getitem = self.getitem__PIL
        self.read_image = self.read_image_from_path_PIL
        for t in transforms:
            _args = t.copy()
            self.transforms.append(PIL_TRANSFORMS[_args.pop('type')](**_args))


        self.debug = debug

        self.query_len = max_query_len
        self.tokenizer = BertTokenizer.from_pretrained(bert_mode, do_lower_case=True)

        # setting datasource
        if self.dataset == 'referit':
            self.dataset_root = osp.join(self.data_root, 'referit')
            self.im_dir = osp.join(self.dataset_root, 'images')
        elif self.dataset == 'flickr':
            self.dataset_root = osp.join(self.data_root, 'Flickr30k')
            self.im_dir = osp.join(self.dataset_root, 'flickr30k-images')
        else:  # refer coco etc.
            self.dataset_root = osp.join(self.data_root, 'other')
            self.im_dir = osp.join(self.dataset_root, 'images', 'mscoco', 'images', 'train2014')

        dataset_split_root = osp.join(self.split_root, self.dataset)
        valid_splits = SUPPORTED_DATASETS[self.dataset]['splits']

        if split not in valid_splits:
            raise ValueError(
                'Dataset {0} does not have split {1}'.format(
                    self.dataset, split))

        # read the image set info
        self.imgset_info = []
        splits = [split]
        if self.dataset != 'referit':
            splits = ['train', 'val'] if split == 'trainval' else [split]
        for split in splits:
            imgset_file = '{0}_{1}.pth'.format(self.dataset, split)
            imgset_path = osp.join(dataset_split_root, imgset_file)
            self.imgset_info += torch.load(imgset_path, map_location="cpu")

        # process the image set info
        if self.dataset == 'flickr':
            self.img_names, self.bboxs, self.phrases = zip(*self.imgset_info)
        else:
            self.img_names, _, self.bboxs, self.phrases, _ = zip(*self.imgset_info)

        self.cache_images = cache_images
        if cache_images:
            self.images_cached = [None] * len(self) #list()
            self.read_image_orig_func = self.read_image
            self.read_image = self.read_image_from_cache

        self.covert_bbox = []
        if not (self.dataset == 'referit' or self.dataset == 'flickr'):  # for refcoco, etc
            # xywh to xyxy
            for bbox in self.bboxs:
                bbox = np.array(bbox, dtype=np.float32)
                bbox[2:] += bbox[:2]
                self.covert_bbox.append(bbox)
        else:
            for bbox in self.bboxs:  # for referit, flickr
                bbox = np.array(bbox, dtype=np.float32)
                self.covert_bbox.append(bbox)


    def __len__(self):
        return len(self.img_names)

    def image_path(self, idx):  # notice: db index is the actual index of data.
        return osp.join(self.im_dir, self.img_names[idx])

    def annotation_box(self, idx):
        return self.covert_bbox[idx].copy()

    def phrase(self, idx):
        return self.phrases[idx]

    def cache(self, idx):
        self.images_cached[idx] = self.read_image_orig_func(idx)

    def read_image_from_path_PIL(self, idx):
        image_path = self.image_path(idx)
        pil_image = Image.open(image_path).convert('RGB')
        return pil_image

    def read_image_from_cache(self, idx):
        image = self.images_cached[idx]
        return image

    def __getitem__(self, idx):
        return self.getitem(idx)


    def getitem__PIL(self, idx):
        # reading images
        image = self.read_image(idx)
        orig_image = image

        # read bbox annotation
        bbox = self.annotation_box(idx)
        bbox = torch.tensor(bbox)
        # read phrase
        phrase = self.phrase(idx)
        phrase = phrase.lower()
        orig_phrase = phrase

        target = {}
        target['phrase'] = phrase
        target['bbox'] = bbox
        
        if self.test or self.debug:
            target['orig_bbox'] = bbox.clone()

        for transform in self.transforms:
            image, target = transform(image, target)


        # For BERT
        examples = read_examples(target['phrase'], idx)
        features = convert_examples_to_features(
            examples=examples, seq_length=self.query_len, tokenizer=self.tokenizer)
        word_id = features[0].input_ids
        word_mask = features[0].input_mask

        target['word_id'] = torch.tensor(word_id, dtype=torch.long)
        target['word_mask'] = torch.tensor(word_mask, dtype=torch.bool)

        if 'mask' in target:
            mask = target.pop('mask')
            return image, mask, target

        return image, target

# data_root, split_root='data', dataset='referit', transforms=[],
#                 debug=False, test=False, split='train', max_query_len=128,
#                 bert_mode='bert-base-uncased', cache_images=False
class Talk2Car(Dataset):
    def __init__(
        self,
        root,
        split_root='data',
        dataset='Talk2car',
        transforms=[],
        debug=False,
        test=False,
        split='train',
        max_query_len=128,
        mask_transform=None,
        bert_mode='bert-base-uncased', 
        cache_images=False,
        glove_path="./Models/utils/dataloader",
        max_len=30,

    ):
        self.cache_images = cache_images
        self.root = root
        self.split = split
        
        self.test = test
        self.debug = debug
        self.tokenizer = BertTokenizer.from_pretrained(bert_mode, do_lower_case=True)
        self.query_len = max_query_len

        self.data = {}
        # if self.split == "test":
        #     data_file = "./Models/utils/dataloader/test.json"
        #     with open(data_file,"rb",) as f:
        #         data = json.load(f)[self.split]
        #         self.data = {int(k): v for k, v in data.items()} 
        # else:
        #     data_file = "./Models/utils/dataloader/talk2car_w_rpn_no_duplicates.json"
        #     with open(data_file,"rb",) as f:#talk2car_w_rpn_no_duplicates
        #         data = json.load(f)[self.split]
                

        #         self.data = {int(k): v for k, v in data.items()}  
        #         print(len(self.data.keys()))
        #          ## self.data["val"] = {int(k): v for k, v in data["val"].items()}  # Map to int
        #          ## self.data["train"] = {int(k): v for k, v in data["train"].items()}
        region_information_path ="../data/region_feature_data.pt"

        self.region_information = torch.load(region_information_path)


        gpt_data_path ="./Talk2Car/data/GPTdata"
        if self.split == "train":
            data_file = "../data/commands/train_commands.json"
            GPT_data = "../GPTdata/train_val/train_final.json"
        elif self.split == "val":
            data_file = "../data/commands/val_commands.json"
            GPT_data = "../GPTdata/train_val/value_final.json"
        elif self.split == "test":
            data_file = "../data/commands/test_commands.json"
            GPT_data = "../GPTdata/train_val/test_final.json"
        else:
            sys.exit(0)

        with open(data_file,"r") as f:
            self.data = json.load(f)['commands']
        
        with open(GPT_data,"r") as f1:
            self.GPT_data = json.load(f1)
         
        self.img_dir = "../data/images" 
        self.mask_dir = os.path.join(self.root, "val_masks_new")


        self.transforms = []
        for t in transforms:
            _args = t.copy()
            self.transforms.append(PIL_TRANSFORMS[_args.pop('type')](**_args))

        self.mask_transform = mask_transform

        self.img_transform = pic_transforms.Compose([
                                pic_transforms.ToTensor(),  # 将图像转换为张量
                                
                                pic_transforms.Resize((640,640))
                                ])
        # self.vocabulary = Vocabulary(vocabulary, glove_path, max_len)

    def __len__(self):
        ## return len(self.annotations)
   
        
        return len(self.data)

    
        # if self.split == "test":
        #     sample = self.data[idx]
        
        #     img_path = os.path.join(self.img_dir, sample["t2c_img"])

        #     with open(img_path, "rb") as f:
        #         image = Image.open(f).convert("RGB")

        #     target['origin_image'] = image
        #     gt = sample["2d_box"] # format [x1,y1,w,h]
            
        #     bbox = torch.tensor([gt[0],gt[1],gt[0]+gt[2],gt[1]+gt[3]]) # translation [x1,y1,w,h] to  [x1,y1,x2,y2]

        #     phrase = sample["command"].lower()

        #     target = {}
        #     target['phrase'] = phrase
        #     target['bbox'] = bbox 
        #     target['command_token'] = sample['command_token'] # 提交官网所需要的指标，bbox格式[x1,,y1,w,h]

        #     phrase = sample["command"].lower()
        #     orig_phrase = phrase


        #     if self.test or self.debug:
        #         target['orig_bbox'] = bbox.clone()


            # for transform in self.transforms:
            #     image, target = transform(image, target)
            
        #     # For BERT
        #     examples = read_examples(target['phrase'], idx)
        #     features = convert_examples_to_features(
        #         examples=examples, seq_length=self.query_len, tokenizer=self.tokenizer)
        #     word_id = features[0].input_ids
        #     word_mask = features[0].input_mask
        #     target['word_id'] = torch.tensor(word_id, dtype=torch.long)
        #     target['word_mask'] = torch.tensor(word_mask, dtype=torch.bool)
        #     target['id'] = torch.tensor(idx)

        #     if 'mask' in target:
        #         mask = target.pop('mask')
        #         return image, mask, target
        #     return image, target
        
        # else:
        """
            {
            'scene_token': f92422ed4b4e427194a4958ccf15709a, # nuScenes scene token
            'sample_token': c32d636e44604d77a1734386b3fe4a0d, # nuScenes sample token
            'translation': [-13.49250542687401, 0.43033061594724364, 59.28095610405408], # Translation
            'size': [0.81, 0.73, 1.959], # Size
            'rotation':  ['-0.38666213835670615', '-0.38076281276237284', '-0.5922192111910205', '0.5956412318459762'], # Rotation,
            'command': 'turn left to pick up the pedestrian at the corner', # Command
            'obj_name': 'human.pedestrian.adult', # Class name of the reffered object 
            'box_token': '0183ed8a474f411f8a3394eb78df7838' # nuScenes box token,
            'command_token': '4175173f5f60d19ecfc3712e960a1103' # A unique command identifier,
            '2d_box': [200, 300, 50, 50] # The 2d bounding box of the referred object in the frontal view. Follows the format [x,y,w,h]
            '':
            }
        """
    def __getitem__(self, idx):
        
        sample = self.data[idx]
    
        img_path = os.path.join(self.img_dir, sample["t2c_img"])

        with open(img_path, "rb") as f:
            image = Image.open(f).convert("RGB")

        
        target = {}
        target["origin_image"] = self.img_transform(image)

        # target['origin_image'] = image
        gt = sample["2d_box"] # format [x1,y1,w,h]
        
        bbox = torch.tensor([gt[0],gt[1],gt[0]+gt[2],gt[1]+gt[3]]) # translation [x1,y1,w,h] to  [x1,y1,x2,y2]

        phrase = sample["command"].lower()

       
        target['phrase'] = phrase
        target['bbox'] = bbox 

        target['origin_boundingbox_xyxy'] = torch.tensor([gt[0]/1600,gt[1]/900,(gt[0]+gt[2])/1600,(gt[1]+gt[3])/900])
        target['command_token'] = sample['command_token'] # ，bbox[x1,,y1,w,h]

        # target['scene_token'] = sample['scene_token']

        if self.test or self.debug:
            target['orig_bbox'] = bbox.clone()

        for transform in self.transforms:
            image, target = transform(image, target)
        
        # For BERT
        examples = read_examples(target['phrase'], idx)
        features = convert_examples_to_features(
            examples=examples, seq_length=self.query_len, tokenizer=self.tokenizer)
        word_id = features[0].input_ids
        word_mask = features[0].input_mask


        target['word_id'] = torch.tensor(word_id, dtype=torch.long)
        target['word_mask'] = torch.tensor(word_mask, dtype=torch.bool)
        target['id'] = torch.tensor(idx)


        # self.region_information = torch.load(region_information_path)
        # 

        
        target["all_region_feature"] = self.region_information[sample["t2c_img"]]["all_region_feature"]

        target["all_region_bbox"] = torch.tensor(self.region_information[sample["t2c_img"]]["all_region_bbox"])
        #target["all_region_bbox"] = torch.tensor(self.region_information[sample["t2c_img"]]["all_region_bbox"])
        target["all_region_iou"] = torch.tensor(self.region_information[sample["t2c_img"]]["all_region_IOU"])
        target["question"] = self.region_information[sample["t2c_img"]]["question"]

        target["region_mask"] = torch.tensor(self.region_information[sample["t2c_img"]]["image_mask"])
        target["input_mask"] = torch.tensor(self.region_information[sample["t2c_img"]]["input_mask"])
        #target["all_region_bbox_ori"] = torch.tensor(self.region_information["all_region_bbox_ori"])

        token = target['command_token']
        # print(self.GPT_data[token].keys())
        # dict_keys(['Traffic Scenario Analysis', 'Weather Conditions', 
        #'Traffic Light Status', 'Signage', 'Lane Line Information', 'Vehicle Count and Description', 'Road Topology', 'Command Keywords', 
        #'Command type', 'Emotion Interpretation', 'Command Feasibility Analysis', 'Summary of Considerations'])


        try:
            target["Command Keywords"] = self.GPT_data[token]['Command Keywords']
            target["Command Analysis"] = self.GPT_data[token]['Command Feasibility Analysis']
            target["Scenario Analysis"]  = self.GPT_data[token]['Traffic Scenario Analysis']
            if 'mask' in target:
                mask = target.pop('mask')
                return image, mask, target
        
            return image, target

        except KeyError as e:          
            target["Command Keywords"] = sample["command"].lower()
            target["Command Analysis"] = sample["command"].lower()
            target["Scenario Analysis"]  = sample["command"].lower()
            
            return self.__getitem__(idx+1)




    def number_of_words(self):
        # Get number of words in the vocabulary
        return self.vocabulary.number_of_words

    def convert_index_to_command_token(self, index):
        return self.data[index]["command_token"]

    def convert_command_to_text(self, command):
        # Takes value from command key and transforms it into human readable text
        return " ".join(self.vocabulary.ix2sent_drop_pad(command.numpy().tolist()))

class Talk2Car_base_v0(Dataset):
    def __init__(
        self,
        root,
        split_root='data',
        dataset='Talk2car',
        transforms=[],
        debug=False,
        test=False,
        split='train',
        max_query_len=128,
        mask_transform=None,
        bert_mode='bert-base-uncased', 
        cache_images=False,
        glove_path="./Models/utils/dataloader",
        max_len=30,

    ):
        self.cache_images = cache_images
        self.root = root
        self.split = split
        
        self.test = test
        self.debug = debug
        self.tokenizer = BertTokenizer.from_pretrained(bert_mode, do_lower_case=True)
        self.query_len = max_query_len

        self.data = {}
        # if self.split == "test":
        #     data_file = "./Models/utils/dataloader/test.json"
        #     with open(data_file,"rb",) as f:
        #         data = json.load(f)[self.split]
        #         self.data = {int(k): v for k, v in data.items()} 
        # else:
        #     data_file = "./Models/utils/dataloader/talk2car_w_rpn_no_duplicates.json"
        #     with open(data_file,"rb",) as f:#talk2car_w_rpn_no_duplicates
        #         data = json.load(f)[self.split]
                

        #         self.data = {int(k): v for k, v in data.items()}  
        #         print(len(self.data.keys()))
        #          ## self.data["val"] = {int(k): v for k, v in data["val"].items()}  # Map to int
        #          ## self.data["train"] = {int(k): v for k, v in data["train"].items()}
        region_information_path ="../data/region_feature_data.pt"

        self.region_information = torch.load(region_information_path)


        gpt_data_path ="./Talk2Car/data/GPTdata"
        if self.split == "train":
            data_file = "../data/commands/train_commands.json"
            GPT_data = "../GPTdata/train_val/train_final.json"
        elif self.split == "val":
            data_file = "../data/commands/val_commands.json"
            GPT_data = "../GPTdata/train_val/value_final.json"
        elif self.split == "test":
            data_file = "../data/commands/test_commands.json"
            GPT_data = "../GPTdata/train_val/test_final.json"
        else:
            sys.exit(0)

        with open(data_file,"r") as f:
            self.data = json.load(f)['commands']
        
         
        self.img_dir = "../data/images" 
        self.mask_dir = os.path.join(self.root, "val_masks_new")

        self.transforms = []
        for t in transforms:
            _args = t.copy()
            self.transforms.append(PIL_TRANSFORMS[_args.pop('type')](**_args))

        self.mask_transform = mask_transform

        self.img_transform = pic_transforms.Compose([
                                pic_transforms.ToTensor(),  # 将图像转换为张量
                                
                                pic_transforms.Resize((640,640))
                                ])
        # self.vocabulary = Vocabulary(vocabulary, glove_path, max_len)

    def __len__(self):
        ## return len(self.annotations)
   
        
        return len(self.data)

    
        # if self.split == "test":
        #     sample = self.data[idx]
        
        #     img_path = os.path.join(self.img_dir, sample["t2c_img"])

        #     with open(img_path, "rb") as f:
        #         image = Image.open(f).convert("RGB")

        #     target['origin_image'] = image
        #     gt = sample["2d_box"] # format [x1,y1,w,h]
            
        #     bbox = torch.tensor([gt[0],gt[1],gt[0]+gt[2],gt[1]+gt[3]]) # translation [x1,y1,w,h] to  [x1,y1,x2,y2]

        #     phrase = sample["command"].lower()

        #     target = {}
        #     target['phrase'] = phrase
        #     target['bbox'] = bbox 
        #     target['command_token'] = sample['command_token'] 

        #     phrase = sample["command"].lower()
        #     orig_phrase = phrase


        #     if self.test or self.debug:
        #         target['orig_bbox'] = bbox.clone()


            # for transform in self.transforms:
            #     image, target = transform(image, target)
            
        #     # For BERT
        #     examples = read_examples(target['phrase'], idx)
        #     features = convert_examples_to_features(
        #         examples=examples, seq_length=self.query_len, tokenizer=self.tokenizer)
        #     word_id = features[0].input_ids
        #     word_mask = features[0].input_mask
        #     target['word_id'] = torch.tensor(word_id, dtype=torch.long)
        #     target['word_mask'] = torch.tensor(word_mask, dtype=torch.bool)
        #     target['id'] = torch.tensor(idx)

        #     if 'mask' in target:
        #         mask = target.pop('mask')
        #         return image, mask, target
        #     return image, target
        
        # else:
        """
            {
            'scene_token': f92422ed4b4e427194a4958ccf15709a, # nuScenes scene token
            'sample_token': c32d636e44604d77a1734386b3fe4a0d, # nuScenes sample token
            'translation': [-13.49250542687401, 0.43033061594724364, 59.28095610405408], # Translation
            'size': [0.81, 0.73, 1.959], # Size
            'rotation':  ['-0.38666213835670615', '-0.38076281276237284', '-0.5922192111910205', '0.5956412318459762'], # Rotation,
            'command': 'turn left to pick up the pedestrian at the corner', # Command
            'obj_name': 'human.pedestrian.adult', # Class name of the reffered object 
            'box_token': '0183ed8a474f411f8a3394eb78df7838' # nuScenes box token,
            'command_token': '4175173f5f60d19ecfc3712e960a1103' # A unique command identifier,
            '2d_box': [200, 300, 50, 50] # The 2d bounding box of the referred object in the frontal view. Follows the format [x,y,w,h]
            '':
            }
        """
    def __getitem__(self, idx):
        
        sample = self.data[idx]
    
        img_path = os.path.join(self.img_dir, sample["t2c_img"])

        with open(img_path, "rb") as f:
            image = Image.open(f).convert("RGB")

        verfi_mask = torch.zeros((3,900,1600))

        target = {}
        target["origin_image"] = self.img_transform(image)

        # target['origin_image'] = image
        gt = sample["2d_box"] # format [x1,y1,w,h]
        
        bbox = torch.tensor([gt[0],gt[1],gt[0]+gt[2],gt[1]+gt[3]]) # translation [x1,y1,w,h] to  [x1,y1,x2,y2]

        verfi_mask[:, bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1
        verfi_mask = verfi_mask.unsqueeze(0) 

        downsampled_image = F.interpolate(verfi_mask, size=(40, 40), mode='bilinear', align_corners=False).squeeze()[0]
        target['downsampled_mask'] = downsampled_image 

        target['mask'] = torch.tensor([0])
        # phrase = sample["command"].lower()
        target['Command']=sample["command"]
        target['image_path'] = img_path
        # target['phrase'] = phrase
        target['bbox'] = bbox 

        target['origin_boundingbox_xyxy'] = torch.tensor([gt[0]/1600,gt[1]/900,(gt[0]+gt[2])/1600,(gt[1]+gt[3])/900])
        target['command_token'] = sample['command_token'] # 提交官网所需要的指标，bbox格式[x1,,y1,w,h]

        # target['scene_token'] = sample['scene_token']

        if self.test or self.debug:
            target['orig_bbox'] = bbox.clone()

        # for transform in self.transforms:
        #     image, target = transform(image, target)
        
        # For BERT
        # examples = read_examples(target['phrase'], idx)
        # features = convert_examples_to_features(
        #     examples=examples, seq_length=self.query_len, tokenizer=self.tokenizer)
        # word_id = features[0].input_ids
        # word_mask = features[0].input_mask


        # target['word_id'] = torch.tensor(word_id, dtype=torch.long)
        # target['word_mask'] = torch.tensor(word_mask, dtype=torch.bool)
        #target['id'] = torch.tensor(idx)



        target["all_region_feature"] = self.region_information[sample["t2c_img"]]["all_region_feature"].to('cpu')

        target["all_region_bbox"] = torch.tensor(self.region_information[sample["t2c_img"]]["all_region_bbox"]).to('cpu')
        #target["all_region_bbox"] = torch.tensor(self.region_information[sample["t2c_img"]]["all_region_bbox"])
        target["all_region_iou"] = torch.tensor(self.region_information[sample["t2c_img"]]["all_region_IOU"]).to('cpu')
        target["question"] = self.region_information[sample["t2c_img"]]["question"].to('cpu')

        target["region_mask"] = torch.tensor(self.region_information[sample["t2c_img"]]["image_mask"]).to('cpu')
        target["input_mask"] = torch.tensor(self.region_information[sample["t2c_img"]]["input_mask"]).to('cpu')


        


            
        #     return self.__getitem__(idx+1)
        image = target['mask']
        if 'mask' in target:
            mask = target.pop('mask')
            return image, mask, target
        else:
            return image, target

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms as pic_transforms
from torch.utils.data import Dataset



class Talk2Car_base_RSD(Dataset):
    def __init__(
        self,
        root,
        split_root='data',
        dataset='Talk2car',
        transforms=[],
        debug=False,
        test=False,
        split='train',
        max_query_len=128,
        mask_transform=None,
        bert_mode='bert-base-uncased', 
        cache_images=False,
        glove_path="./Models/utils/dataloader",
        max_len=30,

    ):
        self.cache_images = cache_images
        self.root = root
        self.split = split
        
        #self.test = test
        #self.debug = debug
        #self.tokenizer = BertTokenizer.from_pretrained(bert_mode, do_lower_case=True)
        #self.query_len = max_query_len

        self.data = {}

        #region_information_path ="../data/region_feature_data.pt"

        #self.region_information = torch.load(region_information_path)
        # 这个部分的数据采用图片名称作为索引，确认过了和原数据集格式一样，内部内容包括


        if self.split == "train":
            data_file = "./data/commands/train_commands.json"

        elif self.split == "val":
            data_file = "./data/commands/val_commands.json"
  
        elif self.split == "test":
            data_file = "./data/commands/train_commands.json"
        else:
            sys.exit(0)

        self.bbox_file = './download/region/CenternetBbox.json'

        with open(data_file,"r") as f:
            self.data = json.load(f)['commands']

        with open(self.bbox_file, 'rb') as f1:
            self.data_rpg = json.load(f1)
        #self.num_rpns_per_image = 32

        #assert(self.num_rpns_per_image < 64)
        #rpns_score_ordered_idx = {k: np.argsort([rpn['score'] for rpn in v]) for k, v in self.data_rpg.items()}

        #self.rpns = {k: [v[idx] for idx in rpns_score_ordered_idx[k][-self.num_rpns_per_image:]] for k, v in self.data_rpg.items()}


        self.img_dir = "./download/image" 

        self.img_transform = pic_transforms.Compose([
                                pic_transforms.ToTensor(),  # 将图像转换为张量
                                pic_transforms.Resize((512,512)),
                                pic_transforms.Normalize((0.47,0.43, 0.39), (0.27, 0.26, 0.27))])


    def __len__(self):


        return len(self.data)


    def __getitem__(self, idx):

        sample = self.data[idx]

        img_path = os.path.join(self.img_dir, sample["t2c_img"])

        with open(img_path, "rb") as f:
            image = Image.open(f).convert("RGB")

        verfi_mask = torch.zeros((3,900,1600))

        target = {}
        img = self.img_transform(image)


        
        # target['origin_image'] = image
        gt = sample["2d_box"] # format [x1,y1,w,h]

        # clear ground truth
        bbox = torch.tensor([gt[0],gt[1],gt[0]+gt[2],gt[1]+gt[3]]) # translation [x1,y1,w,h] to  [x1,y1,x2,y2]
        
        bbox[0] = torch.clamp(bbox[0], 0, 1600) # x1
        bbox[1] = torch.clamp(bbox[1], 0, 900) # y1
        bbox[2] = torch.clamp(bbox[2], 0, 1600) # x2
        bbox[3] = torch.clamp(bbox[3], 0, 900) # y1
        target['gt_bbox'] = bbox
        verfi_mask[:, bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1

        verfi_mask = verfi_mask.unsqueeze(0) 
        #target['map_mask'] = verfi_mask
        downsampled_image = F.interpolate(verfi_mask, size=(32, 23), mode='bilinear', align_corners=False).squeeze()[0]
        target['downsampled_mask'] = downsampled_image 


        target['Command']=sample["command"]
        target['image_path'] = img_path
        # target['phrase'] = phrase
        
        
        target['origin_boundingbox_xyxy'] = torch.tensor([gt[0]/1600,gt[1]/900,(gt[0]+gt[2])/1600,(gt[1]+gt[3])/900])

        target['command_token'] = sample['command_token'] # 提交官网所需要的指标，bbox格式[x1,,y1,w,h]

        # Load region proposals obtained with centernet return bboxes as (xl, yb, xr, yt)
        centernet_boxes = self.data_rpg[sample["t2c_img"]]

        # bbox = torch.stack([torch.LongTensor(centernet_boxes[i]['bbox']) for i in range(self.num_rpns_per_image)]) # num_rpns x 4
        bbox = torch.LongTensor(centernet_boxes['all_region_bbox_ori'])[:,]

        bbox_lbrt = torch.stack([bbox[:,0], bbox[:,1],bbox[:,2],bbox[:,3]], 1)
        
        bbox_lbrt[:,0] = torch.clamp(bbox_lbrt[:,0], 0, 1600) # xl
        bbox_lbrt[:,1] = torch.clamp(bbox_lbrt[:,1], 0, 900) # yb
        bbox_lbrt[:,2] = torch.clamp(bbox_lbrt[:,2], 0, 1600) # xr
        bbox_lbrt[:,3] = torch.clamp(bbox_lbrt[:,3], 0, 900) # yt
        # to x1y1x2y2
        # image_height = 900





        target['rpn_bbox_lbrt'] = bbox_lbrt[1:,]
        mask = torch.tensor([0])
        #iou_array = jaccard(target['rpn_bbox_lbrt'].numpy(), target['gt_bbox'].numpy().reshape(1, -1))
        iou_array = iou(target['gt_bbox'].unsqueeze(0),target['rpn_bbox_lbrt']).numpy()

       
        if np.any(iou_array >= 0.5):
            a=0
            #gt_index = torch.LongTensor([np.argmax(iou_array)]) # Best matching is gt for training
            # target['gt_index'] = gt
            # break
        else:
            
            detection_pred_box_index = np.random.randint(len(target['rpn_bbox_lbrt']))
            target['rpn_bbox_lbrt'][detection_pred_box_index] = target['gt_bbox'] 
            
        N,D = target['rpn_bbox_lbrt'].shape
        # verfi_mask[:, bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1

        verfi_mask_rpgs = torch.zeros((N,32,32))
        for i in range(N):
            regin_bbox = target['rpn_bbox_lbrt'][i]
            verfi_mask_rpg = torch.zeros((900,1600))
            verfi_mask_rpg[regin_bbox[1]:regin_bbox[3],regin_bbox[0]:regin_bbox[2]] = 1

            downsampled_region_i = F.interpolate(verfi_mask_rpg.unsqueeze(0).unsqueeze(0), size=(32, 32), mode='bilinear', align_corners=False).squeeze()
            verfi_mask_rpgs[i] = downsampled_region_i
        target['mask_rpn'] = verfi_mask_rpgs
   
        if 'mask' in target:
            mask = target.pop('mask')
            return image, mask, target
        else:
            return img,mask,target

class Talk2Car_base(Dataset):
    def __init__(
        self,
        root,
        split_root='data',
        dataset='Talk2car',
        transforms=[],
        debug=False,
        test=False,
        split='train',
        max_query_len=128,
        mask_transform=None,
        bert_mode='bert-base-uncased', 
        cache_images=False,
        glove_path="./Models/utils/dataloader",
        max_len=30,

    ):
        self.N = 32
        self.cache_images = cache_images
        self.root = root
        self.split = split
        
        #self.test = test
        #self.debug = debug
        #self.tokenizer = BertTokenizer.from_pretrained(bert_mode, do_lower_case=True)
        #self.query_len = max_query_len

        self.data = {}

        #region_information_path ="../data/region_feature_data.pt"



        if self.split == "train":
            data_file = "./data/commands/train_commands.json"

        elif self.split == "val":
            data_file = "./data/commands/val_commands.json"
  
        elif self.split == "test":
            data_file = "./data/commands/train_commands.json"
        else:
            sys.exit(0)

        self.bbox_file = './baseline/data/centernet_bboxes.json'

        with open(data_file,"r") as f:
            self.data = json.load(f)['commands']

        with open(self.bbox_file, 'rb') as f1:
            self.data_rpg = json.load(f1)[self.split]
        self.num_rpns_per_image = 16

        assert(self.num_rpns_per_image < 64)
        rpns_score_ordered_idx = {k: np.argsort([rpn['score'] for rpn in v]) for k, v in self.data_rpg .items()}
        self.rpns = {k: [v[idx] for idx in rpns_score_ordered_idx[k][-self.num_rpns_per_image:]] for k, v in self.data_rpg.items()}


        self.img_dir = "./download/image" 

        self.img_transform = pic_transforms.Compose([
                                pic_transforms.ToTensor(),  
                                pic_transforms.Resize((self.N*16,self.N*16)),
                                pic_transforms.Normalize((0.47,0.43, 0.39), (0.27, 0.26, 0.27))])


    def __len__(self):


        return len(self.data)


    def __getitem__(self, idx):

        sample = self.data[idx]

        img_path = os.path.join(self.img_dir, sample["t2c_img"])

        with open(img_path, "rb") as f:
            image = Image.open(f).convert("RGB")

        verfi_mask = torch.zeros((3,900,1600))

        target = {}
        img = self.img_transform(image)


        
        # target['origin_image'] = image
        gt = sample["2d_box"] # format [x1,y1,w,h]

        # clear ground truth
        bbox = torch.tensor([gt[0],gt[1],gt[0]+gt[2],gt[1]+gt[3]]) # translation [x1,y1,w,h] to  [x1,y1,x2,y2]
        
        bbox[0] = torch.clamp(bbox[0], 0, 1600) # x1
        bbox[1] = torch.clamp(bbox[1], 0, 900) # y1
        bbox[2] = torch.clamp(bbox[2], 0, 1600) # x2
        bbox[3] = torch.clamp(bbox[3], 0, 900) # y1
        target['gt_bbox'] = bbox
        verfi_mask[:, bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1

        verfi_mask = verfi_mask.unsqueeze(0) 
        #target['map_mask'] = verfi_mask
        downsampled_image = F.interpolate(verfi_mask, size=(self.N, self.N), mode='bilinear', align_corners=False).squeeze()[0]
        target['downsampled_mask'] = downsampled_image 


        target['Command']=sample["command"]
        target['image_path'] = img_path
        # target['phrase'] = phrase
        
        
        target['origin_boundingbox_xyxy'] = torch.tensor([gt[0]/1600,gt[1]/900,(gt[0]+gt[2])/1600,(gt[1]+gt[3])/900])

        target['command_token'] = sample['command_token'] # 提交官网所需要的指标，bbox格式[x1,,y1,w,h]

        # Load region proposals obtained with centernet return bboxes as (xl, yb, xr, yt)
        centernet_boxes = self.rpns[target['command_token']]

        bbox = torch.stack([torch.LongTensor(centernet_boxes[i]['bbox']) for i in range(self.num_rpns_per_image)]) # num_rpns x 4
        #bbox = torch.LongTensor(centernet_boxes['all_region_bbox_ori'])[:,]

        bbox_lbrt = torch.stack([bbox[:,0], 
                                bbox[:,1],
                                bbox[:,0] + bbox[:,2],
                                bbox[:,1] + bbox[:,3]], 1)
        
        bbox_lbrt[:,0] = torch.clamp(bbox_lbrt[:,0], 0, 1600) # xl
        bbox_lbrt[:,1] = torch.clamp(bbox_lbrt[:,1], 0, 900) # yb
        bbox_lbrt[:,2] = torch.clamp(bbox_lbrt[:,2], 0, 1600) # xr
        bbox_lbrt[:,3] = torch.clamp(bbox_lbrt[:,3], 0, 900) # yt


        target['rpn_bbox_lbrt'] = bbox_lbrt[1:,]
        mask = torch.tensor([0])
        #iou_array = jaccard(target['rpn_bbox_lbrt'].numpy(), target['gt_bbox'].numpy().reshape(1, -1))
        iou_array = iou(target['gt_bbox'].unsqueeze(0),target['rpn_bbox_lbrt']).numpy()

       
        if np.any(iou_array >= 0.5):
            a=0
            #gt_index = torch.LongTensor([np.argmax(iou_array)]) # Best matching is gt for training
            # target['gt_index'] = gt
            # break
        else:
            
            detection_pred_box_index = np.random.randint(len(target['rpn_bbox_lbrt']))
            target['rpn_bbox_lbrt'][detection_pred_box_index] = target['gt_bbox'] 
            
        N,D = target['rpn_bbox_lbrt'].shape
        # verfi_mask[:, bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1

        verfi_mask_rpgs = torch.zeros((N,self.N,self.N))
        for i in range(N):
            regin_bbox = target['rpn_bbox_lbrt'][i]
            verfi_mask_rpg = torch.zeros((900,1600))
            verfi_mask_rpg[regin_bbox[1]:regin_bbox[3],regin_bbox[0]:regin_bbox[2]] = 1

            downsampled_region_i = F.interpolate(verfi_mask_rpg.unsqueeze(0).unsqueeze(0), size=(self.N, self.N), mode='bilinear', align_corners=False).squeeze()
            verfi_mask_rpgs[i] = downsampled_region_i
        target['mask_rpn'] = verfi_mask_rpgs
   
        if 'mask' in target:
            mask = target.pop('mask')
            return image, mask, target
        else:
            return img,mask,target

def visualize_and_save_image(image_path, gt_bbox, rpns_bbox_lbrt, command, idx):

    image = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    rect = patches.Rectangle((gt_bbox[0], gt_bbox[1]), gt_bbox[2] - gt_bbox[0], gt_bbox[3] - gt_bbox[1], linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    for i in range(rpns_bbox_lbrt.shape[0]):
        bbox = rpns_bbox_lbrt[i]
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=1, edgecolor='g', facecolor='none', linestyle='--')
        ax.add_patch(rect)

 
    plt.text(0, 0, command, fontsize=9, va='top', ha='left', color='white', bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 5})


    plt.axis('off')
    plt.savefig(f"./Stage1/utils/problem_prcture/visualization_{idx}.png", bbox_inches='tight')
    plt.close()


def iou(anchors, gt_boxes):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (K, 4) ndarray of float
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = anchors.size(0)
    K = gt_boxes.size(0)

    gt_boxes_area = (
        (gt_boxes[:, 2] - gt_boxes[:, 0] + 1) * (gt_boxes[:, 3] - gt_boxes[:, 1] + 1)
    ).view(1, K)

    anchors_area = (
        (anchors[:, 2] - anchors[:, 0] + 1) * (anchors[:, 3] - anchors[:, 1] + 1)
    ).view(N, 1)

    boxes = anchors.view(N, 1, 4).expand(N, K, 4)
    query_boxes = gt_boxes.view(1, K, 4).expand(N, K, 4)
    #print(boxes)
    iw = (
        torch.min(boxes[:, :, 2], query_boxes[:, :, 2])
        - torch.max(boxes[:, :, 0], query_boxes[:, :, 0])
        + 1
    )
    iw[iw < 0] = 0

    ih = (
        torch.min(boxes[:, :, 3], query_boxes[:, :, 3])
        - torch.max(boxes[:, :, 1], query_boxes[:, :, 1])
        + 1
    )
    ih[ih < 0] = 0

    ua = anchors_area + gt_boxes_area - (iw * ih)
    overlaps = iw * ih / ua

    return overlaps

def jaccard(a, b):
    # pairwise jaccard(IoU) botween boxes a and boxes b
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])
    inter = np.clip(rb - lt, 0, None)

    area_i = np.prod(inter, axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)

    area_u = area_a[:, np.newaxis] + area_b - area_i
    return area_i / np.clip(area_u, 1e-7, None)   # len(a) x len(b)

from torchvision.ops.boxes import box_area

def bbox_iou_gt(bboxes, gt_bbox):


    inter_top_left = torch.max(bboxes[:, :2], gt_bbox[:, :2])
    inter_bottom_right = torch.min(bboxes[:, 2:], gt_bbox[:, 2:])
    inter_wh = torch.clamp(inter_bottom_right - inter_top_left, min=0)
    
    inter_area = inter_wh[:, 0] * inter_wh[:, 1]
    bboxes_area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    gt_bbox_area = (gt_bbox[:, 2] - gt_bbox[:, 0]) * (gt_bbox[:, 3] - gt_bbox[:, 1])

    union_area = bboxes_area + gt_bbox_area - inter_area
    
    iou = inter_area / union_area
    return iou

def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou,union

class REFERTALK2CAR:

	def __init__(self, data_root, dataset='talk2car'):
		# provide data_root folder which contains refclef, refcoco, refcoco+ and refcocog
		# also provide dataset name and splitBy information
		# e.g., dataset = 'refcoco', splitBy = 'unc'
		print('loading dataset %s into memory...' % dataset)
		self.ROOT_DIR = osp.abspath(osp.dirname(__file__))
		self.DATA_DIR = osp.join(data_root, dataset)
		#assert dataset == 'talk2car'
		self.IMAGE_DIR = osp.join(data_root, 'images/talk2car/images/')

		# load refs from data/dataset/refs(dataset).json
		tic = time.time()
		ref_file = osp.join(self.DATA_DIR, 'refs_spacy.p')
		self.data = {}
		self.data['dataset'] = dataset
		self.data['refs'] = pickle.load(open(ref_file, 'rb'))

		# load annotations from data/dataset/instances.json
		instances_file = osp.join(self.DATA_DIR, 'instances.json')
		instances = json.load(open(instances_file, 'r'))
		#self.data['images'] = instances['images']
		self.data['annotations'] = instances['annotations']
		#self.data['categories'] = instances['categories']

		# create index
		self.createIndex()
		print('DONE (t=%.2fs)' % (time.time()-tic))

	def createIndex(self):
		# create sets of mapping
		# 1)  Refs: 	 	{ref_id: ref}
		# 2)  Anns: 	 	{ann_id: ann}
		# 3)  Imgs:		 	{image_id: image}
		# 4)  Cats: 	 	{category_id: category_name}
		# 5)  Sents:     	{sent_id: sent}
		# 6)  imgToRefs: 	{image_id: refs}
		# 7)  imgToAnns: 	{image_id: anns}
		# 8)  refToAnn:  	{ref_id: ann}
		# 9)  annToRef:  	{ann_id: ref}
		# 10) catToRefs: 	{category_id: refs}
		# 11) sentToRef: 	{sent_id: ref}
		# 12) sentToTokens: {sent_id: tokens}
		print('creating index...')
		# fetch info from instances
		Anns, Imgs, Cats, imgToAnns = {}, {}, {}, {}
		for ann in self.data['annotations']:
			Anns[ann['id']] = ann
			imgToAnns[ann['image_id']] = imgToAnns.get(ann['image_id'], []) + [ann]
		#for img in self.data['images']:
		#	Imgs[img['id']] = img
#		for cat in self.data['categories']:
#			Cats[cat['id']] = cat['name']

		# fetch info from refs
		Refs, imgToRefs, refToAnn, annToRef, catToRefs = {}, {}, {}, {}, {}
		Sents, sentToRef, sentToTokens = {}, {}, {}
		for ref in self.data['refs']:
			# ids
			ref_id = ref['ref_id']
			ann_id = ref['ann_id']
#			category_id = ref['category_id']
			image_id = ref['image_id']

			# add mapping related to ref
			Refs[ref_id] = ref
			imgToRefs[image_id] = imgToRefs.get(image_id, []) + [ref]
	#		catToRefs[category_id] = catToRefs.get(category_id, []) + [ref]
			refToAnn[ref_id] = Anns[ann_id]
			annToRef[ann_id] = ref

			# add mapping of sent
			for sent in ref['sentences']:
				Sents[sent['sent_id']] = sent
				sentToRef[sent['sent_id']] = ref
				#sentToTokens[sent['sent_id']] = sent['tokens']

		# create class members
		self.Refs = Refs
		self.Anns = Anns
		self.Imgs = Imgs
		self.Cats = Cats
		self.Sents = Sents
		self.imgToRefs = imgToRefs
		self.imgToAnns = imgToAnns
		self.refToAnn = refToAnn
		self.annToRef = annToRef
		self.catToRefs = catToRefs
		self.sentToRef = sentToRef
		self.sentToTokens = sentToTokens
		print('index created.')

	def getRefIds(self, image_ids=[], cat_ids=[], ref_ids=[], split=''):
		image_ids = image_ids if type(image_ids) == list else [image_ids]
		cat_ids = cat_ids if type(cat_ids) == list else [cat_ids]
		ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

		if len(image_ids)==len(cat_ids)==len(ref_ids)==len(split)==0:
			refs = self.data['refs']
		else:
			if not len(image_ids) == 0:
				refs = [self.imgToRefs[image_id] for image_id in image_ids]
			else:
				refs = self.data['refs']
			if not len(cat_ids) == 0:
				refs = [ref for ref in refs if ref['category_id'] in cat_ids]
			if not len(ref_ids) == 0:
				refs = [ref for ref in refs if ref['ref_id'] in ref_ids]
			if not len(split) == 0:
				if split in ['testA', 'testB', 'testC']:
					refs = [ref for ref in refs if split[-1] in ref['split']] # we also consider testAB, testBC, ...
				elif split in ['testAB', 'testBC', 'testAC']:
					refs = [ref for ref in refs if ref['split'] == split]  # rarely used I guess...
				elif split == 'test':
					refs = [ref for ref in refs if 'test' in ref['split']]
				elif split == 'train' or split == 'val':
					refs = [ref for ref in refs if ref['split'] == split]
				else:
					print('No such split [%s]' % split)
					sys.exit()
		ref_ids = [ref['ref_id'] for ref in refs]
		return ref_ids

	def getAnnIds(self, image_ids=[], cat_ids=[], ref_ids=[]):
		image_ids = image_ids if type(image_ids) == list else [image_ids]
		cat_ids = cat_ids if type(cat_ids) == list else [cat_ids]
		ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

		if len(image_ids) == len(cat_ids) == len(ref_ids) == 0:
			ann_ids = [ann['id'] for ann in self.data['annotations']]
		else:
			if not len(image_ids) == 0:
				lists = [self.imgToAnns[image_id] for image_id in image_ids if image_id in self.imgToAnns]  # list of [anns]
				anns = list(itertools.chain.from_iterable(lists))
			else:
				anns = self.data['annotations']
			if not len(cat_ids) == 0:
				anns = [ann for ann in anns if ann['category_id'] in cat_ids]
			ann_ids = [ann['id'] for ann in anns]
			if not len(ref_ids) == 0:
				ids = set(ann_ids).intersection(set([self.Refs[ref_id]['ann_id'] for ref_id in ref_ids]))
		return ann_ids

	def getImgIds(self, ref_ids=[]):
		ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

		if not len(ref_ids) == 0:
			image_ids = list(set([self.Refs[ref_id]['image_id'] for ref_id in ref_ids]))
		else:
			image_ids = self.Imgs.keys()
		return image_ids

	def getCatIds(self):
		return self.Cats.keys()

	def loadRefs(self, ref_ids=[]):
		if type(ref_ids) == list:
			return [self.Refs[ref_id] for ref_id in ref_ids]
		elif type(ref_ids) == int:
			return [self.Refs[ref_ids]]

	def loadAnns(self, ann_ids=[]):
		if type(ann_ids) == list:
			return [self.Anns[ann_id] for ann_id in ann_ids]
		elif type(ann_ids) == int or type(ann_ids) == unicode:
			return [self.Anns[ann_ids]]
	
	def loadImgs(self, image_ids=[]):
		if type(image_ids) == list:
			return [self.Imgs[image_id] for image_id in image_ids]
		elif type(image_ids) == int:
			return [self.Imgs[image_ids]]
	
	
	def loadCats(self, cat_ids=[]):
		if type(cat_ids) == list:
			return [self.Cats[cat_id] for cat_id in cat_ids]
		elif type(cat_ids) == int:
			return [self.Cats[cat_ids]]


	def getRefBox(self, ref_id):
		#ref = self.Refs[ref_id]
		ann = self.refToAnn[ref_id]
		return ann["bbox"]  # [x, y, w, h]

	"""
	def showRef(self, ref, seg_box='seg'):
		ax = plt.gca()
		# show image
		image = self.Imgs[ref['image_id']]
		I = io.imread(osp.join(self.IMAGE_DIR, image['file_name']))
		ax.imshow(I)
		# show refer expression
		for sid, sent in enumerate(ref['sentences']):
			print('%s. %s' % (sid+1, sent['sent']))
		# show segmentations
		if seg_box == 'seg':
			ann_id = ref['ann_id']
			ann = self.Anns[ann_id]
			polygons = []
			color = []
			c = 'none'
			if type(ann['segmentation'][0]) == list:
				# polygon used for refcoco*
				for seg in ann['segmentation']:
					poly = np.array(seg).reshape((len(seg)/2, 2))
					polygons.append(Polygon(poly, True, alpha=0.4))
					color.append(c)
				p = PatchCollection(polygons, facecolors=color, edgecolors=(1,1,0,0), linewidths=3, alpha=1)
				ax.add_collection(p)  # thick yellow polygon
				p = PatchCollection(polygons, facecolors=color, edgecolors=(1,0,0,0), linewidths=1, alpha=1)
				ax.add_collection(p)  # thin red polygon
			else:
				# mask used for refclef
				rle = ann['segmentation']
				m = mask.decode(rle)
				img = np.ones( (m.shape[0], m.shape[1], 3) )
				color_mask = np.array([2.0,166.0,101.0])/255
				for i in range(3):
					img[:,:,i] = color_mask[i]
				ax.imshow(np.dstack( (img, m*0.5) ))
		# show bounding-box
		elif seg_box == 'box':
			ann_id = ref['ann_id']
			ann = self.Anns[ann_id]
			bbox = 	self.getRefBox(ref['ref_id'])
			box_plot = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], fill=False, edgecolor='green', linewidth=3)
			ax.add_patch(box_plot)
	"""
	"""
	def showMask(self, ref):
		M = self.getMask(ref)
		msk = M['mask']
		ax = plt.gca()
		ax.imshow(msk)
	"""

import copy
import lmdb  # install lmdb by "pip install lmdb"
import base64
import pickle
from typing import List
import _pickle as cPickle

class Talk2Car_V1(Dataset):
    def __init__(
        self,
        split='train',
        annotations='./data/talk2car/annotations'
    ):
        self.split = split
        self.data = {}
        region_information_path ="../data/region_feature_data.pt"
        self.region_information = torch.load(region_information_path)

        if self.split == "train":
            data_file = "../data/commands/train_commands.json"
        elif self.split == "val":
            data_file = "../data/commands/val_commands.json"
        elif self.split == "test":
            data_file = "../data/commands/test_commands.json"

        else:
            sys.exit(0)

        with open(data_file,"r") as f:
            self.data = json.load(f)['commands']

        self.img_dir = "../data/images" 
        
        self.transforms = []

        self.img_transform = pic_transforms.Compose([
                                pic_transforms.ToTensor(),  
                                pic_transforms.Resize((640,640))
                                ])
        
        ### ref_talk2car
        self.refer = REFERTALK2CAR(data_root=annotations)
        self.ref_ids = self.refer.getRefIds(split=split)
        print("%s refs are in split [%s]." % (len(self.ref_ids), split))

        
        self._image_features_reader = ImageFeaturesH5Reader
        self.entries = self._load_annotations()

        max_seq_length = 80
        max_region_num =36
        task = 'talk2car'
        
        cache_path = os.path.join(
                annotations,
                "cache",
                task
                + "_"
                + split
                + "_"
                + str(max_seq_length)
                + "_"
                + str(max_region_num)
                + ".pkl",
            )
        
        if not os.path.exists(cache_path):
            self.tokenize()
            self.tensorize()
            cPickle.dump(self.entries, open(cache_path, "wb"))
        else:
            print("loading entries from %s" % (cache_path))
            self.entries = cPickle.load(open(cache_path, "rb"))

    def __len__(self):
        ## return len(self.annotations)
        return len(self.data)
    
    def __getitem__(self, idx):
        
        sample = self.data[idx]
    
        img_path = os.path.join(self.img_dir, sample["t2c_img"])

        with open(img_path, "rb") as f:
            image = Image.open(f).convert("RGB")

        verfi_mask = torch.zeros((3,900,1600))

        target = {}
        target["origin_image"] = self.img_transform(image)

        # target['origin_image'] = image
        gt = sample["2d_box"] # format [x1,y1,w,h]
        
        bbox = torch.tensor([gt[0],gt[1],gt[0]+gt[2],gt[1]+gt[3]]) # translation [x1,y1,w,h] to  [x1,y1,x2,y2]

        verfi_mask[:, bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1
        verfi_mask = verfi_mask.unsqueeze(0) 

        downsampled_image = F.interpolate(verfi_mask, size=(40, 40), mode='bilinear', align_corners=False).squeeze()[0]
        target['downsampled_mask'] = downsampled_image 

        target['mask'] = torch.tensor([0])
        # phrase = sample["command"].lower()
        target['Command']=sample["command"]
        target['image_path'] = img_path
        # target['phrase'] = phrase
        target['bbox'] = bbox 

        target['origin_boundingbox_xyxy'] = torch.tensor([gt[0]/1600,gt[1]/900,(gt[0]+gt[2])/1600,(gt[1]+gt[3])/900])
        target['command_token'] = sample['command_token'] # 提交官网所需要的指标，bbox格式[x1,,y1,w,h]

        # target['scene_token'] = sample['scene_token']

        if self.test or self.debug:
            target['orig_bbox'] = bbox.clone()


        
        target["all_region_feature"] = self.region_information[sample["t2c_img"]]["all_region_feature"].to('cpu')

        target["all_region_bbox"] = torch.tensor(self.region_information[sample["t2c_img"]]["all_region_bbox"]).to('cpu')
        #target["all_region_bbox"] = torch.tensor(self.region_information[sample["t2c_img"]]["all_region_bbox"])
        target["all_region_iou"] = torch.tensor(self.region_information[sample["t2c_img"]]["all_region_IOU"]).to('cpu')
        target["question"] = self.region_information[sample["t2c_img"]]["question"].to('cpu')

        target["region_mask"] = torch.tensor(self.region_information[sample["t2c_img"]]["image_mask"]).to('cpu')
        target["input_mask"] = torch.tensor(self.region_information[sample["t2c_img"]]["input_mask"]).to('cpu')


        image = target['mask']
        if 'mask' in target:
            mask = target.pop('mask')
            return image, mask, target
        else:
            return image, target
        



import numpy as np

class ImageFeaturesH5Reader(object):
    """
    A reader for H5 files containing pre-extracted image features. A typical
    H5 file is expected to have a column named "image_id", and another column
    named "features".

    Example of an H5 file:
    ```
    faster_rcnn_bottomup_features.h5
       |--- "image_id" [shape: (num_images, )]
       |--- "features" [shape: (num_images, num_proposals, feature_size)]
       +--- .attrs ("split", "train")
    ```
    # TODO (kd): Add support to read boxes, classes and scores.

    Parameters
    ----------
    features_h5path : str
        Path to an H5 file containing COCO train / val image features.
    in_memory : bool
        Whether to load the whole H5 file in memory. Beware, these files are
        sometimes tens of GBs in size. Set this to true if you have sufficient
        RAM - trade-off between speed and memory.
    """

    def __init__(self, features_path: str, config, in_memory: bool = False):
        self.features_path = features_path
        self._in_memory = in_memory

        # If not loaded in memory, then list of None.
        self.env = lmdb.open(
            self.features_path,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        with self.env.begin(write=False) as txn:
            self._image_ids = pickle.loads(txn.get("keys".encode()))

        self.features = [None] * len(self._image_ids)
        self.num_boxes = [None] * len(self._image_ids)
        self.boxes = [None] * len(self._image_ids)
        self.boxes_ori = [None] * len(self._image_ids)
        self.feature_size = config.v_feature_size
        self.num_locs = config.num_locs
        self.add_global_imgfeat = config.add_global_imgfeat

    def __len__(self):
        return len(self._image_ids)

    def __getitem__(self, image_id):
        image_id = str(image_id).encode()
        index = self._image_ids.index(image_id)
        if self._in_memory:
            # Load features during first epoch, all not loaded together as it has a slow start.
            if self.features[index] is not None:
                features = self.features[index]
                num_boxes = self.num_boxes[index]
                image_location = self.boxes[index]
                image_location_ori = self.boxes_ori[index]
            else:
                with self.env.begin(write=False) as txn:
                    item = pickle.loads(txn.get(image_id))
                    image_h = int(item["img_h"])
                    image_w = int(item["img_w"])

                    features = np.frombuffer(base64.b64decode(item["features"]), dtype=np.float32).reshape(-1, self.feature_size)
                    boxes = np.frombuffer(base64.b64decode(item['boxes']), dtype=np.float32).reshape(-1, 4)

                    image_location = np.zeros((boxes.shape[0], self.num_locs), dtype=np.float32)
                    image_location[:, :4] = boxes
                    if self.num_locs == 5:
                        image_location[:, 4] = (
                                (image_location[:, 3] - image_location[:, 1])
                                * (image_location[:, 2] - image_location[:, 0])
                                / (float(image_w) * float(image_h))
                        )

                    image_location_ori = copy.deepcopy(image_location)
                    image_location[:, 0] = image_location[:, 0] / float(image_w)
                    image_location[:, 1] = image_location[:, 1] / float(image_h)
                    image_location[:, 2] = image_location[:, 2] / float(image_w)
                    image_location[:, 3] = image_location[:, 3] / float(image_h)

                    num_boxes = features.shape[0]
                    if self.add_global_imgfeat == "first":
                        g_feat = np.sum(features, axis=0) / num_boxes
                        num_boxes = num_boxes + 1
                        features = np.concatenate([np.expand_dims(g_feat, axis=0), features], axis=0)

                        g_location = [0, 0, 1, 1] + [1] * (self.num_locs - 4)
                        image_location = np.concatenate([np.expand_dims(g_location, axis=0), image_location], axis=0)

                        g_location_ori = np.array([0, 0, image_w, image_h] + [image_w * image_h] * (self.num_locs - 4))
                        image_location_ori = np.concatenate(
                            [np.expand_dims(g_location_ori, axis=0), image_location_ori], axis=0
                        )

                    elif self.add_global_imgfeat == "last":
                        g_feat = np.sum(features, axis=0) / num_boxes
                        num_boxes = num_boxes + 1
                        features = np.concatenate([features, np.expand_dims(g_feat, axis=0)], axis=0)

                        g_location = [0, 0, 1, 1] + [1] * (self.num_locs - 4)
                        image_location = np.concatenate([image_location, np.expand_dims(g_location, axis=0)], axis=0)

                        g_location_ori = np.array([0, 0, image_w, image_h] + [image_w * image_h] * (self.num_locs - 4))
                        image_location_ori = np.concatenate(
                            [image_location_ori, np.expand_dims(g_location_ori, axis=0)], axis=0
                        )

                    self.features[index] = features
                    self.boxes[index] = image_location
                    self.boxes_ori[index] = image_location_ori
                    self.num_boxes[index] = num_boxes
        else:
            # Read chunk from file everytime if not loaded in memory.
            with self.env.begin(write=False) as txn:
                item = pickle.loads(txn.get(image_id))
                image_h = int(item["img_h"])
                image_w = int(item["img_w"])

                features = np.frombuffer(base64.b64decode(item["features"]), dtype=np.float32).reshape(-1, self.feature_size)
                boxes = np.frombuffer(base64.b64decode(item['boxes']), dtype=np.float32).reshape(-1, 4)

                image_location = np.zeros((boxes.shape[0], self.num_locs), dtype=np.float32)
                image_location[:, :4] = boxes
                if self.num_locs == 5:
                    image_location[:, 4] = (
                            (image_location[:, 3] - image_location[:, 1])
                            * (image_location[:, 2] - image_location[:, 0])
                            / (float(image_w) * float(image_h))
                    )

                image_location_ori = copy.deepcopy(image_location)
                image_location[:, 0] = image_location[:, 0] / float(image_w)
                image_location[:, 1] = image_location[:, 1] / float(image_h)
                image_location[:, 2] = image_location[:, 2] / float(image_w)
                image_location[:, 3] = image_location[:, 3] / float(image_h)

                num_boxes = features.shape[0]
                if self.add_global_imgfeat == "first":
                    g_feat = np.sum(features, axis=0) / num_boxes
                    num_boxes = num_boxes + 1
                    features = np.concatenate([np.expand_dims(g_feat, axis=0), features], axis=0)

                    g_location = [0, 0, 1, 1] + [1] * (self.num_locs - 4)
                    image_location = np.concatenate([np.expand_dims(g_location, axis=0), image_location], axis=0)

                    g_location_ori = np.array([0, 0, image_w, image_h] + [image_w * image_h] * (self.num_locs - 4))
                    image_location_ori = np.concatenate(
                        [np.expand_dims(g_location_ori, axis=0), image_location_ori], axis=0
                    )

                elif self.add_global_imgfeat == "last":
                    g_feat = np.sum(features, axis=0) / num_boxes
                    num_boxes = num_boxes + 1
                    features = np.concatenate([features, np.expand_dims(g_feat, axis=0)], axis=0)

                    g_location = [0, 0, 1, 1] + [1] * (self.num_locs - 4)
                    image_location = np.concatenate([image_location, np.expand_dims(g_location, axis=0)], axis=0)

                    g_location_ori = np.array([0, 0, image_w, image_h] + [image_w * image_h] * (self.num_locs - 4))
                    image_location_ori = np.concatenate(
                        [image_location_ori, np.expand_dims(g_location_ori, axis=0)], axis=0
                    )

        return features, num_boxes, image_location, image_location_ori

    def keys(self) -> List[int]:
        return self._image_ids

if __name__ == "__main__":
    target =Talk2Car_base(split="train",root='./download/image').__getitem__(1)

    with open(target['image_path'], "rb") as f:
        original_image = Image.open(f).convert("RGB")
    import matplotlib.pyplot as plt
    import numpy as np
    import torchvision.transforms as pic_transforms
    import matplotlib.patches as patches

    
    attention_map_np = target['map_mask'].cpu().detach().numpy()
    downsampled_mask =target['downsampled_mask']
    downsampled_mask = downsampled_mask.unsqueeze(0).unsqueeze(0)
    
    upsampled_attention_map = F.interpolate(downsampled_mask, size=(900, 1600), mode='bilinear', align_corners=False)
    attention_map_np = upsampled_attention_map.cpu().squeeze().numpy()
    attention_map_np = (attention_map_np - attention_map_np.min()) / (attention_map_np.max() - attention_map_np.min())

    image= pic_transforms.Compose([
                                    pic_transforms.ToTensor(), 
                                    ])
    fig, ax = plt.subplots(1)
    for i in range(sample['rpn_bbox_lbrt'].size(0)):
        bbox = sample['rpn_bbox_lbrt'][i].tolist()
        xl, yb, xr, yt = bbox 
        w, h = xr - xl, yt - yb
        rect = patches.Rectangle((xl, yb), w, h, fill = False, edgecolor = 'r')
        ax.add_patch(rect)

    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='black', facecolor='none')
    ax.add_patch(rect)

    ax.imshow(original_image)
    plt.savefig("./utils/example_dataset.jpg")
