# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) 2020, Emanuele Bugliarello (@e-bug).

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import _pickle as cPickle

import numpy as np

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer
from ._image_features_reader import ImageFeaturesH5Reader

from tools.refer.refer import REFER, REFERTALK2CAR
from collections import defaultdict




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


def assert_eq(real, expected):
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)

import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


# (1600, 900)
import os
from PIL import Image
import torch
from torchvision import transforms
import json

# 加载图片 

def load_images_as_dict(root_dir='/home/tam/Documents/RSDLayerAttn/RSDLayerAttn/data/talk2car/images/images/',split='train',if_transform=True,size_w=800,size_h=450):
    image_dict = {}

    transform = transforms.Compose([
        transforms.Resize((size_w, size_h)),
        transforms.ToTensor(),
    ])

    for fname in os.listdir(root_dir):
        if fname.endswith('.jpg'):
            # 解析文件名中的数字
            file_parts = fname.split('_')
            if file_parts[0] == 'img' and file_parts[1]==split:
                try:
                    key = file_parts[2]
                    img_path = os.path.join(root_dir, fname)
                    image = Image.open(img_path)

                    if if_transform:
                        image = transform(image)
                        if type(imge)==NoneType:
                            print("Error imge type")
                    image_dict[key] = image

                except ValueError:
                    pass  # 忽略无效的文件名

    print("successfully load image dataset")
    return image_dict

import os
import sys
def return_image(image_id,split,root_dir='/home/tam/Documents/RSDLayerAttn/RSDLayerAttn/data/talk2car/images/images',if_print=False,if_transform=True,size_w=800,size_h=450):
    image_dict = {}

    transform = transforms.Compose([
        transforms.Resize((size_w, size_h)),
        transforms.ToTensor(),
    ])

    for fname in os.listdir(root_dir):
        if fname.endswith('.jpg'):
            # 解析文件名中的数字
            name,extension = os.path.splitext(fname)
            file_parts = name.split('_')
            #print(file_parts)
            #print(file_parts[2])

            if file_parts[0] == 'img' and file_parts[1]==split and int(file_parts[2])==image_id :

                img_path = os.path.join(root_dir, fname)
                if if_print:
                    print(img_path)

                if os.path.exists(img_path):
                    image = Image.open(img_path)
                else :
                    print(f"could not find the image of{split} {image_id}")
                    sys.exist(1)

                if if_transform:
                    # (1600,900)
                    image = transform(image)
                    
                    return image
                else:
                    print(f"could not find the image of{split} {image_id}")

        else:
            print(f"could not find the image of{split} {image_id}")

#image_dict = load_images_as_dict('/path/to/your/images')


class ReferExpressionDataset(Dataset):
    def __init__(
        self,
        task: str,
        dataroot: str,
        annotations_jsonpath: str,
        split: str,
        image_features_reader: ImageFeaturesH5Reader,
        gt_image_features_reader: ImageFeaturesH5Reader,
        tokenizer: AutoTokenizer,
        bert_model,
        padding_index: int = 0,
        max_seq_length: int = 20,
        max_region_num: int = 60,
        num_locs=5,
        add_global_imgfeat=None,
        append_mask_sep=False,
    ):
        self.split = split

        if task == "refcocog":
            self.refer = REFER(dataroot, dataset=task, splitBy="umd")
        elif task.startswith("talk2car"):
            print("using talk2car REFER data loader")
            self.refer = REFERTALK2CAR(dataroot)
        else:
            self.refer = REFER(dataroot, dataset=task, splitBy="unc")

        if self.split == "mteval":
            self.ref_ids = self.refer.getRefIds(split="train")
        else:
            self.ref_ids = self.refer.getRefIds(split=split)

        print("%s refs are in split [%s]." % (len(self.ref_ids), split))

        self.num_labels = 1
        self._image_features_reader = image_features_reader
        self._gt_image_features_reader = gt_image_features_reader
        self._tokenizer = tokenizer

        self._padding_index = padding_index
        self._max_seq_length = max_seq_length
        self.dataroot = dataroot
        self.entries = self._load_annotations()

        self._max_region_num = max_region_num + int(add_global_imgfeat is not None)
        self._num_locs = num_locs
        self._add_global_imgfeat = add_global_imgfeat

        if "roberta" in bert_model:
            cache_path = os.path.join(
                dataroot,
                "cache",
                task
                + "_"
                + split
                + "_"
                + "roberta"
                + "_"
                + str(max_seq_length)
                + "_"
                + str(max_region_num)
                + ".pkl",
            )
        else:
            cache_path = os.path.join(
                dataroot,
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

    def _load_annotations(self):
        # Build an index which maps image id with a list of caption annotations.
        entries = []
        remove_ids = []
        if self.split == "mteval":
            remove_ids = np.load(
                os.path.join(self.dataroot, "cache", "coco_test_ids.npy")
            )
            remove_ids = [int(x) for x in remove_ids]

        for ref_id in self.ref_ids:
            ref = self.refer.Refs[ref_id]
            image_id = ref["image_id"]
            if self.split == "train" and int(image_id) in remove_ids:
                continue
            elif self.split == "mteval" and int(image_id) not in remove_ids:
                continue
            ref_id = ref["ref_id"]
            refBox = self.refer.getRefBox(ref_id)
            for sent, sent_id in zip(ref["sentences"], ref["sent_ids"]):
                caption = sent["raw"]
                entries.append(
                    {
                        "caption": caption,
                        "sent_id": sent_id,
                        "image_id": image_id,
                        "refBox": refBox,
                        "ref_id": ref_id,
                    }
                )

        return entries

    def tokenize(self):
        """Tokenizes the captions.

        This will add caption_tokens in each entry of the dataset.
        -1 represents nil, and should be treated as padding_idx in embedding.
        """
        for entry in self.entries:
            print(entry["caption"])
            tokens = self._tokenizer.encode(entry["caption"])
            tokens = [tokens[0]] + tokens[1:-1][: self._max_seq_length - 2] + [tokens[-1]]

            segment_ids = [0] * len(tokens)
            input_mask = [1] * len(tokens)

            if len(tokens) < self._max_seq_length:
                # Note here we pad in front of the sentence
                padding = [self._padding_index] * (self._max_seq_length - len(tokens))
                tokens = tokens + padding
                input_mask += padding
                segment_ids += padding

            assert_eq(len(tokens), self._max_seq_length)
            entry["token"] = tokens
            entry["input_mask"] = input_mask
            entry["segment_ids"] = segment_ids

    def tensorize(self):
        for entry in self.entries:
            token = torch.from_numpy(np.array(entry["token"]))
            entry["token"] = token

            input_mask = torch.from_numpy(np.array(entry["input_mask"]))
            entry["input_mask"] = input_mask

            segment_ids = torch.from_numpy(np.array(entry["segment_ids"]))
            entry["segment_ids"] = segment_ids

    def __getitem__(self, index):
        entry = self.entries[index]

        image_id = entry["image_id"]
        ref_box = entry["refBox"]

        ref_box = [
            ref_box[0],
            ref_box[1],
            ref_box[0] + ref_box[2],
            ref_box[1] + ref_box[3],
        ]
        features, num_boxes, boxes, boxes_ori = self._image_features_reader[image_id]

        boxes_ori = boxes_ori[:num_boxes]
        boxes = boxes[:num_boxes]
        features = features[:num_boxes]

        mix_boxes_ori = boxes_ori
        mix_boxes = boxes
        mix_features = features
        mix_num_boxes = min(int(num_boxes), self._max_region_num)
        
        #print("index:{}, ref_box:{}, predict:{}".format(index, ref_box, mix_boxes_ori[:, :1]))
        
        mix_target = iou(
            torch.tensor(mix_boxes_ori[:, :4]).float(),
            torch.tensor([ref_box]).float(),
        )
        
        """
        bbox1 = mix_boxes_ori[torch.argmax(mix_target),:4].tolist()
        int_bbox1=[]
        for k in bbox1:
            k =int(k)
            int_bbox1.append(k)
        #print("index:{}, predict_box:{}".format(index,int_bbox1))
        print(index, int_bbox1)
        """
        
        image_mask = [1] * (mix_num_boxes)
        while len(image_mask) < self._max_region_num:
            image_mask.append(0)
        mix_boxes_pad = np.zeros((self._max_region_num, self._num_locs))
        mix_features_pad = np.zeros((self._max_region_num, 2048))

        mix_boxes_pad[:mix_num_boxes] = mix_boxes[:mix_num_boxes]
        mix_features_pad[:mix_num_boxes] = mix_features[:mix_num_boxes]
                
        # pad mix box ori
        mix_boxes_ori_pad = np.zeros((self._max_region_num, self._num_locs))
        mix_boxes_ori_pad[:mix_num_boxes]  = mix_boxes_ori[:mix_num_boxes]
        
        # appending the target feature.
        features = torch.tensor(mix_features_pad).float()
        image_mask = torch.tensor(image_mask).long()
        spatials = torch.tensor(mix_boxes_pad).float()
        
        
        target = torch.zeros((self._max_region_num, 1)).float()
        target[:mix_num_boxes] = mix_target[:mix_num_boxes]
        
        #bbox2 = mix_boxes_ori[torch.argmax(target[:mix_num_boxes]),:4].tolist()
        
        spatials_ori = torch.tensor(mix_boxes_ori_pad).float()
        #print(spatials_ori)
        caption = entry["token"]
        input_mask = entry["input_mask"]
        segment_ids = entry["segment_ids"]
        
        #print("spatials")
        #print(spatials[1,:4])
        #print("mix_boxes")
        #print(mix_boxes[1,:4])
        #print("spatials_ori")
        #print(spatials_ori[1,:4])
        #print("mix_boxes_ori")
        #print(mix_boxes_ori[1,:4])
        #exit()

        return features, spatials, spatials_ori, image_mask, caption, target, input_mask, segment_ids, image_id

    def __len__(self):
        return len(self.entries)



class ReferExpressionDataset_2(Dataset):
    def __init__(
        self,
        task: str,
        dataroot: str,
        annotations_jsonpath: str,
        split: str,
        image_features_reader: ImageFeaturesH5Reader, # 读2048
        gt_image_features_reader: ImageFeaturesH5Reader,
        tokenizer: AutoTokenizer,
        bert_model,
        padding_index: int = 0,
        max_seq_length: int = 20,
        max_region_num: int = 60,
        num_locs=5,
        add_global_imgfeat=None,
        append_mask_sep=False,
    ):
        self.split = split

        if task == "refcocog":
            self.refer = REFER(dataroot, dataset=task, splitBy="umd")
        elif task.startswith("talk2car"):
            print("using talk2car REFER data loader")
            self.refer = REFERTALK2CAR(dataroot)

        else:
            self.refer = REFER(dataroot, dataset=task, splitBy="unc")
                                                                      
        if self.split == "mteval":
            self.ref_ids = self.refer.getRefIds(split="train")
        else:
            self.ref_ids = self.refer.getRefIds(split=split)
        
        print("%s refs are in split [%s]." % (len(self.ref_ids), split))

        self.num_labels = 1
        self._image_features_reader = image_features_reader
        self._gt_image_features_reader = gt_image_features_reader
        self._tokenizer = tokenizer
        #self.blip_tokenizer = init_tokenizer()

        self._padding_index = padding_index
        self._max_seq_length = max_seq_length
        self.dataroot = dataroot
        self.entries = self._load_annotations()

        self._max_region_num = max_region_num + int(add_global_imgfeat is not None)
        self._num_locs = num_locs
        self._add_global_imgfeat = add_global_imgfeat

        json_path = "/home/tam/Documents/RSDLayerAttn/RSDLayerAttn/data/talk2car/classification_results.json"
        with open(json_path, 'r') as json_file:
            self.classification_dict = json.load(json_file)

        if "roberta" in bert_model:
            cache_path = os.path.join(
                dataroot,
                "cache",
                task
                + "_"
                + split
                + "_"
                + "roberta"
                + "_"
                + str(max_seq_length)
                + "_"
                + str(max_region_num)
                + ".pkl",
            )
        else:
            cache_path = os.path.join(
                dataroot,
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

        
    def _load_annotations(self):
        # Build an index which maps image id with a list of caption annotations.
        entries = []
        remove_ids = []
        if self.split == "mteval":
            remove_ids = np.load(
                os.path.join(self.dataroot, "cache", "coco_test_ids.npy")
            )
            remove_ids = [int(x) for x in remove_ids]

        for ref_id in self.ref_ids:
            ref = self.refer.Refs[ref_id]
            

            image_id = ref["image_id"]
            
            if self.split == "train" and int(image_id) in remove_ids:
                continue
            elif self.split == "mteval" and int(image_id) not in remove_ids:
                continue

            ref_id = ref["ref_id"]
            refBox = self.refer.getRefBox(ref_id)
            for sent, sent_id in zip(ref["sentences"], ref["sent_ids"]):
                caption = sent["raw"]

                entries.append(
                    {   
                        "caption": caption, #
                        "sent_id": sent_id,
                        "image_id": image_id,
                        "refBox": refBox,
                        "ref_id": ref_id,
                    }
                )

        return entries

    def tokenize(self):
        """Tokenizes the captions.

        This will add caption_tokens in each entry of the dataset.
        -1 represents nil, and should be treated as padding_idx in embedding.
        """
        for entry in self.entries:
            tokens = self._tokenizer.encode(entry["caption"])

            tokens = [tokens[0]] + tokens[1:-1][: self._max_seq_length - 2] + [tokens[-1]]

            segment_ids = [0] * len(tokens)
            input_mask = [1] * len(tokens)

            if len(tokens) < self._max_seq_length:
                # Note here we pad in front of the sentence
                padding = [self._padding_index] * (self._max_seq_length - len(tokens))
                tokens = tokens + padding
                input_mask += padding
                segment_ids += padding

            assert_eq(len(tokens), self._max_seq_length)

            entry["token"] = tokens
            entry["input_mask"] = input_mask
            entry["segment_ids"] = segment_ids

    def tensorize(self):
        for entry in self.entries:
            token = torch.from_numpy(np.array(entry["token"]))
            entry["token"] = token

            input_mask = torch.from_numpy(np.array(entry["input_mask"]))
            entry["input_mask"] = input_mask

            segment_ids = torch.from_numpy(np.array(entry["segment_ids"]))
            entry["segment_ids"] = segment_ids

    def __getitem__(self, index):

        entry = self.entries[index]
        image_id = entry["image_id"]
        ref_box = entry["refBox"]

        raw_txt = entry["caption"]
        
       # class_emotion = self.classification_dict[image_id]
        
        # 
        ref_box = [
            ref_box[0],
            ref_box[1],
            ref_box[0] + ref_box[2],
            ref_box[1] + ref_box[3],
        ]
        
        features, num_boxes, boxes, boxes_ori = self._image_features_reader[image_id]
    
        
        if self.split == 'val':
            reshape_id = image_id - 8349
            name = "img_val_"+str(reshape_id)+".jpg"
        elif self.split == 'test':
            reshape_id = image_id - 9512
            name = "img_test_"+str(reshape_id)+".jpg"
        elif self.split == 'train':
            reshape_id = image_id
            name = "img_train_"+str(reshape_id)+".jpg"
     #   print(self.split)
     #   print(reshape_id)
        
        img = return_image(reshape_id,split=f'{self.split}',if_print=False)


        boxes_ori = boxes_ori[:num_boxes]
        boxes = boxes[:num_boxes]
        features = features[:num_boxes]

        mix_boxes_ori = boxes_ori
        mix_boxes = boxes
        mix_features = features
        mix_num_boxes = min(int(num_boxes), self._max_region_num)
       # print("mix_num_boxes",mix_num_boxes)
        #print("index:{}, ref_box:{}, predict:{}".format(index, ref_box, mix_boxes_ori[:, :1]))
        
        mix_target = iou(
            torch.tensor(mix_boxes_ori[:, :4]).float(),
            torch.tensor([ref_box]).float(),
        )

        

        
        image_mask = [1] * (mix_num_boxes)
        while len(image_mask) < self._max_region_num:
            image_mask.append(0)
            print(1)
            
        # print("iamge_mask",image_mask)

        mix_boxes_pad = np.zeros((self._max_region_num, self._num_locs))
        mix_features_pad = np.zeros((self._max_region_num, 2048))

        mix_boxes_pad[:mix_num_boxes] = mix_boxes[:mix_num_boxes]
        mix_features_pad[:mix_num_boxes] = mix_features[:mix_num_boxes]
                
        # pad mix box ori
        mix_boxes_ori_pad = np.zeros((self._max_region_num, self._num_locs))
        mix_boxes_ori_pad[:mix_num_boxes]  = mix_boxes_ori[:mix_num_boxes]
        
        # appending the target feature.
        features = torch.tensor(mix_features_pad).float()
        image_mask = torch.tensor(image_mask).long()
        spatials = torch.tensor(mix_boxes_pad).float()
        
        img = torch.tensor(img).float()
#        print(img.shape)
        
       # text_blip_emd = torch.tensor(text_blip_emd).float()
       # print(text_blip_emd.shape)

        target = torch.zeros((self._max_region_num, 1)).float()
        target[:mix_num_boxes] = mix_target[:mix_num_boxes]
        
        #bbox2 = mix_boxes_ori[torch.argmax(target[:mix_num_boxes]),:4].tolist()
        
        spatials_ori = torch.tensor(mix_boxes_ori_pad).float()
        #print(spatials_ori)
        caption = entry["token"]
        
        input_mask = entry["input_mask"]
        segment_ids = entry["segment_ids"]
        a = 5
        
       # print(image_id,raw_txt)
        #print("image_id",image_id," segment_ids",segment_ids)
        """
        feature: torch.Size([37, 2048])
        spatials torch.Size([37, 5]) 这里应该是这三十六个局部区域的bbox,第一个为无效bbox,但是这里第五个维度是啥
        saptials_ori
        image_mask 
        caption: 编码过后的文本向量
        target torch.Size([37, 1])  tensor([0.0371]) 这三十六的区域的IOU值,用这部分IOU值来拟合选择哪个是真确的区域。
        input_mask
        target 
        """

        # print("saptials",spatials.shape)
        # print(spatials[1])

       
        return features, spatials, spatials_ori, image_mask, caption, target, input_mask, segment_ids, image_id,img,raw_txt,name

    def __len__(self):
        return len(self.entries)





class ReferExpressionTargetObjCategorizationDataset(ReferExpressionDataset):
    def __init__(
            self,
            task: str,
            dataroot: str,
            annotations_jsonpath: str,
            split: str,
            image_features_reader: ImageFeaturesH5Reader,
            gt_image_features_reader: ImageFeaturesH5Reader,
            tokenizer: AutoTokenizer,
            bert_model,
            padding_index: int = 0,
            max_seq_length: int = 20,
            max_region_num: int = 60,
            num_locs=5,
            add_global_imgfeat=None,
            append_mask_sep=False,
    ):
        super(ReferExpressionTargetObjCategorizationDataset, self).__init__(task, dataroot, annotations_jsonpath, split,
                                                                  image_features_reader, gt_image_features_reader,
                                                                  tokenizer,
                                                                  bert_model, padding_index, max_seq_length,
                                                                  max_region_num,
                                                                  num_locs, add_global_imgfeat, append_mask_sep)

        print("ReferExpressionObjClassificationDataset built")

    def _load_annotations(self):
        # Build an index which maps image id with a list of caption annotations.
        entries = []
        remove_ids = []
        if self.split == "mteval":
            remove_ids = np.load(
                os.path.join(self.dataroot, "cache", "coco_test_ids.npy")
            )
            remove_ids = [int(x) for x in remove_ids]

        ref_category_id_counter = np.array([0, 0, 0, 0])
        ref_category_id_list = []

        for ref_id in self.ref_ids:
            ref = self.refer.Refs[ref_id]
            image_id = ref["image_id"]
            if self.split == "train" and int(image_id) in remove_ids:
                continue
            elif self.split == "mteval" and int(image_id) not in remove_ids:
                continue
            ref_id = ref["ref_id"]
            refBox = self.refer.getRefBox(ref_id)
            ref_ann = self.refer.refToAnn[ref_id]

            # simplify category
            # ref_id_simplified {0: human, 1: object, 2: vehicle.other, 3: vehicle.car}
            if 0 < ref_ann["category_id"] <= 7:
                ref_category_id_simplified = 0
            elif 7 < ref_ann["category_id"] <= 12:
                ref_category_id_simplified = 1
            elif 12 < ref_ann["category_id"] <= 15:
                ref_category_id_simplified = 2
            elif ref_ann["category_id"] == 16:
                ref_category_id_simplified = 3
            elif 16 < ref_ann["category_id"]:
                ref_category_id_simplified = 2
            else:
                raise ValueError
            ref_category_id_counter[ref_category_id_simplified] += 1
            ref_category_id_list.append(ref_category_id_simplified)

            for sent, sent_id in zip(ref["sentences"], ref["sent_ids"]):
                caption = sent["raw"]
                entries.append(
                    {
                        "caption": caption,
                        "sent_id": sent_id,
                        "image_id": image_id,
                        "refBox": refBox,
                        #"ref_category_name": ref_ann["category_name"],
                        #"ref_category_id": [ref_ann["category_id"]],
                        "ref_category_id": [ref_category_id_simplified],
                        "ref_id": ref_id,
                    }
                )

        normalized_category_count = ref_category_id_counter / ref_category_id_counter.sum()
        category_weights = 1. / normalized_category_count
        print("ref_category_id_counter")
        print(ref_category_id_counter)
        print(normalized_category_count)
        print("category_weights")
        print(category_weights)
        category_weights = torch.from_numpy(category_weights).float()
        self.sample_category_weights = category_weights[ref_category_id_list]

        return entries

    def tensorize(self):
        for entry in self.entries:
            token = torch.from_numpy(np.array(entry["token"]))
            entry["token"] = token

            input_mask = torch.from_numpy(np.array(entry["input_mask"]))
            entry["input_mask"] = input_mask

            segment_ids = torch.from_numpy(np.array(entry["segment_ids"]))
            entry["segment_ids"] = segment_ids

            # tensorize sequence label
            ref_category_id = torch.from_numpy(np.array(entry["ref_category_id"]))
            entry["ref_category_id"] = ref_category_id


    def __getitem__(self, index):
        entry = self.entries[index]

        image_id = entry["image_id"]
        ref_box = entry["refBox"]
        
        ref_box = [
            ref_box[0],
            ref_box[1],
            ref_box[0] + ref_box[2],
            ref_box[1] + ref_box[3],
        ]
        features, num_boxes, boxes, boxes_ori = self._image_features_reader[image_id]

        boxes_ori = boxes_ori[:num_boxes]
        boxes = boxes[:num_boxes]
        features = features[:num_boxes]

        mix_boxes_ori = boxes_ori
        mix_boxes = boxes
        mix_features = features
        mix_num_boxes = min(int(num_boxes), self._max_region_num)

        # print("index:{}, ref_box:{}, predict:{}".format(index, ref_box, mix_boxes_ori[:, :1]))

        mix_target = iou(
            torch.tensor(mix_boxes_ori[:, :4]).float(),
            torch.tensor([ref_box]).float(),
        )

        """
        bbox1 = mix_boxes_ori[torch.argmax(mix_target),:4].tolist()
        int_bbox1=[]
        for k in bbox1:
            k =int(k)
            int_bbox1.append(k)
        #print("index:{}, predict_box:{}".format(index,int_bbox1))
        print(index, int_bbox1)
        """

        image_mask = [1] * (mix_num_boxes)
        while len(image_mask) < self._max_region_num:
            image_mask.append(0)
        mix_boxes_pad = np.zeros((self._max_region_num, self._num_locs))
        mix_features_pad = np.zeros((self._max_region_num, 2048))

        mix_boxes_pad[:mix_num_boxes] = mix_boxes[:mix_num_boxes]
        mix_features_pad[:mix_num_boxes] = mix_features[:mix_num_boxes]

        # pad mix box ori
        mix_boxes_ori_pad = np.zeros((self._max_region_num, self._num_locs))
        mix_boxes_ori_pad[:mix_num_boxes] = mix_boxes_ori[:mix_num_boxes]

        # appending the target feature.
        features = torch.tensor(mix_features_pad).float()
        image_mask = torch.tensor(image_mask).long()
        spatials = torch.tensor(mix_boxes_pad).float()

        target = torch.zeros((self._max_region_num, 1)).float()
        target[:mix_num_boxes] = mix_target[:mix_num_boxes]

        # bbox2 = mix_boxes_ori[torch.argmax(target[:mix_num_boxes]),:4].tolist()

        spatials_ori = torch.tensor(mix_boxes_ori_pad).float()
        # print(spatials_ori)
        caption = entry["token"]
        input_mask = entry["input_mask"]
        segment_ids = entry["segment_ids"]
        ref_category_id = entry["ref_category_id"]

        # print("spatials")
        # print(spatials[1,:4])
        # print("mix_boxes")
        # print(mix_boxes[1,:4])
        # print("spatials_ori")
        # print(spatials_ori[1,:4])
        # print("mix_boxes_ori")
        # print(mix_boxes_ori[1,:4])
        # exit()
        

        return features, spatials, spatials_ori, image_mask, caption, target, input_mask, segment_ids, image_id, ref_category_id


class ReferExpressionSequenceLabelDataset(ReferExpressionDataset):
    def __init__(
        self,
        task: str,
        dataroot: str,
        annotations_jsonpath: str,
        split: str,
        image_features_reader: ImageFeaturesH5Reader,
        gt_image_features_reader: ImageFeaturesH5Reader,
        tokenizer: AutoTokenizer,
        bert_model,
        padding_index: int = 0,
        max_seq_length: int = 20,
        max_region_num: int = 60,
        num_locs=5,
        add_global_imgfeat=None,
        append_mask_sep=False,
    ):

        self.sequence_label_to_id = defaultdict(int)  # the default value is 0
        self.sequence_label_to_id["PROPN"] = 1
        self.sequence_label_to_id["NOUN"] = 1
        #self.sequence_label_to_id["ADJ"] = 1
        #self.sequence_label_to_id["DET"] = 1
        #self.sequence_label_to_id["ADP"] = 1
        # PROPN, NOUN, ADJ, ADV, DET, ADP

        super(ReferExpressionSequenceLabelDataset, self).__init__(task, dataroot, annotations_jsonpath, split,
                                                            image_features_reader, gt_image_features_reader, tokenizer,
                                                            bert_model, padding_index, max_seq_length, max_region_num,
                                                            num_locs, add_global_imgfeat, append_mask_sep)

        print("ReferExpressionSequenceLabelDataset built")

    def _load_annotations(self):
        # Build an index which maps image id with a list of caption annotations.
        entries = []
        remove_ids = []
        if self.split == "mteval":
            remove_ids = np.load(
                os.path.join(self.dataroot, "cache", "coco_test_ids.npy")
            )
            remove_ids = [int(x) for x in remove_ids]

        for ref_id in self.ref_ids:
            ref = self.refer.Refs[ref_id]
            image_id = ref["image_id"]
            if self.split == "train" and int(image_id) in remove_ids:
                continue
            elif self.split == "mteval" and int(image_id) not in remove_ids:
                continue
            ref_id = ref["ref_id"]
            refBox = self.refer.getRefBox(ref_id)
            for sent, sent_id in zip(ref["sentences"], ref["sent_ids"]):
                caption = sent["raw"]
                sequence_labels = sent["pos_labels_simple"]
                tokenized_sent = sent["spacy_tokenized_sent"]
                entries.append(
                    {
                        "caption": caption,
                        "tokenized_sent": tokenized_sent,
                        "sequence_labels": sequence_labels,
                        "sent_id": sent_id,
                        "image_id": image_id,
                        "refBox": refBox,
                        "ref_id": ref_id,
                    }
                )

        return entries

    # Tokenize all texts and align the labels with them.
    def tokenize(self):
        for entry in self.entries:
            # We use is_split_into_words because the texts in our dataset are lists of words (with a label for each word).
            transformers_tokenized_sent = self._tokenizer(entry["tokenized_sent"], padding=False, is_split_into_words=True)
            label = entry["sequence_labels"]
            assert len(entry["tokenized_sent"]) == len(entry["sequence_labels"])
            #print("raw")
            #print(entry["caption"])
            #print("tokenized_sent")
            #print(entry["tokenized_sent"])
            #print("label")
            #print(label)
            # construct label ids
            # Adapted from Huggingface Transformers
            word_ids = transformers_tokenized_sent.word_ids()
            #print("word_ids")
            #print(word_ids)
            #print("tokens")
            #print(transformers_tokenized_sent["input_ids"])
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(self.sequence_label_to_id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(self.sequence_label_to_id[label[word_idx]])
                    #label_ids.append(label_to_id[label[word_idx]] if data_args.label_all_tokens else -100)
                previous_word_idx = word_idx
            #print("label_ids")
            #print(label_ids)
            tokens = transformers_tokenized_sent["input_ids"]
            #print("tokens")
            #print(tokens)
            #exit()

            # truncate to max len
            tokens = [tokens[0]] + tokens[1:-1][: self._max_seq_length - 2] + [tokens[-1]]
            label_ids = [label_ids[0]] + label_ids[1:-1][: self._max_seq_length - 2] + [label_ids[-1]]

            assert len(tokens) == len(label_ids)

            segment_ids = [0] * len(tokens)
            input_mask = [1] * len(tokens)

            # padding
            if len(tokens) < self._max_seq_length:
                # Note here we pad in front of the sentence
                padding = [self._padding_index] * (self._max_seq_length - len(tokens))
                tokens = tokens + padding
                # Pad label
                label_padding = [-100] * (self._max_seq_length - len(label_ids))
                label_ids = label_ids + label_padding

                input_mask += padding
                segment_ids += padding

            assert_eq(len(tokens), self._max_seq_length)
            entry["token"] = tokens
            entry["input_mask"] = input_mask
            entry["segment_ids"] = segment_ids
            entry["sequence_label_ids"] = label_ids

    def tensorize(self):
        for entry in self.entries:
            token = torch.from_numpy(np.array(entry["token"]))
            entry["token"] = token

            input_mask = torch.from_numpy(np.array(entry["input_mask"]))
            entry["input_mask"] = input_mask

            segment_ids = torch.from_numpy(np.array(entry["segment_ids"]))
            entry["segment_ids"] = segment_ids

            # tensorize sequence label
            sequence_label_ids = torch.from_numpy(np.array(entry["sequence_label_ids"]))
            entry["sequence_label_ids"] = sequence_label_ids

    def __getitem__(self, index):
        entry = self.entries[index]

        image_id = entry["image_id"]
        ref_box = entry["refBox"]

        ref_box = [
            ref_box[0],
            ref_box[1],
            ref_box[0] + ref_box[2],
            ref_box[1] + ref_box[3],
        ]
        features, num_boxes, boxes, boxes_ori = self._image_features_reader[image_id]

        boxes_ori = boxes_ori[:num_boxes]
        boxes = boxes[:num_boxes]
        features = features[:num_boxes]

        mix_boxes_ori = boxes_ori
        mix_boxes = boxes
        mix_features = features
        mix_num_boxes = min(int(num_boxes), self._max_region_num)

        # print("index:{}, ref_box:{}, predict:{}".format(index, ref_box, mix_boxes_ori[:, :1]))

        mix_target = iou(
            torch.tensor(mix_boxes_ori[:, :4]).float(),
            torch.tensor([ref_box]).float(),
        )

        """
        bbox1 = mix_boxes_ori[torch.argmax(mix_target),:4].tolist()
        int_bbox1=[]
        for k in bbox1:
            k =int(k)
            int_bbox1.append(k)
        #print("index:{}, predict_box:{}".format(index,int_bbox1))
        print(index, int_bbox1)
        """

        image_mask = [1] * (mix_num_boxes)
        while len(image_mask) < self._max_region_num:
            image_mask.append(0)
        mix_boxes_pad = np.zeros((self._max_region_num, self._num_locs))
        mix_features_pad = np.zeros((self._max_region_num, 2048))

        mix_boxes_pad[:mix_num_boxes] = mix_boxes[:mix_num_boxes]
        mix_features_pad[:mix_num_boxes] = mix_features[:mix_num_boxes]

        # pad mix box ori
        mix_boxes_ori_pad = np.zeros((self._max_region_num, self._num_locs))
        mix_boxes_ori_pad[:mix_num_boxes] = mix_boxes_ori[:mix_num_boxes]

        # appending the target feature.
        features = torch.tensor(mix_features_pad).float()
        image_mask = torch.tensor(image_mask).long()
        spatials = torch.tensor(mix_boxes_pad).float()

        target = torch.zeros((self._max_region_num, 1)).float()
        target[:mix_num_boxes] = mix_target[:mix_num_boxes]

        # bbox2 = mix_boxes_ori[torch.argmax(target[:mix_num_boxes]),:4].tolist()

        spatials_ori = torch.tensor(mix_boxes_ori_pad).float()
        # print(spatials_ori)
        caption = entry["token"]
        input_mask = entry["input_mask"]
        segment_ids = entry["segment_ids"]
        sequence_labels = entry["sequence_label_ids"].float().unsqueeze(1)

        # print("spatials")
        # print(spatials[1,:4])
        # print("mix_boxes")
        # print(mix_boxes[1,:4])
        # print("spatials_ori")
        # print(spatials_ori[1,:4])
        # print("mix_boxes_ori")
        # print(mix_boxes_ori[1,:4])
        # exit()

        return features, spatials, spatials_ori, image_mask, caption, target, input_mask, segment_ids, image_id, sequence_labels

