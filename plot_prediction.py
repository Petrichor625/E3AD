import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
# import argparse
import os

def main():
    output_result_path = "./resultes/prediction.json"
    train_data_path ="../Talk2Car/data/commands/train_commands.json"
    val_data_path = "../Talk2Car/data/commands/val_commands.json"
    test_data_path = "../Talk2Car/data/commands/test_commands.json"
    img_data_path = "../Talk2Car/data/image"
    output_dir ="./resultes/output_bbox_show"

    with open(output_result_path) as f1:
        Preded_bboxes = json.load(f1) 
    with open(train_data_path) as f2:
        train_data = json.load(f2)

    with open(val_data_path) as f3:
        val_data = json.load(f3)

    with open(test_data_path) as f4:
        test_data = json.load(f4)   

    flag = True
    print(val_data['commands'][0].keys())
    
    for i,data in enumerate(test_data['commands'][:50]):
        img_path = os.path.join(img_data_path,data['t2c_img'])
        with open(img_path, 'rb') as f_img:
            img = Image.open(f_img).convert('RGB')
        gt_bbox = data['2d_box']
        x1,y1,w1,h1 = gt_bbox[0],gt_bbox[1],gt_bbox[2],gt_bbox[3]# gt_bbox[0],gt_bbox[1],gt_bbox[0]+gt_bbox[2],gt_bbox[1]+gt_bbox[3]
        fig, ax = plt.subplots(1)
        ax.imshow(img)
        rect = patches.Rectangle((x1, y1), w1, h1, fill=False, edgecolor='r')
        ax.add_patch(rect)

        bbox = Preded_bboxes[data['command_token']]

        x2,y2,w2,h2 = bbox[0],bbox[1],bbox[2],bbox[3]

        rect = patches.Rectangle((x2, y2), w2, h2, fill=False, edgecolor='g')
        ax.add_patch(rect)
        
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'val_{}_bboxes.png'.format(i)), bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    main()
