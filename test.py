import os
import json
import torch
import cv2 as cv
import numpy as np
from tqdm import tqdm
from metrics.map import coco_map
from datasets.coco import COCODataSets
from torch.utils.data.dataloader import DataLoader
from utils.boxs import fcos_non_max_suppression
from utils.augmentations import ScalePadding
from nets.fcos import FCOS

rgb_mean = [0.485, 0.456, 0.406]
rgb_std = [0.229, 0.224, 0.225]

coco_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33,
            34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
            62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]


@torch.no_grad()
def valid_model():
    model = FCOS()
    weights = torch.load("weights/fcos/0_focs_last.pth")['ema']
    model.load_state_dict(weights)
    model.cuda().eval()
    predict_list = list()
    target_list = list()
    vdata = COCODataSets(img_root="/home/huffman/data/val2017",
                         annotation_path="/home/huffman/data/annotations/instances_val2017.json",
                         use_crowd=False,
                         augments=False,
                         debug=False,
                         remove_blank=False,
                         img_size=640
                         )
    vloader = DataLoader(dataset=vdata,
                         batch_size=4,
                         num_workers=4,
                         collate_fn=vdata.collate_fn,
                         shuffle=False
                         )
    pbar = tqdm(vloader)
    for img_tensor, targets_tensor, _ in pbar:
        _, _, h, w = img_tensor.shape
        targets_tensor[:, 3:] = targets_tensor[:, 3:] * torch.tensor(data=[w, h, w, h])
        img_tensor = img_tensor.cuda()
        targets_tensor = targets_tensor.cuda()
        predicts = model(img_tensor)

        for i in range(len(predicts)):
            predicts[i][:, [0, 2]] = predicts[i][:, [0, 2]].clamp(min=0, max=w)
            predicts[i][:, [1, 3]] = predicts[i][:, [1, 3]].clamp(min=0, max=h)
        predicts = fcos_non_max_suppression(predicts,
                                            conf_thresh=0.05,
                                            iou_thresh=0.6,
                                            max_det=300)
        for i, predict in enumerate(predicts):
            predict_list.append(predict)
            targets_sample = targets_tensor[targets_tensor[:, 0] == i][:, 2:]
            target_list.append(targets_sample)
    mp, mr, map50, map = coco_map(predict_list, target_list)
    print(map50, map)


def write_coco_json():
    from pycocotools.coco import COCO
    img_root = "/home/huffman/data/val2017"
    model = FCOS()
    weights = torch.load("weights/fcos/0_focs_last.pth")['ema']
    model.load_state_dict(weights)
    model.cuda().eval()

    basic_transform = ScalePadding(target_size=(640, 640), padding_val=(103, 116, 123))
    coco = COCO("/home/huffman/data/annotations/instances_val2017.json")
    coco_predict_list = list()
    for img_id in tqdm(coco.imgs.keys()):
        file_name = coco.imgs[img_id]['file_name']
        img_path = os.path.join(img_root, file_name)
        img = cv.imread(img_path)
        # ori_img = img.copy()
        img, ratio, (left, top) = basic_transform.make_border(img)
        h, w = img.shape[:2]
        img_out = img[:, :, [2, 1, 0]].astype(np.float32) / 255.0
        img_out = ((img_out - np.array(rgb_mean)) / np.array(rgb_std)).transpose(2, 0, 1).astype(np.float32)
        img_out = torch.from_numpy(np.ascontiguousarray(img_out)).unsqueeze(0).float().cuda()
        predicts = model(img_out)
        for i in range(len(predicts)):
            predicts[i][:, [0, 2]] = predicts[i][:, [0, 2]].clamp(min=0, max=w)
            predicts[i][:, [1, 3]] = predicts[i][:, [1, 3]].clamp(min=0, max=h)
        box = fcos_non_max_suppression(predicts,
                                       conf_thresh=0.05,
                                       iou_thresh=0.5,
                                       max_det=300)[0]
        if box is None:
            continue
        box[:, [0, 2]] = (box[:, [0, 2]] - left) / ratio[0]
        box[:, [1, 3]] = (box[:, [1, 3]] - top) / ratio[1]
        box = box.detach().cpu().numpy()
        # ret_img = draw_box(ori_img, box[:, [4, 5, 0, 1, 2, 3]], colors=coco_colors)
        # cv.imwrite(file_name, ret_img)
        coco_box = box[:, :4]
        coco_box[:, 2:] = coco_box[:, 2:] - coco_box[:, :2]
        for p, b in zip(box.tolist(), coco_box.tolist()):
            coco_predict_list.append({'image_id': img_id,
                                      'category_id': coco_ids[int(p[5])],
                                      'bbox': [round(x, 3) for x in b],
                                      'score': round(p[4], 5)})
    with open("predicts.json", 'w') as file:
        json.dump(coco_predict_list, file)


def coco_eval():
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
    cocoGt = COCO("/home/huffman/data/annotations/instances_val2017.json")  # initialize COCO ground truth api

    cocoDt = cocoGt.loadRes("predicts.json")  # initialize COCO pred api
    imgIds = [img_id for img_id in cocoGt.imgs.keys()]
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.params.imgIds = imgIds  # image IDs to evaluate
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


def coco_val_eval():
    write_coco_json()
    coco_eval()


if __name__ == '__main__':
    # valid_model()
    coco_val_eval()
    # coco_eavl()
