import torch
import cv2 as cv
import numpy as np
from torchvision.ops.boxes import nms


def draw_box(img, labels, colors, names):
    if labels.shape[0] < 1:
        return img
    ret_img = img.copy()
    for weights, label, x1, y1, x2, y2 in labels:
        cv.rectangle(ret_img, (int(x1), int(y1)), (int(x2), int(y2)), color=colors[int(label)], thickness=2)
        cv.putText(ret_img, "{:s}".format(names[int(label)]), (int(x1), int(y1)), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                   colors[int(label)], 2)
        # cv.putText(ret_img, "{:.2f}".format(float(weights)), (int(x1), int(y1 + 5)), cv.FONT_HERSHEY_SIMPLEX, 0.5,
        #            colors[int(label)],
        #            1)
    return ret_img


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def bbox_giou(box1, box2, x1y1x2y2=True):
    """
    :param box1: [num,4]
    :param box2: [num,4]
    :param x1y1x2y2:
    :return:
    """
    box1 = box1.t()
    box2 = box2.t()
    if x1y1x2y2:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = (w1 * h1 + 1e-16) + w2 * h2 - inter
    iou = inter / union
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
    c_area = cw * ch + 1e-16
    return iou - (c_area - union) / c_area


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.t())
    area2 = box_area(box2.t())

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def retina_non_max_suppression(prediction: torch.Tensor,
                               conf_thresh=0.05,
                               iou_thresh=0.5,
                               max_det=300,
                               max_box=2048,
                               max_layer_num=1000
                               ):
    """
    :param max_layer_num:
    :param prediction:
    :param conf_thresh:
    :param iou_thresh:
    :param max_det:
    :param max_box:
    :return:
    """
    for i in range(len(prediction)):
        if prediction[i].dtype == torch.float16:
            prediction[i] = prediction[i].float()
    bs = prediction[0].shape[0]
    out = [None] * bs
    for bi in range(bs):
        batch_predicts_list = [torch.zeros(size=(0, 6), device=prediction[0].device).float()] * len(prediction)
        for lj in range(len(prediction)):
            one_layer_bath_predict = prediction[lj][bi]
            reg_predicts = one_layer_bath_predict[:, :4]
            cls_predicts = one_layer_bath_predict[:, 4:].sigmoid()

            max_val, max_idx = cls_predicts.max(dim=1)
            valid_bool_idx = max_val > conf_thresh
            if valid_bool_idx.sum() == 0:
                continue
            valid_val = max_val[valid_bool_idx]
            sorted_idx = valid_val.argsort(descending=True)
            valid_val = valid_val[sorted_idx]
            valid_box = reg_predicts[valid_bool_idx, :][sorted_idx]
            valid_cls = max_idx[valid_bool_idx][sorted_idx]
            if 0 < max_layer_num < valid_box.shape[0]:
                valid_val = valid_val[:max_layer_num]
                valid_box = valid_box[:max_layer_num, :]
                valid_cls = valid_cls[:max_layer_num]
            batch_predicts = torch.cat([valid_box, valid_val[:, None], valid_cls[:, None]], dim=-1)
            batch_predicts_list[lj] = batch_predicts
        x = torch.cat(batch_predicts_list, dim=0)
        if x.shape[0] == 0:
            continue
        c = x[:, 5:6] * max_box
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = nms(boxes, scores, iou_thresh)
        if i.shape[0] > max_det:
            i = i[:max_det]
        out[bi] = x[i]
    return out


def fcos_non_max_suppression(prediction: torch.Tensor,
                             conf_thresh=0.05,
                             iou_thresh=0.5,
                             max_det=300,
                             max_box=2048,
                             max_layer_num=1000
                             ):
    """
    :param max_layer_num:
    :param prediction:
    :param conf_thresh:
    :param iou_thresh:
    :param max_det:
    :param max_box:
    :return: (x1,y1,x2,y2,score,cls_id)
    """
    for i in range(len(prediction)):
        if prediction[i].dtype == torch.float16:
            prediction[i] = prediction[i].float()
    bs = prediction[0].shape[0]
    out = [None] * bs
    for bi in range(bs):
        batch_predicts_list = [torch.zeros(size=(0, 6), device=prediction[0].device).float()] * len(prediction)
        for lj in range(len(prediction)):
            one_layer_bath_predict = prediction[lj][bi]
            reg_predicts = one_layer_bath_predict[:, :4]
            center_predicts = one_layer_bath_predict[:, 4].sigmoid()
            cls_predicts = one_layer_bath_predict[:, 5:].sigmoid()

            max_val, max_idx = cls_predicts.max(dim=1)
            valid_bool_idx = max_val > conf_thresh
            if valid_bool_idx.sum() == 0:
                continue
            valid_val = max_val[valid_bool_idx] * center_predicts[valid_bool_idx]
            sorted_idx = valid_val.argsort(descending=True)
            valid_val = valid_val[sorted_idx]
            valid_box = reg_predicts[valid_bool_idx, :][sorted_idx]
            valid_cls = max_idx[valid_bool_idx][sorted_idx]
            if 0 < max_layer_num < valid_box.shape[0]:
                valid_val = valid_val[:max_layer_num]
                valid_box = valid_box[:max_layer_num, :]
                valid_cls = valid_cls[:max_layer_num]
            batch_predicts = torch.cat([valid_box, valid_val[:, None], valid_cls[:, None]], dim=-1)
            batch_predicts_list[lj] = batch_predicts
        x = torch.cat(batch_predicts_list, dim=0)
        if x.shape[0] == 0:
            continue
        c = x[:, 5:6] * max_box
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = nms(boxes, scores, iou_thresh)
        if i.shape[0] > max_det:
            i = i[:max_det]
        out[bi] = x[i]
    return out
