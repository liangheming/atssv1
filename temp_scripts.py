import torch

from datasets.coco import COCODataSets
from torch.utils.data.dataloader import DataLoader
from utils.model import rand_seed

rand_seed(1024)


def fcos_temp():
    from nets.fcos import FCOS
    from losses.atss_fcos_loss import FCOSLoss
    dataset = COCODataSets(img_root="/home/huffman/data/val2017",
                           annotation_path="/home/huffman/data/annotations/instances_val2017.json",
                           use_crowd=True,
                           augments=True,
                           remove_blank=True,
                           img_size=640
                           )
    dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=dataset.collate_fn)
    net = FCOS(
        backbone="resnet18"
    )
    creterion = FCOSLoss()
    for img_input, targets, _ in dataloader:
        _, _, h, w = img_input.shape
        targets[:, 3:] = targets[:, 3:] * torch.tensor(data=[w, h, w, h])
        cls_outputs, reg_outputs, center_outputs, grids = net(img_input)
        creterion(cls_outputs, reg_outputs, center_outputs, grids,
                  targets)
        # total_loss, detail_loss = creterion(cls_outputs, reg_outputs, center_outputs, grids,
        #                                     targets)
        # cls_loss, reg_loss, center_loss = detail_loss
        # print(total_loss)
        # print(cls_loss, reg_loss, center_loss)
        break


def retina_temp():
    from nets.retinanet import RetinaNet
    from losses.atss_retina_loss import ATSSRetinaLoss
    dataset = COCODataSets(img_root="/home/huffman/data/val2017",
                           annotation_path="/home/huffman/data/annotations/instances_val2017.json",
                           use_crowd=True,
                           augments=True,
                           remove_blank=True,
                           img_size=640
                           )
    dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=dataset.collate_fn)
    net = RetinaNet(
        backbone="resnet18"
    )
    creterion = ATSSRetinaLoss()
    for img_input, targets, _ in dataloader:
        _, _, h, w = img_input.shape
        targets[:, 3:] = targets[:, 3:] * torch.tensor(data=[w, h, w, h])
        cls_outputs, reg_outputs, anchors = net(img_input)
        creterion(cls_outputs, reg_outputs, anchors, targets)
        # total_loss, detail_loss, pos_num = creterion(cls_outputs, reg_outputs, anchors, targets)
        # cls_loss, reg_loss = detail_loss
        # print(total_loss)
        # print(cls_loss, reg_loss, pos_num)
        break


if __name__ == '__main__':
    fcos_temp()
