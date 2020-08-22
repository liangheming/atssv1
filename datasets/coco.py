import os
import torch
import cv2 as cv
import numpy as np
from torch.utils.data.dataset import Dataset
from pycocotools.coco import COCO
from utils.boxs import draw_box, xyxy2xywh
from utils.augmentations import Compose, OneOf, \
    ScalePadding, RandNoise, Mosaic, MixUp, RandPerspective, HSV, Identity, LRFlip, RandCutOut

coco_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33,
            34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
            62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

coco_names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
              "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
              "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
              "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
              "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
              "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
              "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
              "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]
colors = [(67, 68, 113), (130, 45, 169), (2, 202, 130), (127, 111, 90), (92, 136, 113),
          (33, 250, 7), (238, 92, 104), (0, 151, 197), (134, 9, 145), (253, 181, 88),
          (246, 11, 137), (55, 72, 220), (136, 8, 253), (56, 73, 180), (85, 241, 53),
          (153, 207, 15), (187, 183, 180), (149, 32, 71), (92, 113, 184), (131, 7, 201),
          (56, 20, 219), (243, 201, 77), (13, 74, 96), (79, 14, 44), (195, 150, 66),
          (2, 249, 42), (195, 135, 43), (105, 70, 66), (120, 107, 116), (122, 241, 22),
          (17, 19, 179), (162, 185, 124), (31, 65, 117), (88, 200, 80), (232, 49, 154),
          (72, 1, 46), (59, 144, 187), (200, 193, 118), (123, 165, 219), (194, 84, 34),
          (91, 184, 108), (252, 64, 153), (251, 121, 27), (105, 93, 210), (89, 85, 81),
          (58, 12, 154), (81, 3, 50), (200, 40, 236), (155, 147, 180), (73, 29, 176),
          (193, 19, 175), (157, 225, 121), (128, 195, 235), (146, 251, 108), (13, 146, 186),
          (231, 118, 145), (253, 15, 105), (187, 149, 62), (121, 247, 158), (34, 8, 142),
          (83, 61, 48), (119, 218, 69), (197, 94, 130), (222, 176, 142), (21, 20, 77),
          (6, 42, 17), (136, 33, 156), (39, 252, 211), (52, 50, 40), (183, 115, 34),
          (107, 80, 164), (195, 215, 74), (7, 154, 135), (136, 35, 24), (131, 241, 125),
          (208, 99, 208), (5, 4, 129), (137, 156, 175), (29, 141, 67), (44, 20, 99)]
default_aug_cfg = {
    'hsv_h': 0.014,
    'hsv_s': 0.68,
    'hsv_v': 0.36,
    'degree': 5,
    'translate': 0.1,
    'scale': (0.6, 1.5),
    'shear': 0.0,
    'beta': 1.5,
    'pad_val': (103, 116, 123),
    # 'pad_val': (114, 114, 114)
}

rgb_mean = [0.485, 0.456, 0.406]
rgb_std = [0.229, 0.224, 0.225]


# noinspection PyTypeChecker
class COCODataSets(Dataset):
    def __init__(self, img_root, annotation_path,
                 img_size=640,
                 augments=True,
                 use_crowd=True,
                 debug=False,
                 remove_blank=True,
                 aug_cfg=None,
                 ):
        """
        :param img_root: 图片根目录
        :param annotation_path: 标注（json）文件的路径
        :param img_size: 长边的size
        :param augments: 是否进行数据增强
        :param use_crowd: 是否使用crowed的标注
        :param debug: debug模式(少量数据)
        :param remove_blank: 是否过滤掉没有标注的数据
        :param aug_cfg: 数据增强中配置
        """
        super(COCODataSets, self).__init__()
        self.coco = COCO(annotation_path)
        self.img_size = img_size
        self.img_root = img_root
        self.use_crowd = use_crowd
        self.remove_blank = remove_blank
        self.data_len = len(self.coco.imgs.keys())
        self.img_paths = [None] * self.data_len
        self.shapes = [None] * self.data_len
        # [label_weights, label_index, x1, y1, x2, y2]
        self.labels = [np.zeros((0, 6), dtype=np.float32)] * self.data_len
        self.augments = augments
        if aug_cfg is None:
            aug_cfg = default_aug_cfg
        self.aug_cfg = aug_cfg
        self.debug = debug
        self.empty_images_len = 0
        valid_len = self.__load_data()
        if valid_len != self.data_len:
            print("valid data len: ", valid_len)
            self.data_len = valid_len
            self.img_paths = self.img_paths[:valid_len]
            self.shapes = self.shapes[:valid_len]
            self.labels = self.labels[:valid_len]
        if self.debug:
            assert debug <= valid_len, "not enough data to debug"
            print("debug")
            self.img_paths = self.img_paths[:debug]
            self.shapes = self.shapes[:debug]
            self.labels = self.labels[:debug]
        self.transform = None
        self.set_transform()

    def __load_data(self):
        index = 0
        for img_id in self.coco.imgs.keys():
            file_name = self.coco.imgs[img_id]['file_name']
            width, height = self.coco.imgs[img_id]['width'], self.coco.imgs[img_id]['height']
            file_path = os.path.join(self.img_root, file_name)
            if not os.path.exists(file_path):
                print("img {:s} is not exist".format(file_path))
                continue

            assert width > 1 and height > 1, "invalid width or heights"

            anns = self.coco.imgToAnns[img_id]
            label_list = list()
            for ann in anns:
                category_id, box, iscrowd = ann['category_id'], ann['bbox'], ann['iscrowd']
                label_id = coco_ids.index(category_id)
                assert label_id >= 0, 'error label_id'
                if not self.use_crowd and iscrowd == 1:
                    continue
                x1, y1 = box[:2]
                x2, y2 = x1 + box[2], y1 + box[3]
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                if x2 - x1 < 1 or y2 - y1 < 1:
                    print("not a valid box ", box)
                    continue
                if x1 < 0 or x2 > width or y1 < 0 or y2 > height:
                    print("warning box ", box)
                label_list.append((1., label_id, x1, y1, x2, y2))
            if self.remove_blank:
                if len(label_list) < 1:
                    self.empty_images_len += 1
                    continue
            if label_list:
                self.labels[index] = np.array(label_list, dtype=np.float32)

            self.img_paths[index] = file_path
            self.shapes[index] = (width, height)
            index += 1
        return index

    def __getitem__(self, item):
        img_path, label = self.img_paths[item], self.labels[item]
        img = cv.imread(img_path)
        img, label = self.transform(img, label)
        if self.debug:
            import uuid
            ret_img = draw_box(img, label, colors, coco_names)
            cv.imwrite("{:d}_{:s}.jpg".format(item, str(uuid.uuid4()).replace('-', "")), ret_img)
        label_num = len(label)
        if label_num:
            # [weight,label,x1,y1,x2,y2]
            label[:, [3, 5]] /= img.shape[0]  # height
            label[:, [2, 4]] /= img.shape[1]
        img_out = img[:, :, [2, 1, 0]].astype(np.float32) / 255.0
        img_out = ((img_out - np.array(rgb_mean)) / np.array(rgb_std)).transpose(2, 0, 1).astype(np.float32)
        img_out = np.ascontiguousarray(img_out)
        assert not np.any(np.isnan(img_out)), "invalid input"
        labels_out = torch.zeros((label_num, 7))
        if label_num:
            labels_out[:, 1:] = torch.from_numpy(label)
        return torch.from_numpy(img_out).float(), labels_out, self.img_paths[item]

    def set_transform(self):
        if self.augments:
            self.transform = Compose(transforms=[
                OneOf(transforms=[
                    (0.6, Compose(transforms=[
                        OneOf(transforms=[Identity(),
                                          HSV(p=1,
                                              hgain=self.aug_cfg['hsv_h'],
                                              sgain=self.aug_cfg['hsv_s'],
                                              vgain=self.aug_cfg['hsv_v']),
                                          RandNoise()
                                          ]),
                        RandCutOut(),
                        ScalePadding(target_size=self.img_size, padding_val=self.aug_cfg['pad_val']),
                        RandPerspective(target_size=(self.img_size, self.img_size),
                                        scale=self.aug_cfg['scale'],
                                        degree=self.aug_cfg['degree'],
                                        translate=self.aug_cfg['translate'],
                                        shear=self.aug_cfg['shear'],
                                        pad_val=self.aug_cfg['pad_val'])])),
                    (0.4, Mosaic(self.img_paths,
                                 self.labels,
                                 color_gitter=OneOf(transforms=[Identity(),
                                                                HSV(p=1,
                                                                    hgain=self.aug_cfg['hsv_h'],
                                                                    sgain=self.aug_cfg['hsv_s'],
                                                                    vgain=self.aug_cfg['hsv_v']),
                                                                RandNoise()]),
                                 target_size=self.img_size,
                                 pad_val=self.aug_cfg['pad_val']))
                ]),

                LRFlip()])
        else:
            self.transform = ScalePadding(target_size=(self.img_size, self.img_size),
                                          padding_val=self.aug_cfg['pad_val'])

    def __len__(self):
        return len(self.img_paths)

    @staticmethod
    def collate_fn(batch):
        """
        :param batch:
        :return: images shape[bs,3,h,w] targets[bs,7] (bs_idx,weights,label_idx,x1,y1,x2,y2)
        """
        img, label, path = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path


if __name__ == '__main__':
    from torch.utils.data.dataloader import DataLoader

    dataset = COCODataSets(img_root="/home/huffman/data/val2017",
                           annotation_path="/home/huffman/data/annotations/instances_val2017.json",
                           use_crowd=True,
                           augments=True,
                           debug=60
                           )
    dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=True, num_workers=4, collate_fn=dataset.collate_fn)
    for img_tensor, target_tensor, _ in dataloader:
        for weights in target_tensor[:, 1].unique():
            nonzero_index = torch.nonzero((target_tensor[:, 1] == weights), as_tuple=True)
            print(target_tensor[nonzero_index].shape)
        print("=" * 20)
    for img_tensor, target_tensor, _ in dataloader:
        for weights in target_tensor[:, 1].unique():
            nonzero_index = torch.nonzero((target_tensor[:, 1] == weights), as_tuple=True)
            print(target_tensor[nonzero_index].shape)
        print("=" * 20)
