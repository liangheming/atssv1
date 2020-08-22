import math
import random
import cv2 as cv
import numpy as np


class DetectAugment(object):
    def __init__(self, p=0.5):
        super(DetectAugment, self).__init__()
        self.p = p

    def img_aug(self, img: np.ndarray) -> np.ndarray:
        pass

    def aug(self, img: np.ndarray, labels: np.ndarray) -> tuple:
        pass

    def __call__(self, img: np.ndarray, labels: np.ndarray):
        """
        :param img: np.ndarray (BGR) 模型 (直接使用opencv进行读取)
        :param labels: [box_num,6] (weights,label_idx,x1,y1,x2,y2) 其中坐标为原图上坐标(不需要normalize到[0，1])
        :return:
        """
        aug_p = np.random.uniform()
        labels = labels.copy()
        if aug_p <= self.p:
            img, labels = self.aug(img, labels)
        return img, labels


class Identity(DetectAugment):
    """
    恒等增强
    """

    def __init__(self, **kwargs):
        super(Identity, self).__init__(**kwargs)
        self.p = 1

    def img_aug(self, img: np.ndarray) -> np.ndarray:
        return img

    def aug(self, img: np.ndarray, labels: np.ndarray) -> tuple:
        return img, labels


class RandNoise(DetectAugment):
    """
    随机加入噪声
    """

    def __init__(self, **kwargs):
        super(RandNoise, self).__init__(**kwargs)

    def img_aug(self, img: np.ndarray) -> np.ndarray:
        mu = 0
        sigma = np.random.uniform(1, 15)
        img = np.array(img, dtype=np.float32)
        img += np.random.normal(mu, sigma, img.shape)
        img = img.clip(0., 255.).astype(dtype=np.uint8)
        return img

    def aug(self, img: np.ndarray, labels: np.ndarray) -> tuple:
        img = self.img_aug(img)
        return img, labels


class RandBlur(DetectAugment):
    """
    随机进行模糊
    """

    def __init__(self, **kwargs):
        super(RandBlur, self).__init__(**kwargs)

    @staticmethod
    def gaussian_blur(img):
        kernel_size = np.random.choice([3, 5])
        img = cv.GaussianBlur(img, (kernel_size, kernel_size), 0)
        return img

    @staticmethod
    def median_blur(img):
        kernel_size = np.random.choice([3, 5])
        img = cv.medianBlur(img, kernel_size, 0)
        return img

    @staticmethod
    def blur(img):
        kernel_size = np.random.choice([3, 5])
        img = cv.blur(img, (kernel_size, kernel_size))
        return img

    def img_aug(self, img: np.ndarray) -> np.ndarray:
        aug_blur = np.random.choice([self.gaussian_blur, self.median_blur, self.blur])
        img = aug_blur(img)
        return img

    def aug(self, img: np.ndarray, labels: np.ndarray) -> tuple:
        img = self.img_aug(img)
        return img, labels


class HSV(DetectAugment):
    """
    color jitter
    """

    def __init__(self, hgain=0.5, sgain=0.5, vgain=0.5, **kwargs):
        super(HSV, self).__init__(**kwargs)
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain

    def img_aug(self, img: np.ndarray) -> np.ndarray:
        r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1
        hue, sat, val = cv.split(cv.cvtColor(img, cv.COLOR_BGR2HSV))
        dtype = img.dtype
        x = np.arange(0, 256, dtype=np.int16)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        img_hsv = cv.merge((cv.LUT(hue, lut_hue), cv.LUT(sat, lut_sat), cv.LUT(val, lut_val))).astype(dtype)
        ret_img = cv.cvtColor(img_hsv, cv.COLOR_HSV2BGR)
        return ret_img

    def aug(self, img: np.ndarray, labels: np.ndarray) -> tuple:
        img = self.img_aug(img)
        return img, labels


class ScaleNoPadding(DetectAugment):
    """
    等比缩放长边至指定尺寸，不进行padding
    """

    def __init__(self, target_size=640, **kwargs):
        super(ScaleNoPadding, self).__init__(**kwargs)
        self.p = 1
        self.target_size = target_size

    def scale_img(self, img):
        h, w = img.shape[:2]
        r = min(self.target_size / h, self.target_size / w)
        if r != 1:
            img = cv.resize(img, (int(round(w * r)), int(round(h * r))), interpolation=cv.INTER_LINEAR)
        return img, r

    def img_aug(self, img: np.ndarray) -> np.ndarray:
        img, _ = self.scale_img(img)
        return img

    def aug(self, img: np.ndarray, labels: np.ndarray) -> tuple:
        img, ratio = self.scale_img(img)
        if labels.shape[0] > 0:
            labels[:, 2:] = ratio * labels[:, 2:]
        return img, labels


class ScalePadding(DetectAugment):
    """
    等比缩放长边至指定尺寸，padding短边部分
    """

    def __init__(self, target_size=(640, 640),
                 padding_val=(114, 114, 114),
                 minimum_rectangle=False,
                 scale_up=True, **kwargs):
        super(ScalePadding, self).__init__(**kwargs)
        self.p = 1
        self.new_shape = target_size
        self.padding_val = padding_val
        self.minimum_rectangle = minimum_rectangle
        self.scale_up = scale_up

    def make_border(self, img: np.ndarray):
        # h,w
        shape = img.shape[:2]
        if isinstance(self.new_shape, int):
            self.new_shape = (self.new_shape, self.new_shape)
        r = min(self.new_shape[1] / shape[0], self.new_shape[0] / shape[1])
        if not self.scale_up:
            r = min(r, 1.0)
        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = self.new_shape[0] - new_unpad[0], self.new_shape[1] - new_unpad[1]
        if self.minimum_rectangle:
            dw, dh = np.mod(dw, 64), np.mod(dh, 64)

        dw /= 2
        dh /= 2
        if shape[::-1] != new_unpad:
            img = cv.resize(img, new_unpad, interpolation=cv.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT, value=self.padding_val)
        return img, ratio, (left, top)

    def img_aug(self, img: np.ndarray) -> np.ndarray:
        img, _, _ = self.make_border(img)
        return img

    def aug(self, img: np.ndarray, labels: np.ndarray) -> tuple:
        img, ratio, (left, top) = self.make_border(img)
        if labels.shape[0] > 0:
            labels[:, [2, 4]] = ratio[0] * labels[:, [2, 4]] + left
            labels[:, [3, 5]] = ratio[1] * labels[:, [3, 5]] + top
        return img, labels


class LRFlip(DetectAugment):
    """
    左右翻转
    """

    def __init__(self, **kwargs):
        super(LRFlip, self).__init__(**kwargs)

    def img_aug(self, img: np.ndarray) -> np.ndarray:
        img = np.fliplr(img)
        return img

    def aug(self, img: np.ndarray, labels: np.ndarray) -> tuple:
        h, w = img.shape[:2]
        img = self.img_aug(img)
        if labels.shape[0] > 0:
            labels[:, [4, 2]] = w - labels[:, [2, 4]]
        return img, labels


class UDFlip(DetectAugment):
    """
    上下翻转
    """

    def __init__(self, **kwargs):
        super(UDFlip, self).__init__(**kwargs)

    def img_aug(self, img: np.ndarray) -> np.ndarray:
        img = np.flipud(img)
        return img

    def aug(self, img: np.ndarray, labels: np.ndarray) -> tuple:
        h, w = img.shape[:2]
        img = self.img_aug(img)
        if labels.shape[0] > 0:
            labels[:, [5, 3]] = h - labels[:, [3, 5]]
        return img, labels


class RandAffine(DetectAugment):
    """
    随即进行仿射变换
    """

    def __init__(self, target_size=(640, 640),
                 degree=(-10, 10),
                 translate=0.1,
                 scale=0.1,
                 shear=5,
                 pad_val=(114, 114, 114),
                 border_translate=(0, 0),
                 **kwargs):
        super(RandAffine, self).__init__(**kwargs)
        self.target_size = target_size
        if isinstance(degree, float) or isinstance(degree, int):
            assert degree >= 0, 'degree should be positive'
            degree = (-degree, degree)
        self.degree = degree
        self.translate = translate
        if isinstance(scale, float) or isinstance(scale, int):
            assert 0 <= scale < 1
            scale = (1 - scale, 1 + scale)
        self.scale = scale
        self.shear = shear
        self.pad_val = pad_val
        self.border_translate = border_translate

    def get_transform_matrix(self, img):
        width, height = self.target_size
        matrix_r = np.eye(3)
        angle = np.random.uniform(self.degree[0], self.degree[1])
        scale = np.random.uniform(self.scale[0], self.scale[1])
        matrix_r[:2] = cv.getRotationMatrix2D(angle=angle, center=(img.shape[1] / 2, img.shape[0] / 2), scale=scale)

        matrix_t = np.eye(3)
        matrix_t[0, 2] = np.random.uniform(-self.translate, self.translate) * img.shape[1] + self.border_translate[0]
        matrix_t[1, 2] = np.random.uniform(-self.translate, self.translate) * img.shape[0] + self.border_translate[1]

        matrix_s = np.eye(3)
        matrix_s[0, 1] = math.tan(np.random.uniform(-self.shear, self.shear) * math.pi / 180)
        matrix_s[1, 0] = math.tan(np.random.uniform(-self.shear, self.shear) * math.pi / 180)
        return matrix_s @ matrix_t @ matrix_r, width, height, scale

    def img_aug(self, img: np.ndarray) -> np.ndarray:
        transform_matrix, width, height, scale = self.get_transform_matrix(img)
        # if (transform_matrix != np.eye(3)).any():  # image changed
        img = cv.warpAffine(img, transform_matrix[:2],
                            dsize=(width, height),
                            flags=cv.INTER_LINEAR,
                            borderValue=self.pad_val)
        return img

    def aug(self, img: np.ndarray, labels: np.ndarray) -> tuple:
        transform_matrix, width, height, scale = self.get_transform_matrix(img)
        # image changed
        img = cv.warpAffine(img, transform_matrix[:2],
                            dsize=(width, height),
                            flags=cv.INTER_LINEAR,
                            borderValue=self.pad_val)
        n = len(labels)
        if n:
            xy = np.ones((n * 4, 3))
            xy[:, :2] = labels[:, [2, 3, 4, 5, 2, 5, 4, 3]].reshape(n * 4, 2)
            xy = (xy @ transform_matrix.T)[:, :2].reshape(n, 8)
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
            xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
            xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
            w = xy[:, 2] - xy[:, 0]
            h = xy[:, 3] - xy[:, 1]
            area = w * h
            area0 = (labels[:, 4] - labels[:, 2]) * (labels[:, 5] - labels[:, 3])
            ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
            i = (w > 2) & (h > 2) & (area / (area0 * scale + 1e-16) > 0.2) & (ar < 20)
            labels = labels[i]
            labels[:, 2:6] = xy[i]
        return img, labels


class RandPerspective(DetectAugment):
    """
    随即进行透视变换
    """

    def __init__(self, target_size=(640, 640),
                 degree=(-10, 10),
                 translate=0.1,
                 scale=0.1,
                 shear=5,
                 pad_val=(114, 114, 114),
                 perspective=0.0,
                 **kwargs):
        super(RandPerspective, self).__init__(**kwargs)
        self.target_size = target_size
        if isinstance(degree, float) or isinstance(degree, int):
            assert degree >= 0, 'degree should be positive'
            degree = (-degree, degree)
        self.degree = degree
        self.translate = translate
        if isinstance(scale, float) or isinstance(scale, int):
            assert 0 <= scale < 1
            scale = (1 - scale, 1 + scale)
        self.scale = scale
        self.shear = shear
        self.pad_val = pad_val
        self.perspective = perspective

    def get_transform_matrix(self, img):
        width, height = self.target_size

        matrix_c = np.eye(3)
        matrix_c[0, 2] = -img.shape[1] / 2
        matrix_c[1, 2] = -img.shape[0] / 2

        matrix_p = np.eye(3)
        matrix_p[2, 0] = random.uniform(-self.perspective, self.perspective)
        matrix_p[2, 1] = random.uniform(-self.perspective, self.perspective)

        matrix_r = np.eye(3)
        angle = np.random.uniform(self.degree[0], self.degree[1])
        scale = np.random.uniform(self.scale[0], self.scale[1])
        matrix_r[:2] = cv.getRotationMatrix2D(angle=angle, center=(0, 0), scale=scale)

        matrix_t = np.eye(3)
        matrix_t[0, 2] = np.random.uniform(0.5 - self.translate, 0.5 + self.translate) * width
        matrix_t[1, 2] = np.random.uniform(0.5 - self.translate, 0.5 + self.translate) * height

        matrix_s = np.eye(3)
        matrix_s[0, 1] = math.tan(np.random.uniform(-self.shear, self.shear) * math.pi / 180)
        matrix_s[1, 0] = math.tan(np.random.uniform(-self.shear, self.shear) * math.pi / 180)
        return matrix_t @ matrix_s @ matrix_r @ matrix_p @ matrix_c, width, height, scale

    def img_aug(self, img: np.ndarray) -> np.ndarray:
        transform_matrix, width, height, scale = self.get_transform_matrix(img)
        # if (transform_matrix != np.eye(3)).any():  # image changed
        if self.perspective:
            img = cv.warpPerspective(img, transform_matrix, dsize=(width, height), borderValue=self.pad_val)
        else:  # affine
            img = cv.warpAffine(img, transform_matrix[:2], dsize=(width, height), borderValue=self.pad_val)
        return img

    def aug(self, img: np.ndarray, labels: np.ndarray) -> tuple:
        transform_matrix, width, height, scale = self.get_transform_matrix(img)
        if self.perspective:
            img = cv.warpPerspective(img, transform_matrix, dsize=(width, height), borderValue=self.pad_val)
        else:  # affine
            img = cv.warpAffine(img, transform_matrix[:2], dsize=(width, height), borderValue=self.pad_val)
        n = len(labels)
        if n:
            xy = np.ones((n * 4, 3))
            xy[:, :2] = labels[:, [2, 3, 4, 5, 2, 5, 4, 3]].reshape(n * 4, 2)
            xy = (xy @ transform_matrix.T)
            if self.perspective:
                xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
            else:  # affine
                xy = xy[:, :2].reshape(n, 8)
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
            xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
            xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
            w = xy[:, 2] - xy[:, 0]
            h = xy[:, 3] - xy[:, 1]
            area = w * h
            area0 = (labels[:, 4] - labels[:, 2]) * (labels[:, 5] - labels[:, 3])
            ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
            i = (w > 2) & (h > 2) & (area / (area0 * scale + 1e-16) > 0.2) & (ar < 20)
            labels = labels[i]
            labels[:, 2:6] = xy[i]
        return img, labels


class RandCutOut(DetectAugment):
    def __init__(self, max_cut_time=8, mask_scale=None, **kwargs):
        super(RandCutOut, self).__init__(**kwargs)
        assert max_cut_time >= 1
        self.max_cut_time = max_cut_time
        if mask_scale is None:
            mask_scale = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16
        self.mask_scale = mask_scale

    @staticmethod
    def bbox_ioa(box1, box2):
        # Returns the intersection over box2 area given box1, box2. box1 is 4, box2 is nx4. boxes are x1y1x2y2
        box2 = box2.transpose()

        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

        # Intersection area
        inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                     (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

        # box2 area
        box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + 1e-16

        # Intersection over box2 area
        return inter_area / box2_area

    def img_aug(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        cut_times = random.randint(1, self.max_cut_time + 1)
        ret_img = img.copy()
        for _ in range(cut_times):
            s = np.random.choice(self.mask_scale)
            mask_h = random.randint(1, int(h * s))
            mask_w = random.randint(1, int(w * s))

            x_min = max(0, random.randint(0, w) - mask_w // 2)
            y_min = max(0, random.randint(0, h) - mask_h // 2)
            x_max = min(w, x_min + mask_w)
            y_max = min(h, y_min + mask_h)
            ret_img[y_min:y_max, x_min, x_max] = [random.randint(64, 191) for _ in range(3)]
        return ret_img

    def aug(self, img: np.ndarray, labels: np.ndarray) -> tuple:
        """
        :param img:
        :param labels:[weights,label_id,x1,y1,x2,y2]
        :return:
        """
        h, w = img.shape[:2]
        cut_times = random.randint(1, self.max_cut_time + 1)
        ret_img = img.copy()
        for _ in range(cut_times):
            s = np.random.choice(self.mask_scale)
            mask_w = random.randint(1, int(w * s))
            mask_h = random.randint(1, int(h * s))

            x_min = max(0, random.randint(0, w) - mask_w // 2)
            y_min = max(0, random.randint(0, h) - mask_h // 2)
            x_max = min(w, x_min + mask_w)
            y_max = min(h, y_min + mask_h)
            ret_img[y_min:y_max, x_min:x_max] = [random.randint(64, 191) for _ in range(3)]

            if len(labels) and s > 0.02:
                box = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
                ioa = self.bbox_ioa(box, labels[:, 2:6])
                labels = labels[ioa < 0.60]
        return ret_img, labels


class Mosaic(DetectAugment):
    """
    使用马赛克数据增强
    """

    def __init__(self, candidate_imgs, candidate_labels,
                 color_gitter=None,
                 target_size=640,
                 pad_val=(114, 114, 114), **kwargs):
        """
        :param candidate_imgs: 候选图片路径列表
        :param candidate_labels: 图片对应标注列表
        :param color_gitter: 单张图片进行马赛克增强前使用的color gitter
        :param target_size: 长边缩放尺寸
        :param pad_val: padding部分填充的数值
        :param kwargs:
        """
        super(Mosaic, self).__init__(**kwargs)
        self.p = 1
        self.candidate_imgs = candidate_imgs
        self.candidate_labels = candidate_labels
        self.target_size = target_size
        self.pad_val = pad_val
        self.scale_no_pad = ScaleNoPadding(target_size)
        self.mosaic_border = (-self.target_size // 2, -self.target_size // 2)
        if color_gitter is None:
            color_gitter = Identity()
        self.color_gitter = color_gitter
        self.affine = RandPerspective(p=1,
                                      target_size=(target_size, target_size),
                                      degree=0,
                                      translate=0,
                                      scale=(0.6, 1.2),
                                      shear=0,
                                      perspective=0.0,
                                      pad_val=pad_val,
                                      )

    def img_aug(self, img: np.ndarray) -> np.ndarray:
        yc, xc = [int(random.uniform(-x, 2 * self.target_size + x)) for x in self.mosaic_border]
        indices = [random.randint(0, len(self.candidate_labels) - 1) for _ in range(3)]
        img4 = np.ones(shape=(self.target_size * 2, self.target_size * 2, 3))
        img4 = (img4 * (np.array(self.pad_val)[None, None, :])).astype(np.uint8)
        for i, index in enumerate([1] + indices):
            img_i = img if i == 0 else cv.imread(self.candidate_imgs[index])
            img_i, _ = self.scale_no_pad.scale_img(img_i)
            img_i, _ = self.color_gitter(img_i, np.zeros(shape=(0, 5), dtype=np.float32))
            h, w = img_i.shape[:2]
            if i == 0:
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, self.target_size * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(self.target_size * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            else:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, self.target_size * 2), min(self.target_size * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            img4[y1a:y2a, x1a:x2a] = img_i[y1b:y2b, x1b:x2b]
        img4, _ = self.affine(img4, np.zeros((0, 6), dtype=np.float32))
        return img4

    def aug(self, img: np.ndarray, labels: np.ndarray) -> tuple:
        yc, xc = [int(random.uniform(-x, 2 * self.target_size + x)) for x in self.mosaic_border]
        indices = [random.randint(0, len(self.candidate_labels) - 1) for _ in range(3)]

        img4 = np.ones(shape=(self.target_size * 2, self.target_size * 2, 3))
        img4 = (img4 * (np.array(self.pad_val)[None, None, :])).astype(np.uint8)
        labels4 = list()

        for i, index in enumerate([1] + indices):
            img_i = img if i == 0 else cv.imread(self.candidate_imgs[index])
            labels_i = labels if i == 0 else self.candidate_labels[index]
            img_i, ratio = self.scale_no_pad.scale_img(img_i)
            img_i, labels_i = self.color_gitter(img_i, labels_i)
            h, w = img_i.shape[:2]
            if i == 0:
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, self.target_size * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(self.target_size * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            else:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, self.target_size * 2), min(self.target_size * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            img4[y1a:y2a, x1a:x2a] = img_i[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b
            if labels_i.shape[0] > 0:
                labels_i[:, [2, 4]] = ratio * labels_i[:, [2, 4]] + padw
                labels_i[:, [3, 5]] = ratio * labels_i[:, [3, 5]] + padh
                labels4.append(labels_i)
        if len(labels4):
            labels4 = np.concatenate(labels4, 0)
            # np.clip(labels4[:, 1:] - s / 2, 0, s, out=labels4[:, 1:])  # use with center crop
            np.clip(labels4[:, 2:], 0, 2 * self.target_size, out=labels4[:, 2:])  # use with random_affine
        else:
            img4, labels = self.affine(img4, np.zeros((0, 6), dtype=np.float32))
            return img4, labels

        valid_index = np.bitwise_and((labels4[:, 4] - labels4[:, 2]) > 2, (labels4[:, 5] - labels4[:, 3]) > 2)
        labels4 = labels4[valid_index, :]
        img4, labels4 = self.affine(img4, labels4)
        return img4, labels4


class MixUp(DetectAugment):
    """
    使用双图混合增强
    """

    def __init__(self, candidate_imgs,
                 candidate_labels,
                 color_gitter=None,
                 alpha=0.6,
                 target_size=(640, 640),
                 pad_val=(114, 114, 114),
                 **kwargs):
        """
        :param candidate_imgs: 候选图片路径列表
        :param candidate_labels: 图片对应标注列表
        :param color_gitter: 单张图片进行混合增强前使用的color gitter
        :param alpha: 混合因子
        :param target_size: 长边缩放尺寸
        :param pad_val: padding部分填充的数值
        :param kwargs:
        """
        super(MixUp, self).__init__(**kwargs)
        self.p = 1
        self.candidate_imgs = candidate_imgs
        self.candidate_labels = candidate_labels
        if color_gitter is None:
            color_gitter = Identity()
        self.color_gitter = color_gitter
        self.alpha = alpha
        self.pad_val = pad_val
        self.scale_padding = ScalePadding(target_size=target_size, padding_val=pad_val, scale_up=True)

    def img_aug(self, img: np.ndarray) -> np.ndarray:
        index = random.randint(0, len(self.candidate_labels) - 1)
        # alpha = round(np.random.uniform(0.36, 0.64), 2)
        alpha = self.alpha
        append_img = cv.imread(self.candidate_imgs[index])
        append_img, _ = self.color_gitter(append_img, np.zeros(shape=(0, 6), dtype=np.float32))
        h1, w1 = img.shape[:2]
        h2, w2 = append_img.shape[:2]
        temp_img = np.ones(shape=(max(h1, h2), max(w1, w2), 3), dtype=np.float32)
        img1 = (temp_img * (np.array(self.pad_val)[None, None, :]))
        img2 = img1.copy()
        img1[0:h1, 0:w1, :] = img.astype(np.float32)
        img2[0:h2, 0:w2, :] = append_img.astype(np.float32)
        mix_img = img1 * alpha + img2 * (1 - alpha)
        mix_img = mix_img.astype(np.uint8)
        mix_img, _ = self.scale_padding(mix_img, np.zeros(shape=(0, 6), dtype=np.float32))
        return mix_img

    def aug(self, img: np.ndarray, labels: np.ndarray) -> tuple:
        index = random.randint(0, len(self.candidate_labels) - 1)
        alpha = self.alpha
        append_img = cv.imread(self.candidate_imgs[index])
        append_labels = self.candidate_labels[index]
        append_img, append_labels = self.color_gitter(append_img, append_labels)
        h1, w1 = img.shape[:2]
        h2, w2 = append_img.shape[:2]
        temp_img = np.ones(shape=(max(h1, h2), max(w1, w2), 3), dtype=np.float32)
        img1 = (temp_img * (np.array(self.pad_val)[None, None, :]))
        img2 = img1.copy()
        img1[0:h1, 0:w1, :] = img.astype(np.float32)
        img2[0:h2, 0:w2, :] = append_img.astype(np.float32)

        mix_img = img1 * alpha + img2 * (1 - alpha)
        mix_img = mix_img.astype(np.uint8)
        if len(labels):
            labels[:, 0] = alpha
        if len(append_labels):
            append_labels[:, 0] = 1 - alpha
        ret_labels = np.concatenate([labels, append_labels], axis=0)
        mix_img, ret_labels = self.scale_padding(mix_img, ret_labels)
        return mix_img, ret_labels


class OneOf(DetectAugment):
    """
    随即一个增强方式进行增强
    """

    def __init__(self, transforms, **kwargs):
        super(OneOf, self).__init__(**kwargs)
        self.p = 1
        if isinstance(transforms[0], DetectAugment):
            prob = float(1 / len(transforms))
            transforms = [(prob, transform) for transform in transforms]
        probs, transforms = zip(*transforms)
        probs, transforms = list(probs), list(transforms)
        for item in probs:
            assert item > 0, "prob > 0"
        assert np.sum(probs) == 1.0, 'sum of prob should be equal to 1'
        probs.insert(0, 0)
        self.flag = np.cumsum(probs)[:-1]
        self.transforms = transforms

    def img_aug(self, img: np.ndarray) -> np.ndarray:
        p = np.random.uniform(0, 1)
        index = (self.flag < p).sum() - 1
        img = self.transforms[index].img_aug(img)
        return img

    def aug(self, img: np.ndarray, labels: np.ndarray) -> tuple:
        p = np.random.uniform(0, 1)
        index = (self.flag < p).sum() - 1
        img, labels = self.transforms[index](img, labels)
        return img, labels


class Compose(DetectAugment):
    """
    串行数据增强的方式
    """

    def __init__(self, transforms, **kwargs):
        super(Compose, self).__init__(**kwargs)
        self.transforms = transforms
        self.p = 1

    def img_aug(self, img: np.ndarray) -> np.ndarray:
        for transform in self.transforms:
            img = transform.img_aug(img)
            return img

    def aug(self, img: np.ndarray, labels: np.ndarray) -> tuple:
        for transform in self.transforms:
            img, labels = transform(img, labels)
        return img, labels
