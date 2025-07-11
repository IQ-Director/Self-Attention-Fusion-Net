# -*- coding: utf-8 -*-
# @annotation    : 数据读取类

import logging
from os.path import splitext
from pathlib import Path
from typing import Union

import random
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import torchvision.transforms.functional as F


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


class SyncTransform:
    def __init__(self, resize, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        """
        数据增强同步，避免双眼图像执行不同的数据增强方法
        Args:
            resize (): 调整的尺寸
            brightness (): 亮度
            contrast (): 对比度
            saturation (): 饱和度
            hue (): 色调
        """
        self.resize = resize
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.normalize = transforms.Normalize(mean=[0.444, 0.284, 0.155], std=[0.285, 0.206, 0.144])

    def colorjitter(self, img, color_params):
        transform_order, brightness_factor, contrast_factor, saturation_factor, hue_factor = color_params
        for t in transform_order:
            if t == 0 and brightness_factor is not None:
                img = F.adjust_brightness(img, brightness_factor)
            elif t == 1 and contrast_factor is not None:
                img = F.adjust_contrast(img, contrast_factor)
            elif t == 2 and saturation_factor is not None:
                img = F.adjust_saturation(img, saturation_factor)
            elif t == 3 and hue_factor is not None:
                img = F.adjust_hue(img, hue_factor)
        return img

    def __call__(self, l_img, r_img):
        # Resize
        l_img = F.resize(l_img, (self.resize, self.resize))
        r_img = F.resize(r_img, (self.resize, self.resize))

        # Random horizontal flip
        if random.random() > 0.5:
            l_img = F.hflip(l_img)
            r_img = F.hflip(r_img)

        # Random color jitter (same params)
        color_params = transforms.ColorJitter.get_params(
            brightness=[1 - self.brightness, 1 + self.brightness],
            contrast=[1 - self.contrast, 1 + self.contrast],
            saturation=[1 - self.saturation, 1 + self.saturation],
            hue=[-self.hue, self.hue]
        )

        l_img = self.colorjitter(l_img,color_params)
        r_img = self.colorjitter(r_img,color_params)

        # To tensor
        l_img = F.to_tensor(l_img)
        r_img = F.to_tensor(r_img)

        # 使用自己的数据集计算均值和标准差
        l_img = self.normalize(l_img)
        r_img = self.normalize(r_img)

        return l_img, r_img


class DoubleDataset(Dataset):
    def __init__(self, images_dir: str, label_path: str, resize: int = 224, transform=None):
        '''
        Args:
            images_dir (): 图片文件夹路径
            label_path (): 标签文件路径
            resize (): 调整尺寸大小
            transform (): 图片处理方法

        '''
        self.images_dir = Path(images_dir)
        self.label_path = Path(label_path)

        self.resize = resize
        self.transform = transform
        if self.transform is None:
            self.transform = SyncTransform(resize)

        self.df_label = pd.read_excel(label_path)
        self.ids = self.df_label['ID'].tolist()
        if not self.ids:
            raise RuntimeError(f'Empty label file: {label_path}')

        label_columns = self.df_label.columns[7:]
        self.multi_labels = {}

        for _, row in self.df_label.iterrows():
            id_value = row['ID']
            labels = row[label_columns].tolist()  # 获取标签列的列表
            labels = list(map(int, labels))
            self.multi_labels[id_value] = labels

        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]
        left_img_file = list(self.images_dir.glob(str(name) + '_left.*'))
        right_img_file = list(self.images_dir.glob(str(name) + '_right.*'))

        assert left_img_file != [], f'No input file found in {left_img_file}, make sure you put your images there'
        assert right_img_file != [], f'No input file found in {right_img_file}, make sure you put your images there'

        l_img = load_image(left_img_file[0])
        r_img = load_image(right_img_file[0])

        l_img,r_img = self.transform(l_img,r_img)

        return {
            'left_image': l_img.contiguous(),
            'right_image': r_img.contiguous(),
            'labels': torch.tensor(self.multi_labels[name], dtype=torch.int8).contiguous()
        }
