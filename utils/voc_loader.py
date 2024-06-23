import os
import numpy as np
from PIL import Image
import torch
import torchvision

VOC_PATH = "./VOCdevkit/VOC2012"  # VOC数据集的根目录路径

# 定义一个函数，用于获取数据加载器
def get_dataloader(batch_size, crop_size=(256, 256), shuffle=True):
    # 创建一个VOCDataset实例
    dataset = VOCDataset(crop_size=crop_size)
    # 返回一个DataLoader对象
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1)

# 定义一个VOCDataset类，继承自torch.utils.data.Dataset
class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, crop_size=(256, 256), path=VOC_PATH):
        self.crop_size = crop_size  # 存储裁剪尺寸
        with open("./ImageSets/train.txt") as f:  # 打开训练集文件列表
            lines = f.readlines()  # 读取所有行
            # 获取所有图像文件的路径
            self.img_files = map(
                lambda l: os.path.join(path, "JPEGImages", l.replace('\n', '.jpg')),
                lines)
            # 获取所有分割标签文件的路径
            self.seg_files = map(
                lambda l: os.path.join(path, "SegmentationClass", l.replace('\n', '.png')),
                lines)

            # 定义一个函数，用于检查图像尺寸是否满足要求
            def size_checker(f):
                img = Image.open(f)  # 打开图像文件
                w, h = img.size  # 获取图像尺寸
                return w >= crop_size[0] & h >= crop_size[1]  # 检查宽和高是否都大于等于裁剪尺寸

            # 过滤掉不满足尺寸要求的图像文件
            self.img_files = filter(size_checker, self.img_files)
            # 过滤掉不满足尺寸要求的分割标签文件
            self.seg_files = filter(size_checker, self.seg_files)
            self.img_files = list(self.img_files)  # 将过滤后的图像文件列表转换为列表
            self.seg_files = list(self.seg_files)  # 将过滤后的分割标签文件列表转换为列表

    # 返回数据集的大小
    def __len__(self):
        return len(self.img_files)

    # 返回指定索引的数据样本
    def __getitem__(self, idx):
        img = Image.open(self.img_files[idx])  # 打开图像文件
        seg = Image.open(self.seg_files[idx])  # 打开分割标签文件

        # 获取随机裁剪的参数
        t, l, h, w = torchvision.transforms.RandomCrop.get_params(img, output_size=self.crop_size)
        # 对图像进行裁剪
        img = torchvision.transforms.functional.crop(img, t, l, h, w)
        # 对分割标签进行裁剪
        seg = torchvision.transforms.functional.crop(seg, t, l, h, w)

        # 将图像转换为张量
        toTensor = torchvision.transforms.ToTensor()
        img = toTensor(img)
        # 对图像进行标准化
        # normalize = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # img = normalize(img)

        # 将分割标签转换为张量
        seg = torch.tensor(np.array(seg), dtype=torch.long)
        seg[seg == 255] = 0  # 将“无效”类别转换为“背景”类别

        return img, seg  # 返回图像和分割标签