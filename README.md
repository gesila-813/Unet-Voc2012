# unet_voc
基于PASCAL VOC 2012图像分割任务的U-Net PyTorch实现。

## U-Net
U-Net是一种用于图像分割任务的神经网络架构。
它具有“U形”结构，由编码器（“收缩路径”）、解码器（“扩展路径”）和它们之间的快捷路径组成。
有关详细信息，请参阅原始论文。

# 用法
下载数据集
下载并解压PASCAL VOC 2012训练/验证数据集到unet_voc/目录中。
训练和验证图像应放置在./VOCdevkit/VOC2012/JPEGImages/目录中。

我们从PASCAL VOC 2012训练/验证集中包含分割数据的2913张图像中随机选择2713张作为训练集。
剩下的200张图像用作测试集。
图像文件名列在unet_voc/ImageSets/*.txt中。

## 训练
直接运行train.py使用默认参数，或者用以下方式在终端运行
python3 train.py epochs batch-size learning-rate amp classes device

epochs：整数，指定训练的轮数。
batch-size：整数，每次训练的批大小
learning-rate：浮点数，学习率
amp：布尔值，是否采用混合精度
classes：类别数（算入背景）
device：训练设备

每10个轮次生成一次训练模型，并保存为ckpt/tag_name/model_epoch*.pth。

## 测试
python3 test.py

每张图像的准确率及其平均值会打印在标准输出中。
生成的分割图保存在./SegImage/目录中。