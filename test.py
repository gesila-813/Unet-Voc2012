import argparse  # 导入参数解析模块
import os  # 导入操作系统接口模块
import sys  # 导入系统模块
import torch  # 导入PyTorch深度学习库
import torchvision  # 导入PyTorch视觉工具库
import model  # 导入模型定义文件
import numpy as np  # 导入NumPy库
from PIL import Image  # 导入图像处理库中的图像类
from utils.dice_score import dice_loss  # 导入Dice损失函数
VOC_PATH = "./VOCdevkit/VOC2012"  # VOC数据集路径

ckpt_path = 'ckpt/models/model_epochbest1.pth'  # 检查点路径


def class2rgb(idx, rgb):  # 定义将类别索引转换为RGB颜色的函数
    def bit1(x, bit):  # 辅助函数，返回二进制位的值
        return int(x & (1 << bit) != 0)

    if rgb == 'r':  # 红色通道
        return 0x80 * bit1(idx, 0) + 0x40 * bit1(idx, 3)
    elif rgb == 'g':  # 绿色通道
        return 0x80 * bit1(idx, 1) + 0x40 * bit1(idx, 4)
    elif rgb == 'b':  # 蓝色通道
        return 0x80 * bit1(idx, 2) + 0x40 * bit1(idx, 5)


def test():  # 定义测试函数
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 获取设备

    net = model.UNet()  # 创建UNet模型
    net.load_state_dict(torch.load(ckpt_path))  # 加载模型权重
    net = net.to(device)  # 将模型移动到设备
    net.eval()  # 设置为评估模式

    toTensor = torchvision.transforms.ToTensor()  # 定义图像转张量的变换
    class2rgb_np = np.vectorize(class2rgb)  # 使用NumPy向量化函数处理类别到RGB颜色的转换

    sum_accuracy = 0.0  # 总准确率
    sum_dice = 0.0  # 总Dice系数
    num_classes = 20  # 类别数

    list_f = open("./ImageSets/train.txt").readlines()  # 读取测试集文件列表
    for line in list_f:  # 遍历每个测试样本
        img_path = os.path.join(VOC_PATH, "JPEGImages", line.replace('\n', '.jpg'))  # 图像路径
        input_img = Image.open(img_path)  # 打开图像
        input_img = toTensor(input_img).to(device)  # 转为张量并移动到设备
        input_img = input_img.unsqueeze(0)  # 添加批次维度

        img_path = os.path.join(VOC_PATH, "SegmentationClass", line.replace('\n', '.png'))  # 分割标签路径
        gt_img = Image.open(img_path)  # 打开分割标签
        gt_img = np.array(gt_img)  # 转为NumPy数组
        gt_img[gt_img == 255] = 0  # 处理标签中的255值

        # normalize = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化变换
        # input_img = normalize(input_img)  # 应用归一化变换

        out_seg = net(input_img)  # 模型推理
        out_seg = out_seg.argmax(1)[0].to('cpu').numpy()  # 获取预测结果

        out_seg_r = np.expand_dims(class2rgb_np(out_seg, 'r'), -1)  # 获取红色通道
        out_seg_g = np.expand_dims(class2rgb_np(out_seg, 'g'), -1)  # 获取绿色通道
        out_seg_b = np.expand_dims(class2rgb_np(out_seg, 'b'), -1)  # 获取蓝色通道
        out_seg_rgb = np.concatenate([out_seg_r, out_seg_g, out_seg_b], axis=-1)  # 合并通道
        out_img = Image.fromarray(out_seg_rgb.astype(np.uint8))  # 创建输出图像
        out_img.save(os.path.join("./result/test", line.replace('\n', '.png')))  # 保存输出图像

        accuracy = np.sum(gt_img == out_seg) / gt_img.size  # 计算准确率
        sum_accuracy += accuracy  # 累加准确率

        dice_scores = []  # 存储每个类别的Dice系数
        for i in range(1, num_classes + 1):  # 遍历每个类别
            gt_class = (gt_img == i).astype(np.float32)  # 创建当前类别的真实标签
            out_class = (out_seg == i).astype(np.float32)  # 创建当前类别的预测标签
            gt_sum = np.sum(gt_class)  # 计算当前类别的真实像素数
            pred_sum = np.sum(out_class)  # 计算当前类别的预测像素数
            if (gt_sum != 0):  # 避免除0错误
                intersection = np.sum(gt_class * out_class)  # 计算交集
                dice = (2. * intersection) / (gt_sum + pred_sum)  # 计算Dice系数
                dice_scores.append(dice)  # 添加到列表中
        avg_dice = np.mean(dice_scores)  # 计算平均Dice系数
        sum_dice += avg_dice  # 累加Dice系数

    print("Average Accuracy: %f" % (sum_accuracy / len(list_f)))  # 输出平均准确率
    print("Average Dice: %f" % (sum_dice / len(list_f)))  # 输出平均Dice系数


if __name__ == '__main__':  # 程序入口
    test()  # 调用测试函数