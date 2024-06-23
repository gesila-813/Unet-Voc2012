import argparse  # 导入参数解析模块
import os  # 导入操作系统接口模块
import torch  # 导入PyTorch深度学习库
import torchvision  # 导入PyTorch视觉工具库
from matplotlib import pyplot as plt  # 导入绘图模块

import model  # 导入模型定义文件
import numpy as np  # 导入NumPy库
from PIL import Image  # 导入图像处理库中的图像类


def class2rgb(idx, rgb):  # 定义将类别索引转换为RGB颜色的函数
    def bit1(x, bit):  # 辅助函数，返回二进制位的值
        return int(x & (1 << bit) != 0)

    if rgb == 'r':  # 红色通道
        return 0x80 * bit1(idx, 0) + 0x40 * bit1(idx, 3)
    elif rgb == 'g':  # 绿色通道
        return 0x80 * bit1(idx, 1) + 0x40 * bit1(idx, 4)
    elif rgb == 'b':  # 蓝色通道
        return 0x80 * bit1(idx, 2) + 0x40 * bit1(idx, 5)


def get_args():  # 定义获取命令行参数的函数
    parser = argparse.ArgumentParser(description='Evaluate the UNet on an image and its ground truth')
    parser.add_argument('--ckpt_path', type=str, help='Path to the checkpoint file', default='ckpt/models/model_epochbest.pth')
    parser.add_argument('--img_path', type=str, help='Path to the image file', default='./VOCdevkit/VOC2012/JPEGImages/2011_002575.jpg')
    parser.add_argument('--gt_path', type=str, help='Path to the ground truth file', default='./VOCdevkit/VOC2012/SegmentationClass/2011_002575.png')
    return parser.parse_args()


def evaluate():  # 定义评估函数
    args = get_args()  # 获取命令行参数
    ckpt_path = args.ckpt_path  # 检查点路径
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 设置设备

    net = model.UNet()  # 创建UNet模型
    net.load_state_dict(torch.load(ckpt_path))  # 加载模型权重
    net = net.to(device)  # 将模型移动到设备
    net.eval()  # 设置为评估模式

    toTensor = torchvision.transforms.ToTensor()  # 定义图像转张量的变换
    class2rgb_np = np.vectorize(class2rgb)  # 使用NumPy向量化函数处理类别到RGB颜色的转换

    img_path = args.img_path  # 图像路径
    input_img = Image.open(img_path)  # 打开图像
    input_img = toTensor(input_img).to(device)  # 转为张量并移动到设备
    input_img = input_img.unsqueeze(0)  # 添加批次维度

    img_path = args.gt_path  # 分割标签路径
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
    out_img.save(os.path.join("./result/evaluate", img_path.split('/')[-1]) + '.png')  # 保存输出图像

    accuracy = np.sum(gt_img == out_seg) / gt_img.size  # 计算准确率
    print("Accuracy: %f" % accuracy)  # 输出准确率

    num_classes = 20  # 类别数
    dice_scores = []  # 存储每个类别的Dice系数
    for i in range(1, num_classes + 1):  # 遍历每个类别
        gt_class = (gt_img == i).astype(np.float32)  # 创建当前类别的真实标签
        out_class = (out_seg == i).astype(np.float32)  # 创建当前类别的预测标签
        gt_sum = np.sum(gt_class)  # 计算当前类别的真实像素数
        pred_sum = np.sum(out_class)  # 计算当前类别的预测像素数
        if (gt_sum != 0):  # 避免除0错误
            intersection = np.sum(gt_class * out_class)  # 计算交集
            dice = (2. * intersection) / (gt_sum + pred_sum)  # 计算Dice系数
            dice_scores.append(dice)  # 将Dice系数添加到列表中
    avg_dice = np.mean(dice_scores)  # 计算平均Dice系数
    print("Average Dice: %f" % avg_dice)  # 输出平均Dice系数

    # 可视化结果
    fig, ax = plt.subplots(1, 3, figsize=(10, 5))  # 创建图像显示窗口
    ax[0].imshow(input_img[0].permute(1, 2, 0).cpu().numpy())  # 显示输入图像
    ax[0].set_title('Input Image')  # 设置标题
    ax[1].imshow(out_img, cmap='viridis')  # 显示预测分割结果
    ax[1].set_title('Predicted Mask')  # 设置标题
    ax[2].imshow(gt_img, cmap='viridis')  # 显示真实分割结果
    ax[2].set_title('Ground Truth Mask')  # 设置标题
    plt.show()  # 显示图像

if __name__ == '__main__':
    evaluate()  # 执行评估函数