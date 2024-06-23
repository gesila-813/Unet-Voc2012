import argparse
import logging
import os

import torch
import torch.nn as nn
import torchsummary
from utils.dice_score import dice_loss
from tqdm import tqdm
import torch.nn.functional as F
from utils import voc_loader, ckpt_saver
import model
import matplotlib.pyplot as plt

# 日志文件路径
log_file = './logs/train.log'
# 模型存储路径
models_path = './ckpt/models'

# 配置日志记录器
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


# 定义训练函数
def train(num_epochs,
          batch_size=8,
          lr=0.001,
          num_classes=21,
          device='cuda'):
    # 定义裁剪尺寸
    crop_size = (256, 256)

    # 定义损失记录文件路径
    loss_path = './result/loss/loss_{}.txt'.format(num_epochs)

    # 获取数据加载器
    dataloader = voc_loader.get_dataloader(batch_size, crop_size=crop_size, shuffle=True)

    # 初始化UNet模型
    net = model.UNet()
    net = net.to(device)
    # 打印模型概述
    torchsummary.summary(net, (3, crop_size[0], crop_size[1]))

    # 初始化模型保存器
    csaver = ckpt_saver.CkptSaver(net)

    # 定义交叉熵损失函数
    ce_loss = nn.CrossEntropyLoss()
    ce_loss = ce_loss.to(device)
    # 定义优化器
    optim = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0.0)

    # 初始化损失记录
    total_loss = []
    min_loss = 1e10  # 初始化最小损失

    # 开始训练循环
    for epoch in range(num_epochs):
        running_loss = 0.0  # 初始化当前损失
        tqdm_loader = tqdm(dataloader, total=len(dataloader))
        for i, data in enumerate(tqdm_loader):
            img, seg = data
            img, seg = img.to(device), seg.to(device)

            # 前向传播
            outputs = net(img)

            # 计算损失
            loss = ce_loss(outputs, seg) + dice_loss(F.softmax(outputs, dim=1).float(),
                                                     F.one_hot(seg, num_classes).permute(0, 3, 1, 2).float(),
                                                     multiclass=True)

            # 反向传播和优化
            optim.zero_grad()
            loss.backward()
            optim.step()

            # 更新运行损失
            running_loss += loss.item()
            tqdm_loader.set_postfix(loss=running_loss / (i + 1))

        # 打印每个epoch的平均损失
        print('[%d] loss: %.3f' % (epoch + 1, running_loss / len(dataloader)))

        # 记录每个epoch的损失
        total_loss.append(running_loss / len(dataloader))
        with open(loss_path, 'w') as f:
            f.write('\n'.join(map(str, total_loss)))

        # 保存当前最小损失模型
        if running_loss / len(dataloader) < min_loss:
            min_loss = running_loss / len(dataloader)
            csaver.save('best')

        # 每10个epoch保存一次模型
        if epoch % 10 == 0:
            updateModels()
            csaver.save(epoch)

    # 训练完成后保存最终模型
    csaver.save(num_epochs)

    # 绘制损失函数图像
    plt.plot(range(1, num_epochs + 1), total_loss, 'b-', label='Training Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./result/loss{}.png'.format(num_epochs))
    plt.show()


# 更新模型文件，删除旧的模型文件，只保留最近的10个模型
def updateModels():
    model_path_list = [os.path.join(models_path, file) for file in os.listdir(models_path)
                       if 'best' not in file]

    if len(model_path_list) > 9:
        model_path_list.sort(key=os.path.getctime)
        os.remove(model_path_list[0])


# 获取命令行参数
def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--classes', '-c', type=int, default=21, help='Number of classes')
    parser.add_argument('--device', '-d', default='cuda', help='Device to use (cpu or cuda)')

    return parser.parse_args()


if __name__ == '__main__':
    # 获取参数
    args = get_args()

    # 记录参数
    logging.info(args)

    # 训练模型
    train(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_classes=args.classes,
        device=args.device
    )
    updateModels()