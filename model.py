import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义一个下采样层类，继承自nn.Module
class DownLayer(nn.Module):
    def __init__(self, inc):
        super(DownLayer, self).__init__()
        # 定义一个2x2的最大池化层
        self.pool = nn.MaxPool2d(2, 2)
        # 定义第一个卷积层，将输入通道数inc变为2倍
        self.conv1 = nn.Conv2d(inc, inc * 2, 3, padding=1)
        # 定义第一个BatchNorm层，对应第一个卷积层的输出通道数
        self.bn1 = nn.BatchNorm2d(inc * 2, track_running_stats=True)
        # 定义第二个卷积层，输入输出通道数不变
        self.conv2 = nn.Conv2d(inc * 2, inc * 2, 3, padding=1)
        # 定义第二个BatchNorm层，对应第二个卷积层的输出通道数
        self.bn2 = nn.BatchNorm2d(inc * 2, track_running_stats=True)

    # 定义前向传播过程
    def forward(self, x):
        # 先对输入进行池化
        net = self.pool(x)
        # 第一个卷积层 -> 批归一化 -> ReLU激活
        net = F.relu(self.bn1(self.conv1(net)))
        # 第二个卷积层 -> 批归一化 -> ReLU激活
        net = F.relu(self.bn2(self.conv2(net)))
        # 返回下采样后的特征图
        return net

# 定义一个上采样层类，继承自nn.Module
class UpLayer(nn.Module):
    def __init__(self, inc):
        super(UpLayer, self).__init__()
        # 定义一个转置卷积层，用于上采样，将通道数从inc变为inc//2
        self.convt = nn.ConvTranspose2d(inc, inc // 2, 2, stride=2)
        # 定义第一个卷积层，将输入通道数从inc变为inc//2
        self.conv1 = nn.Conv2d(inc, inc // 2, 3, padding=1)
        # 定义第一个BatchNorm层，对应第一个卷积层的输出通道数
        self.bn1 = nn.BatchNorm2d(inc // 2, track_running_stats=True)
        # 定义第二个卷积层，输入输出通道数不变
        self.conv2 = nn.Conv2d(inc // 2, inc // 2, 3, padding=1)
        # 定义第二个BatchNorm层，对应第二个卷积层的输出通道数
        self.bn2 = nn.BatchNorm2d(inc // 2, track_running_stats=True)

    # 定义前向传播过程
    def forward(self, up_x, shortcut_x):
        # 对输入进行转置卷积上采样
        net = self.convt(up_x)

        # 获取跳跃连接特征图的形状
        out_shape = shortcut_x.shape  # [B, C, H, W]
        # 获取上采样后特征图的形状
        up_shape = net.shape
        # 计算需要填充的高度和宽度
        pad_h = out_shape[2] - up_shape[2]
        pad_w = out_shape[3] - up_shape[3]
        # 对上采样后的特征图进行填充
        net = F.pad(net, (0, pad_w, 0, pad_h))

        # 将跳跃连接特征图与上采样后的特征图在通道维度上拼接
        net = torch.cat((shortcut_x, net), 1)
        # 第一个卷积层 -> 批归一化 -> ReLU激活
        net = F.relu(self.bn1(self.conv1(net)))
        # 第二个卷积层 -> 批归一化 -> ReLU激活
        net = F.relu(self.bn2(self.conv2(net)))

        # 返回上采样后的特征图
        return net

# 定义UNet模型类，继承自nn.Module
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # 定义输入层的两个卷积层
        self.input_conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.input_conv2 = nn.Conv2d(64, 64, 3, padding=1)

        # 定义下采样层
        self.ld2 = DownLayer(64)
        self.ld3 = DownLayer(128)
        self.ld4 = DownLayer(256)
        self.ld5 = DownLayer(512)

        # 定义上采样层
        self.lu4 = UpLayer(1024)
        self.lu3 = UpLayer(512)
        self.lu2 = UpLayer(256)
        self.lu1 = UpLayer(128)

        # 定义输出层的卷积层
        self.output_conv = nn.Conv2d(64, 21, 1)

    # 定义前向传播方法
    def forward(self, x):
        # 输入层卷积 -> ReLU激活
        ld1 = F.relu(self.input_conv1(x))
        ld1 = F.relu(self.input_conv2(ld1))

        # 下采样层
        ld2 = self.ld2(ld1)
        ld3 = self.ld3(ld2)
        ld4 = self.ld4(ld3)
        ld5 = self.ld5(ld4)

        # 上采样层，使用跳跃连接
        lu4 = self.lu4(ld5, ld4)
        lu3 = self.lu3(lu4, ld3)
        lu2 = self.lu2(lu3, ld2)
        lu1 = self.lu1(lu2, ld1)

        # 输出层卷积
        out = self.output_conv(lu1)
        return out