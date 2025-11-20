import torch
from torchsummary import summary
from torch import nn
from torch.nn import functional as F

# 第一个类：卷积块
class Conv_Block(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(Conv_Block, self).__init__()
        # 3*3的卷积核，步长为1
        self.layer = nn.Sequential(
            # 第一个卷积层
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(),

            # 第二个卷积层
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU())

    def forward(self, x):
        return self.layer(x)


# 第二个类：下采样类,下采样不改变通道数
class DownSample(nn.Module):
    def __init__(self, channel):
        super(DownSample, self).__init__()
        # 卷积核大小为3*3,步长为2,padding=1
        self.layer = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 2, 1, padding_mode="reflect", bias=False),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU())

    def forward(self, x):
        return self.layer(x)


# 第三个类：上采样类，也不会改变通道数
# 而且这里上采样层里有拼接contact
class UpSample(nn.Module):
    def __init__(self, channel):
        super(UpSample, self).__init__()
        # 采用最邻近插值法,变化为原来的一半
        self.layer = nn.Conv2d(channel, channel // 2, 1, 1)

    def forward(self, x, feature_map):
        # 插值函数，变成原来的2倍
        up = F.interpolate(x, scale_factor=2, mode="nearest")
        out = self.layer(up)

        # 将左边的特征图和我们经过上采样后的特征图进行拼接
        # 这里dim=1,说明是按（N,C,H,W)第一个维度C(channel)进行拼接
        return torch.cat((out, feature_map), dim=1)


# 第四个类 Unet类
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # 图片刚输入进来，经过一个双卷积层,通道数由3个通道变为64个通道
        self.c1 = Conv_Block(3, 64)
        # 然后经过一个下采样
        self.d1 = DownSample(64)
        # 又经过一个双卷积,通道数由64个通道变为128个通道
        self.c2 = Conv_Block(64, 128)
        # 然后又经过一个下采样
        self.d2 = DownSample(128)
        # 又经过一个双卷积,通道数由128个通道变为256个通道
        self.c3 = Conv_Block(128, 256)
        # 然后又经过一个下采样
        self.d3 = DownSample(256)
        # 又经过一个双卷积,通道数由256个通道变为512个通道
        self.c4 = Conv_Block(256, 512)
        # 然后又经过一个下采样
        self.d4 = DownSample(512)
        # 又经过一个双卷积,通道数由512个通道变为1024个通道
        self.c5 = Conv_Block(512, 1024)

        # 下面开始上采样,上采样通道数不变
        # 这里上采样层里有拼接contact
        self.u1 = UpSample(1024)
        # 又经过一个双卷积,通道数由1024个通道变为512个通道
        self.c6 = Conv_Block(1024, 512)
        # 上采样通道数不变
        self.u2 = UpSample(512)
        # 又经过一个双卷积,通道数由512个通道变成256个通道
        self.c7 = Conv_Block(512, 256)
        # 上采样通道数不变
        self.u3 = UpSample(256)
        # 又经过一个双卷积,通道数由512个通道变成256个通道
        self.c8 = Conv_Block(256, 128)
        # 上采样通道数不变
        self.u4 = UpSample(128)
        # 又经过一个双卷积,通道数由128个通道变成64个通道
        self.c9 = Conv_Block(128, 64)
        # 开始输出,要输出的是一个彩色图片，输入64通道，输出3通道，卷积核3*3,步长和padding都为1
        self.out = nn.Conv2d(64, 3, 3, 1, 1)
        # 激活函数，这里也可以用softmax
        self.Th = nn.Sigmoid()

    def forward(self, x):
        R1 = self.c1(x)
        R2 = self.c2(self.d1(R1))
        R3 = self.c3(self.d2(R2))
        R4 = self.c4(self.d3(R3))
        R5 = self.c5(self.d4(R4))

        # 下面开始上采样
        O1 = self.c6(self.u1(R5, R4))
        O2 = self.c7(self.u2(O1, R3))
        O3 = self.c8(self.u3(O2, R2))
        O4 = self.c9(self.u4(O3, R1))

        # 对输出结果再加上一个sigmoid层
        return self.Th(self.out(O4))


if __name__ == "__main__":
    net = UNet()
    summary(net, input_size=(3, 256, 256))