# 同时集成 Focal Convolution + Edge-Aware Pooling
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

"""
本文实现了两个创新组件：
1. EdgeAwarePool (可变形池化)：使用 DeformConv2d 实现自适应空间采样的池化操作
2. FocalConv2d (焦点卷积)：将特征图在高度方向分割成多个条带，独立进行卷积运算

关于可变形池化的数学原理：
相比于传统池化，可变形池化具有以下数学特性：
1.自适应采样：采样位置不再固定，而是根据输入特征动态调整
2.连续空间变换：通过双线性插值实现了连续的空间变换，而不是离散的下采样
3.边缘保持能力：能够更好地保持物体边缘信息，因为采样位置可以向边缘区域偏移
数学过程表示：
1.对于每个输出位置 (i, j) 和每个卷积核采样点 (k, l) ，计算变形后的采样坐标
2.通过双线性插值从输入特征图x中获取位置p_kl处的特征值:
3.将所有采样点的特征值进行聚合，常用的方法有平均池化和最大池化。
4.将聚合后的特征值作为输出的特征图。

"""


class EdgeAwarePool(nn.Module):
    """可变形池化：用 DeformConv2d 代替传统 MaxPool2d，实现边缘保持的自适应池化"""

    def __init__(self, in_ch, k=3, s=2, p=1):
        super().__init__()
        # 偏移量预测卷积层：输入通道数为in_ch，输出通道数为2×k×k
        # 输出的偏移量表示每个输出位置上k×k个采样点的x和y方向偏移
        self.offset = nn.Conv2d(in_ch, 2 * k * k, 3, s, p, bias=True)
        # 输出的 offset 张量形状为 (N, 2×k×k, H', W') ,表示每个输出位置上k×k个采样点的x和y方向偏移
        self.dconv = DeformConv2d(in_ch, in_ch, k, s, p, bias=False)

    def forward(self, x):
        # 数学逻辑：
        # 1. 首先通过offset卷积层生成偏移量：offset = self.offset(x)
        # 2. 然后使用生成的偏移量对输入特征图进行可变形卷积
        # 3. 可变形卷积会根据偏移量动态调整采样位置
        return self.dconv(x, self.offset(x))
        # x 是输入特征图，形状为 (N, C, H, W)
        # self.offset(x) 生成的偏移量，形状为 (N, 2×k×k, H', W')
        # self.dconv(x, self.offset(x)) 进行可变形卷积，输出形状为 (N, C, H', W')
        # 其中，H' = floor((H + 2*p - k)/s + 1), W' = floor((W + 2*p - k)/s + 1)
        # self.dconv(x, self.offset(x)) 进行可变形卷积，输出形状为 (N, C, H', W')


class FocalConv2d(nn.Module):
    """焦点卷积:将特征图在高度方向分割成p个条带,每个条带独立进行卷积运算
    这种设计可以更好地捕获人体不同部位的特征，适合于人体姿态识别任务
    """

    def __init__(self, in_c, out_c, k_size, p=1, **kwargs):
        super().__init__()
        assert p >= 1, "分割数量p必须大于等于1"
        self.p = p  # 在高度方向分割的条带数量
        padding = kwargs.pop('padding', 0)  # 获取或设置默认padding值

        # 为每个条带创建独立的卷积层
        self.convs = nn.ModuleList([
            nn.Conv2d(in_c, out_c, k_size, padding=padding, **kwargs)
            for _ in range(p)
        ])

    def forward(self, x):
        N, C, H, W = x.shape  # 获取输入特征图的维度：批次大小，通道数，高度，宽度
        assert H % self.p == 0, f'高度 {H} 不能整除 p={self.p}，无法均匀分割'

        strip_h = H // self.p  # 每个条带的高度
        strips = torch.split(x, strip_h, dim=2)  # 在高度方向(维度2)分割成p个条带

        # 对每个条带应用独立的卷积操作
        outs = [self.convs[i](strip) for i, strip in enumerate(strips)]

        # 在高度方向(维度2)将处理后的条带拼接回完整特征图
        return torch.cat(outs, dim=2)


class BasicConv2d(nn.Module):
    """基础卷积模块：普通卷积 + LeakyReLU激活函数
    用于特征提取和维度变换
    """

    def __init__(self, ic, oc, ks, **k):
        super().__init__()
        # 创建卷积层，输入通道数ic，输出通道数oc，核大小ks，不使用偏置
        self.conv = nn.Conv2d(ic, oc, ks, bias=False, **k)

    def forward(self, x):
        # 先进行卷积操作，然后应用LeakyReLU激活函数
        return nn.LeakyReLU(inplace=True)(self.conv(x))


class SetBlock(nn.Module):
    """集合操作模块:将常规的4D卷积/池化操作(适用于单帧)
    封装成适用于5D输入(N,S,C,H,W)的操作,其中S表示序列长度
    """

    def __init__(self, forward_block, pooling=None):
        super().__init__()
        self.fb = forward_block  # 前向操作模块，如BasicConv2d或FocalConv2d
        self.pool = pooling  # 池化操作，可以是None / EdgeAwarePool / nn.MaxPool2d

    def forward(self, x):
        # 输入形状：(N, S, C, H, W)，其中N是批次大小，S是序列长度
        n, s, c, h, w = x.size()

        # 将5D输入重塑为4D，以便应用常规卷积操作
        x = self.fb(x.view(-1, c, h, w))

        # 如果有池化操作，则应用池化
        if self.pool is not None:
            x = self.pool(x)

        # 获取处理后的特征图维度
        _, c, h, w = x.size()

        # 重塑回5D输出形状：(N, S, C, H, W)
        return x.view(n, s, c, h, w)
