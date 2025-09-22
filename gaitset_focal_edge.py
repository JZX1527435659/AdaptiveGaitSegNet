
# gaitset_focal_edge.py
# 集成焦点卷积(Focal Convolution)和边缘感知池化(Edge-Aware Pooling)的GaitSet网络实现
# 该网络结合了焦点卷积和可变形池化来增强步态识别性能
import torch
import torch.nn as nn
import numpy as np
# 从focal_conv_edge模块导入自定义组件
from .focal_conv_edge import SetBlock, FocalConv2d, BasicConv2d, EdgeAwarePool


class SetNet(nn.Module):
    """集成焦点卷积和边缘感知池化的GaitSet网络架构
    主要创新点：
    1. 使用焦点卷积(FocalConv2d)在不同尺度捕获人体不同部位特征
    2. 使用边缘感知池化(EdgeAwarePool)替代传统池化,更好地保留边缘信息
    3. 采用双分支结构(SetBranch和GlobalBranch)提取多尺度特征
    4. 使用多尺度金字塔池化聚合特征
    """

    def __init__(self, hidden_dim, num_classes):
        super(SetNet, self).__init__()
        self.hidden_dim = hidden_dim  # 输出特征维度
        self.num_classes = num_classes #新增
        self.batch_frame = None  # 用于存储变长序列的帧索引

        # SetBranch(帧级分支)配置
        _set_in_channels = 1  # 输入通道数(单通道轮廓)
        _set_channels = [32, 64, 128]  # 各层输出通道数

        # Block1：浅层特征提取 - 使用普通卷积和边缘感知池化
        # 目的：提取底层特征并降低空间维度
        self.set_layer1 = SetBlock(BasicConv2d(_set_in_channels, _set_channels[0], 5, padding=2))
        self.set_layer2 = SetBlock(BasicConv2d(_set_channels[0], _set_channels[0], 3, padding=1),
                                   pooling=EdgeAwarePool(_set_channels[0]))  # 使用边缘感知池化

        # Block2：中层特征提取 - 使用焦点卷积(p=4)和边缘感知池化
        # 目的：将特征图在高度方向分为4个条带,独立卷积以捕获不同身体部位特征
        self.set_layer3 = SetBlock(FocalConv2d(_set_channels[0], _set_channels[1], 3, p=4, padding=1))
        self.set_layer4 = SetBlock(FocalConv2d(_set_channels[1], _set_channels[1], 3, p=4, padding=1),
                                   pooling=EdgeAwarePool(_set_channels[1]))

        # Block3：深层特征提取 - 使用焦点卷积(p=8)进一步精细化特征
        # 目的：将特征图分为8个条带,捕获更精细的身体部位特征
        self.set_layer5 = SetBlock(FocalConv2d(_set_channels[1], _set_channels[2], 3, p=8, padding=1))
        self.set_layer6 = SetBlock(FocalConv2d(_set_channels[2], _set_channels[2], 3, p=8, padding=1))

        # GlobalBranch(全局分支)配置
        # 目的：提取全局特征,与SetBranch特征互补
        _gl_in_channels = 32
        _gl_channels = [64, 128]
        self.gl_layer1 = BasicConv2d(_gl_in_channels, _gl_channels[0], 3, padding=1)
        self.gl_layer2 = BasicConv2d(_gl_channels[0], _gl_channels[0], 3, padding=1)
        self.gl_layer3 = BasicConv2d(_gl_channels[0], _gl_channels[1], 3, padding=1)
        self.gl_layer4 = BasicConv2d(_gl_channels[1], _gl_channels[1], 3, padding=1)
        self.gl_pooling = nn.MaxPool2d(2)

        # 多尺度分箱配置 - 用于多尺度金字塔池化
        self.bin_num = [1, 2, 4, 8, 16]  # 不同尺度的分箱数量
        # 全连接权重参数 - 用于多尺度特征映射到隐藏空间
        self.fc_bin = nn.ParameterList([
            nn.Parameter(
                nn.init.xavier_uniform_(
                    torch.zeros(sum(self.bin_num) * 2, 128, hidden_dim)))])

        # 新增分类头
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self._initialize_weights()

        # 权重初始化 - 使用Xavier均匀分布初始化卷积层和全连接层
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.normal(m.weight.data, 1.0, 0.02)
                nn.init.constant(m.bias.data, 0.0)

    def _initialize_weights(self):  # 新增
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

    def frame_max(self, x):
        """帧维度最大值池化,支持变长序列处理
        参数：
            x: 输入特征,形状为(N, S, C, H, W),其中N为批次大小,S为最大帧数
        返回：
            max_list: 帧维度最大值特征,形状为(N, C, H, W)
            arg_max_list: 最大值对应的帧索引,形状为(N, C, H, W)
        """
        if self.batch_frame is None:  # 无变长处理,批次中所有样本帧数相同
            return torch.max(x, 1)  # (N,S,C,H,W)→(N,C,H,W) 最大值及索引
        else:  # 有变长处理,self.batch_frame存储各样本的帧索引边界
            # self.batch_frame是一个长度为N+1的列表：[0, f1, f1+f2, … , f_total]
            _tmp = [
                torch.max(x[:, self.batch_frame[i]:self.batch_frame[i + 1], :, :, :], 1)
                for i in range(len(self.batch_frame) - 1)
            ]  # 逐段求最大值

            max_list = torch.cat([_tmp[i][0] for i in range(len(_tmp))], 0)  # 拼接最大值
            arg_max_list = torch.cat([_tmp[i][1] for i in range(len(_tmp))], 0)  # 拼接索引

            return max_list, arg_max_list  # 返回最大值和索引

    def frame_median(self, x):
        """帧维度中位数池化(结构与frame_max相同,但实际未使用)"""
        if self.batch_frame is None:
            return torch.median(x, 1)
        else:
            _tmp = [
                torch.median(x[:, self.batch_frame[i]:self.batch_frame[i + 1], :, :, :], 1)
                for i in range(len(self.batch_frame) - 1)
            ]
            median_list = torch.cat([_tmp[i][0] for i in range(len(_tmp))], 0)
            arg_median_list = torch.cat([_tmp[i][1] for i in range(len(_tmp))], 0)
            return median_list, arg_median_list

    def forward(self, silho, batch_frame=None):
            """网络前向传播函数
            参数：
                silho: 输入步态轮廓序列,形状为(N, S, H, W)
                batch_frame: 可选,各样本实际帧数列表
            返回：
                feature: 提取的步态特征,形状为(N, 62, hidden_dim)
                None: 占位返回值
            """
            if self.batch_frame is None:  # 无变长,batch 里所有样本长度相同 ,即帧数相同
                return torch.max(x, 1)  # (N,S,C,H,W)→(N,C,H,W) 最大值及索引
            else:  # self.batch_frame 是一个长度为 N+1 的 list：[0, f1, f1+f2, … , f_total],
                # f_i 表示第 i 个样本的真实帧数
                _tmp = [
                    torch.max(x[:, self.batch_frame[i]:self.batch_frame[i + 1], :, :, :], 1)
                    for i in range(len(self.batch_frame) - 1)
                ]  # 逐段求最大

                max_list = torch.cat([_tmp[i][0] for i in range(len(_tmp))], 0)  # 拼接最大值
                arg_max_list = torch.cat([_tmp[i][1] for i in range(len(_tmp))], 0)  # 拼接索引
            
                return max_list, arg_max_list  # max_list：最大值 Tensor (N, C, H, W)
                # arg_max_list：最大值在帧维度的索引 (N, C, H, W)

    

    def frame_median(self, x):
        """帧维度中位数池化(结构与frame_max相同,但实际未使用)"""
        if self.batch_frame is None:
            return torch.median(x, 1)
        else:
            _tmp = [
                torch.median(x[:, self.batch_frame[i]:self.batch_frame[i + 1], :, :, :], 1)
                for i in range(len(self.batch_frame) - 1)
            ]
            median_list = torch.cat([_tmp[i][0] for i in range(len(_tmp))], 0)
            arg_median_list = torch.cat([_tmp[i][1] for i in range(len(_tmp))], 0)
            return median_list, arg_median_list

    # ---------------------------------------------------------------------
    def forward(self, silho, batch_frame=None):  # 网络前向
        # 处理变长序列输入
        if batch_frame is not None:  # 若提供了各样本实际帧数
            batch_frame = batch_frame[0].data.cpu().numpy().tolist()  # 转换为Python列表
            _ = len(batch_frame)
            # 移除尾部的零值(填充帧)
            for i in range(len(batch_frame)):
                if batch_frame[-(i + 1)] != 0:
                    break
                else:
                    _ -= 1
            batch_frame = batch_frame[:_]  # 截断列表,移除尾部零值
            frame_sum = np.sum(batch_frame)  # 计算有效总帧数
            if frame_sum < silho.size(1):  # 如果有效帧数小于输入帧数
                silho = silho[:, :frame_sum, :, :]  # 移除尾部空帧
            # 生成帧索引边界列表 [0, f1, f1+f2, ..., f_total]
            self.batch_frame = [0] + np.cumsum(batch_frame).tolist()
        else:
            self.batch_frame = None

        n = silho.size(0)  # 当前批次大小
        x = silho.unsqueeze(2)  # 增加通道维度,形状变为(N,S,1,H,W)
        del silho  # 及时释放显存

        # ========== SetBranch(帧级分支)前向传播 ==========
        # 第1组卷积块 - 提取浅层特征
        x = self.set_layer1(x)  # (N,S,1,H,W) → (N,S,32,H,W)
        x = self.set_layer2(x)  # 应用边缘感知池化 → (N,S,32,H/2,W/2)

        # 初始化GlobalBranch(全局分支)特征
        gl = self.gl_layer1(self.frame_max(x)[0])  # 从SetBranch提取全局特征 → (N,64,H/2,W/2)
        gl = self.gl_layer2(gl)  # 全局分支卷积 → (N,64,H/2,W/2)
        gl = self.gl_pooling(gl)  # 全局分支池化 → (N,64,H/4,W/4)

        # 第2组卷积块 - 使用焦点卷积(p=4)提取中层特征
        x = self.set_layer3(x)  # (N,S,32,H/2,W/2) → (N,S,64,H/2,W/2)
        x = self.set_layer4(x)  # 应用焦点卷积和边缘感知池化 → (N,S,64,H/4,W/4)

        # 全局分支与SetBranch特征融合
        gl = self.gl_layer3(gl + self.frame_max(x)[0])  # 残差连接融合 → (N,128,H/4,W/4)
        gl = self.gl_layer4(gl)  # 全局分支卷积 → (N,128,H/4,W/4)

        # 第3组卷积块 - 使用焦点卷积(p=8)提取深层特征
        x = self.set_layer5(x)  # (N,S,64,H/4,W/4) → (N,S,128,H/4,W/4)
        x = self.set_layer6(x)  # 应用焦点卷积 → (N,S,128,H/4,W/4)

        # 帧级全局最大池化 - 聚合帧维度信息
        x = self.frame_max(x)[0]  # (N,S,128,H/4,W/4) → (N,128,H/4,W/4)

        # 最终特征融合 - SetBranch与GlobalBranch特征相加
        gl = gl + x  # 残差连接 → (N,128,H/4,W/4)

        # ========== 多尺度金字塔池化 ==========
        """
        多尺度金字塔池化的核心作用：
        1. 将特征图在高度方向按不同尺度(1,2,4,8,16)分割成多个条带
        2. 对每个条带分别进行均值池化和最大值池化,捕获不同尺度的特征
        3. 将所有尺度的特征拼接并映射到统一维度,增强特征表达能力
        """
        feature = []  # 初始化特征列表,用于存储多尺度特征
        n, c, h, w = gl.size()  # 获取特征图维度：n=N(批次), c=128(通道), h=H/4(高度), w=W/4(宽度)

        # 遍历每个分块尺度 [1,2,4,8,16]
        for num_bin in self.bin_num:
            # 处理SetBranch特征
            z = x.view(n, c, num_bin, -1)  # 将特征图重塑为4D张量(N, 128, num_bin, p)
            # 其中p = (H/4 * W/4) // num_bin,表示每个条带的像素数

            # 对每个条带应用均值池化+最大值池化融合
            z = z.mean(3) + z.max(3)[0]  # 沿最后一维融合,得到形状(N, 128, num_bin)
            feature.append(z)  # 存储SetBranch当前尺度特征

            # 处理GlobalBranch特征(与SetBranch相同操作)
            z = gl.view(n, c, num_bin, -1)  # 重塑GlobalBranch特征
            z = z.mean(3) + z.max(3)[0]  # 均值+最大值池化融合
            feature.append(z)  # 存储GlobalBranch当前尺度特征

        # 拼接所有尺度和分支的特征
        # 每个尺度有2个特征(SetBranch+GlobalBranch),5个尺度共10个特征
        # 沿第2维(通道方向)拼接,得到形状(N, 128, 62),其中62=2*(1+2+4+8+16)
        feature = torch.cat(feature, 2)

        # 调整维度顺序,便于后续矩阵乘法
        feature = feature.permute(2, 0, 1).contiguous()  # 形状变为(62, N, 128)

        # ========== 全连接特征映射 ==========
        # 对每个特征块应用独立的全连接映射
        # 权重形状为(62, 128, hidden_dim),表示对62个特征块分别进行映射
        feature = feature.matmul(self.fc_bin[0])  # (62, N, 128) × (62, 128, hidden_dim) → (62, N, hidden_dim)

        # 恢复批次维度为第一维,得到最终特征形状(N, 62, hidden_dim)
        feature = feature.permute(1, 0, 2).contiguous()

        # 新增分类预测 - 输出原始logits，Softmax在损失函数中处理
        label_prob = self.classifier(feature.mean(dim=1))

        # 返回提取的步态特征和概率分布
        return feature, label_prob
