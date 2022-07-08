import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from lipreading.models.resnet import ResNet, BasicBlock
from lipreading.models.resnet1D import ResNet1D, BasicBlock1D
from lipreading.models.shufflenetv2 import ShuffleNetV2
from lipreading.models.tcn import MultibranchTemporalConvNet, TemporalConvNet


# -- auxiliary functions (B, 64, 29, 22, 22)
def threeD_to_2D_tensor(x):
    n_batch, n_channels, s_time, sx, sy = x.shape
    x = x.transpose(1, 2)
    return x.reshape(n_batch * s_time, n_channels, sx, sy)
    # (B*29,64,22,22)


# 在长度维度取特征平均值
def _average_batch(x, lengths, B):
    return torch.stack([torch.mean(x[index][:, 0:i], 1) for index, i in enumerate(lengths)], 0)


# APFF
def APFF(out, out1, out2, out3, lengths, B):
    w1 = torch.mean(out * out1, 1, True)  # [8,1]
    w2 = torch.mean(out * out2, 1, True)  # [8,1]
    w3 = torch.mean(out * out3, 1, True)  # [8,1]
    w = F.softmax(torch.cat((w1, w2, w3), 1), dim=1)  # [8,3]
    out_branch = out1 * w[:, 0:1] + out2 * w[:, 1:2] + out3 * w[:, 2:]  # [8,768]
    return out_branch


# MS-TCN
class MultiscaleMultibranchTCN(nn.Module):
    def __init__(self, input_size, num_channels, num_classes, tcn_options, dropout, relu_type, dwpw=False):
        super(MultiscaleMultibranchTCN, self).__init__()

        self.kernel_sizes = tcn_options['kernel_size']  # [3,5,7]
        self.num_kernels = len(self.kernel_sizes)  # 3

        self.mb_ms_tcn = MultibranchTemporalConvNet(input_size, num_channels, tcn_options, dropout=dropout,
                                                    relu_type=relu_type, dwpw=dwpw)
        self.tcn_output = nn.Linear(num_channels[-1], num_classes)  # 将B*768 调整为B*500张量即500分类概率

        self.consensus_func = _average_batch

        self.APFF = APFF

    def forward(self, x, a, b, c, lengths, B):
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        # 总分支
        xtrans = x.transpose(1, 2)  # 输入B×29×512 输出B×512×29
        # 部分分支
        a_trans = a.transpose(1, 2)
        b_trans = b.transpose(1, 2)
        c_trans = c.transpose(1, 2)
        # print("a_trans:", a_trans.shape)
        # print("xtrans * a_trans:", (xtrans * a_trans).shape)
        # 计算部分分支与总分支的权重
        # w1 = torch.mean(xtrans * a_trans, 1, True)  # [8, 1, 29]
        # w2 = torch.mean(xtrans * b_trans, 1, True)  # [8, 1, 29]
        # w3 = torch.mean(xtrans * c_trans, 1, True)  # [8, 1, 29]
        # w = F.softmax(torch.cat((w1, w2, w3), 1), dim=1)  # [8, 3, 29]
        # w_branch = a_trans * w[:, 0:1] + b_trans * w[:, 1:2] + c_trans * w[:, 2:]  # [8, 512, 29]
        # 总分支
        out = self.mb_ms_tcn(xtrans)  # out=(B,768,29)
        out = self.consensus_func(out, lengths, B)  # (B,768)

        # 部分分支
        out1 = self.mb_ms_tcn(a_trans)
        out1 = self.consensus_func(out1, lengths, B)
        out2 = self.mb_ms_tcn(b_trans)
        out2 = self.consensus_func(out2, lengths, B)
        out3 = self.mb_ms_tcn(c_trans)
        out3 = self.consensus_func(out3, lengths, B)
        # APFF
        out_branch = self.APFF(out, out1, out2, out3, lengths, B)
        # 部分分支
        # out_branch = self.mb_ms_tcn(w_branch)
        # out_branch = self.consensus_func(out_branch, lengths, B)
        # 全连接层
        main = self.tcn_output(out)
        branch = self.tcn_output(out_branch)
        # print("main,branch:", main, branch)
        # return self.tcn_output(out)
        return main, branch


class TCN(nn.Module):
    """Implements Temporal Convolutional Network (TCN)
    __https://arxiv.org/pdf/1803.01271.pdf
    """

    def __init__(self, input_size, num_channels, num_classes, tcn_options, dropout, relu_type, dwpw=False):
        super(TCN, self).__init__()
        self.tcn_trunk = TemporalConvNet(input_size, num_channels, dropout=dropout, tcn_options=tcn_options,
                                         relu_type=relu_type, dwpw=dwpw)
        self.tcn_output = nn.Linear(num_channels[-1], num_classes)

        self.consensus_func = _average_batch

        self.has_aux_losses = False

    def forward(self, x, lengths, B):
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        x = self.tcn_trunk(x.transpose(1, 2))
        x = self.consensus_func(x, lengths, B)
        return self.tcn_output(x)


# 输入：B×1×29×88×88
class Lipreading(nn.Module):
    def __init__(self, modality='video', hidden_dim=256, backbone_type='resnet', num_classes=500,
                 relu_type='prelu', tcn_options={}, width_mult=1.0, extract_feats=False):
        super(Lipreading, self).__init__()
        self.extract_feats = extract_feats
        self.backbone_type = backbone_type  # backbone
        self.modality = modality  # 数据模态

        if self.modality == 'raw_audio':
            self.frontend_nout = 1
            self.backend_out = 512
            self.trunk = ResNet1D(BasicBlock1D, [2, 2, 2, 2], relu_type=relu_type)
        elif self.modality == 'video':
            if self.backbone_type == 'resnet':
                self.frontend_nout = 64
                self.backend_out = 512
                # 使用ResNet作为backbone
                self.trunk = ResNet(BasicBlock, [2, 2, 2, 2], relu_type=relu_type)
            elif self.backbone_type == 'shufflenet':
                assert width_mult in [0.5, 1.0, 1.5, 2.0], "Width multiplier not correct"
                shufflenet = ShuffleNetV2(input_size=96, width_mult=width_mult)
                self.trunk = nn.Sequential(shufflenet.features, shufflenet.conv_last, shufflenet.globalpool)
                self.frontend_nout = 24
                self.backend_out = 1024 if width_mult != 2.0 else 2048
                self.stage_out_channels = shufflenet.stage_out_channels[-1]
            # 使用prelu
            frontend_relu = nn.PReLU(num_parameters=self.frontend_nout) if relu_type == 'prelu' else nn.ReLU()
            # 输入 (B,1,29,88,88)
            self.frontend3D = nn.Sequential(
                nn.Conv3d(1, self.frontend_nout, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3),
                          bias=False),  # 输出 => (B,64,29,44,44)
                nn.BatchNorm3d(self.frontend_nout),
                frontend_relu,
                nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)))  # 输出 => (B,64,29,22,22)
        else:
            raise NotImplementedError
        # TCN类别：TCN or MS-TCN 这里使用的是MS-TCN
        tcn_class = TCN if len(tcn_options['kernel_size']) == 1 else MultiscaleMultibranchTCN
        self.tcn = tcn_class(input_size=self.backend_out,
                             num_channels=[hidden_dim * len(tcn_options['kernel_size']) * tcn_options['width_mult']] *
                                          tcn_options['num_layers'],
                             num_classes=num_classes,
                             tcn_options=tcn_options,
                             dropout=tcn_options['dropout'],
                             relu_type=relu_type,
                             dwpw=tcn_options['dwpw'],
                             )
        # -- initialize
        self._initialize_weights_randomly()

    def forward(self, x, lengths):
        if self.modality == 'video':
            B, C, T, H, W = x.size()  # Batch_size 、color-channel 、Frames、Height、width (32,1,29,88,88)
            x = self.frontend3D(x)  # 3D 卷积  => (B,64,29,22,22)
            Tnew = x.shape[2]  # output  should be B x C2 x Tnew x H x W   Tnew=29
            x = threeD_to_2D_tensor(x)  # 3D 转 2D  out(B*29,64,22,22)
            x, a, b, c = self.trunk(x)  # 送入主干网络中ResNet18 输出(B*29,512)
            # print("output_resnet:", x.shape, a.shape, b.shape, c.shape)
            if self.backbone_type == 'shufflenet':
                x = x.view(-1, self.stage_out_channels)
            x = x.view(B, Tnew, x.size(1))  # (B,29,512)
            a, b, c = a.view(B, Tnew, a.size(1)), b.view(B, Tnew, b.size(1)), c.view(B, Tnew, c.size(1))
            # print("output_view:", x.shape, a.shape, b.shape, c.shape)
        elif self.modality == 'raw_audio':
            B, C, T = x.size()
            x = self.trunk(x)
            x = x.transpose(1, 2)
            lengths = [_ // 640 for _ in lengths]
        # if self.extract_feats:
        #     return x
        # else:
        #     return self.tcn(x, lengths, B), self.tcn(a, lengths, B), self.tcn(b, lengths, B), self.tcn(c, lengths, B)
        return x if self.extract_feats else self.tcn(x, a, b, c, lengths, B)

    def _initialize_weights_randomly(self):
        use_sqrt = True
        if use_sqrt:
            def f(n):
                return math.sqrt(2.0 / float(n))
        else:
            def f(n):
                return 2.0 / float(n)
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                n = np.prod(m.kernel_size) * m.out_channels
                m.weight.data.normal_(0, f(n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = float(m.weight.data[0].nelement())
                m.weight.data = m.weight.data.normal_(0, f(n))
