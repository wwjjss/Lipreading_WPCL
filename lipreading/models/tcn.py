import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import pdb

"""Implements Temporal Convolutional Network (TCN)

__https://arxiv.org/pdf/1803.01271.pdf
"""


# 裁剪 对卷积后的多余padding裁剪
# x.contiguous()——把tensor变成在内存中连续分布的形式
class Chomp1d(nn.Module):
    def __init__(self, chomp_size, symm_chomp):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
        self.symm_chomp = symm_chomp
        if self.symm_chomp:
            assert self.chomp_size % 2 == 0, "If symmetric chomp, chomp size needs to be even"

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        if self.symm_chomp:
            return x[:, :, self.chomp_size // 2:-self.chomp_size // 2].contiguous()
        else:
            return x[:, :, :-self.chomp_size].contiguous()


# cbcr：Conv BN Chomp1d relu
# dwpw：depthwise seperable convolution
class ConvBatchChompRelu(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, relu_type, dwpw=False):
        super(ConvBatchChompRelu, self).__init__()
        self.dwpw = dwpw
        if dwpw:
            self.conv = nn.Sequential(
                # -- dw
                nn.Conv1d(n_inputs, n_inputs, kernel_size, stride=stride,
                          padding=padding, dilation=dilation, groups=n_inputs, bias=False),
                nn.BatchNorm1d(n_inputs),
                Chomp1d(padding, True),
                nn.PReLU(num_parameters=n_inputs) if relu_type == 'prelu' else nn.ReLU(inplace=True),
                # -- pw
                nn.Conv1d(n_inputs, n_outputs, 1, 1, 0, bias=False),
                nn.BatchNorm1d(n_outputs),
                nn.PReLU(num_parameters=n_outputs) if relu_type == 'prelu' else nn.ReLU(inplace=True)
            )
        else:
            self.conv = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                  stride=stride, padding=padding, dilation=dilation)
            self.batchnorm = nn.BatchNorm1d(n_outputs)
            self.chomp = Chomp1d(padding, True)
            self.non_lin = nn.PReLU(num_parameters=n_outputs) if relu_type == 'prelu' else nn.ReLU()

    def forward(self, x):
        if self.dwpw:
            return self.conv(x)
        else:
            out = self.conv(x)
            out = self.batchnorm(out)
            out = self.chomp(out)
            return self.non_lin(out)


# --------- MULTI-BRANCH VERSION ---------------
class MultibranchTemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_sizes, stride, dilation, padding, dropout=0.2,
                 relu_type='relu', dwpw=False):
        super(MultibranchTemporalBlock, self).__init__()

        self.kernel_sizes = kernel_sizes  # [3,5,7]
        self.num_kernels = len(kernel_sizes)  # 3
        self.n_outputs_branch = n_outputs // self.num_kernels  # 256
        assert n_outputs % self.num_kernels == 0, "Number of output channels needs to be divisible by number of kernels"

        for k_idx, k in enumerate(self.kernel_sizes):  # 构建第一层Sub-Block
            # print(n_inputs, self.n_outputs_branch, k, stride, dilation, padding[k_idx])
            cbcr = ConvBatchChompRelu(n_inputs, self.n_outputs_branch, k, stride, dilation, padding[k_idx], relu_type,
                                      dwpw=dwpw)
            setattr(self, 'cbcr0_{}'.format(k_idx), cbcr)
        self.dropout0 = nn.Dropout(dropout)

        for k_idx, k in enumerate(self.kernel_sizes):  # 构建第二层Sub-Block
            # print(n_outputs, self.n_outputs_branch, k, stride, dilation, padding[k_idx])
            cbcr = ConvBatchChompRelu(n_outputs, self.n_outputs_branch, k, stride, dilation, padding[k_idx], relu_type,
                                      dwpw=dwpw)
            setattr(self, 'cbcr1_{}'.format(k_idx), cbcr)
        self.dropout1 = nn.Dropout(dropout)

        # downsample? 使用1*1的卷积控制输入和输出维度相同
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if (n_inputs // self.num_kernels) != n_outputs else None

        # final relu
        if relu_type == 'relu':
            self.relu_final = nn.ReLU()
        elif relu_type == 'prelu':
            self.relu_final = nn.PReLU(num_parameters=n_outputs)

    def forward(self, x):

        # first multi-branch set of convolutions
        outputs = []
        for k_idx in range(self.num_kernels):
            branch_convs = getattr(self, 'cbcr0_{}'.format(k_idx))
            outputs.append(branch_convs(x))
        out0 = torch.cat(outputs, 1)
        out0 = self.dropout0(out0)

        # second multi-branch set of convolutions
        outputs = []
        for k_idx in range(self.num_kernels):
            branch_convs = getattr(self, 'cbcr1_{}'.format(k_idx))
            outputs.append(branch_convs(out0))
        out1 = torch.cat(outputs, 1)
        out1 = self.dropout1(out1)

        # downsample?
        res = x if self.downsample is None else self.downsample(x)

        return self.relu_final(out1 + res)


# 构造4层，三分支，kernel为3，5，7的MS-TCN模型
# num_inputs=512  num_channels=[768, 768, 768, 768]
class MultibranchTemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, tcn_options, dropout=0.2, relu_type='relu', dwpw=False):
        super(MultibranchTemporalConvNet, self).__init__()

        self.ksizes = tcn_options['kernel_size']  # [3,5,7]
        layers = []
        num_levels = len(num_channels)  # 4
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            padding = [(s - 1) * dilation_size for s in self.ksizes]
            layers.append(MultibranchTemporalBlock(in_channels, out_channels, self.ksizes,
                                                   stride=1, dilation=dilation_size, padding=padding, dropout=dropout,
                                                   relu_type=relu_type,
                                                   dwpw=dwpw))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # print("tcn_forward:", x.shape)
        # print("tcn_network:", self.network(x).shape)
        return self.network(x)
    # --------------------------------


# --------------- STANDARD VERSION (SINGLE BRANCH) ------------------------
"""
相当于一个残差块
       :param n_inputs: int, 输入通道数
       :param n_outputs: int, 输出通道数
       :param kernel_size: int, 卷积核尺寸
       :param stride: int, 步长，一般为1
       :param dilation: int, 膨胀系数
       :param padding: int, 填充系数
       :param dropout: float, dropout比率
"""


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2,
                 symm_chomp=False, no_padding=False, relu_type='relu', dwpw=False):
        super(TemporalBlock, self).__init__()

        self.no_padding = no_padding
        if self.no_padding:
            downsample_chomp_size = 2 * padding - 4
            padding = 1  # hack-ish thing so that we can use 3 layers

        if dwpw:
            self.net = nn.Sequential(
                # -- first conv set within block
                # -- dw
                nn.Conv1d(n_inputs, n_inputs, kernel_size, stride=stride,
                          padding=padding, dilation=dilation, groups=n_inputs, bias=False),
                nn.BatchNorm1d(n_inputs),
                Chomp1d(padding, True),
                nn.PReLU(num_parameters=n_inputs) if relu_type == 'prelu' else nn.ReLU(inplace=True),
                # -- pw
                nn.Conv1d(n_inputs, n_outputs, 1, 1, 0, bias=False),
                nn.BatchNorm1d(n_outputs),
                nn.PReLU(num_parameters=n_outputs) if relu_type == 'prelu' else nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                # -- second conv set within block
                # -- dw
                nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride,
                          padding=padding, dilation=dilation, groups=n_outputs, bias=False),
                nn.BatchNorm1d(n_outputs),
                Chomp1d(padding, True),
                nn.PReLU(num_parameters=n_outputs) if relu_type == 'prelu' else nn.ReLU(inplace=True),
                # -- pw
                nn.Conv1d(n_outputs, n_outputs, 1, 1, 0, bias=False),
                nn.BatchNorm1d(n_outputs),
                nn.PReLU(num_parameters=n_outputs) if relu_type == 'prelu' else nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
        else:
            self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                   stride=stride, padding=padding, dilation=dilation)
            self.batchnorm1 = nn.BatchNorm1d(n_outputs)
            self.chomp1 = Chomp1d(padding, symm_chomp) if not self.no_padding else None
            if relu_type == 'relu':
                self.relu1 = nn.ReLU()
            elif relu_type == 'prelu':
                self.relu1 = nn.PReLU(num_parameters=n_outputs)
            self.dropout1 = nn.Dropout(dropout)

            self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                   stride=stride, padding=padding, dilation=dilation)
            self.batchnorm2 = nn.BatchNorm1d(n_outputs)
            self.chomp2 = Chomp1d(padding, symm_chomp) if not self.no_padding else None
            if relu_type == 'relu':
                self.relu2 = nn.ReLU()
            elif relu_type == 'prelu':
                self.relu2 = nn.PReLU(num_parameters=n_outputs)
            self.dropout2 = nn.Dropout(dropout)

            if self.no_padding:
                self.net = nn.Sequential(self.conv1, self.batchnorm1, self.relu1, self.dropout1,
                                         self.conv2, self.batchnorm2, self.relu2, self.dropout2)
            else:
                self.net = nn.Sequential(self.conv1, self.batchnorm1, self.chomp1, self.relu1, self.dropout1,
                                         self.conv2, self.batchnorm2, self.chomp2, self.relu2, self.dropout2)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        if self.no_padding:
            self.downsample_chomp = Chomp1d(downsample_chomp_size, True)
        if relu_type == 'relu':
            self.relu = nn.ReLU()
        elif relu_type == 'prelu':
            self.relu = nn.PReLU(num_parameters=n_outputs)

    def forward(self, x):
        out = self.net(x)
        if self.no_padding:
            x = self.downsample_chomp(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, tcn_options, dropout=0.2, relu_type='relu', dwpw=False):
        super(TemporalConvNet, self).__init__()
        self.ksize = tcn_options['kernel_size'][0] if isinstance(tcn_options['kernel_size'], list) else tcn_options[
            'kernel_size']
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(TemporalBlock(in_channels, out_channels, self.ksize, stride=1, dilation=dilation_size,
                                        padding=(self.ksize - 1) * dilation_size, dropout=dropout, symm_chomp=True,
                                        no_padding=False, relu_type=relu_type, dwpw=dwpw))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
# --------------------------------
