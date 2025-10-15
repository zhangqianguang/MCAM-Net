import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class MSCBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MSCBlock, self).__init__()

        # 使用1x7和7x1的卷积替代7x7和3x3的卷积
        self.dconv1_7 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 7), padding=(0, 3), groups=in_channels)  # 1x7深度卷积
        self.dconv7_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(7, 1), padding=(3, 0), groups=in_channels)  # 7x1深度卷积

        self.dconv1_3 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 3), padding=(0, 1), groups=in_channels)  # 1x3深度卷积
        self.dconv3_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 1), padding=(1, 0), groups=in_channels)  # 3x1深度卷积

        # 逐点卷积
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)  # 逐点卷积

        # 批归一化
        self.bn = nn.BatchNorm2d(2 * out_channels)  # 确保此处的通道数与拼接输出一致

        # 使用1x1卷积替代逐点卷积
        self.conv1 = nn.Conv2d(2 * out_channels, out_channels, kernel_size=1, padding=0)  # 1x1卷积
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)  # 1x1卷积

        # 激活函数
        self.relu = nn.ReLU(inplace=True)
        self.default_act = nn.SiLU()

    def forward(self, x):
        # 使用1x7和7x1的深度卷积
        dw_output_1 = self.dconv1_7(x)  # 1x7深度卷积
        dw_output_2 = self.dconv7_1(dw_output_1)  # 7x1深度卷积

        # 使用1x3和3x1的深度卷积
        dw_output_3 = self.dconv1_3(x)  # 1x3深度卷积
        dw_output_4 = self.dconv3_1(dw_output_3)  # 3x1深度卷积

        # 将四个卷积的输出拼接起来
        concatenated_output = torch.cat([dw_output_2,dw_output_4], dim=1)

        # 批归一化
        bn_output = self.bn(concatenated_output)

        # 使用1x1普通卷积
        conv_output_1 = self.conv1(bn_output)  # 1x1卷积
        # relu_output = self.relu(conv_output_1)
        # conv_output_2 = self.conv2(relu_output)  # 1x1卷积

        # 残差连接
        output = x + conv_output_1

        return output


class C2f_MSCBlock(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(MSCBlock(self.c, self.c) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))