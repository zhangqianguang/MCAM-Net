import torch
import torch.nn.functional as F
from torch import nn
import math

# FReLU Implementation
class FReLU(nn.Module):
    def __init__(self, c1):
        super(FReLU, self).__init__()
        self.conv = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, groups=c1)
        self.bn = nn.GroupNorm(num_groups=32, num_channels=c1)  # GroupNorm 适用于小批量和小尺寸
        # 或者使用 LayerNorm: self.bn = nn.LayerNorm([c1, 1, 1])

    def forward(self, x):
        return torch.max(x, self.bn(self.conv(x)))



# New ChannelAttention implementation (using channel_att logic)
class CA(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(CA, self).__init__()
        self.fc1 = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1,
                             bias=True)
        self.fc2 = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1,
                             bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x1 = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        x1 = self.fc1(x1)
        x1 = F.relu(x1, inplace=True)
        x1 = self.fc2(x1)
        x1 = torch.sigmoid(x1)

        x2 = F.adaptive_max_pool2d(inputs, output_size=(1, 1))
        x2 = self.fc1(x2)
        x2 = F.relu(x2, inplace=True)
        x2 = self.fc2(x2)
        x2 = torch.sigmoid(x2)

        x = x1 + x2
        x = x.view(-1, self.input_channels, 1, 1)
        return x


# Define SubSpace and SA modules as given earlier
class SubSpace(nn.Module):
    def __init__(self, nin: int):
        super(SubSpace, self).__init__()
        # Multi-scale depthwise convolutions
        self.dconv5_5 = nn.Conv2d(nin, nin, kernel_size=5, padding=2, groups=nin)
        self.dconv1_7 = nn.Conv2d(nin, nin, kernel_size=(1, 7), padding=(0, 3), groups=nin)
        self.dconv7_1 = nn.Conv2d(nin, nin, kernel_size=(7, 1), padding=(3, 0), groups=nin)
        self.dconv1_11 = nn.Conv2d(nin, nin, kernel_size=(1, 11), padding=(0, 5), groups=nin)
        self.dconv11_1 = nn.Conv2d(nin, nin, kernel_size=(11, 1), padding=(5, 0), groups=nin)
        self.dconv1_21 = nn.Conv2d(nin, nin, kernel_size=(1, 21), padding=(0, 10), groups=nin)
        self.dconv21_1 = nn.Conv2d(nin, nin, kernel_size=(21, 1), padding=(10, 0), groups=nin)
        self.conv = nn.Conv2d(nin, nin, kernel_size=(1, 1), padding=0)

    def forward(self, x):
        # Multi-scale convolution operations
        out_5_5 = self.dconv5_5(x)
        out_1 = self.dconv1_7(out_5_5)
        out_1 = self.dconv7_1(out_1)
        out_2 = self.dconv1_11(out_5_5)
        out_2 = self.dconv11_1(out_2)
        out_3 = self.dconv1_21(out_5_5)
        out_3 = self.dconv21_1(out_3)

        # Add outputs from different kernel sizes
        out = out_3 + out_2 + out_1
        out = torch.mul(out, x)
        out = out + x
        return out

class SA(nn.Module):
    """
    Grouped Attention Block having multiple (num_splits) Subspaces.
    num_splits : int
        number of subspaces
    """
    def __init__(self, nin: int, nout: int, h: int, w: int, num_splits: int):
        super(SA, self).__init__()
        assert nin % num_splits == 0  # Ensure nin is divisible by num_splits

        self.nin = nin
        self.nout = nout
        self.h = h
        self.w = w
        self.num_splits = num_splits

        self.subspaces = nn.ModuleList(
            [SubSpace(int(self.nin / self.num_splits)) for _ in range(self.num_splits)]
        )

    def forward(self, x):
        group_size = int(self.nin / self.num_splits)
        sub_feat = torch.chunk(x, self.num_splits, dim=1)

        out = []
        for idx, l in enumerate(self.subspaces):
            out.append(self.subspaces[idx](sub_feat[idx]))

        out = torch.cat(out, dim=1)
        return out

# CPCA module with ULSAM replacing spatial attention
class CPMSSA(nn.Module):
    def __init__(self, in_channels, out_channels, channelAttention_reduce=4, num_splits=4):
        super(CPMSSA, self).__init__()
        self.C = in_channels
        self.O = out_channels
        assert in_channels == out_channels

        # Using new ChannelAttention
        # self.ca = ChannelAttention(channel=in_channels)
        # self.ca = ChannelAttention(input_channels=in_channels)
        self.ca = CA(input_channels=in_channels, internal_neurons=in_channels // channelAttention_reduce)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1), padding=0)
        self.frelu = FReLU(c1=in_channels)  # Use FReLU instead of GELU
        self.act = nn.GELU()
        # Replacing the spatial attention part with ULSAM
        self.SA = SA(in_channels, out_channels, h=128, w=128, num_splits=num_splits)

        self.relu = nn.ReLU(inplace=True)
    def forward(self, inputs):
        # inputs = self.conv(inputs)
        # inputs = self.frelu(inputs)  # FReLU activation
        # inputs = self.relu(inputs)

        # Channel attention
        channel_att_vec = self.ca(inputs)
        inputs = channel_att_vec * inputs

        # Using SA instead of the original spatial attention mechanism
        out = self.SA(inputs)

        return out

# Test the module
if __name__ == "__main__":
    cpca_module = CPMSSA(in_channels=64, out_channels=64, num_splits=4)
    input_tensor = torch.randn(1, 64, 128, 128)
    output_tensor = cpca_module(input_tensor)
    print('Input size:', input_tensor.size())
    print('Output size:', output_tensor.size())
