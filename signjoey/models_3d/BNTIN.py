import torch
from torch import nn
import torch.nn.functional as F


class Inception(nn.Module):

    def __init__(
        self,
        in_channels: int,
        ch1x1: int,
        ch3x3red: int,
        ch3x3: int,
        ch5x5red: int,
        ch5x5: int,
        pool_proj: int,
        conv_block=None
    ) -> None:
        super(Inception, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1 = conv_block(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            conv_block(in_channels, ch3x3red, kernel_size=1),
            conv_block(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, ch5x5red, kernel_size=1),
            # Here, kernel_size=3 instead of kernel_size=5 is a known bug.
            # Please see https://github.com/pytorch/vision/issues/906 for details.
            conv_block(ch5x5red, ch5x5, kernel_size=3, padding=1)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            conv_block(in_channels, pool_proj, kernel_size=1)
        )

    def _forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class BasicConv2d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        **kwargs
    ) -> None:
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(TemporalBlock, self).__init__()
        self.conv1d = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1d = torch.nn.BatchNorm1d(num_features=out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv1d(x)
        x = self.bn1d(x)
        x = self.relu(x)
        return x

class BN_TIN(nn.Module):
    def __init__(self):
        super(BN_TIN, self).__init__()
        conv_block, inception_block = BasicConv2d, Inception
        #https://pytorch.org/vision/stable/_modules/torchvision/models/googlenet.html
        self.conv1 = conv_block(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = conv_block(64, 64, kernel_size=1)
        self.conv3 = conv_block(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = inception_block(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.temporal_1 = TemporalBlock(in_channels=1024, out_channels=512, kernel_size=5, stride=1, padding=2)
        self.maxpool1d_1 = torch.nn.MaxPool1d(kernel_size=2, stride=2, padding=1)

        self.temporal_2 = TemporalBlock(in_channels=512, out_channels=512, kernel_size=5, stride=1, padding=2)
        self.maxpool1d_2 = torch.nn.MaxPool1d(kernel_size=2, stride=2, padding=1)

    def gather_from_src(self, x, mask):
        #x B,C,T,H,W -> N,C,H,W
        x = x.transpose(1,2) # B,T,C,H,W
        reshaped = x.reshape([-1, self.Cin, self.H, self.W]) #B*T,C,H,W
        reshaped_mask = mask.reshape([-1, 1, 1, 1]) > 0 #B*T, 1, 1, 1
        selected = torch.masked_select(reshaped, reshaped_mask).reshape(-1, self.Cin, self.H, self.W) #->N
        return selected

    def scatter_from_output(self, x, mask):
        N, D = x.shape
        zeros = torch.zeros([self.batch_size*self.Tin, D],dtype=x.dtype, device=x.device) #B*T,D
        reshaped_mask = mask.reshape([-1,1])>0 #B*T,1
        y = zeros.masked_scatter(reshaped_mask, x) #number of ones in reshaped_mask = # elements in x
        y = y.reshape([self.batch_size, self.Tin, D]) #B,T,D
        return y.transpose(1,2) # B, D, T for 1d convolution

    def forward(self, x, sgn_lengths):
        #x B, C, T, H, W
        #sgn_lengths B
        #mask B, L
        self.batch_size, self.Cin, self.Tin, self.H, self.W = x.shape
        mask = torch.zeros([self.batch_size, self.Tin], dtype=torch.bool, device=x.device)
        for bi in range(self.batch_size):
            mask[bi, :sgn_lengths[bi]] = True
        x = self.gather_from_src(x, mask)
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14

        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        x = x.squeeze(-1).squeeze(-1) # N, 1024
        # N x 1024 x 1 x 1
        x = self.scatter_from_output(x, mask)
        # B, D, T padded with zeros

        x = self.temporal_1(x)
        x = self.maxpool1d_1(x)
        x = self.temporal_2(x)
        x = self.maxpool1d_2(x)        
        return x




        
