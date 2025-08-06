import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import init


class ResBlock(nn.Module):
    def __init__(self, dim):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))


class FeaMod1(nn.Module):
    def __init__(self, dim):
        super(FeaMod1, self).__init__()

        self.res1 = ResBlock(dim)

    def forward(self, x):
        x = self.res1(x)

        return x


class FeaMod2(nn.Module):
    def __init__(self, dim):
        super(FeaMod2, self).__init__()

        self.dowm = nn.Conv2d(dim, dim * 2, kernel_size=3, padding=1, stride=2, bias=True)
        self.res1 = ResBlock(dim * 2)

    def forward(self, x):
        x = self.res1(self.dowm(x))
        x = F.interpolate(x, scale_factor=2, mode='bilinear')

        return x


class FeaMod(nn.Module):
    def __init__(self, dim):
        super(FeaMod, self).__init__()

        self.fea1 = FeaMod1(dim)
        self.fea2 = FeaMod2(dim)
        self.conv1 = nn.Conv2d(dim * 3, dim, kernel_size=1, padding=0, stride=1, bias=True)

    def forward(self, x):
        x1 = self.fea1(x)
        x2 = self.fea2(x)
        x = torch.cat((x1, x2), dim=1)
        x = self.conv1(x)

        return x


class IgMod(nn.Module):
    def __init__(self, dim):
        super(IgMod, self).__init__()

        self.conv1 = nn.Conv2d(dim, 1, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(1, dim, kernel_size=3, padding=1, bias=True)
        self.res1 = ResBlock(dim)

    def forward(self, x, y, Phi, PhiT):
        x_pixel = self.conv1(x)
        Phix = F.conv2d(x_pixel, Phi, padding=0, stride=32, bias=None)
        delta = y - Phix
        x_pixel = nn.PixelShuffle(32)(F.conv2d(delta, PhiT, padding=0, bias=None))
        x_delta = self.conv2(x_pixel)
        x = self.res1(x_delta) + x

        return x


class AttBlock(nn.Module):
    def __init__(self, dim):
        super(AttBlock, self).__init__()

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=7, padding=3, stride=1, bias=True, groups=dim)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, bias=True)
        self.conv3 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, bias=True)
        self.scale = dim ** -0.5
        self.conv4 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, bias=True)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.conv1(x)
        x = x.reshape(B, C, H // 32, 32, W // 32, 32).permute(0, 2, 4, 1, 3, 5).reshape(-1, C, 32, 32)
        x1 = self.conv2(x).reshape(-1, C, 32 * 32)
        x2 = self.conv3(x).reshape(-1, C, 32 * 32).transpose(1, 2)
        att = (x2 @ x1) * self.scale
        att = att.softmax(dim=1)
        x = (x.reshape(-1, C, 32 * 32) @ att).reshape(-1, C, 32, 32)
        x = self.conv4(x)
        x = x.reshape(B, H // 32, W // 32, C, 32, 32).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)

        return x


class LCNet(nn.Module):
    def __init__(self, sensing_rate, LayerNo):
        super(LCNet, self).__init__()

        self.measurement = int(sensing_rate * 1024)
        self.base = 16

        self.Phi = nn.Parameter(init.xavier_normal_(torch.Tensor(self.measurement, 1024)))
        self.conv1 = nn.Conv2d(1, self.base, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(self.base, 2, kernel_size=1, padding=0, bias=True)
        layer1 = []
        layer2 = []
        self.LayerNo = LayerNo
        for i in range(LayerNo):
            layer1.append(FeaMod(self.base))
            layer2.append(IgMod(self.base))
        self.fcs1 = nn.ModuleList(layer1)
        self.fcs2 = nn.ModuleList(layer2)

    def forward(self, x):

        Phi = self.Phi.contiguous().view(self.measurement, 1, 32, 32)#######102,1,32,32

        PhiT = self.Phi.t().contiguous().view(1024, self.measurement, 1, 1)

        y = F.conv2d(x, Phi, padding=0, stride=32, bias=None)##########1,102,8,8

        x = F.conv2d(y, PhiT, padding=0, bias=None)############1,1024,8,8

        x = nn.PixelShuffle(32)(x)######1,1,256,256

        x = self.conv1(x)##########1,16,256,256

        for i in range(self.LayerNo):
            x = self.fcs2[i](x, y, Phi, PhiT)
            x = self.fcs1[i](x)#############1,16,256,256

        x = self.conv2(x)############,1,2,256,256

        phi_cons = torch.mm(self.Phi, self.Phi.t())

        return x, phi_cons,y,Phi
