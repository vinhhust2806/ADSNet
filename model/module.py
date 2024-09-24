import torch 
import torch.nn as nn
import torch.nn.functional as F
import timm

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1):
        super(BasicConv2d, self).__init__()

        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.batchnorm(self.conv2d(x)))

class PASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PASPP, self).__init__()
        
        self.conv1x1_1 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, stride=1, padding=0)
        self.conv1x1_2 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, stride=1, padding=0)
        self.conv1x1_3 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, stride=1, padding=0)
        self.conv1x1_4 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1, stride=1, padding=0)
        
        self.bn1_1 = nn.BatchNorm2d(out_channels // 4)
        self.bn1_2 = nn.BatchNorm2d(out_channels // 4)
        self.bn1_3 = nn.BatchNorm2d(out_channels // 4)
        self.bn1_4 = nn.BatchNorm2d(out_channels // 4)

        self.conv3x3_1 = nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        self.conv3x3_2 = nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.conv3x3_3 = nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, stride=1, padding=4, dilation=4, bias=False)
        self.conv3x3_4 = nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=3, stride=1, padding=8, dilation=8, bias=False)

        self.bn2_1 = nn.BatchNorm2d(out_channels // 4)
        self.bn2_2 = nn.BatchNorm2d(out_channels // 4)
        self.bn2_3 = nn.BatchNorm2d(out_channels // 4)
        self.bn2_4 = nn.BatchNorm2d(out_channels // 4)

        self.conv1x1_out1 = nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=1, stride=1, padding=0)
        self.conv1x1_out2 = nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=1, stride=1, padding=0)
        self.bn_out1 = nn.BatchNorm2d(out_channels // 2)
        self.bn_out2 = nn.BatchNorm2d(out_channels // 2)

        self.conv1x1_final = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn_final = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x1 = F.relu(self.bn1_1(self.conv1x1_1(x)))
        x2 = F.relu(self.bn1_2(self.conv1x1_2(x)))
        x3 = F.relu(self.bn1_3(self.conv1x1_3(x)))
        x4 = F.relu(self.bn1_4(self.conv1x1_4(x)))

        x1_2 = x1 + x2
        x3_4 = x3 + x4

        x1 = F.relu(self.bn2_1(self.conv3x3_1(x1)))
        x1 = x1 + x1_2

        x2 = F.relu(self.bn2_2(self.conv3x3_2(x2)))
        x2 = x2 + x1_2

        x3 = F.relu(self.bn2_3(self.conv3x3_3(x3)))
        x3 = x3 + x3_4

        x4 = F.relu(self.bn2_4(self.conv3x3_4(x4)))
        x4 = x4 + x3_4

        x1_2 = torch.cat((x1, x2), dim=1)
        x3_4 = torch.cat((x3, x4), dim=1)

        x1_2 = F.relu(self.bn_out1(self.conv1x1_out1(x1_2)))
        x3_4 = F.relu(self.bn_out2(self.conv1x1_out2(x3_4)))

        y = torch.cat((x1_2, x3_4), dim=1)
        y = F.relu(self.bn_final(self.conv1x1_final(y)))

        return y

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        
        return self.sigmoid(x)

class AttentionGate(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(AttentionGate,self).__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        g1 = self.pool(g1)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi
    
class Decoder(nn.Module):
    def __init__(self, channel):
        super(Decoder, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, channel, 3, padding=1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x1 = self.conv4(x3_2)

        return x1