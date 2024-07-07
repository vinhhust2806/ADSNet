import torch
import torch.nn as nn
import torch.nn.functional as F
import os

torch.manual_seed(31)

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

class PASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureTransform, self).__init__()
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


class Decoder(nn.Module):
    def __init__(self, channel):
        super(CFM, self).__init__()
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

class Model(nn.Module):
    def __init__(self, channel=32):
        super(Model, self).__init__()

        self.encoder = timm.create_model('tf_efficientnetv2_s.in21k_ft_in1k', pretrained=True, features_only=True) 
        
        self.Translayer2_1 = PASPP(64, channel)
        self.Translayer3_1 = PASPP(160, channel)
        self.Translayer4_1 = PASPP(256, channel1)
        
        self.decoder = Decoder(channel)
        self.ca = ChannelAttention(48)
        self.sa = SpatialAttention()
        
        self.out_decoder = nn.Conv2d(channel, 1, 1)
        self.attention_gate = AttentionGate()
        self.out = nn.Conv2d(3,1,1)

    def forward(self, x):

        # Encoder
        encoder = self.backbone(x)
        x1 = encoder[1]
        x2 = encoder[2]
        x3 = encoder[3]
        x4 = encoder[4]
        
        # CIM
        x1 = self.ca(x1) * x1 # channel attention
        cim_feature = self.sa(x1) * x1 # spatial attention

        x2_t = self.Translayer2_1(x2)  
        x3_t = self.Translayer3_1(x3)  
        x4_t = self.Translayer4_1(x4) 
      
        # Decoder
        cfm_feature = self.decoder(x4_t, x3_t, x2_t)
        prediction1 = self.out_decoder(cfm_feature)
        
        out1_resized = nn.functional.interpolate(prediction1, size=(64,64), mode='bilinear', align_corners=False)
        out1_sigmoid = torch.sigmoid(out1_resized)
        threshold = 0.00001
        out1_s2 = (out1_sigmoid > threshold).float()

        # Continuous Attention
        p1_s1 = cim_feature*(1-out1_sigmoid)
        a2_s1 = self.attention_gate(p1_s1,x2_t)
        a3_s1 = self.attention_gate(a2_s1,x3_t)
        a4_s1 = self.attention_gate(a3_s1,x4_t)

        # Decoder
        cfm_feature1 = self.decoder(a4_s1, a3_s1, a2_s1)
        prediction2 = self.out_decoder(cfm_feature1)
        
        # Continuous Attention
        p1_s2 = cim_feature*(1-out1_s2)
        a2_s2 = self.attention_gate(p1_s2,x2_t)
        a3_s2 = self.attention_gate(a2_s2,x3_t)
        a4_s2 = self.attention_gate(a3_s2,x4_t)
        
        # Decoder
        cfm_feature2 = self.decoder(a4_s2, a3_s2, a2_s2)
        prediction3 = self.out_decoder(cfm_feature2)

        prediction1 = F.interpolate(prediction1, scale_factor=8, mode='bilinear') 
        prediction2 = F.interpolate(prediction2, scale_factor=8, mode='bilinear')  
        prediction3 = F.interpolate(prediction3, scale_factor=8, mode='bilinear')  
        out = torch.cat([prediction1_8, prediction2_8, prediction3_8], dim=1)
        out = self.out(out)
        
        return torch.sigmoid(out)
