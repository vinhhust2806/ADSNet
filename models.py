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

class AttentionBlock(nn.Module):
    def __init__(self, in_channels=32):
        super(AttentionBlock, self).__init__()
        self.in_channels = in_channels
        
        self.bn_g = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv_g = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn_x = nn.BatchNorm2d(in_channels)
        self.conv_x = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv_gc = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        
    def forward(self, g, x):
        g_conv = self.bn_g(g)
        g_conv = self.relu(g_conv)
        g_conv = self.conv_g(g_conv)
        g_pool = self.pool(g_conv)
        
        x_conv = self.bn_x(x)
        x_conv = self.relu(x_conv)
        x_conv = self.conv_x(x_conv)
        
        gc_sum = g_pool + x_conv
        
        gc_conv = self.bn_g(gc_sum)  
        gc_conv = self.relu(gc_conv)
        gc_conv = self.conv_gc(gc_conv)
        
        gc_mul = gc_conv * x
        
        return gc_mul

class Model(nn.Module):
    def __init__(self, channel=32):
        super(Model, self).__init__()

        self.encoder = timm.create_model('caformer_s18.sail_in22k_ft_in1k_384', pretrained=True, features_only=True) 

        self.Translayer2_0 = BasicConv2d(64, channel, 1)
        self.Translayer2_1 = BasicConv2d(128, channel, 1)
        self.Translayer3_1 = BasicConv2d(320, channel, 1)
        self.Translayer4_1 = BasicConv2d(512, channel, 1)

        self.decoder = Decoder(channel)
        self.ca = ChannelAttention(64)
        self.sa = SpatialAttention()
        
        self.out_CFM = nn.Conv2d(channel, 1, 1)
        self.attention_block = AttentionBlock()
        self.out = nn.Conv2d(3,1,1)

    def forward(self, x):

        # backbone
        encoder = self.backbone(x)
        x1 = encoder[0]
        x2 = encoder[1]
        x3 = encoder[2]
        x4 = encoder[3]
        
        # CIM
        x1 = self.ca(x1) * x1 # channel attention
        cim_feature = self.sa(x1) * x1 # spatial attention

        x2_t = self.Translayer2_1(x2)  
        x3_t = self.Translayer3_1(x3)  
        x4_t = self.Translayer4_1(x4) 
      
        # Decoder
        cfm_feature = self.decoder(x4_t, x3_t, x2_t)
        
        out1_resized = nn.functional.interpolate(cfm_feature, size=(64,64), mode='bilinear', align_corners=False)
        out1_sigmoid = torch.sigmoid(out1_resized)
        threshold = 0.00001
        out1_s2 = (out1_sigmoid > threshold).float()

        # Continuous Attention
        p1_s1 = cim_feature*(1-out1_sigmoid)
        a2_s1 = self.attention_block(p1_s1,x2_t)
        a3_s1 = self.attention_block(a2_s1,x3_t)
        a4_s1 = self.attention_block(a3_s1,x4_t)

        # Decoder
        cfm_feature1 = self.decoder(a4_s1, a3_s1, a2_s1)
        # Continuous Attention
        p1_s2 = cim_feature*(1-out1_s2)
        a2_s2 = self.attention_block(p1_s2,x2_t)
        a3_s2 = self.attention_block(a2_s2,x3_t)
        a4_s2 = self.attention_block(a3_s2,x4_t)
        
        # Decoder
        cfm_feature2 = self.decoder(a4_s2, a3_s2, a2_s2)

        prediction1 = self.out_CFM(cfm_feature)
        prediction2 = self.out_CFM(cfm_feature1)
        prediction3 = self.out_CFM(cfm_feature2)

        prediction1 = F.interpolate(prediction1, scale_factor=8, mode='bilinear') 
        prediction2 = F.interpolate(prediction2, scale_factor=8, mode='bilinear')  
        prediction3 = F.interpolate(prediction3, scale_factor=8, mode='bilinear')  
        out = torch.cat([prediction1_8, prediction2_8, prediction3_8], dim=1)
        out = self.out(out)
        return torch.sigmoid(out)
