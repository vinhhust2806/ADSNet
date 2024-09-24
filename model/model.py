from model.module import *

class Model(nn.Module):
    def __init__(self, args, channel=32):
        super(Model, self).__init__()
        
        self.args = args
        self.encoder = timm.create_model('tf_efficientnetv2_s.in21k_ft_in1k', pretrained=True, features_only=True) 
        
        self.Translayer2_1 = PASPP(64, channel)
        self.Translayer3_1 = PASPP(160, channel)
        self.Translayer4_1 = PASPP(256, channel)
        
        self.decoder = Decoder(channel)
        self.ca = ChannelAttention(48)
        self.sa = SpatialAttention()
        
        self.attention_gate = AttentionGate(48, channel, channel)
        self.attention_gate1 = AttentionGate(channel, channel, channel)

        self.conv = nn.Conv2d(3*channel, channel, 3, 1, 1)
        self.conv1 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.mask = nn.Conv2d(channel, 1, 1)
        self.out = nn.Conv2d(3,1,1)
    def forward(self, x):

        # Encoder
        encoder = self.encoder(x)
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
      
        # Early Global Map
        cfm_feature = self.decoder(x4_t, x3_t, x2_t)
        prediction1 = self.mask(cfm_feature)
        
        out1_resized = F.interpolate(prediction1, size=(self.args.image_size//4, self.args.image_size//4), mode='bilinear', align_corners=False)
        out1_sigmoid = torch.sigmoid(out1_resized)
    
        out1_s2 = (out1_sigmoid > self.args.threshold).float()

        # Continuous Attention
        p1_s1 = cim_feature*(1-out1_sigmoid)
        a2_s1 = self.attention_gate(p1_s1,x2_t)
        a3_s1 = self.attention_gate1(a2_s1,x3_t)
        a4_s1 = self.attention_gate1(a3_s1,x4_t)
        
        a3_s1 = F.interpolate(a3_s1, size=(self.args.image_size//8, self.args.image_size//8), mode = 'bilinear')
        a4_s1 = F.interpolate(a4_s1, size=(self.args.image_size//8, self.args.image_size//8), mode = 'bilinear')

        # BS
        cfm_feature1 = self.conv1(self.conv(torch.cat([a4_s1, a3_s1, a2_s1], dim=1)))
        prediction2 = self.mask(cfm_feature1)
        
        # Continuous Attention
        p1_s2 = cim_feature*(1-out1_s2)
        a2_s2 = self.attention_gate(p1_s2,x2_t)
        a3_s2 = self.attention_gate1(a2_s2,x3_t)
        a4_s2 = self.attention_gate1(a3_s2,x4_t)
        
        a3_s2 = F.interpolate(a3_s2, size=(self.args.image_size//8, self.args.image_size//8), mode = 'bilinear')
        a4_s2 = F.interpolate(a4_s2, size=(self.args.image_size//8, self.args.image_size//8), mode = 'bilinear')
        
        # OS
        cfm_feature2 = self.conv1(self.conv(torch.cat([a4_s2, a3_s2, a2_s2], dim=1)))
        prediction3 = self.mask(cfm_feature2)

        prediction1 = F.interpolate(prediction1, size=(self.args.image_size, self.args.image_size), mode='bilinear') 
        prediction2 = F.interpolate(prediction2, size=(self.args.image_size, self.args.image_size), mode='bilinear')  
        prediction3 = F.interpolate(prediction3, size=(self.args.image_size, self.args.image_size), mode='bilinear')  
        out = torch.cat([prediction1, prediction2, prediction3], dim=1)
        out = self.out(out)
        
        return torch.sigmoid(out)
