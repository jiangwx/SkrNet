import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .region_loss import * 

class ReorgLayer(nn.Module):
    def __init__(self, stride=2):
        super(ReorgLayer, self).__init__()
        self.stride = stride
    def forward(self, x):
        stride = self.stride
        assert(x.data.dim() == 4)
        B, C, H, W = x.shape
        assert(H % stride == 0)
        assert(W % stride == 0)
        ws = stride
        hs = stride
        x = x.view([B, C, H//hs, hs, W//ws, ws]).transpose(3, 4).contiguous()
        x = x.view([B, C, H//hs*W//ws, hs*ws]).transpose(2, 3).contiguous()
        x = x.view([B, C, hs*ws, H//hs, W//ws]).transpose(1, 2).contiguous()
        x = x.view([B, hs*ws*C, H//hs, W//ws])
        return x

class SkrNet(nn.Module):
    def __init__(self, num_class = 102, detection = True):
        super(SkrNet, self).__init__()
        self.num_class = int(num_class)
        self.detection = detection
        self.reorg = ReorgLayer(stride=2)
        self.fc = nn.Linear(1024, num_class)
        self.bbox = nn.Conv2d(1024, 10, 1, 1,bias=False)
        self.loss = RegionLoss()
        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU6(inplace=True),
                
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),
            )

        self.model_p1 = nn.Sequential(
            conv_dw( 3, 96, 1),    #dw1
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_dw( 96,  192, 1),   #dw2
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_dw( 192, 384, 1),   #dw3
        )    
        self.model_p2 = nn.Sequential(    
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_dw(384, 768, 1),   #dw4
            conv_dw(768, 1024, 1),   #dw5
        )
        self.model_p3 = nn.Sequential(  #cat dw3(ch:192 -> 768) and dw5(ch:512)
            conv_dw(2560, 1024, 1),
        )
        self.initialize_weights()
        # self.loss = RegionLoss()
    def forward(self, x):
        x_p1 = self.model_p1(x)
        x_p1_reorg = self.reorg(x_p1)
        x_p2 = self.model_p2(x_p1)
        x_p3_in = torch.cat([x_p1_reorg, x_p2], 1)
        x = self.model_p3(x_p3_in)
        if(self.detection):
            x = self.bbox(x)
        else:
            x = nn.AvgPool2d(x.size(2))(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)  