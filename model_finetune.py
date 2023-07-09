import torch
import torch.nn as nn


class WaveNet_Finetune(nn.Module):
    def __init__(self, params, backbone, backbone_size = (720, 1440), target_size = (360, 720), target_channel = 1):
        super().__init__()
        self.params = params

        # Backbone Settings
        self.backbone = backbone
        self.backbone_out_chans = params.N_out_channels

        # Target domain settings
        self.target_size = (params.target_size_h, params.target_size_w)
        self.target_channel = params.target_chans

        # Fine-tune structure
        # Downscaling to Target Size
        self.conv0 = torch.nn.Conv2d(self.backbone_out_chans, self.backbone_out_chans, kernel_size=1, stride=2, padding=0)

        # Mapping to Target Channel Dimension
        if params.target_in_chans:
            self.conv = nn.Conv2d(self.backbone_out_chans + params.target_in_chans, self.target_channel, kernel_size=3, stride=1, padding=0, bias=True)
        else:
            self.conv = nn.Conv2d(self.backbone_out_chans, self.target_channel, kernel_size=3, stride=1, padding=0, bias=True)

        self.act = nn.ReLU()
        self.norm = nn.GroupNorm(1, self.target_channel)
        
    def forward(self, x, x_init = 0):
        x = self.backbone(x)

        # Downscaling to Target Size
        x = self.conv0(self.act(x))

        # Initial Condiction for Target Variable
        if self.params.target_in_chans:
            x = torch.cat((x_init, x), dim = 1)

        # Mapping to Target Prediction
        x = self.conv(x)
        x = nn.functional.adaptive_avg_pool2d(x, (self.target_size))
        x = self.act(self.norm(x))

        return x
    
    
class BiochemicalNet_Finetune(nn.Module):
    def __init__(self, params, backbone, backbone_size = (720, 1440), target_size = (180, 360), target_channel = 8):
        super().__init__()
        self.params = params
        # Backbone Settings
        self.backbone = backbone
        self.backbone_out_chans = params.N_out_channels
        # Target domain settings
        self.target_size = (params.target_size_h, params.target_size_w)
        self.target_channel = params.target_chans
        # Fine-tune structure
        # Downscaling to Target Size
        self.conv0 = torch.nn.Conv2d(self.backbone_out_chans, self.backbone_out_chans, kernel_size=1, stride=2, padding=0)
        self.conv1 = torch.nn.Conv2d(self.backbone_out_chans, self.backbone_out_chans, kernel_size=1, stride=2, padding=0)
        # Mapping to Target Channel Dimension
        if params.target_in_chans:
            self.conv = nn.Conv2d(self.backbone_out_chans + params.target_in_chans, self.target_channel, kernel_size=3, stride=1, padding=0, bias=True)
        else:
            self.conv = nn.Conv2d(self.backbone_out_chans, self.target_channel, kernel_size=3, stride=1, padding=0, bias=True)
        self.act = nn.ReLU()
        self.norm = nn.GroupNorm(1, self.target_channel)
        
    def forward(self, x, x_init = 0):
        x = self.backbone(x)

        # Downscaling to Target Size
        x = self.conv0(self.act(x))
        x = self.conv1(self.act(x))

        # Initial Condiction for Target Variable
        if self.params.target_in_chans:
            x = torch.cat((x_init, x), dim = 1)
        # Mapping to Target Prediction
        x = self.conv(x)
        x = self.act(self.norm(x))
        
        return x

