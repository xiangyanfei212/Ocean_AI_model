import torch
import torch.nn as nn
from einops import rearrange
import utils.bicubic as bicubic
import torch.nn.functional as F
from downstream_networks.blocks import ResConv

class BiochemicalNet(nn.Module):
    def __init__(self, backbone, params, target_in_chans=False):
        super().__init__()
        # Params
        self.params = params
        self.backbone_decoder_pred = backbone.decoder_pred
        self.backbone = backbone
        self.backbone.decoder_pred = nn.Sequential()

        self.backbone_decoder_embed_dim = params.decoder_embed_dim

        self.target_size_h = params.downstream_target_size_h
        self.target_size_w = params.downstream_target_size_w

        self.target_in_chans = params.downstream_n_in_chans
        self.target_channel = params.downstream_n_out_chans

        self.hidden_channel = self.target_channel * 3 + self.target_in_chans
        self.tconv = nn.ConvTranspose2d(
            self.backbone_decoder_embed_dim,
            self.target_channel * 3,
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=1,
            output_padding=1,
        )
        self.conv_merge = nn.Conv2d(
            self.target_channel * 3 + self.target_in_chans,
            self.hidden_channel,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )  # Merge

        self.conv_block = ResConv(
            self.hidden_channel, self.hidden_channel, 4, self.hidden_channel // 2
        )

        # Linear Project to Target Size
        self.conv_proj = nn.Conv2d(
            self.hidden_channel,
            self.target_channel,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.act = nn.ReLU()

    def forward(self, x, x_init):
        x = self.backbone(x)  # [B, decoder_embed_dim, h, w]
        x_backbone = self.backbone_decoder_pred(x)  # [B, out_chans, H, W]

        # Merge Input Plan a
        x = self.tconv(self.act(x))
        x = torch.cat((x, x_init), dim=1)
        x = self.conv_merge(self.act(x))

        # Downstream Block
        x = self.conv_block(x)
        x = self.conv_proj(x)

        return x_backbone, x

