import torch
import torch.nn as nn
from einops import rearrange
import utils.bicubic as bicubic
from networks.fourier_block_Masked_AE_Ocean import Block


class WaveNet(nn.Module):
    def __init__(self, backbone, params, target_in_chans=False):
        super().__init__()
        self.params = params

        self.backbone_decoder_pred = backbone.decoder_pred
        self.backbone = backbone
        self.backbone.decoder_pred = nn.Sequential()

        self.backbone_decoder_embed_dim = params.decoder_embed_dim

        # Target domain settings
        self.target_size = (
            params.finetune_target_size_h,
            params.finetune_target_size_w,
        )
        self.target_size_h = params.finetune_target_size_h
        self.target_size_w = params.finetune_target_size_w
        self.target_in_chans = params.finetune_n_in_chans
        self.target_channel = params.finetune_n_out_chans

        self.hidden_channel = self.backbone_decoder_embed_dim * 2
        self.conv1 = nn.Conv2d(
            in_channels=self.target_in_chans,
            out_channels=self.backbone_decoder_embed_dim // 8,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=self.backbone_decoder_embed_dim // 8,
            out_channels=self.backbone_decoder_embed_dim,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.conv_merge = nn.Conv2d(
            self.backbone_decoder_embed_dim * 2,
            self.hidden_channel,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )  # Merge

        # Conv Block
        self.act = nn.ReLU()
        self.norm = nn.LayerNorm(self.hidden_channel)

        self.finetune_blocks = nn.ModuleList(
            [
                Block(
                    dim=self.hidden_channel,
                    mlp_ratio=4.0,
                    norm_layer=nn.LayerNorm,
                    num_blocks=8,
                    sparsity_threshold=0.01,
                    hard_thresholding_fraction=1.0,
                )
                for i in range(2)
            ]
        )

        # Linear Project to Target Size
        self.tconv_proj1 = nn.ConvTranspose2d(
            self.hidden_channel,
            self.hidden_channel // 8,
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=1,
            output_padding=1,
        )
        self.tconv_proj2 = nn.ConvTranspose2d(
            self.hidden_channel // 8,
            self.target_channel,
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=1,
            output_padding=1,
        )

    def forward(self, x, x_init):
        x = self.backbone(x)  # [B, decoder_embed_dim, h, w]
        x_backbone = self.backbone_decoder_pred(x)  # [B, out_chans, H, W]
        B, c, h, w = x.shape
        # Merge Input
        # x_init [360,720]
        x_init = self.conv1(self.act(x_init))
        x_init = self.conv2(self.act(x_init))  # [90, 180]
        x = torch.cat((x, x_init), dim=1)  # [1024, 90, 180]
        x = self.conv_merge(self.act(x))  # [1024, 90, 180]

        # Downstream Block
        x = rearrange(x, "B C H W -> B (H W) C")
        for blk in self.finetune_blocks:
            x = blk(x)
        x = self.norm(x)
        x = rearrange(x, "B (H W) C -> B C H W", H=h)

        # Mapping to Target Prediction
        x = self.tconv_proj1(x)  # [1024 // 8, 180, 360]
        x = self.tconv_proj2(x)  # [target_channel, 360, 720]
        
        return x_backbone, x
