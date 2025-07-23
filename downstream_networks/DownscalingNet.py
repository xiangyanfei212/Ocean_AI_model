import torch
import torch.nn as nn
from einops import rearrange
import utils.bicubic as bicubic
from networks.fourier_block import Block

class DownScalingNet(nn.Module):
    def __init__(self, backbone, params, target_in_chans=False):
        super().__init__()
        self.params = params

        self.backbone_decoder_pred = backbone.decoder_pred
        self.backbone = backbone
        self.backbone.decoder_pred = nn.Sequential()

        self.backbone_decoder_embed_dim = params.decoder_embed_dim

        self.target_size = (
            params.downstream_target_size_h,
            params.downstream_target_size_w,
        )
        self.target_size_h = params.downstream_target_size_h
        self.target_size_w = params.downstream_target_size_w

        self.target_in_chans = params.downstream_n_in_chans
        self.target_channel = params.downstream_n_out_chans

        self.target_hidden = 128
        self.hidden_channel = 256

        self.tconv1 = nn.ConvTranspose2d(
            self.backbone_decoder_embed_dim,
            self.backbone_decoder_embed_dim // 4,
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=1,
            output_padding=1,
        ) 
        self.conv_hidden = nn.Conv2d(
            self.target_in_chans,
            self.target_hidden,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )

        self.conv_merge = nn.Conv2d(
            self.backbone_decoder_embed_dim // 4 + self.target_hidden,
            self.hidden_channel,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )  # Merge

        # downstream Block
        self.act = nn.ReLU()

        self.norm = nn.LayerNorm(self.hidden_channel)

        self.downstream_blocks = nn.ModuleList(
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
        self.upscale3x = nn.Sequential(
            nn.Conv2d(
                self.hidden_channel,
                9 * self.hidden_channel,
                kernel_size=3,
                padding=(3 // 2),
                bias=False,
            ),
            nn.PixelShuffle(3),
        )
        self.conv_proj = nn.Conv2d(
            self.hidden_channel,
            self.target_channel,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )

    def forward(self, x, x_init):
        x = self.backbone(x)  # [B, decoder_embed_dim, h, w]
        x_backbone = self.backbone_decoder_pred(x)  # [B, out_chans, H, W]

        x = self.tconv1(x)  # [90, 180] -> [180,360]
        _, _, h, w = x_init.shape
        x = bicubic.imresize(x, sizes=(h, w))

        x_init = self.conv_hidden(x_init)

        # concat backbone'latent feature(t+1) and downstream's input (low resolution, t+1)
        x = torch.cat((x, x_init), dim=1)

        x = self.conv_merge(self.act(x))

        # downstream Block
        B, C, H, W = x.shape
        x = rearrange(x, "B C H W -> B (H W) C")
        for blk in self.downstream_blocks:
            x = blk(x)
        x = self.norm(x)
        x = rearrange(x, "B (H W) C -> B C H W", H=H)

        x = self.upscale3x(x)
        x = self.conv_proj(x)

        return x_backbone, x
