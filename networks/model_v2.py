import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from networks.fourier_block import Block, PatchEmbed
from networks.modules import *

# input_feature shape: [B, C, H, W]
class Encoder(nn.Module):
    def __init__(self,C_in, C_hid, N_S):
        super(Encoder,self).__init__()
        strides = stride_generator(N_S)
        self.enc = nn.Sequential(
            ConvSC(C_in, C_hid, stride=strides[0]),
            *[ConvSC(C_hid, C_hid, stride=s) for s in strides[1:]]
        )
    
    def forward(self,x):# B*4, 3, 128, 128
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1,len(self.enc)):
            latent = self.enc[i](latent)
        return latent,enc1


class Decoder(nn.Module):
    def __init__(self, C_hid, C_out, N_S):
        super(Decoder, self).__init__()
        strides = stride_generator(N_S, reverse=True)
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, stride=s, transpose=True) for s in strides[:-1]],
            ConvSC(2*C_hid, C_hid, stride=strides[-1], transpose=True)
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, hid, enc1=None):
        for i in range(0, len(self.dec) - 1):
            hid = self.dec[i](hid)
        _, _, h, w = enc1.size()
        hid = F.interpolate(hid, size=(h, w), mode='bilinear', align_corners=False)
        Y = self.dec[-1](torch.cat([hid, enc1], dim=1))
        Y = self.readout(Y)
        return Y

    
    
class Temporal_Evolution_Module(nn.Module):
    def __init__(self, channel_in, channel_hid, N_T, incep_ker=[3,5,7,11], groups=8):
        super(Temporal_Evolution_Module, self).__init__()

        self.N_T = N_T
        enc_layers = [Inception(channel_in, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups)]
        for i in range(1, N_T-1):
            enc_layers.append(Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))
        enc_layers.append(Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))

        dec_layers = [Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups)]
        for i in range(1, N_T-1):
            dec_layers.append(Inception(2*channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))
        dec_layers.append(Inception(2*channel_hid, channel_hid//2, channel_in, incep_ker= incep_ker, groups=groups))

        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, input_state):
        B, C, H, W = input_state.shape
        input_state = input_state.reshape(B, C, H, W)

        # encoder
        skips = []
        state = input_state
        for i in range(self.N_T):
            state = self.enc[i](state)
            if i < self.N_T - 1:
                skips.append(state)

        # decoder
        state = self.dec[0](state)
        for i in range(1, self.N_T):
            state = self.dec[i](torch.cat([state, skips[-i]], dim=1))

        return state




class Model_iter(nn.Module):
    def __init__(self, params, in_chans = 3, out_chans = 3, input_shape = (720,1440), embed_dim=256, hid_S=16, hid_T=256, Num_S=4, Num_T=4, incep_ker=[3,5,7,11], groups=1):
        super(Model_iter, self).__init__()
        self.in_chans = params.in_chans
        self.out_chans = params.out_chans
        self.H = params.img_size_h
        self.W = params.img_size_w
        self.Patch_Mixing = Patch_Fourier_Mixing(img_size=(self.H, self.W), patch_size=(16, 16), in_chans=self.in_chans, embed_dim=embed_dim)
        self.encoder = Encoder(self.in_chans, hid_S, Num_S)
        self.temporal_evolution = Temporal_Evolution_Module(hid_S, hid_T, Num_T, incep_ker, groups)
        self.decoder = Decoder(hid_S, self.out_chans, Num_S)


    def forward(self, input_image):

        # Encoder
        encoder_embedd, skip_embed = self.encoder(input_image)
        _, C_, H_, W_ = encoder_embedd.shape

        # Latent Space
        latent_embed = self.temporal_evolution(encoder_embedd)

        # Decoder
        predict_feature = self.decoder(latent_embed, skip_embed)

        return predict_feature
    
    
class Patch_Fourier_Mixing(nn.Module):
    def __init__(self, img_size=(720, 1440), patch_size=(16, 16), in_chans=2, embed_dim=768,
            depth=12,
            mlp_ratio=4.,
            drop_rate=0.,
            drop_path_rate=0.,
            num_blocks=16,
            sparsity_threshold=0.01,
            hard_thresholding_fraction=1.0):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 1)]
        self.filter = Block(dim=embed_dim, mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            num_blocks=self.num_blocks, sparsity_threshold=sparsity_threshold, hard_thresholding_fraction=hard_thresholding_fraction)
    
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=self.patch_size, in_chans=self.in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.h = img_size[0] // self.patch_size[0]
        self.w = img_size[1] // self.patch_size[1]
        self.head = nn.Linear(embed_dim, self.in_chans*self.patch_size[0]*self.patch_size[1], bias=False)
    def forward(self, x):
        B,C,H,W = x.shape
        # Patch Embedding & Mixing
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)        
        x = x.reshape(B, self.h, self.w, self.embed_dim)
        x = self.filter(x)
        x = self.head(x)
        x = rearrange(
            x,
            "b h w (p1 p2 c_out) -> b c_out (h p1) (w p2)",
            p1=self.patch_size[0],
            p2=self.patch_size[1],
            h=self.img_size[0] // self.patch_size[0],
            w=self.img_size[1] // self.patch_size[1],
        )
        return x

if __name__ == '__main__':

    x = torch.randn(1, 3, 720, 1440)
    model = Model_iter(input_shape=(3, 720, 1440))
    print("输入矩阵维度:", x.shape)
    output = model(x)
    print("输出矩阵维度:", output.shape)
