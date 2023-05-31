from functools import partial
import math
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from timm.models.vision_transformer import PatchEmbed
from networks.fourier_block import Block
import torch.nn.functional as F

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class AutoencoderViT(nn.Module):
    def __init__(self, params, img_size=(720, 1440), patch_size=(16,16), in_chans=20, out_chans=20,
                 embed_dim=768, depth=4, num_heads=16, mlp_ratio=4.):
        super().__init__()

        # --------------------------------------------------------------------------
        # Common Parameter
        self.params = params
        self.img_size = (params.img_size_h, params.img_size_w)
        self.patch_size = (params.patch_size_h, params.patch_size_w)
        self.h = self.img_size[0] // self.patch_size[0]
        self.w = self.img_size[1] // self.patch_size[1]
        self.num_patches = self.h * self.w
        self.in_chans = params.in_chans
        self.out_chans = params.out_chans
        self.op_size = params.op_size
        self.hidden_dim = params.hidden_dim
        # Encoder
        self.embed_dim = params.embed_dim
        self.patch_embed = PatchEmbed(self.img_size, self.patch_size, params.in_chans, params.embed_dim)
        self.fc0_encoder = nn.Linear(self.embed_dim, self.hidden_dim)
        self.fc1_encoder = nn.Linear(self.hidden_dim, self.op_size)
        self.num_blocks = params.num_blocks
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.dpr = [x.item() for x in torch.linspace(0, params.drop_path_rate, params.depth)]
        self.norm = self.norm_layer(self.op_size)

        # --------------------------------------------------------------------------
        # Decoder
        self.fc0_decoder = nn.Linear(self.op_size, self.hidden_dim)
        self.fc1_decoder = nn.Linear(self.hidden_dim, self.embed_dim)
        self.decoder_norm = self.norm_layer(self.embed_dim)

        # --------------------------------------------------------------------------

        self.initialize_weights()
        # Koopman Parameter
        self.blocks = nn.ModuleList([
            Block(dim=self.op_size, mlp_ratio=params.mlp_ratio, drop=params.drop_rate, drop_path=self.dpr[i], 
                  norm_layer=self.norm_layer, num_blocks=self.num_blocks, sparsity_threshold=params.sparsity_threshold, 
                  hard_thresholding_fraction=params.hard_thresholding_fraction)
            for i in range(params.depth)])
        
        self.fc_norm = self.norm_layer(self.embed_dim)
        self.fc = nn.Linear(self.embed_dim, self.out_chans*self.patch_size[0]*self.patch_size[1], bias=False)
    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], (self.h, self.w), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

#         decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], (self.h, self.w), cls_token=False)
#         self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
#         torch.nn.init.normal_(self.pos_embed, std=.02)
#         torch.nn.init.normal_(self.decoder_pos_embed, std=.02)
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
#         torch.nn.init.normal_(self.cls_token, std=.02)
#         torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def no_weight_decay(self):
#         return {'pos_embed', 'decoder_pos_embed'}
        return {'pos_embed'}

    def encoder(self, x):
        B = x.shape[0]
        # embed patches
        x = self.patch_embed(x) # [1, 196, 1024]

        # add pos embed w/o cls token
        x = x + self.pos_embed # [1, 196, 1024]
        x = x.reshape(B, self.h, self.w, self.embed_dim)
        # Transformation
        x = self.fc0_encoder(x)
        x = torch.tanh(x)
        x = self.fc1_encoder(x)
        x = torch.tanh(x)
        
        return x

    def decoder(self, x):
        B = x.shape[0]
        
        x = self.fc0_decoder(x)
        x = torch.tanh(x)
        x = self.fc1_decoder(x)
        x = torch.tanh(x)       
        
        x = self.decoder_norm(x)
        
        x = self.fc(x)
        x = rearrange(
            x,
            "b h w (p1 p2 c_out) -> b c_out (h p1) (w p2)",
            p1=self.patch_size[0],
            p2=self.patch_size[1],
            h=self.img_size[0] // self.patch_size[0],
            w=self.img_size[1] // self.patch_size[1],
        )
        
        return x

    def koopman_core(self, x):
        # AFNO to Koopman
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        
        return x
    
    def forward(self, x):
        x = self.encoder(x)
        
        x_recons = self.decoder(x)

        x = self.koopman_core(x)

        x = self.decoder(x)
        
        return x, x_recons
