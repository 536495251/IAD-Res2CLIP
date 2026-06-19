import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ResidualAdapter(nn.Module):
    '''Lightweight bottleneck adapter for global/text residuals (A_t, A^res_t, A^res_{v,cls}).'''
    def __init__(self, embed_dim=768, reduction=4):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, embed_dim // reduction, bias=False)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(embed_dim // reduction, embed_dim, bias=False)
        nn.init.zeros_(self.fc2.weight)

    def forward(self, x):
        return x + self.fc2(self.act(self.fc1(self.norm(x))))


class MultiScaleConvAdapter(nn.Module):
    '''Multi-scale conv adapter for local visual residuals (A_{v,l}, A^res_{v,l}).'''
    def __init__(self, embed_dim=768, reduction=4):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)

        down_dim = embed_dim // reduction
        self.down = nn.Linear(embed_dim, down_dim, bias=False)
        self.act_down = nn.ReLU()

        self.conv_fine   = nn.Conv2d(down_dim, down_dim, kernel_size=3, padding=1,
                                     groups=down_dim, bias=False)
        self.conv_coarse = nn.Conv2d(down_dim, down_dim, kernel_size=3, padding=2,
                                     dilation=2, groups=down_dim, bias=False)

        self.internal_norm = nn.LayerNorm(down_dim * 3)
        self.blend    = nn.Linear(down_dim * 3, down_dim, bias=False)
        self.act_fuse = nn.ReLU()

        self.up = nn.Linear(down_dim, embed_dim, bias=False)
        nn.init.zeros_(self.up.weight)
        nn.init.kaiming_normal_(self.conv_fine.weight,   mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv_coarse.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        B, L, D = x.shape
        H = int(math.sqrt(L))

        x_norm = self.norm(x)
        x_down = self.act_down(self.down(x_norm))
        x_2d   = x_down.transpose(1, 2).view(B, -1, H, H)

        feat_fine   = self.conv_fine(x_2d)
        feat_coarse = self.conv_coarse(x_2d)

        feat_multi = torch.cat([x_2d, feat_fine, feat_coarse], dim=1).flatten(2).transpose(1, 2)
        feat_fuse  = self.act_fuse(self.blend(self.internal_norm(feat_multi)))

        return x + self.up(feat_fuse)
