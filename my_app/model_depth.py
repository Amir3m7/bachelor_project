import math
from dataclasses import dataclass
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utils
# -----------------------------
def conv_bn_act(in_ch, out_ch, k=3, s=1, p=1, groups=1, act=True):
    layers = [
        nn.Conv2d(in_ch, out_ch, k, s, p, groups=groups, bias=False),
        nn.BatchNorm2d(out_ch),
    ]
    if act:
        layers.append(nn.ReLU6(inplace=True))
    return nn.Sequential(*layers)


# -----------------------------
# MobileNetV2-like Inverted Residual (lightweight CNN)
# -----------------------------
class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expand_ratio):
        super().__init__()
        assert stride in [1, 2]
        hidden = int(round(in_ch * expand_ratio))
        self.use_res = (stride == 1 and in_ch == out_ch)

        layers = []
        if expand_ratio != 1:
            layers.append(conv_bn_act(in_ch, hidden, k=1, s=1, p=0))

        layers.append(conv_bn_act(hidden, hidden, k=3, s=stride, p=1, groups=hidden))
        layers.append(nn.Conv2d(hidden, out_ch, 1, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(out_ch))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        y = self.block(x)
        return y + x if self.use_res else y


class TSTEncoder3(nn.Module):
    """
    Outputs ONLY 3 scales like TST paper's {F1,F2,F3}:
      f1: /8
      f2: /16
      f3: /32
    """
    def __init__(self, c_f1=64, c_f2=128, c_f3=160, width_mult=1.0):
        super().__init__()
        stem = int(32 * width_mult)
        self.stem = conv_bn_act(3, stem, k=3, s=2, p=1)  # /2

        # internal /4 stage (not exposed)
        c4 = int(64 * width_mult)
        self.s1 = nn.Sequential(
            InvertedResidual(stem, c4, stride=2, expand_ratio=1),  # /4
            InvertedResidual(c4, c4, stride=1, expand_ratio=6),
        )
        # /8 -> F1
        self.s2 = nn.Sequential(
            InvertedResidual(c4, c_f1, stride=2, expand_ratio=6),  # /8
            InvertedResidual(c_f1, c_f1, stride=1, expand_ratio=6),
        )
        # /16 -> F2
        self.s3 = nn.Sequential(
            InvertedResidual(c_f1, c_f2, stride=2, expand_ratio=6),  # /16
            InvertedResidual(c_f2, c_f2, stride=1, expand_ratio=6),
        )
        # /32 -> F3
        self.s4 = nn.Sequential(
            InvertedResidual(c_f2, c_f3, stride=2, expand_ratio=6),  # /32
            InvertedResidual(c_f3, c_f3, stride=1, expand_ratio=6),
        )

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)   # /2
        x = self.s1(x)     # /4 (hidden)
        f1 = self.s2(x)    # /8
        f2 = self.s3(f1)   # /16
        f3 = self.s4(f2)   # /32
        return f1, f2, f3


# -----------------------------
# Token-Sharing Transformer (Connection Module)
# FIX: K,V are projected from global channels (can differ from local channels)
# -----------------------------
class CrossAttention(nn.Module):
    def __init__(self, c_q: int, c_kv: int, heads: int, dim_qk=16, dim_v=16):
        super().__init__()
        self.heads = heads
        self.dim_qk = dim_qk
        self.dim_v = dim_v

        # Q from local
        self.q = nn.Conv2d(c_q, heads * dim_qk, 1, bias=False)
        self.q_bn = nn.BatchNorm2d(heads * dim_qk)

        # K,V from global (may have different channels)
        self.k = nn.Conv2d(c_kv, heads * dim_qk, 1, bias=False)
        self.v = nn.Conv2d(c_kv, heads * dim_v,  1, bias=False)
        self.k_bn = nn.BatchNorm2d(heads * dim_qk)
        self.v_bn = nn.BatchNorm2d(heads * dim_v)

        # project back to local channels
        self.proj = nn.Conv2d(heads * dim_v, c_q, 1, bias=False)
        self.proj_bn = nn.BatchNorm2d(c_q)

    def forward(self, x_local, x_global):
        B, _, H, W = x_local.shape
        L = H * W

        q = self.q_bn(self.q(x_local)).view(B, self.heads, self.dim_qk, L)
        k = self.k_bn(self.k(x_global)).view(B, self.heads, self.dim_qk, L)
        v = self.v_bn(self.v(x_global)).view(B, self.heads, self.dim_v,  L)

        attn = torch.matmul(q.permute(0, 1, 3, 2), k) / math.sqrt(self.dim_qk)  # (B,h,L,L)
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v.permute(0, 1, 3, 2))  # (B,h,L,dim_v)
        out = out.permute(0, 1, 3, 2).contiguous().view(B, self.heads * self.dim_v, H, W)

        out = self.proj_bn(self.proj(out))
        return out


class FFNLite(nn.Module):
    # 1x1 -> DW 3x3 -> 1x1 (light FFN)
    def __init__(self, c: int, expansion=4):
        super().__init__()
        hidden = max(c, c * expansion)
        self.pw1 = conv_bn_act(c, hidden, k=1, s=1, p=0)
        self.dw  = conv_bn_act(hidden, hidden, k=3, s=1, p=1, groups=hidden)
        self.pw2 = nn.Sequential(
            nn.Conv2d(hidden, c, 1, bias=False),
            nn.BatchNorm2d(c),
        )

    def forward(self, x):
        return self.pw2(self.dw(self.pw1(x)))


class TSTBlock(nn.Module):
    def __init__(self, c_local: int, c_global: int, heads: int, dim_qk=16, dim_v=16):
        super().__init__()
        self.attn = CrossAttention(c_q=c_local, c_kv=c_global, heads=heads, dim_qk=dim_qk, dim_v=dim_v)
        self.ffn  = FFNLite(c_local)

    def forward(self, x_local_ds, x_global):
        x = x_local_ds + self.attn(x_local_ds, x_global)
        x = x + self.ffn(x)
        return x


class TokenSharingConnection(nn.Module):
    """
    Inputs:
      f1 (/8), f2 (/16), f3 (/32)
    Global token is f3.
    """
    def __init__(self, c_f1: int, c_f2: int, c_f3: int):
        super().__init__()
        self.b1 = TSTBlock(c_local=c_f1, c_global=c_f3, heads=2, dim_qk=16, dim_v=16)
        self.b2 = TSTBlock(c_local=c_f2, c_global=c_f3, heads=4, dim_qk=16, dim_v=16)
        self.b3 = TSTBlock(c_local=c_f3, c_global=c_f3, heads=5, dim_qk=16, dim_v=32)

    def forward(self, f1, f2, f3):
        Hg, Wg = f3.shape[-2:]
        g = f3

        f1_ds = F.adaptive_avg_pool2d(f1, (Hg, Wg))
        f2_ds = F.adaptive_avg_pool2d(f2, (Hg, Wg))

        o1 = self.b1(f1_ds, g)
        o2 = self.b2(f2_ds, g)
        o3 = self.b3(f3,    g)

        o1_up = F.interpolate(o1, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        o2_up = F.interpolate(o2, size=f2.shape[-2:], mode="bilinear", align_corners=False)

        return f1 + o1_up, f2 + o2_up, f3 + o3


# -----------------------------
# GLPDepth SFF + 3-scale Decoder
# -----------------------------
class SelectiveFeatureFusion(nn.Module):
    def __init__(self, c=64):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(2 * c, c, 3, 1, 1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(c, c // 2, 3, 1, 1),
            nn.BatchNorm2d(c // 2),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Conv2d(c // 2, 2, 3, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_local, x_global):
        x = torch.cat([x_local, x_global], dim=1)
        attn = self.sigmoid(self.conv3(self.conv2(self.conv1(x))))
        return x_local * attn[:, 0:1] + x_global * attn[:, 1:2]


class GLPDecoder3(nn.Module):
    """
    Decoder adapted to 3 scales:
      f3: /32 -> 64 -> up to /16
      fuse with f2 (/16) -> up to /8
      fuse with f1 (/8)  -> upsample to full res
    """
    def __init__(self, c_f1: int, c_f2: int, c_f3: int, out_c: int = 64):
        super().__init__()
        self.bot   = nn.Conv2d(c_f3, out_c, 1)
        self.skip2 = nn.Conv2d(c_f2, out_c, 1)
        self.skip1 = nn.Conv2d(c_f1, out_c, 1)

        self.fuse2 = SelectiveFeatureFusion(out_c)  # /16
        self.fuse1 = SelectiveFeatureFusion(out_c)  # /8

    def forward(self, f1, f2, f3):
        out = F.interpolate(self.bot(f3), scale_factor=2, mode="bilinear", align_corners=False)  # /32->/16
        out = self.fuse2(self.skip2(f2), out)
        out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=False)          # /16->/8

        out = self.fuse1(self.skip1(f1), out)

        # /8 -> /4 -> /2 -> /1
        out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=False)
        out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=False)
        out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=False)
        return out


# -----------------------------
# Full TST Depth Model (3-scale)
# -----------------------------
@dataclass
class TSTConfig:
    c_f1: int = 64   # /8
    c_f2: int = 128  # /16
    c_f3: int = 160  # /32
    dec_c: int = 64
    max_depth: float = 10.0


class TSTDepthModel3(nn.Module):
    def __init__(self, cfg: TSTConfig = TSTConfig()):
        super().__init__()
        self.max_depth = cfg.max_depth

        self.encoder = TSTEncoder3(cfg.c_f1, cfg.c_f2, cfg.c_f3)
        self.connection = TokenSharingConnection(cfg.c_f1, cfg.c_f2, cfg.c_f3)

        # optional conv on last feature before decoder
        self.last_conv = conv_bn_act(cfg.c_f3, cfg.c_f3, k=3, s=1, p=1)

        self.decoder = GLPDecoder3(cfg.c_f1, cfg.c_f2, cfg.c_f3, out_c=cfg.dec_c)

        self.depth_head = nn.Sequential(
            nn.Conv2d(cfg.dec_c, cfg.dec_c, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(cfg.dec_c, 1, 3, 1, 1),
        )

    def forward(self, x) -> Dict[str, torch.Tensor]:
        f1, f2, f3 = self.encoder(x)                 # /8, /16, /32
        f1p, f2p, f3p = self.connection(f1, f2, f3)  # token sharing
        f3p = self.last_conv(f3p)

        out = self.decoder(f1p, f2p, f3p)
        depth = torch.sigmoid(self.depth_head(out)) * self.max_depth
        return {"pred_d": depth}


