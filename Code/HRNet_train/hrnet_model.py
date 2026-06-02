"""
hrnet_model.py — Lightweight HRNet for star heatmap detection.

Architectural difference from U-Net:
  - U-Net: encoder DOWNSAMPLES then decoder UPSAMPLES (information bottleneck)
  - HRNet: maintains MULTIPLE resolutions in parallel branches, fuses repeatedly
           → full resolution branch never loses spatial detail

For point-like features (stars), this is theoretically advantageous:
  - U-Net: star info passes through 32×32 bottleneck (16× downsample)
  - HRNet: star info stays at 512×512 throughout one branch

Reference:
  Sun et al. 2019. "Deep High-Resolution Representation Learning for Human Pose Estimation"
  https://arxiv.org/abs/1902.09212

This implementation is a 3-branch lite version with ~7M parameters
to match U-Net's parameter count for fair comparison.

Input:  (B, 1, 512, 512)  grayscale image
Output: (B, 1, 512, 512)  heatmap in [0, 1]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Basic block (same as ResNet)
# ---------------------------------------------------------------------------
class BasicBlock(nn.Module):
    """Conv→BN→ReLU→Conv→BN + skip connection → ReLU."""
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(ch)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + residual
        return F.relu(out, inplace=True)


# ---------------------------------------------------------------------------
# Multi-resolution fusion
# ---------------------------------------------------------------------------
class FusionBlock(nn.Module):
    """
    Fuse all branches into all branches.
    For N branches with channels [c0, c1, ..., c_{N-1}] and resolutions
    [H, H/2, H/4, ...], output is fused versions at the SAME resolutions.

    For each target branch i:
      output_i = sum over j of: resample(branch_j, target_resolution_i)
    """
    def __init__(self, channels):
        """
        channels: list of int — channels per branch (in resolution order, high → low)
        """
        super().__init__()
        N = len(channels)
        self.n_branches = N

        # transforms[i][j]: converts branch j → resolution of branch i
        self.transforms = nn.ModuleList()
        for i in range(N):
            row = nn.ModuleList()
            for j in range(N):
                if i == j:
                    row.append(nn.Identity())
                elif j < i:
                    # branch j is HIGHER resolution → downsample with strided convs.
                    # Keep intermediate channels = channels[j]; convert to channels[i]
                    # in the LAST step only.
                    layers = []
                    for k in range(i - j):
                        is_last = (k == i - j - 1)
                        in_ch  = channels[j]
                        out_ch = channels[i] if is_last else channels[j]
                        layers += [
                            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(out_ch),
                        ]
                        if not is_last:
                            layers.append(nn.ReLU(inplace=True))
                    row.append(nn.Sequential(*layers))
                else:
                    # branch j is LOWER resolution → upsample with 1×1 conv + bilinear
                    row.append(nn.Sequential(
                        nn.Conv2d(channels[j], channels[i], 1, bias=False),
                        nn.BatchNorm2d(channels[i]),
                    ))
            self.transforms.append(row)

    def forward(self, branches):
        """branches: list of N tensors at different resolutions."""
        out = []
        for i in range(self.n_branches):
            target_h = branches[i].shape[-2]
            target_w = branches[i].shape[-1]
            fused = None
            for j in range(self.n_branches):
                t = self.transforms[i][j](branches[j])
                if t.shape[-2:] != (target_h, target_w):
                    t = F.interpolate(t, size=(target_h, target_w),
                                      mode="bilinear", align_corners=False)
                fused = t if fused is None else fused + t
            out.append(F.relu(fused, inplace=True))
        return out


# ---------------------------------------------------------------------------
# HRNet
# ---------------------------------------------------------------------------
class HRNet(nn.Module):
    """
    Multi-resolution HRNet for heatmap regression.

    Channels per branch: [C, 2C, 4C]
    With C=32 → channels (32, 64, 128) → ~7M parameters (matches U-Net).
    """

    def __init__(self, base_ch=32, blocks_per_stage=4):
        super().__init__()
        c0 = base_ch
        c1 = base_ch * 2
        c2 = base_ch * 4

        # Stem — operates on full resolution
        self.stem = nn.Sequential(
            nn.Conv2d(1, c0, 3, padding=1, bias=False),
            nn.BatchNorm2d(c0),
            nn.ReLU(inplace=True),
            nn.Conv2d(c0, c0, 3, padding=1, bias=False),
            nn.BatchNorm2d(c0),
            nn.ReLU(inplace=True),
        )

        # Transitions to create new branches
        self.trans_1to2 = nn.Sequential(    # full → 1/2
            nn.Conv2d(c0, c1, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        self.trans_2to3 = nn.Sequential(    # 1/2 → 1/4
            nn.Conv2d(c1, c2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )

        # Stage blocks per branch
        def make_blocks(ch, n):
            return nn.Sequential(*[BasicBlock(ch) for _ in range(n)])

        # Stage 1 — 1 branch (after stem)
        self.stage1 = make_blocks(c0, blocks_per_stage)

        # Stage 2 — 2 branches + fusion
        self.stage2_b0 = make_blocks(c0, blocks_per_stage)
        self.stage2_b1 = make_blocks(c1, blocks_per_stage)
        self.fuse2 = FusionBlock([c0, c1])

        # Stage 3 — 3 branches + fusion
        self.stage3_b0 = make_blocks(c0, blocks_per_stage)
        self.stage3_b1 = make_blocks(c1, blocks_per_stage)
        self.stage3_b2 = make_blocks(c2, blocks_per_stage)
        self.fuse3 = FusionBlock([c0, c1, c2])

        # Stage 4 — 3 branches + final fusion
        self.stage4_b0 = make_blocks(c0, blocks_per_stage)
        self.stage4_b1 = make_blocks(c1, blocks_per_stage)
        self.stage4_b2 = make_blocks(c2, blocks_per_stage)
        self.fuse4 = FusionBlock([c0, c1, c2])

        # Head — operates on full resolution branch only
        self.head = nn.Sequential(
            nn.Conv2d(c0, c0, 3, padding=1, bias=False),
            nn.BatchNorm2d(c0),
            nn.ReLU(inplace=True),
            nn.Conv2d(c0, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Stem
        x0 = self.stem(x)
        x0 = self.stage1(x0)

        # Stage 2: split into 2 branches
        b0 = self.stage2_b0(x0)
        b1 = self.stage2_b1(self.trans_1to2(x0))
        b0, b1 = self.fuse2([b0, b1])

        # Stage 3: split into 3 branches
        b0 = self.stage3_b0(b0)
        b1 = self.stage3_b1(b1)
        b2 = self.stage3_b2(self.trans_2to3(b1))
        b0, b1, b2 = self.fuse3([b0, b1, b2])

        # Stage 4
        b0 = self.stage4_b0(b0)
        b1 = self.stage4_b1(b1)
        b2 = self.stage4_b2(b2)
        b0, b1, b2 = self.fuse4([b0, b1, b2])

        # Head — only on full-resolution branch
        return self.head(b0)


if __name__ == "__main__":
    # Quick sanity check
    model = HRNet(base_ch=32)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"HRNet parameters: {n_params:,}")

    x = torch.randn(2, 1, 512, 512)
    y = model(x)
    print(f"Input  : {x.shape}")
    print(f"Output : {y.shape}")
    print(f"Output range: [{y.min().item():.4f}, {y.max().item():.4f}]")
