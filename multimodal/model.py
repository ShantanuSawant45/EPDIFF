"""
multimodal/model.py
===================
MultiModal-UNet: a 3-D U-Net backbone with pluggable multimodal fusion
and an optional classification head for tumour grade prediction.

Architecture overview:
    - Independent per-modality encoders (shared or separate weights)
    - Fusion at every encoder stage with a chosen fusion strategy
    - Single shared decoder with skip connections from fused features
    - Segmentation head: 4-class (background, necrotic, edema, enhancing)
    - Classification head: tumour grade (LGG / HGG / no-tumour)
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fusion import build_fusion, SEBlock3d


# ──────────────────────────────────────────────────────────────────────
#  Basic building blocks
# ──────────────────────────────────────────────────────────────────────

class ConvBlock3d(nn.Module):
    """Two stacked Conv3d → InstanceNorm3d → LeakyReLU residual block."""

    def __init__(self, in_ch: int, out_ch: int,
                 stride: int = 1, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride,
                               padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_ch, affine=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_ch, affine=True)
        self.act   = nn.LeakyReLU(0.01, inplace=True)
        self.drop  = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

        # Residual projection if dims change
        self.residual = (
            nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False)
            if (in_ch != out_ch or stride != 1) else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.drop(x)
        x = self.norm2(self.conv2(x))
        return self.act(x + res)


class UpBlock3d(nn.Module):
    """Trilinear upsample → conv → ConvBlock."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int,
                 dropout: float = 0.0):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="trilinear",
                                 align_corners=False)
        self.conv = ConvBlock3d(in_ch + skip_ch, out_ch, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle spatial mismatch from odd dims
        diff = [skip.shape[i] - x.shape[i] for i in range(2, 5)]
        x = F.pad(x, [0, diff[2], 0, diff[1], 0, diff[0]])
        return self.conv(torch.cat([skip, x], dim=1))


# ──────────────────────────────────────────────────────────────────────
#  Per-modality encoder
# ──────────────────────────────────────────────────────────────────────

class ModalityEncoder(nn.Module):
    """
    Encodes a single MRI modality (1-channel input) into a pyramid
    of feature maps at 4 resolution scales.

    Returns:
        list of feature tensors [s0, s1, s2, s3, bottleneck]
        where each sN is 2× smaller than the previous.
    """

    def __init__(self, base_ch: int = 32, dropout: float = 0.1):
        super().__init__()
        c = base_ch
        self.enc0 = ConvBlock3d(1, c,         dropout=dropout)   # full res
        self.enc1 = ConvBlock3d(c, c * 2,  stride=2, dropout=dropout)
        self.enc2 = ConvBlock3d(c * 2, c * 4, stride=2, dropout=dropout)
        self.enc3 = ConvBlock3d(c * 4, c * 8, stride=2, dropout=dropout)
        self.bot  = ConvBlock3d(c * 8, c * 16, stride=2, dropout=dropout)

        self.channels = [c, c*2, c*4, c*8, c*16]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # x: (B, 1, D, H, W)
        s0 = self.enc0(x)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        bt = self.bot(s3)
        return [s0, s1, s2, s3, bt]


# ──────────────────────────────────────────────────────────────────────
#  Multimodal U-Net
# ──────────────────────────────────────────────────────────────────────

class MultimodalUNet(nn.Module):
    """
    3-D U-Net with multimodal feature fusion at every encoder level.

    Parameters
    ----------
    num_modalities    : number of input MRI channels (typically 4 for BraTS)
    num_seg_classes   : segmentation output classes (default 4)
    num_grade_classes : tumour grade classes (0=none,1=LGG,2=HGG) or 0 to disable
    base_channels     : base feature width (doubles at each level)
    fusion_type       : fusion strategy ('concat'|'se_modality'|'cross_modal'|
                        'moe'|'robust')
    shared_encoder    : if True all modalities share encoder weights
    dropout           : spatial dropout rate in conv blocks
    """

    def __init__(
        self,
        num_modalities: int = 4,
        num_seg_classes: int = 4,
        num_grade_classes: int = 3,
        base_channels: int = 32,
        fusion_type: str = "se_modality",
        shared_encoder: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.M = num_modalities
        self.num_seg  = num_seg_classes
        self.num_grade = num_grade_classes
        self.shared_encoder = shared_encoder

        # ---- Encoders ----
        if shared_encoder:
            enc = ModalityEncoder(base_channels, dropout)
            self.encoders = nn.ModuleList([enc] * num_modalities)
        else:
            self.encoders = nn.ModuleList([
                ModalityEncoder(base_channels, dropout)
                for _ in range(num_modalities)
            ])

        enc_chs = self.encoders[0].channels  # [c, 2c, 4c, 8c, 16c]

        # ---- Fusion modules at each scale ----
        self.fusions = nn.ModuleList()
        for in_ch in enc_chs:
            self.fusions.append(
                build_fusion(
                    fusion_type,
                    num_modalities=num_modalities,
                    in_channels=in_ch,
                    out_channels=in_ch,          # keep channel count
                )
            )

        # ---- Decoder ----
        c = base_channels
        self.dec3 = UpBlock3d(c * 16, c * 8,  c * 8,  dropout=dropout)
        self.dec2 = UpBlock3d(c * 8,  c * 4,  c * 4,  dropout=dropout)
        self.dec1 = UpBlock3d(c * 4,  c * 2,  c * 2,  dropout=dropout)
        self.dec0 = UpBlock3d(c * 2,  c,      c,       dropout=dropout)

        # ---- Segmentation head ----
        self.seg_head = nn.Sequential(
            nn.Conv3d(c, c // 2, 3, padding=1, bias=False),
            nn.InstanceNorm3d(c // 2, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(c // 2, num_seg_classes, 1),
        )

        # ---- Classification head (optional) ----
        if num_grade_classes > 0:
            self.grade_head = nn.Sequential(
                SEBlock3d(c * 16),
                nn.AdaptiveAvgPool3d(1),
                nn.Flatten(),
                nn.Dropout(0.3),
                nn.Linear(c * 16, c * 8),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(c * 8, num_grade_classes),
            )
        else:
            self.grade_head = None

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.InstanceNorm3d, nn.BatchNorm3d)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def encode_modalities(
        self, images: torch.Tensor
    ) -> List[List[torch.Tensor]]:
        """
        images: (B, M, D, H, W)
        Returns list[scale][modality]: feature tensors
        """
        all_feats = []  # [modality][scale]
        for m_idx in range(self.M):
            x_m = images[:, m_idx:m_idx+1, ...]  # (B, 1, D, H, W)
            all_feats.append(self.encoders[m_idx](x_m))

        # Transpose to [scale][modality]
        num_scales = len(all_feats[0])
        per_scale = [
            [all_feats[m][s] for m in range(self.M)]
            for s in range(num_scales)
        ]
        return per_scale

    # ------------------------------------------------------------------
    def forward(
        self,
        images: torch.Tensor,
        available_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        images         : (B, M, D, H, W)
        available_mask : (B, M) bool  [for robust fusion only]

        Returns
        -------
        dict with keys:
            "seg"         : (B, num_seg_classes, D, H, W)  logits
            "grade"       : (B, num_grade_classes)          logits  [optional]
            "aux_loss"    : scalar auxiliary loss from MoE  [optional]
        """
        per_scale = self.encode_modalities(images)

        # Fuse each scale
        fused_scales = []
        aux_loss = torch.tensor(0.0, device=images.device)

        for s, (fusion, modality_feats) in enumerate(
                zip(self.fusions, per_scale)):
            fusion_kwargs = {}
            if available_mask is not None and hasattr(fusion, "forward") and \
               "available_mask" in fusion.forward.__code__.co_varnames:
                fusion_kwargs["available_mask"] = available_mask

            result = fusion(modality_feats, **fusion_kwargs)

            if isinstance(result, tuple):
                fused, al = result
                aux_loss = aux_loss + al
            else:
                fused = result
            fused_scales.append(fused)

        # fused_scales: [s0, s1, s2, s3, bottleneck]
        s0, s1, s2, s3, bt = fused_scales

        # Decoder
        d3 = self.dec3(bt, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        d0 = self.dec0(d1, s0)

        seg_logits = self.seg_head(d0)   # (B, C_seg, D, H, W)

        out = {"seg": seg_logits, "aux_loss": aux_loss}

        if self.grade_head is not None:
            out["grade"] = self.grade_head(bt)

        return out

    # ------------------------------------------------------------------
    def predict_segmentation(
        self, images: torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """Argmax segmentation prediction (B, D, H, W)."""
        with torch.no_grad():
            out = self.forward(images)
            return torch.argmax(out["seg"], dim=1)

    # ------------------------------------------------------------------
    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ──────────────────────────────────────────────────────────────────────
#  Lighter variant: 2-D MultimodalUNet for slice-wise inference
# ──────────────────────────────────────────────────────────────────────

class ConvBlock2d(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride,
                               padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(out_ch, affine=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(out_ch, affine=True)
        self.act   = nn.LeakyReLU(0.01, inplace=True)
        self.drop  = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.residual = (
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)
            if (in_ch != out_ch or stride != 1) else nn.Identity()
        )

    def forward(self, x):
        res = self.residual(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.drop(x)
        x = self.norm2(self.conv2(x))
        return self.act(x + res)


class MultimodalUNet2D(nn.Module):
    """
    Lightweight 2-D version for slice-based training / fast prototyping.
    Accepts (B, M, H, W) input where M is the number of modalities.
    """

    def __init__(self, num_modalities: int = 4, num_seg_classes: int = 4,
                 num_grade_classes: int = 3, base_channels: int = 32,
                 dropout: float = 0.1):
        super().__init__()
        c = base_channels
        in_ch = num_modalities   # fuse at input by treating mods as channels

        self.enc0 = ConvBlock2d(in_ch, c,        dropout=dropout)
        self.enc1 = ConvBlock2d(c, c*2,  stride=2, dropout=dropout)
        self.enc2 = ConvBlock2d(c*2, c*4, stride=2, dropout=dropout)
        self.enc3 = ConvBlock2d(c*4, c*8, stride=2, dropout=dropout)
        self.bot  = ConvBlock2d(c*8, c*16, stride=2, dropout=dropout)

        # Decoder
        self.dec3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock2d(c*16 + c*8, c*8, dropout=dropout))
        self.dec2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock2d(c*8 + c*4, c*4, dropout=dropout))
        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock2d(c*4 + c*2, c*2, dropout=dropout))
        self.dec0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock2d(c*2 + c, c, dropout=dropout))

        self.seg_head = nn.Conv2d(c, num_seg_classes, 1)

        if num_grade_classes > 0:
            self.grade_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3),
                nn.Linear(c*16, num_grade_classes))
        else:
            self.grade_head = None

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        # images: (B, M, H, W)
        s0 = self.enc0(images)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        bt = self.bot(s3)

        def up_cat(dec, x, skip):
            out = dec[0](x)                  # upsample
            diff = [skip.shape[i] - out.shape[i] for i in range(2, 4)]
            out = F.pad(out, [0, diff[1], 0, diff[0]])
            return dec[1](torch.cat([skip, out], dim=1))

        d3 = up_cat(self.dec3, bt, s3)
        d2 = up_cat(self.dec2, d3, s2)
        d1 = up_cat(self.dec1, d2, s1)
        d0 = up_cat(self.dec0, d1, s0)

        out = {"seg": self.seg_head(d0)}
        if self.grade_head is not None:
            out["grade"] = self.grade_head(bt)
        return out


# ──────────────────────────────────────────────────────────────────────
#  Model factory
# ──────────────────────────────────────────────────────────────────────

def build_model(cfg: dict) -> nn.Module:
    """
    Build model from config dict.

    Required keys:
        model_type   : "3d" | "2d"
        num_modalities, num_seg_classes, base_channels, fusion_type
    Optional:
        num_grade_classes, shared_encoder, dropout
    """
    mtype = cfg.get("model_type", "3d")
    if mtype == "3d":
        model = MultimodalUNet(
            num_modalities=cfg.get("num_modalities", 4),
            num_seg_classes=cfg.get("num_seg_classes", 4),
            num_grade_classes=cfg.get("num_grade_classes", 3),
            base_channels=cfg.get("base_channels", 32),
            fusion_type=cfg.get("fusion_type", "se_modality"),
            shared_encoder=cfg.get("shared_encoder", False),
            dropout=cfg.get("dropout", 0.1),
        )
    elif mtype == "2d":
        model = MultimodalUNet2D(
            num_modalities=cfg.get("num_modalities", 4),
            num_seg_classes=cfg.get("num_seg_classes", 4),
            num_grade_classes=cfg.get("num_grade_classes", 3),
            base_channels=cfg.get("base_channels", 32),
            dropout=cfg.get("dropout", 0.1),
        )
    else:
        raise ValueError(f"Unknown model_type: {mtype}")
    return model