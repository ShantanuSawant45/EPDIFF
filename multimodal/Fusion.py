"""
multimodal/fusion.py
====================
Feature-level fusion strategies for combining multi-modal MRI representations.
Implements concatenation, attention-based, cross-modal attention and
mixture-of-experts fusion, all compatible with 3-D feature maps.
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────
#  Helper blocks
# ──────────────────────────────────────────────────────────────────────

class ConvBnRelu3d(nn.Module):
    """Conv3d → BN → ReLU building block."""

    def __init__(self, in_ch: int, out_ch: int,
                 kernel: int = 3, stride: int = 1, padding: int = 1,
                 groups: int = 1, bias: bool = False):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel, stride=stride,
                      padding=padding, groups=groups, bias=bias),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SEBlock3d(nn.Module):
    """
    Squeeze-and-Excitation block (channel recalibration).
    Works on (B, C, D, H, W) tensors.
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1, 1)
        return x * w


# ──────────────────────────────────────────────────────────────────────
#  1. Concatenation fusion (baseline)
# ──────────────────────────────────────────────────────────────────────

class ConcatFusion(nn.Module):
    """
    Concatenate features from all modalities along channel dim,
    then project back to `out_channels` with a 1×1×1 conv.
    """

    def __init__(self, num_modalities: int, in_channels: int,
                 out_channels: int):
        super().__init__()
        self.proj = ConvBnRelu3d(num_modalities * in_channels,
                                  out_channels, kernel=1, padding=0)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        # features: list of (B, C, D, H, W)
        x = torch.cat(features, dim=1)
        return self.proj(x)


# ──────────────────────────────────────────────────────────────────────
#  2. Squeeze-and-Excitation based modality attention
# ──────────────────────────────────────────────────────────────────────

class SEModalityFusion(nn.Module):
    """
    Compute a soft weight per modality via SE-style gating,
    then perform weighted sum of modality features.

    Architecture:
        concat → global avg pool → MLP → softmax → per-modality scale
        → weighted sum → SE recalibration
    """

    def __init__(self, num_modalities: int, in_channels: int,
                 out_channels: int, reduction: int = 4):
        super().__init__()
        self.M = num_modalities
        self.in_ch = in_channels
        hidden = max(in_channels // reduction, 8)

        # Modality gating network
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),           # (B*M, C, 1,1,1) → (B*M,C)
            nn.Flatten(),
            nn.Linear(in_channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1, bias=False),  # one scalar per modality
        )

        self.proj = ConvBnRelu3d(in_channels, out_channels, kernel=1, padding=0)
        self.se   = SEBlock3d(out_channels)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        B = features[0].shape[0]

        # Stack along extra dim: (B, M, C, D, H, W)
        stacked = torch.stack(features, dim=1)
        # Flatten batch×modality for gating
        flat = stacked.view(B * self.M, *stacked.shape[2:])
        gates = self.gate(flat).view(B, self.M, 1, 1, 1, 1)  # (B,M,1,1,1,1)
        weights = torch.softmax(gates, dim=1)                 # modality softmax

        # Weighted sum: (B, C, D, H, W)
        fused = (stacked * weights).sum(dim=1)
        fused = self.proj(fused)
        fused = self.se(fused)
        return fused


# ──────────────────────────────────────────────────────────────────────
#  3. Cross-modal attention fusion
# ──────────────────────────────────────────────────────────────────────

class CrossModalAttention(nn.Module):
    """
    Each modality acts as query against all other modalities (keys/values).
    Outputs are summed and projected.

    For computational efficiency the spatial dimensions are flattened and
    a single-head dot-product attention is used.  For full multi-head
    attention set `num_heads > 1`.
    """

    def __init__(self, num_modalities: int, in_channels: int,
                 out_channels: int, num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.M  = num_modalities
        self.nh = num_heads
        assert in_channels % num_heads == 0, \
            "in_channels must be divisible by num_heads"
        self.head_dim = in_channels // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Per-modality Q, K, V projections (1×1×1 convs to keep spatial)
        self.W_q = nn.ModuleList(
            [nn.Conv3d(in_channels, in_channels, 1, bias=False)
             for _ in range(num_modalities)])
        self.W_k = nn.ModuleList(
            [nn.Conv3d(in_channels, in_channels, 1, bias=False)
             for _ in range(num_modalities)])
        self.W_v = nn.ModuleList(
            [nn.Conv3d(in_channels, in_channels, 1, bias=False)
             for _ in range(num_modalities)])

        self.out_proj = ConvBnRelu3d(in_channels, out_channels,
                                      kernel=1, padding=0)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        B, C, *spatial = features[0].shape
        N = math.prod(spatial)          # number of spatial tokens

        attended = []
        for i, query_feat in enumerate(features):
            # Q from modality i
            Q = self.W_q[i](query_feat)   # (B, C, *spatial)
            Q = Q.view(B, self.nh, self.head_dim, N).permute(0, 1, 3, 2)
            # (B, nh, N, head_dim)

            # K, V from all modalities
            K_all, V_all = [], []
            for j, kv_feat in enumerate(features):
                K_all.append(
                    self.W_k[j](kv_feat).view(B, self.nh, self.head_dim, N)
                    .permute(0, 1, 3, 2))
                V_all.append(
                    self.W_v[j](kv_feat).view(B, self.nh, self.head_dim, N)
                    .permute(0, 1, 3, 2))
            K = torch.cat(K_all, dim=2)  # (B, nh, M*N, head_dim)
            V = torch.cat(V_all, dim=2)

            attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
            attn = torch.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            out  = torch.matmul(attn, V)  # (B, nh, N, head_dim)

            # Reshape back to spatial
            out = out.permute(0, 1, 3, 2).contiguous()
            out = out.view(B, C, *spatial)
            attended.append(out)

        # Sum over modalities and project
        fused = sum(attended)
        return self.out_proj(fused)


# ──────────────────────────────────────────────────────────────────────
#  4. Mixture-of-Experts (MoE) fusion
# ──────────────────────────────────────────────────────────────────────

class MoEFusion(nn.Module):
    """
    Mixture-of-Experts fusion: each expert specialises in combining
    a subset of modalities.  A router network selects which expert(s)
    to use given the concatenated modality features.

    Top-k routing with an auxiliary load-balancing loss.
    """

    def __init__(self, num_modalities: int, in_channels: int,
                 out_channels: int, num_experts: int = 4,
                 top_k: int = 2):
        super().__init__()
        self.M = num_modalities
        self.E = num_experts
        self.k = top_k

        # Experts: simple 1×1×1 conv per expert
        self.experts = nn.ModuleList([
            ConvBnRelu3d(num_modalities * in_channels, out_channels,
                         kernel=1, padding=0)
            for _ in range(num_experts)
        ])

        # Router: global average → MLP → expert logits
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(num_modalities * in_channels, num_experts, bias=True),
        )

    def forward(self, features: List[torch.Tensor]
                ) -> tuple:    # (fused_tensor, aux_loss)
        concat = torch.cat(features, dim=1)  # (B, M*C, D, H, W)

        # Routing
        logits = self.router(concat)         # (B, E)
        probs  = torch.softmax(logits, dim=-1)
        topk_vals, topk_idx = torch.topk(probs, self.k, dim=-1)
        # Normalise top-k weights
        topk_w = topk_vals / topk_vals.sum(dim=-1, keepdim=True)

        # Auxiliary load-balancing loss (encourage uniform expert utilisation)
        # Following Switch Transformer: L_aux = E * sum_e(f_e * p_e)
        # f_e = fraction of tokens routed to expert e
        # p_e = mean routing probability for expert e
        E = self.E
        # Compute f_e: average of one-hot top-1 over batch
        top1_idx = topk_idx[:, 0]
        one_hot  = F.one_hot(top1_idx, num_classes=E).float()  # (B, E)
        f_e = one_hot.mean(dim=0)       # (E,)
        p_e = probs.mean(dim=0)         # (E,)
        aux_loss = E * (f_e * p_e).sum()

        # Weighted sum of top-k expert outputs
        B = concat.shape[0]
        out = torch.zeros(B, self.experts[0].block[0].out_channels,
                          *concat.shape[2:], device=concat.device,
                          dtype=concat.dtype)
        for ki in range(self.k):
            for bi in range(B):
                eidx = topk_idx[bi, ki].item()
                w    = topk_w[bi, ki]
                out[bi] = out[bi] + w * self.experts[eidx](concat[bi:bi+1])[0]

        return out, aux_loss


# ──────────────────────────────────────────────────────────────────────
#  5. Missing-modality robust fusion
# ──────────────────────────────────────────────────────────────────────

class RobustModalityFusion(nn.Module):
    """
    Handles missing modalities at inference time by masking out their
    contribution.  During training, randomly drops modalities to simulate
    missing data.

    Each available modality is encoded independently; a learned mask
    embedding is used as a surrogate for absent modalities.
    """

    def __init__(self, num_modalities: int, in_channels: int,
                 out_channels: int, drop_prob: float = 0.15):
        super().__init__()
        self.M = num_modalities
        self.drop_prob = drop_prob

        # Per-modality projection heads
        self.encoders = nn.ModuleList([
            ConvBnRelu3d(in_channels, out_channels, kernel=1, padding=0)
            for _ in range(num_modalities)
        ])

        # Learnable substitute embedding for missing modality
        self.missing_embed = nn.ParameterList([
            nn.Parameter(torch.zeros(1, out_channels, 1, 1, 1))
            for _ in range(num_modalities)
        ])

        # Attention pooling over available modalities
        self.agg = SEBlock3d(out_channels)
        self.final_proj = ConvBnRelu3d(out_channels, out_channels,
                                        kernel=1, padding=0)

    def forward(self, features: List[torch.Tensor],
                available_mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """
        Parameters
        ----------
        features       : list of (B, C, D, H, W) per modality
        available_mask : (B, M) bool tensor; True = modality is present.
                         If None, all modalities are assumed present
                         (training mode may still randomly drop some).
        """
        B, C, *spatial = features[0].shape

        if available_mask is None:
            if self.training and self.drop_prob > 0:
                # Randomly mask modalities
                available_mask = torch.ones(B, self.M,
                                             dtype=torch.bool,
                                             device=features[0].device)
                for m in range(self.M):
                    drop = torch.rand(B) < self.drop_prob
                    available_mask[:, m] = ~drop
            else:
                available_mask = torch.ones(B, self.M,
                                             dtype=torch.bool,
                                             device=features[0].device)

        encoded = []
        for m, feat in enumerate(features):
            enc = self.encoders[m](feat)            # (B, out_C, *spatial)
            # Replace missing slots with learned substitute
            for bi in range(B):
                if not available_mask[bi, m]:
                    enc[bi] = self.missing_embed[m].expand_as(enc[bi])
            encoded.append(enc)

        # Mean of encoded modalities (ignoring fully-dropped cases per sample)
        fused = torch.stack(encoded, dim=1)          # (B, M, out_C, *spatial)
        mask  = available_mask.float().view(B, self.M, 1, 1, 1, 1)
        fused = (fused * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        fused = self.agg(fused)
        return self.final_proj(fused)


# ──────────────────────────────────────────────────────────────────────
#  Factory
# ──────────────────────────────────────────────────────────────────────

FUSION_REGISTRY = {
    "concat":       ConcatFusion,
    "se_modality":  SEModalityFusion,
    "cross_modal":  CrossModalAttention,
    "moe":          MoEFusion,
    "robust":       RobustModalityFusion,
}


def build_fusion(fusion_type: str, **kwargs) -> nn.Module:
    """
    Build a fusion module by name.

    Parameters
    ----------
    fusion_type : one of "concat", "se_modality", "cross_modal", "moe", "robust"
    kwargs      : passed to the fusion module constructor
    """
    if fusion_type not in FUSION_REGISTRY:
        raise ValueError(
            f"Unknown fusion type '{fusion_type}'. "
            f"Choose from {list(FUSION_REGISTRY.keys())}")
    return FUSION_REGISTRY[fusion_type](**kwargs)