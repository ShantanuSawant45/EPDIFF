"""
multimodal/losses.py
====================
Loss functions for multimodal brain tumour detection.

Includes:
- Dice loss (per-class and mean)
- Focal loss
- Combined Dice + Focal / Dice + CE
- Hausdorff distance-aware loss
- Deep supervision loss wrapper
- Joint segmentation + classification loss
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────
#  Dice Loss
# ──────────────────────────────────────────────────────────────────────

def dice_coefficient(pred: torch.Tensor, target: torch.Tensor,
                     smooth: float = 1e-6,
                     exclude_bg: bool = True) -> torch.Tensor:
    """
    Soft Dice per class, averaged.

    Parameters
    ----------
    pred   : (B, C, *spatial) – soft probabilities after softmax
    target : (B, *spatial)    – integer class labels
    """
    B, C = pred.shape[:2]
    num_classes = C

    # One-hot encode target
    target_oh = F.one_hot(target, num_classes).float()     # (B, *sp, C)
    # Move class dim to position 1
    dims = list(range(target_oh.ndim))
    dims = [0, dims[-1]] + dims[1:-1]
    target_oh = target_oh.permute(*dims).contiguous()       # (B, C, *sp)

    start_c = 1 if exclude_bg else 0
    dice_per_class = []
    for c in range(start_c, num_classes):
        p = pred[:, c]
        t = target_oh[:, c]
        inter = (p * t).sum()
        denom = p.sum() + t.sum()
        dice_per_class.append((2.0 * inter + smooth) / (denom + smooth))

    return torch.stack(dice_per_class).mean()


class DiceLoss(nn.Module):
    """
    Soft Dice loss = 1 - Dice.

    Parameters
    ----------
    weight       : optional per-class weight tensor of shape (C,)
    exclude_bg   : whether to exclude background (class 0) from Dice
    smooth       : Laplace smoothing factor
    """

    def __init__(self, weight: Optional[torch.Tensor] = None,
                 exclude_bg: bool = True, smooth: float = 1e-6):
        super().__init__()
        self.register_buffer("weight", weight)
        self.exclude_bg = exclude_bg
        self.smooth = smooth

    def forward(self, logits: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        return 1.0 - dice_coefficient(probs, target, self.smooth,
                                      self.exclude_bg)


# ──────────────────────────────────────────────────────────────────────
#  Focal Loss
# ──────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Multi-class focal loss.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Parameters
    ----------
    gamma   : focusing parameter (0 = standard CE)
    alpha   : per-class weight tensor (C,) or scalar
    """

    def __init__(self, gamma: float = 2.0,
                 alpha: Optional[torch.Tensor] = None,
                 reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        # logits: (B, C, *spatial) | target: (B, *spatial)
        B, C = logits.shape[:2]
        spatial = logits.shape[2:]

        log_p = F.log_softmax(logits, dim=1)           # (B, C, *sp)
        p     = torch.exp(log_p)

        # Gather per-voxel probabilities of true class
        # target: (B, *sp) → need to index along dim=1
        t_flat  = target.view(B, 1, -1)                # (B, 1, N)
        lp_flat = log_p.view(B, C, -1)                 # (B, C, N)
        p_flat  = p.view(B, C, -1)

        log_pt = lp_flat.gather(1, t_flat).squeeze(1)  # (B, N)
        pt     = p_flat.gather(1, t_flat).squeeze(1)   # (B, N)

        focal_factor = (1.0 - pt) ** self.gamma
        loss = -focal_factor * log_pt                  # (B, N)

        if self.alpha is not None:
            # Gather alpha weight for true class
            alpha = self.alpha.view(1, C, 1).expand_as(p_flat)
            alpha_t = alpha.gather(1, t_flat).squeeze(1)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss.view(B, *spatial)


# ──────────────────────────────────────────────────────────────────────
#  Combined Dice + Focal / Dice + CE
# ──────────────────────────────────────────────────────────────────────

class DiceFocalLoss(nn.Module):
    """
    Segmentation loss = λ_dice * Dice + λ_focal * Focal.
    """

    def __init__(self, lambda_dice: float = 1.0,
                 lambda_focal: float = 1.0,
                 gamma: float = 2.0,
                 class_weights: Optional[torch.Tensor] = None,
                 exclude_bg: bool = True):
        super().__init__()
        self.dice  = DiceLoss(weight=class_weights, exclude_bg=exclude_bg)
        self.focal = FocalLoss(gamma=gamma, alpha=class_weights)
        self.ld    = lambda_dice
        self.lf    = lambda_focal

    def forward(self, logits: torch.Tensor,
                target: torch.Tensor) -> Dict[str, torch.Tensor]:
        d = self.dice(logits, target)
        f = self.focal(logits, target)
        total = self.ld * d + self.lf * f
        return {"total": total, "dice": d, "focal": f}


class DiceCELoss(nn.Module):
    """
    Segmentation loss = λ_dice * Dice + λ_ce * CrossEntropy.
    Simpler to tune than focal; CE with class weights handles imbalance.
    """

    def __init__(self, lambda_dice: float = 1.0,
                 lambda_ce: float = 1.0,
                 class_weights: Optional[torch.Tensor] = None,
                 exclude_bg: bool = True):
        super().__init__()
        self.dice = DiceLoss(weight=class_weights, exclude_bg=exclude_bg)
        self.ce   = nn.CrossEntropyLoss(weight=class_weights)
        self.ld   = lambda_dice
        self.lc   = lambda_ce

    def forward(self, logits: torch.Tensor,
                target: torch.Tensor) -> Dict[str, torch.Tensor]:
        d = self.dice(logits, target)
        c = self.ce(logits, target)
        total = self.ld * d + self.lc * c
        return {"total": total, "dice": d, "ce": c}


# ──────────────────────────────────────────────────────────────────────
#  Boundary-aware (approximate Hausdorff) loss
# ──────────────────────────────────────────────────────────────────────

def distance_map(binary: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Approximate distance transform for a binary 3-D mask using iterated
    pooling.  Not identical to exact EDT but suitable as a loss signal.
    binary : (B, 1, D, H, W) float in {0, 1}
    """
    # Foreground = 1, background = 0
    neg = 1.0 - binary
    dist = torch.zeros_like(binary)
    kernel = 3
    pad = kernel // 2
    current = neg.clone()
    for step in range(1, 30):
        eroded = -F.max_pool3d(-current, kernel, stride=1, padding=pad)
        changed = (eroded < current).float()
        dist += changed * step
        current = eroded
        if changed.sum() < 1:
            break
    return dist


class HausdorffLoss(nn.Module):
    """
    Soft Hausdorff distance loss (Karimi et al., 2019).
    Penalises boundary voxels misclassified by the model.

    Only applied to foreground classes; background is excluded.
    """

    def __init__(self, alpha: float = 2.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, logits: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        B, C = logits.shape[:2]
        probs = F.softmax(logits, dim=1)   # (B, C, *sp)
        total = 0.0
        for c in range(1, C):             # skip background
            t_c  = (target == c).float().unsqueeze(1)   # (B,1,*sp)
            p_c  = probs[:, c:c+1]                      # (B,1,*sp)
            dt   = distance_map(t_c)                    # (B,1,*sp)
            dp   = distance_map(p_c > 0.5)              # (B,1,*sp)
            loss = ((p_c - t_c) ** 2 * (dt ** self.alpha + dp ** self.alpha))
            total = total + loss.mean()
        return total / (C - 1)


# ──────────────────────────────────────────────────────────────────────
#  Deep supervision loss
# ──────────────────────────────────────────────────────────────────────

class DeepSupervisionLoss(nn.Module):
    """
    Weighted sum of losses at multiple decoder levels.
    Weights decrease geometrically from finest to coarsest.

    Parameters
    ----------
    base_loss  : loss module to apply at each scale
    weights    : list of floats summing to 1 (finest-to-coarsest)
    """

    def __init__(self, base_loss: nn.Module,
                 weights: Optional[List[float]] = None):
        super().__init__()
        self.base_loss = base_loss
        self.weights   = weights or [0.5, 0.25, 0.15, 0.1]

    def forward(self, logits_list: List[torch.Tensor],
                target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        logits_list : [finest, ..., coarsest] scale logits
        target      : (B, D, H, W) at finest resolution
        """
        total = 0.0
        per_scale = {}
        for i, (logits, w) in enumerate(zip(logits_list, self.weights)):
            if logits.shape[2:] != target.shape[1:]:
                # Downsample target to match current scale
                tgt = F.interpolate(
                    target.unsqueeze(1).float(),
                    size=logits.shape[2:],
                    mode="nearest").squeeze(1).long()
            else:
                tgt = target
            loss_dict = self.base_loss(logits, tgt)
            scale_loss = loss_dict["total"] if isinstance(loss_dict, dict) \
                else loss_dict
            total = total + w * scale_loss
            per_scale[f"scale_{i}"] = scale_loss.detach()

        per_scale["total"] = total
        return per_scale


# ──────────────────────────────────────────────────────────────────────
#  Joint segmentation + classification loss
# ──────────────────────────────────────────────────────────────────────

class MultiTaskLoss(nn.Module):
    """
    Combined loss for simultaneous segmentation and grade classification.

    L_total = λ_seg * L_seg + λ_grade * L_grade + λ_aux * L_aux

    L_seg   : DiceFocalLoss on segmentation logits
    L_grade : CrossEntropy on grade logits (ignores grade == -1)
    L_aux   : Auxiliary MoE load-balancing loss (passthrough scalar)
    """

    def __init__(self,
                 lambda_seg: float = 1.0,
                 lambda_grade: float = 0.5,
                 lambda_aux: float = 0.01,
                 class_weights: Optional[torch.Tensor] = None,
                 gamma: float = 2.0,
                 label_smoothing: float = 0.1):
        super().__init__()
        self.l_seg   = lambda_seg
        self.l_grade = lambda_grade
        self.l_aux   = lambda_aux

        self.seg_loss = DiceFocalLoss(
            class_weights=class_weights, gamma=gamma)
        self.grade_ce = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing, ignore_index=-1)

    def forward(self,
                seg_logits: torch.Tensor,
                seg_target: torch.Tensor,
                grade_logits: Optional[torch.Tensor] = None,
                grade_target: Optional[torch.Tensor] = None,
                aux_loss: Optional[torch.Tensor] = None,
                ) -> Dict[str, torch.Tensor]:

        seg_dict  = self.seg_loss(seg_logits, seg_target)
        total     = self.l_seg * seg_dict["total"]

        out = {
            "seg_total": seg_dict["total"],
            "dice":      seg_dict["dice"],
            "focal":     seg_dict["focal"],
        }

        if grade_logits is not None and grade_target is not None:
            valid = grade_target != -1
            if valid.any():
                gl = self.grade_ce(grade_logits[valid], grade_target[valid])
                total = total + self.l_grade * gl
                out["grade"] = gl

        if aux_loss is not None and aux_loss > 0:
            total = total + self.l_aux * aux_loss
            out["aux"] = aux_loss

        out["total"] = total
        return out


# ──────────────────────────────────────────────────────────────────────
#  Loss factory
# ──────────────────────────────────────────────────────────────────────

def build_loss(cfg: dict,
               class_weights: Optional[torch.Tensor] = None) -> nn.Module:
    """
    Build loss module from config dict.

    Expected keys: loss_type, lambda_seg, lambda_grade, lambda_aux,
                   gamma, label_smoothing, lambda_dice, lambda_focal, lambda_ce
    """
    loss_type = cfg.get("loss_type", "multitask")

    if loss_type == "multitask":
        return MultiTaskLoss(
            lambda_seg=cfg.get("lambda_seg", 1.0),
            lambda_grade=cfg.get("lambda_grade", 0.5),
            lambda_aux=cfg.get("lambda_aux", 0.01),
            class_weights=class_weights,
            gamma=cfg.get("gamma", 2.0),
            label_smoothing=cfg.get("label_smoothing", 0.1),
        )
    elif loss_type == "dice_focal":
        return DiceFocalLoss(
            lambda_dice=cfg.get("lambda_dice", 1.0),
            lambda_focal=cfg.get("lambda_focal", 1.0),
            gamma=cfg.get("gamma", 2.0),
            class_weights=class_weights,
        )
    elif loss_type == "dice_ce":
        return DiceCELoss(
            lambda_dice=cfg.get("lambda_dice", 1.0),
            lambda_ce=cfg.get("lambda_ce", 1.0),
            class_weights=class_weights,
        )
    elif loss_type == "dice":
        return DiceLoss(weight=class_weights)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")