"""
multimodal/metrics.py
=====================
Medical image segmentation and classification metrics.

Segmentation:
    - Dice score (per-class and mean)
    - Hausdorff Distance 95th percentile (HD95)
    - Sensitivity / Specificity per class
    - Intersection over Union (IoU / Jaccard)

Classification:
    - Accuracy, balanced accuracy
    - Macro-F1, per-class precision/recall
    - AUROC (one-vs-rest)

BraTS-specific:
    - Whole Tumour (WT): labels 1+2+4 (=1+2+3 after remap)
    - Tumour Core (TC): labels 1+4   (=1+3)
    - Enhancing Tumour (ET): label 4  (=3)
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from scipy.ndimage import binary_erosion, generate_binary_structure


# ──────────────────────────────────────────────────────────────────────
#  Dice & IoU
# ──────────────────────────────────────────────────────────────────────

def dice_score(pred: np.ndarray, target: np.ndarray,
               smooth: float = 1e-6) -> float:
    """
    Binary Dice score.
    pred, target : boolean or 0/1 np arrays of the same shape.
    """
    pred   = pred.astype(bool)
    target = target.astype(bool)
    inter  = (pred & target).sum()
    denom  = pred.sum() + target.sum()
    return float((2.0 * inter + smooth) / (denom + smooth))


def iou_score(pred: np.ndarray, target: np.ndarray,
              smooth: float = 1e-6) -> float:
    pred   = pred.astype(bool)
    target = target.astype(bool)
    inter  = (pred & target).sum()
    union  = (pred | target).sum()
    return float((inter + smooth) / (union + smooth))


def multiclass_dice(pred: np.ndarray,
                    target: np.ndarray,
                    num_classes: int,
                    exclude_bg: bool = True,
                    smooth: float = 1e-6) -> Dict[str, float]:
    """
    Per-class and mean Dice.
    pred, target : integer label arrays of identical shape.
    """
    start = 1 if exclude_bg else 0
    scores = {}
    for c in range(start, num_classes):
        scores[f"class_{c}"] = dice_score(pred == c, target == c, smooth)
    valid = [v for v in scores.values() if not np.isnan(v)]
    scores["mean"] = float(np.mean(valid)) if valid else float("nan")
    return scores


def multiclass_iou(pred: np.ndarray,
                   target: np.ndarray,
                   num_classes: int,
                   exclude_bg: bool = True) -> Dict[str, float]:
    start = 1 if exclude_bg else 0
    scores = {}
    for c in range(start, num_classes):
        scores[f"class_{c}"] = iou_score(pred == c, target == c)
    valid = [v for v in scores.values()]
    scores["mean"] = float(np.mean(valid))
    return scores


# ──────────────────────────────────────────────────────────────────────
#  BraTS sub-regions
# ──────────────────────────────────────────────────────────────────────

def brats_subregions(pred: np.ndarray,
                     target: np.ndarray) -> Dict[str, float]:
    """
    Compute BraTS Dice for WT, TC, ET using remapped labels:
        0 = background
        1 = necrotic core  (NCR/NET)
        2 = edema          (ED)
        3 = enhancing tumour (ET)

    WT = {1,2,3}, TC = {1,3}, ET = {3}
    """
    def subregion_dice(p, t, labels):
        pm = np.isin(p, labels)
        tm = np.isin(t, labels)
        return dice_score(pm, tm)

    return {
        "WT": subregion_dice(pred, target, [1, 2, 3]),
        "TC": subregion_dice(pred, target, [1, 3]),
        "ET": subregion_dice(pred, target, [3]),
    }


# ──────────────────────────────────────────────────────────────────────
#  Hausdorff distance 95th percentile
# ──────────────────────────────────────────────────────────────────────

def surface_points(binary: np.ndarray) -> np.ndarray:
    """
    Extract surface voxels of a binary mask using morphological erosion.
    Returns (N, ndim) array of surface voxel coordinates.
    """
    binary = binary.astype(bool)
    if not binary.any():
        return np.empty((0, binary.ndim), dtype=np.int64)
    struct = generate_binary_structure(binary.ndim, 1)
    eroded = binary_erosion(binary, structure=struct)
    surface = binary & ~eroded
    return np.argwhere(surface)


def hausdorff_95(pred: np.ndarray, target: np.ndarray,
                 voxel_spacing: Tuple[float, ...] = (1.0, 1.0, 1.0)
                 ) -> float:
    """
    95th percentile Hausdorff distance in mm (or voxels if spacing=(1,1,1)).
    Returns np.inf if either surface is empty.
    """
    sp = surface_points(pred)
    st = surface_points(target)

    if len(sp) == 0 or len(st) == 0:
        return float("inf")

    # Scale by voxel spacing
    sp = sp * np.array(voxel_spacing)
    st = st * np.array(voxel_spacing)

    # Pairwise L2 distances – this can be slow for large volumes.
    # For efficiency we use scipy cdist if available.
    try:
        from scipy.spatial.distance import cdist
        dist_sp = cdist(sp, st).min(axis=1)
        dist_st = cdist(st, sp).min(axis=1)
    except MemoryError:
        # Fallback: random sub-sample
        MAX = 5000
        sp_ = sp[np.random.choice(len(sp), min(MAX, len(sp)), replace=False)]
        st_ = st[np.random.choice(len(st), min(MAX, len(st)), replace=False)]
        from scipy.spatial.distance import cdist
        dist_sp = cdist(sp_, st_).min(axis=1)
        dist_st = cdist(st_, sp_).min(axis=1)

    return float(np.percentile(np.concatenate([dist_sp, dist_st]), 95))


def multiclass_hd95(pred: np.ndarray,
                    target: np.ndarray,
                    num_classes: int,
                    exclude_bg: bool = True,
                    voxel_spacing: Tuple = (1.0, 1.0, 1.0)) -> Dict[str, float]:
    start = 1 if exclude_bg else 0
    scores = {}
    for c in range(start, num_classes):
        scores[f"class_{c}"] = hausdorff_95(pred == c, target == c,
                                             voxel_spacing)
    finite = [v for v in scores.values() if np.isfinite(v)]
    scores["mean"] = float(np.mean(finite)) if finite else float("inf")
    return scores


# ──────────────────────────────────────────────────────────────────────
#  Sensitivity / Specificity
# ──────────────────────────────────────────────────────────────────────

def sensitivity_specificity(pred: np.ndarray,
                             target: np.ndarray) -> Tuple[float, float]:
    """
    Binary sensitivity (TPR) and specificity (TNR).
    """
    p = pred.astype(bool)
    t = target.astype(bool)
    TP = float((p &  t).sum())
    TN = float((~p & ~t).sum())
    FP = float((p & ~t).sum())
    FN = float((~p &  t).sum())
    sens = TP / (TP + FN + 1e-8)
    spec = TN / (TN + FP + 1e-8)
    return sens, spec


# ──────────────────────────────────────────────────────────────────────
#  Classification metrics
# ──────────────────────────────────────────────────────────────────────

def classification_report(preds: np.ndarray,
                           targets: np.ndarray,
                           probs: Optional[np.ndarray] = None,
                           num_classes: int = 3) -> Dict[str, float]:
    """
    preds   : (N,) predicted class labels
    targets : (N,) ground-truth class labels
    probs   : (N, C) class probabilities (for AUROC)

    Returns accuracy, balanced_accuracy, macro_f1, per-class P/R/F1, AUROC
    """
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                  f1_score, precision_score, recall_score,
                                  roc_auc_score, classification_report as cr)

    mask = targets != -1
    p    = preds[mask]
    t    = targets[mask]

    result: Dict[str, float] = {
        "accuracy":          float(accuracy_score(t, p)),
        "balanced_accuracy": float(balanced_accuracy_score(t, p)),
        "macro_f1":          float(f1_score(t, p, average="macro",
                                             zero_division=0)),
        "macro_precision":   float(precision_score(t, p, average="macro",
                                                    zero_division=0)),
        "macro_recall":      float(recall_score(t, p, average="macro",
                                                 zero_division=0)),
    }

    # Per-class F1
    per_f1 = f1_score(t, p, average=None, zero_division=0,
                      labels=list(range(num_classes)))
    for c, f in enumerate(per_f1):
        result[f"f1_class_{c}"] = float(f)

    # AUROC
    if probs is not None:
        probs_filt = probs[mask]
        try:
            if num_classes == 2:
                result["auroc"] = float(
                    roc_auc_score(t, probs_filt[:, 1]))
            else:
                result["auroc"] = float(
                    roc_auc_score(t, probs_filt, multi_class="ovr",
                                  average="macro"))
        except ValueError:
            pass   # not enough classes in batch

    return result


# ──────────────────────────────────────────────────────────────────────
#  Aggregated per-epoch metric tracker
# ──────────────────────────────────────────────────────────────────────

class MetricTracker:
    """
    Accumulates per-subject metrics across an epoch and computes means.

    Usage:
        tracker = MetricTracker(num_seg_classes=4, num_grade_classes=3)
        for batch in loader:
            pred_seg   = ...   # (B, D, H, W) numpy int
            target_seg = ...
            pred_grade = ...   # (B,) numpy int, -1 for unknown
            target_grade = ...
            pred_probs = ...   # (B, C) numpy float
            tracker.update(pred_seg, target_seg, pred_grade,
                           target_grade, pred_probs)

        results = tracker.compute()
    """

    def __init__(self, num_seg_classes: int = 4,
                 num_grade_classes: int = 3,
                 voxel_spacing: Tuple = (1.0, 1.0, 1.0),
                 compute_hd95: bool = True):
        self.C = num_seg_classes
        self.G = num_grade_classes
        self.spacing = voxel_spacing
        self.do_hd95 = compute_hd95
        self.reset()

    def reset(self):
        self._dice_records: List[Dict] = []
        self._hd95_records: List[Dict] = []
        self._brats_records: List[Dict] = []
        self._grade_preds: List[int]   = []
        self._grade_targets: List[int] = []
        self._grade_probs: List[np.ndarray] = []

    def update(self,
               pred_seg: np.ndarray,
               target_seg: np.ndarray,
               pred_grade: Optional[np.ndarray] = None,
               target_grade: Optional[np.ndarray] = None,
               grade_probs: Optional[np.ndarray] = None):
        """
        All arrays are for a single subject (unbatched).
        pred_seg, target_seg : (D, H, W) int
        """
        # Segmentation
        dc = multiclass_dice(pred_seg, target_seg, self.C)
        self._dice_records.append(dc)

        br = brats_subregions(pred_seg, target_seg)
        self._brats_records.append(br)

        if self.do_hd95:
            hd = multiclass_hd95(pred_seg, target_seg, self.C,
                                  voxel_spacing=self.spacing)
            self._hd95_records.append(hd)

        # Classification
        if pred_grade is not None and target_grade is not None:
            self._grade_preds.extend(
                pred_grade.tolist() if hasattr(pred_grade, "tolist")
                else [int(pred_grade)])
            self._grade_targets.extend(
                target_grade.tolist() if hasattr(target_grade, "tolist")
                else [int(target_grade)])
            if grade_probs is not None:
                self._grade_probs.append(grade_probs)

    def compute(self) -> Dict[str, float]:
        results = {}

        # Mean Dice per class
        if self._dice_records:
            keys = self._dice_records[0].keys()
            for k in keys:
                vals = [r[k] for r in self._dice_records
                        if not np.isnan(r.get(k, float("nan")))]
                results[f"dice/{k}"] = float(np.mean(vals)) if vals else float("nan")

        # BraTS sub-regions
        if self._brats_records:
            for region in ["WT", "TC", "ET"]:
                vals = [r[region] for r in self._brats_records]
                results[f"brats/{region}"] = float(np.mean(vals))

        # HD95
        if self._hd95_records:
            keys = self._hd95_records[0].keys()
            for k in keys:
                vals = [r[k] for r in self._hd95_records
                        if np.isfinite(r.get(k, float("inf")))]
                results[f"hd95/{k}"] = float(np.mean(vals)) if vals else float("inf")

        # Classification
        if self._grade_preds:
            p  = np.array(self._grade_preds, dtype=np.int64)
            t  = np.array(self._grade_targets, dtype=np.int64)
            pr = (np.vstack(self._grade_probs)
                  if self._grade_probs else None)
            cls_r = classification_report(p, t, pr, self.G)
            for k, v in cls_r.items():
                results[f"grade/{k}"] = v

        return results

    def summary_str(self) -> str:
        r = self.compute()
        lines = []
        lines.append("── Segmentation (Dice) ──")
        for c in range(1, self.C):
            v = r.get(f"dice/class_{c}", float("nan"))
            lines.append(f"  Class {c}: {v:.4f}")
        lines.append(f"  Mean:     {r.get('dice/mean', float('nan')):.4f}")
        lines.append("── BraTS Sub-regions ──")
        for reg in ["WT", "TC", "ET"]:
            lines.append(f"  {reg}: {r.get(f'brats/{reg}', float('nan')):.4f}")
        if "grade/accuracy" in r:
            lines.append("── Grade Classification ──")
            lines.append(f"  Accuracy: {r['grade/accuracy']:.4f}")
            lines.append(f"  Macro-F1: {r['grade/macro_f1']:.4f}")
        return "\n".join(lines)