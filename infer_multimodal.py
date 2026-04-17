"""
infer_multimodal.py
===================
Inference script: runs MultiModal-UNet on new subjects, saves
predicted segmentation masks as NIfTI files and a JSON report.

Usage:
    python infer_multimodal.py \\
        --checkpoint output/multimodal/best.pth \\
        --config     configs/multimodal_config.yaml \\
        --data_root  /path/to/test/subjects \\
        --out_dir    predictions/ \\
        [--tta]          # test-time augmentation (flip ensemble)
        [--sliding_window]  # overlap-tile inference for large volumes
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from multimodal.dataset import (
    MODALITIES, load_nifti, normalise_volume,
    extract_brain_mask, pad_or_crop_to, remap_brats_labels,
    build_subject_index,
)
from multimodal.model import build_model
from multimodal.metrics import MetricTracker, brats_subregions

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-8s %(message)s")


# ──────────────────────────────────────────────────────────────────────
#  TTA helpers
# ──────────────────────────────────────────────────────────────────────

def flip_augment(image: torch.Tensor) -> List[torch.Tensor]:
    """Returns the original + 7 axis-flip variants (all combinations)."""
    variants = [image]
    for axes in [(2,), (3,), (4,), (2, 3), (2, 4), (3, 4), (2, 3, 4)]:
        variants.append(torch.flip(image, dims=axes))
    return variants


def tta_predict(model: torch.nn.Module,
                image: torch.Tensor,
                device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run 8-fold TTA (original + 7 flips), average softmax probabilities.
    Returns (prob_seg, logit_grade).
    """
    augmented = flip_augment(image)
    probs_list = []
    grade_list = []

    for aug_img in augmented:
        inp = aug_img.to(device)
        with torch.no_grad():
            out = model(inp)
        probs = F.softmax(out["seg"], dim=1)  # (1, C, D, H, W)
        probs_list.append(probs.cpu())
        if "grade" in out:
            grade_list.append(out["grade"].cpu())

    # Undo flips on probs and average
    restored = []
    variants_axes = [
        (), (2,), (3,), (4,), (2, 3), (2, 4), (3, 4), (2, 3, 4)
    ]
    for prob, axes in zip(probs_list, variants_axes):
        if axes:
            prob = torch.flip(prob, dims=list(axes))
        restored.append(prob)

    avg_prob = torch.stack(restored).mean(dim=0)   # (1, C, D, H, W)
    avg_grade = (torch.stack(grade_list).mean(dim=0)
                 if grade_list else None)
    return avg_prob, avg_grade


# ──────────────────────────────────────────────────────────────────────
#  Sliding window inference (overlap-tile)
# ──────────────────────────────────────────────────────────────────────

def sliding_window_predict(
        model: torch.nn.Module,
        image: torch.Tensor,
        patch_size: Tuple[int, int, int],
        overlap: float = 0.5,
        device: torch.device = torch.device("cpu"),
        batch_size: int = 1,
) -> torch.Tensor:
    """
    Patch-wise inference with Gaussian weight blending to reduce boundary
    artefacts.

    image : (1, C, D, H, W)
    Returns (1, num_classes, D, H, W) probability tensor.
    """
    _, C, D, H, W = image.shape
    pD, pH, pW = patch_size
    stride = [max(1, int(p * (1 - overlap))) for p in patch_size]
    sD, sH, sW = stride

    num_classes = None
    out_sum  = None
    out_cnt  = None

    # Gaussian weight kernel
    def gaussian_kernel(size):
        c = (size - 1) / 2.0
        k = torch.zeros(size)
        for i in range(size):
            k[i] = np.exp(-0.5 * ((i - c) / (c / 3)) ** 2)
        return k / k.max()

    gD = gaussian_kernel(pD)
    gH = gaussian_kernel(pH)
    gW = gaussian_kernel(pW)
    weight = (gD[:, None, None] * gH[None, :, None] * gW[None, None, :])  # (pD,pH,pW)
    weight = weight.unsqueeze(0).unsqueeze(0)   # (1,1,pD,pH,pW)

    model.eval()
    with torch.no_grad():
        d_starts = list(range(0, D - pD + 1, sD))
        if d_starts[-1] < D - pD:
            d_starts.append(D - pD)
        h_starts = list(range(0, H - pH + 1, sH))
        if h_starts[-1] < H - pH:
            h_starts.append(H - pH)
        w_starts = list(range(0, W - pW + 1, sW))
        if w_starts[-1] < W - pW:
            w_starts.append(W - pW)

        patches, coords = [], []
        for d0 in d_starts:
            for h0 in h_starts:
                for w0 in w_starts:
                    patch = image[:, :, d0:d0+pD, h0:h0+pH, w0:w0+pW]
                    patches.append(patch)
                    coords.append((d0, h0, w0))

        for i in range(0, len(patches), batch_size):
            batch = torch.cat(patches[i:i+batch_size], dim=0).to(device)
            out   = model(batch)
            probs = F.softmax(out["seg"], dim=1).cpu()

            if num_classes is None:
                num_classes = probs.shape[1]
                out_sum = torch.zeros(1, num_classes, D, H, W)
                out_cnt = torch.zeros(1, 1,           D, H, W)

            for j, (d0, h0, w0) in enumerate(coords[i:i+batch_size]):
                p = probs[j:j+1]                    # (1, C, pD, pH, pW)
                w = weight                           # (1, 1, pD, pH, pW)
                out_sum[:, :, d0:d0+pD, h0:h0+pH, w0:w0+pW] += p * w
                out_cnt[:,  :, d0:d0+pD, h0:h0+pH, w0:w0+pW] += w

    return out_sum / (out_cnt + 1e-8)


# ──────────────────────────────────────────────────────────────────────
#  Single subject inference
# ──────────────────────────────────────────────────────────────────────

def infer_subject(
        model: torch.nn.Module,
        subject: Dict,
        cfg: Dict,
        device: torch.device,
        use_tta: bool = False,
        use_sliding_window: bool = False,
) -> Dict:
    """
    Run inference on one subject.  Returns a dict with:
        pred_seg       : (D, H, W) int numpy array
        grade_probs    : (num_classes,) numpy array or None
        brats_metrics  : dict with WT/TC/ET Dice (if ground-truth available)
    """
    modalities = cfg.get("modalities", MODALITIES)
    norm_method = cfg.get("norm_method", "z_score")
    target_shape = tuple(cfg.get("target_shape", [240, 240, 155]))
    patch_size   = tuple(cfg.get("patch_size",   [128, 128, 128]))

    # Load + preprocess
    volumes = []
    for mod in modalities:
        vol = load_nifti(subject["paths"][mod])
        vol = pad_or_crop_to(vol, target_shape)
        mask = extract_brain_mask(vol)
        vol  = normalise_volume(vol, norm_method, mask=mask)
        volumes.append(vol)

    # Stack: (1, M, D, H, W)
    image = torch.from_numpy(np.stack(volumes, axis=0)).unsqueeze(0)

    # ── Inference ──
    t0 = time.time()
    if use_sliding_window:
        prob = sliding_window_predict(
            model, image, patch_size,
            overlap=cfg.get("sw_overlap", 0.5),
            device=device,
            batch_size=cfg.get("sw_batch_size", 1),
        )
        grade_logits = None
    elif use_tta:
        prob, grade_logits = tta_predict(model, image, device)
    else:
        image = image.to(device)
        with torch.no_grad():
            out = model(image)
        prob = F.softmax(out["seg"], dim=1).cpu()
        grade_logits = out.get("grade")

    infer_time = time.time() - t0

    pred_seg = torch.argmax(prob, dim=1).squeeze(0).numpy()  # (D, H, W)

    grade_probs = None
    if grade_logits is not None:
        grade_probs = F.softmax(grade_logits, dim=1).squeeze(0).cpu().numpy()

    result = {
        "subject_id": subject["subject_id"],
        "pred_seg":   pred_seg,
        "grade_probs": grade_probs,
        "pred_grade": int(grade_probs.argmax()) if grade_probs is not None else -1,
        "infer_time_s": infer_time,
    }

    # Optional GT evaluation
    if subject["paths"]["seg"] is not None:
        seg_gt = remap_brats_labels(
            load_nifti(subject["paths"]["seg"]).astype(np.int64))
        seg_gt = pad_or_crop_to(seg_gt, target_shape)
        result["brats_metrics"] = brats_subregions(pred_seg, seg_gt)
    else:
        result["brats_metrics"] = {}

    return result


# ──────────────────────────────────────────────────────────────────────
#  Save predictions
# ──────────────────────────────────────────────────────────────────────

def save_nifti_prediction(pred: np.ndarray,
                          ref_path: str,
                          out_path: str):
    """Save prediction using the reference NIfTI's header/affine."""
    ref  = nib.load(ref_path)
    nii  = nib.Nifti1Image(pred.astype(np.uint8), ref.affine, ref.header)
    nib.save(nii, out_path)


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Inference for MultiModal Brain Tumour Detection")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--config", default="configs/multimodal_config.yaml",
                        help="Config YAML file")
    parser.add_argument("--data_root", required=True,
                        help="Root directory with subject folders")
    parser.add_argument("--split", default="test",
                        help="Data split: train / val / test")
    parser.add_argument("--out_dir", default="predictions",
                        help="Output directory for predictions")
    parser.add_argument("--tta", action="store_true",
                        help="Use test-time augmentation (8 flips)")
    parser.add_argument("--sliding_window", action="store_true",
                        help="Use overlap-tile sliding window inference")
    parser.add_argument("--save_nifti", action="store_true",
                        help="Save predicted masks as .nii.gz files")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cfg["data_root"] = args.data_root

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Build & load model
    model = build_model(cfg).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    logger.info(f"Loaded model from {args.checkpoint}  "
                f"(epoch={ckpt.get('epoch','?')}, "
                f"best_dice={ckpt.get('best_dice',0):.4f})")

    # Build subject index
    subjects = build_subject_index(
        args.data_root, args.split,
        cfg.get("modalities", MODALITIES),
        cfg.get("label_file", None),
    )

    if not subjects:
        logger.error("No subjects found.  Check --data_root and --split.")
        return

    # Metric tracker for aggregate reporting
    tracker = MetricTracker(
        num_seg_classes=cfg.get("num_seg_classes", 4),
        num_grade_classes=cfg.get("num_grade_classes", 3),
        compute_hd95=cfg.get("compute_hd95_test", True),
    )

    report = []

    for subj in tqdm(subjects, desc="Inferring"):
        result = infer_subject(
            model, subj, cfg, device,
            use_tta=args.tta,
            use_sliding_window=args.sliding_window,
        )

        # Optionally save NIfTI
        if args.save_nifti:
            ref = subj["paths"][cfg.get("modalities", MODALITIES)[0]]
            nii_out = str(out_dir / f"{subj['subject_id']}_pred_seg.nii.gz")
            save_nifti_prediction(result["pred_seg"], ref, nii_out)

        # Update tracker
        if result["brats_metrics"]:
            # Only if GT available
            seg_gt = remap_brats_labels(
                load_nifti(subj["paths"]["seg"]).astype(np.int64))
            target_shape = tuple(cfg.get("target_shape", [240, 240, 155]))
            from multimodal.dataset import pad_or_crop_to as _pcc
            seg_gt = _pcc(seg_gt, target_shape)
            tracker.update(
                result["pred_seg"], seg_gt,
                np.array([result["pred_grade"]]) if result["pred_grade"] >= 0 else None,
                np.array([subj["grade"]]) if subj["grade"] >= 0 else None,
                result["grade_probs"][np.newaxis] if result["grade_probs"] is not None else None,
            )

        entry = {
            "subject_id":    result["subject_id"],
            "pred_grade":    result["pred_grade"],
            "grade_probs":   (result["grade_probs"].tolist()
                              if result["grade_probs"] is not None else None),
            "infer_time_s":  round(result["infer_time_s"], 3),
            "brats_metrics": result["brats_metrics"],
        }
        report.append(entry)
        logger.info(
            f"{subj['subject_id']}  "
            f"grade={result['pred_grade']}  "
            f"WT={result['brats_metrics'].get('WT',float('nan')):.4f}  "
            f"TC={result['brats_metrics'].get('TC',float('nan')):.4f}  "
            f"ET={result['brats_metrics'].get('ET',float('nan')):.4f}  "
            f"t={result['infer_time_s']:.1f}s"
        )

    # Save JSON report
    report_path = str(out_dir / "inference_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report saved → {report_path}")

    # Print aggregate summary
    print("\n" + "=" * 50)
    print(" Aggregate Results")
    print("=" * 50)
    print(tracker.summary_str())
    print("=" * 50)

    # Save aggregate metrics
    agg = tracker.compute()
    with open(str(out_dir / "aggregate_metrics.json"), "w") as f:
        # Convert nan/inf to strings for JSON serialisation
        clean = {k: (v if np.isfinite(v) else str(v))
                 for k, v in agg.items()}
        json.dump(clean, f, indent=2)


if __name__ == "__main__":
    main()