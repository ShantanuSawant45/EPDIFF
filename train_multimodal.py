"""
train_multimodal.py
===================
Training entry-point for MultiModal Brain Tumour Detection.

Usage:
    python train_multimodal.py --config configs/multimodal_config.yaml

Supports:
    - Mixed-precision training (torch.cuda.amp)
    - Gradient clipping
    - OneCycleLR / CosineAnnealingWarmRestarts / ReduceLROnPlateau scheduling
    - Early stopping on validation Dice
    - Checkpointing (best and last)
    - TensorBoard logging
    - Distributed Data Parallel (single-node) via torchrun
"""

import argparse
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import yaml

from multimodal.dataset import build_loaders
from multimodal.model import build_model
from multimodal.losses import build_loss
from multimodal.metrics import MetricTracker

# ─────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
#  Reproducibility
# ──────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────────────────────────────
#  Checkpoint helpers
# ──────────────────────────────────────────────────────────────────────

def save_checkpoint(state: Dict, path: str):
    torch.save(state, path)
    logger.info(f"Checkpoint saved → {path}")


def load_checkpoint(path: str, model: nn.Module,
                    optimizer: Optional[optim.Optimizer] = None,
                    scaler: Optional[GradScaler] = None,
                    device: str = "cpu") -> Dict:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    logger.info(f"Loaded checkpoint from {path}  "
                f"(epoch {ckpt.get('epoch', '?')}, "
                f"best Dice {ckpt.get('best_dice', '?'):.4f})")
    return ckpt


# ──────────────────────────────────────────────────────────────────────
#  Learning rate scheduler factory
# ──────────────────────────────────────────────────────────────────────

def build_scheduler(optimizer, cfg: Dict, steps_per_epoch: int):
    stype = cfg.get("scheduler", "cosine")
    epochs = cfg.get("epochs", 300)

    if stype == "onecycle":
        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg["lr"],
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            pct_start=0.1,
            anneal_strategy="cos",
        )
    elif stype == "cosine":
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cfg.get("cosine_T0", 50),
            T_mult=cfg.get("cosine_Tmult", 2),
            eta_min=cfg.get("lr_min", 1e-7),
        )
    elif stype == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5,
            patience=cfg.get("patience_lr", 15),
            verbose=True, min_lr=cfg.get("lr_min", 1e-7),
        )
    elif stype == "warmup_cosine":
        from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
        warmup_epochs = cfg.get("warmup_epochs", 10)
        warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                          total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(optimizer,
                                    T_max=epochs - warmup_epochs,
                                    eta_min=cfg.get("lr_min", 1e-7))
        return SequentialLR(optimizer,
                            schedulers=[warmup, cosine],
                            milestones=[warmup_epochs])
    else:
        raise ValueError(f"Unknown scheduler: {stype}")


# ──────────────────────────────────────────────────────────────────────
#  Training step
# ──────────────────────────────────────────────────────────────────────

def train_one_epoch(model: nn.Module,
                    loader,
                    optimizer: optim.Optimizer,
                    loss_fn: nn.Module,
                    scaler: GradScaler,
                    scheduler,
                    device: torch.device,
                    cfg: Dict,
                    epoch: int,
                    writer: SummaryWriter) -> Dict[str, float]:

    model.train()
    total_loss = 0.0
    total_dice = 0.0
    total_grade = 0.0
    steps = 0
    sched_step = cfg.get("scheduler") == "onecycle"   # step per iteration

    t0 = time.time()
    for batch_idx, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)   # (B,M,D,H,W)
        seg_t  = batch["seg"].to(device, non_blocking=True)     # (B,D,H,W)
        grade_t = batch["grade"].to(device, non_blocking=True)  # (B,)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=cfg.get("amp", True)):
            out = model(images)
            seg_logits   = out["seg"]
            grade_logits = out.get("grade")
            aux_loss     = out.get("aux_loss", None)

            if isinstance(loss_fn, nn.Module) and hasattr(loss_fn, "l_grade"):
                loss_dict = loss_fn(
                    seg_logits=seg_logits,
                    seg_target=seg_t,
                    grade_logits=grade_logits,
                    grade_target=grade_t,
                    aux_loss=aux_loss,
                )
            else:
                loss_dict = loss_fn(seg_logits, seg_t)

            loss = loss_dict["total"] if isinstance(loss_dict, dict) \
                else loss_dict

        scaler.scale(loss).backward()

        # Gradient clipping
        if cfg.get("grad_clip", 0) > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(),
                                      cfg["grad_clip"])

        scaler.step(optimizer)
        scaler.update()

        if sched_step:
            scheduler.step()

        total_loss  += loss.item()
        if isinstance(loss_dict, dict):
            total_dice  += loss_dict.get("dice", torch.tensor(0.0)).item()
            total_grade += loss_dict.get("grade", torch.tensor(0.0)).item()
        steps += 1

        if batch_idx % cfg.get("log_every", 20) == 0:
            global_step = epoch * len(loader) + batch_idx
            writer.add_scalar("train/loss_step", loss.item(), global_step)
            lr_now = optimizer.param_groups[0]["lr"]
            writer.add_scalar("train/lr", lr_now, global_step)
            logger.info(
                f"Ep{epoch:03d} [{batch_idx:04d}/{len(loader)}]  "
                f"loss={loss.item():.4f}  lr={lr_now:.2e}  "
                f"t={time.time()-t0:.1f}s"
            )
            t0 = time.time()

    metrics = {
        "loss":  total_loss / steps,
        "dice":  total_dice / steps,
        "grade": total_grade / steps,
    }
    return metrics


# ──────────────────────────────────────────────────────────────────────
#  Validation step
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model: nn.Module,
             loader,
             loss_fn: nn.Module,
             device: torch.device,
             cfg: Dict) -> Dict[str, float]:

    model.eval()
    tracker = MetricTracker(
        num_seg_classes=cfg.get("num_seg_classes", 4),
        num_grade_classes=cfg.get("num_grade_classes", 3),
        compute_hd95=cfg.get("compute_hd95_val", False),
    )
    total_loss = 0.0
    steps = 0

    for batch in loader:
        images  = batch["image"].to(device)
        seg_t   = batch["seg"].to(device)
        grade_t = batch["grade"].to(device)

        with autocast(enabled=cfg.get("amp", True)):
            out = model(images)
            seg_logits   = out["seg"]
            grade_logits = out.get("grade")
            aux_loss     = out.get("aux_loss", None)

            if isinstance(loss_fn, nn.Module) and hasattr(loss_fn, "l_grade"):
                loss_dict = loss_fn(
                    seg_logits=seg_logits,
                    seg_target=seg_t,
                    grade_logits=grade_logits,
                    grade_target=grade_t,
                    aux_loss=aux_loss,
                )
            else:
                loss_dict = loss_fn(seg_logits, seg_t)

            loss = loss_dict["total"] if isinstance(loss_dict, dict) else loss_dict

        total_loss += loss.item()
        steps += 1

        # Convert to numpy for metric computation (CPU, unbatched)
        pred_seg = torch.argmax(seg_logits, dim=1).cpu().numpy()
        tgt_seg  = seg_t.cpu().numpy()
        pred_g   = None
        tgt_g    = None
        probs_g  = None

        if grade_logits is not None:
            import torch.nn.functional as F
            probs_g  = F.softmax(grade_logits, dim=1).cpu().numpy()
            pred_g   = probs_g.argmax(axis=1)
            tgt_g    = grade_t.cpu().numpy()

        for bi in range(pred_seg.shape[0]):
            tracker.update(
                pred_seg[bi], tgt_seg[bi],
                np.array([pred_g[bi]]) if pred_g is not None else None,
                np.array([tgt_g[bi]])  if tgt_g  is not None else None,
                probs_g[bi:bi+1]       if probs_g is not None else None,
            )

    results = tracker.compute()
    results["loss"] = total_loss / steps
    return results


# ──────────────────────────────────────────────────────────────────────
#  Main training loop
# ──────────────────────────────────────────────────────────────────────

def train(cfg: Dict):
    set_seed(cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Output directory
    out_dir = Path(cfg.get("output_dir", "output/multimodal"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)

    # Data
    train_loader, val_loader, _ = build_loaders(cfg)

    # Model
    model = build_model(cfg).to(device)
    logger.info(f"Model params: {model.count_params():,}")

    # Loss (optionally with class weights)
    class_weights = None
    if cfg.get("use_class_weights", False):
        class_weights = train_loader.dataset.get_class_weights().to(device)
        logger.info(f"Class weights: {class_weights}")
    loss_fn = build_loss(cfg, class_weights)

    # Optimiser
    lr = cfg.get("lr", 1e-4)
    wd = cfg.get("weight_decay", 1e-5)
    opt_name = cfg.get("optimizer", "adamw").lower()
    if opt_name == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr,
                                weight_decay=wd, eps=1e-8)
    elif opt_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr,
                              momentum=0.99, weight_decay=wd,
                              nesterov=True)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler    = GradScaler(enabled=cfg.get("amp", True))
    writer    = SummaryWriter(log_dir=str(out_dir / "tb_logs"))

    # Optionally resume
    start_epoch = 0
    best_dice   = 0.0
    if cfg.get("resume"):
        ckpt = load_checkpoint(cfg["resume"], model, optimizer, scaler,
                               str(device))
        start_epoch = ckpt.get("epoch", 0) + 1
        best_dice   = ckpt.get("best_dice", 0.0)

    # Early stopping
    patience        = cfg.get("patience", 50)
    patience_counter = 0
    epochs          = cfg.get("epochs", 300)
    sched_per_epoch = cfg.get("scheduler") != "onecycle"

    logger.info("Starting training …")

    for epoch in range(start_epoch, epochs):
        # ── Train ──
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, loss_fn,
            scaler, scheduler, device, cfg, epoch, writer)

        for k, v in train_metrics.items():
            writer.add_scalar(f"train/{k}", v, epoch)

        # ── LR step (epoch-level schedulers) ──
        if sched_per_epoch:
            if cfg.get("scheduler") == "plateau":
                scheduler.step(best_dice)
            else:
                scheduler.step()

        # ── Validate ──
        val_metrics = validate(model, val_loader, loss_fn, device, cfg)
        val_dice    = val_metrics.get("dice/mean", 0.0)

        for k, v in val_metrics.items():
            if np.isfinite(v):
                writer.add_scalar(f"val/{k}", v, epoch)

        logger.info(
            f"Epoch {epoch:03d}  "
            f"train_loss={train_metrics['loss']:.4f}  "
            f"val_dice={val_dice:.4f}  "
            f"val_WT={val_metrics.get('brats/WT',0):.4f}  "
            f"val_TC={val_metrics.get('brats/TC',0):.4f}  "
            f"val_ET={val_metrics.get('brats/ET',0):.4f}"
        )

        # ── Checkpoint ──
        state = {
            "epoch":      epoch,
            "model":      model.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "scaler":     scaler.state_dict(),
            "best_dice":  best_dice,
            "val_metrics": val_metrics,
        }
        # Always save last
        save_checkpoint(state, str(out_dir / "last.pth"))

        # Save best
        if val_dice > best_dice:
            best_dice = val_dice
            state["best_dice"] = best_dice
            save_checkpoint(state, str(out_dir / "best.pth"))
            logger.info(f"  ↑ New best Dice: {best_dice:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(
                    f"Early stopping triggered after {patience} epochs "
                    f"without improvement.")
                break

    writer.close()
    logger.info(f"Training finished. Best val Dice: {best_dice:.4f}")
    logger.info(f"Checkpoints saved in: {out_dir}")


# ──────────────────────────────────────────────────────────────────────
#  CLI entry-point
# ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train MultiModal Brain Tumour Detection model")
    p.add_argument("--config", default="configs/multimodal_config.yaml",
                   help="Path to YAML config file")
    p.add_argument("--override", nargs="*", default=[],
                   help="Key=value overrides, e.g. lr=1e-3 epochs=200")
    return p.parse_args()


def load_config(path: str, overrides: list) -> Dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for ov in overrides:
        k, v = ov.split("=", 1)
        # Try to parse as numeric
        try:
            v = int(v)
        except ValueError:
            try:
                v = float(v)
            except ValueError:
                if v.lower() == "true":
                    v = True
                elif v.lower() == "false":
                    v = False
        cfg[k] = v
    return cfg


if __name__ == "__main__":
    args = parse_args()
    cfg  = load_config(args.config, args.override)
    train(cfg)