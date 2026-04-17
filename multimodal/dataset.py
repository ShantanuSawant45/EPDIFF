"""
multimodal/dataset.py
=====================
BraTS-style multimodal MRI dataset for brain tumour detection.
Handles T1, T1ce, T2 and FLAIR modalities simultaneously.
Supports 3-D volumetric patches, 2-D slice extraction and
optional on-the-fly augmentation.
"""

import os
import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import nibabel as nib
from scipy.ndimage import zoom, rotate as ndrotate

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────
MODALITIES = ["t1", "t1ce", "t2", "flair"]
BRATS_LABEL_MAP = {
    0: "background",
    1: "necrotic_core",          # NCR / NET
    2: "edema",                  # ED
    3: "enhancing_tumour",       # ET
}
# For BraTS 2020/2021 the label values used in segmentation masks
BRATS_SEG_LABELS = {
    0: 0,   # background
    1: 1,   # necrotic core
    2: 2,   # edema
    4: 3,   # enhancing tumour  (note: value 3 is unused in BraTS)
}

TUMOUR_CLASSES = {
    0: "no_tumour",
    1: "LGG",    # low-grade glioma
    2: "HGG",    # high-grade glioma
}


# ──────────────────────────────────────────────
#  Utility helpers
# ──────────────────────────────────────────────

def load_nifti(path: Union[str, Path]) -> np.ndarray:
    """Load a NIfTI file and return a float32 numpy array (H, W, D)."""
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    return data


def normalise_volume(volume: np.ndarray,
                     method: str = "z_score",
                     mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Intensity normalisation.

    Parameters
    ----------
    volume : np.ndarray  (H, W, D)
    method : "z_score" | "min_max" | "percentile"
    mask   : optional brain mask – stats are computed inside the mask.
    """
    if mask is not None:
        roi = volume[mask > 0]
    else:
        roi = volume[volume > 0]

    if roi.size == 0:
        return volume

    if method == "z_score":
        mu, sigma = roi.mean(), roi.std()
        if sigma < 1e-8:
            sigma = 1.0
        out = (volume - mu) / sigma

    elif method == "min_max":
        lo, hi = roi.min(), roi.max()
        span = hi - lo if (hi - lo) > 1e-8 else 1.0
        out = (volume - lo) / span

    elif method == "percentile":
        lo, hi = np.percentile(roi, 1), np.percentile(roi, 99)
        span = hi - lo if (hi - lo) > 1e-8 else 1.0
        out = np.clip((volume - lo) / span, 0.0, 1.0)

    else:
        raise ValueError(f"Unknown normalisation method: {method}")

    return out.astype(np.float32)


def remap_brats_labels(seg: np.ndarray) -> np.ndarray:
    """
    Convert BraTS segmentation mask (values 0,1,2,4) to
    contiguous labels (0,1,2,3).
    """
    out = np.zeros_like(seg, dtype=np.int64)
    for src, dst in BRATS_SEG_LABELS.items():
        out[seg == src] = dst
    return out


def extract_brain_mask(volume: np.ndarray,
                       threshold: float = 0.0) -> np.ndarray:
    """Simple thresholding brain mask from any MRI modality."""
    return (volume > threshold).astype(np.uint8)


def random_crop_3d(volumes: List[np.ndarray],
                   seg: np.ndarray,
                   patch_size: Tuple[int, int, int],
                   fg_ratio: float = 0.5) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Random 3-D patch crop with foreground oversampling.

    Parameters
    ----------
    volumes    : list of (H, W, D) arrays, one per modality
    seg        : (H, W, D) segmentation mask
    patch_size : (pH, pW, pD)
    fg_ratio   : probability of sampling a patch that contains tumour voxels
    """
    H, W, D = volumes[0].shape
    pH, pW, pD = patch_size

    # Foreground voxels
    fg_voxels = np.argwhere(seg > 0)

    if len(fg_voxels) > 0 and random.random() < fg_ratio:
        # Sample a random foreground voxel as anchor
        anchor = fg_voxels[random.randint(0, len(fg_voxels) - 1)]
        # Shift anchor so patch stays within bounds
        h0 = int(np.clip(anchor[0] - pH // 2, 0, H - pH))
        w0 = int(np.clip(anchor[1] - pW // 2, 0, W - pW))
        d0 = int(np.clip(anchor[2] - pD // 2, 0, D - pD))
    else:
        h0 = random.randint(0, max(H - pH, 0))
        w0 = random.randint(0, max(W - pW, 0))
        d0 = random.randint(0, max(D - pD, 0))

    crops_v = [v[h0:h0+pH, w0:w0+pW, d0:d0+pD] for v in volumes]
    crops_s = seg[h0:h0+pH, w0:w0+pW, d0:d0+pD]
    return crops_v, crops_s


def pad_or_crop_to(volume: np.ndarray,
                   target_shape: Tuple[int, int, int]) -> np.ndarray:
    """Centre-pad or centre-crop a 3-D volume to target_shape."""
    out = np.zeros(target_shape, dtype=volume.dtype)
    src_slices, dst_slices = [], []
    for dim in range(3):
        src_sz = volume.shape[dim]
        tgt_sz = target_shape[dim]
        if src_sz >= tgt_sz:
            s_start = (src_sz - tgt_sz) // 2
            src_slices.append(slice(s_start, s_start + tgt_sz))
            dst_slices.append(slice(0, tgt_sz))
        else:
            d_start = (tgt_sz - src_sz) // 2
            src_slices.append(slice(0, src_sz))
            dst_slices.append(slice(d_start, d_start + src_sz))
    out[tuple(dst_slices)] = volume[tuple(src_slices)]
    return out


# ──────────────────────────────────────────────
#  Augmentation helpers
# ──────────────────────────────────────────────

class MultimodalAugmenter:
    """
    Applies the same geometric transform to all modalities + the mask.
    Intensity transforms are applied per-modality independently.
    """

    def __init__(self,
                 p_flip: float = 0.5,
                 p_rotate: float = 0.3,
                 max_angle: float = 15.0,
                 p_noise: float = 0.3,
                 noise_std: float = 0.05,
                 p_scale: float = 0.2,
                 scale_range: Tuple[float, float] = (0.9, 1.1),
                 p_bias: float = 0.2,
                 bias_range: Tuple[float, float] = (-0.1, 0.1)):
        self.p_flip = p_flip
        self.p_rotate = p_rotate
        self.max_angle = max_angle
        self.p_noise = p_noise
        self.noise_std = noise_std
        self.p_scale = p_scale
        self.scale_range = scale_range
        self.p_bias = p_bias
        self.bias_range = bias_range

    def __call__(self,
                 volumes: List[np.ndarray],
                 seg: np.ndarray
                 ) -> Tuple[List[np.ndarray], np.ndarray]:
        # ---- geometric transforms (same for all channels) ----

        # Random flips
        for axis in range(3):
            if random.random() < self.p_flip:
                volumes = [np.flip(v, axis=axis).copy() for v in volumes]
                seg = np.flip(seg, axis=axis).copy()

        # Random rotation (in-plane)
        if random.random() < self.p_rotate:
            angle = random.uniform(-self.max_angle, self.max_angle)
            axes = random.choice([(0, 1), (0, 2), (1, 2)])
            volumes = [ndrotate(v, angle, axes=axes, reshape=False, order=1)
                       for v in volumes]
            seg = ndrotate(seg.astype(np.float32), angle, axes=axes,
                           reshape=False, order=0).astype(np.int64)

        # ---- intensity transforms (independent per modality) ----
        out_volumes = []
        for v in volumes:
            # Additive Gaussian noise
            if random.random() < self.p_noise:
                v = v + np.random.randn(*v.shape).astype(np.float32) * self.noise_std

            # Multiplicative scaling (simulate scanner gain variation)
            if random.random() < self.p_scale:
                scale = random.uniform(*self.scale_range)
                v = v * scale

            # Additive bias (simulate field inhomogeneity)
            if random.random() < self.p_bias:
                bias = random.uniform(*self.bias_range)
                v = v + bias

            out_volumes.append(v.astype(np.float32))

        return out_volumes, seg


# ──────────────────────────────────────────────
#  Subject index builder
# ──────────────────────────────────────────────

def build_subject_index(data_root: Union[str, Path],
                        split: str = "train",
                        modalities: List[str] = MODALITIES,
                        label_file: Optional[str] = None
                        ) -> List[Dict]:
    """
    Walk `data_root/{split}/` and collect per-subject file paths.
    Expects BraTS-style naming:
        {subject_id}/{subject_id}_t1.nii.gz
        {subject_id}/{subject_id}_t1ce.nii.gz
        {subject_id}/{subject_id}_t2.nii.gz
        {subject_id}/{subject_id}_flair.nii.gz
        {subject_id}/{subject_id}_seg.nii.gz   (optional)

    Parameters
    ----------
    label_file : optional JSON mapping subject_id -> tumour grade (int)
    """
    data_root = Path(data_root)
    split_dir = data_root / split

    grade_map: Dict[str, int] = {}
    if label_file and Path(label_file).exists():
        with open(label_file) as f:
            grade_map = json.load(f)

    subjects = []
    for subj_dir in sorted(split_dir.iterdir()):
        if not subj_dir.is_dir():
            continue
        sid = subj_dir.name

        entry: Dict = {"subject_id": sid, "paths": {}}

        # Collect modality paths
        missing = False
        for mod in modalities:
            candidate = subj_dir / f"{sid}_{mod}.nii.gz"
            if not candidate.exists():
                # Try without .gz
                candidate = subj_dir / f"{sid}_{mod}.nii"
            if not candidate.exists():
                logger.warning(f"Missing {mod} for {sid}, skipping.")
                missing = True
                break
            entry["paths"][mod] = str(candidate)

        if missing:
            continue

        # Optional segmentation
        seg_path = subj_dir / f"{sid}_seg.nii.gz"
        if not seg_path.exists():
            seg_path = subj_dir / f"{sid}_seg.nii"
        if seg_path.exists():
            entry["paths"]["seg"] = str(seg_path)
        else:
            entry["paths"]["seg"] = None

        # Optional classification label
        entry["grade"] = grade_map.get(sid, -1)

        subjects.append(entry)

    logger.info(f"Found {len(subjects)} subjects in {split_dir}")
    return subjects


# ──────────────────────────────────────────────
#  Main Dataset class
# ──────────────────────────────────────────────

class MultimodalBraTSDataset(Dataset):
    """
    Volumetric multimodal MRI dataset for brain tumour segmentation
    and grade classification.

    Each sample returns:
        image  : (C, H, W, D) float32 tensor   — one channel per modality
        seg    : (H, W, D)    int64  tensor     — 4-class segmentation mask
        grade  : ()           int64  scalar     — tumour grade (0/1/2) or -1
        meta   : dict with subject_id and shape info

    Parameters
    ----------
    data_root      : root directory containing {train,val,test} sub-folders
    split          : "train" | "val" | "test"
    modalities     : list of modality strings to load
    patch_size     : 3-D patch size for training; None = full volume
    norm_method    : intensity normalisation strategy
    augment        : whether to apply augmentation (training only)
    fg_ratio       : foreground oversampling ratio during patch extraction
    cache_data     : if True, volumes are loaded once and kept in RAM
    label_file     : optional JSON with {subject_id: grade} mapping
    slice_dim      : if not None, extract 2-D slices along this dimension
                     (0=axial from H, 1=coronal, 2=sagittal)
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        modalities: List[str] = None,
        patch_size: Optional[Tuple[int, int, int]] = (128, 128, 128),
        norm_method: str = "z_score",
        augment: bool = False,
        fg_ratio: float = 0.5,
        cache_data: bool = False,
        label_file: Optional[str] = None,
        slice_dim: Optional[int] = None,
        target_shape: Optional[Tuple[int, int, int]] = (240, 240, 155),
    ):
        super().__init__()
        self.modalities = modalities or MODALITIES
        self.patch_size = patch_size
        self.norm_method = norm_method
        self.augment = augment
        self.fg_ratio = fg_ratio
        self.cache_data = cache_data
        self.slice_dim = slice_dim
        self.target_shape = target_shape

        self.subjects = build_subject_index(data_root, split,
                                            self.modalities, label_file)
        if len(self.subjects) == 0:
            raise RuntimeError(f"No subjects found in {data_root}/{split}")

        self.augmenter = MultimodalAugmenter() if augment else None
        self._cache: Dict[str, Dict] = {}

        # Build flat index for 2-D slice mode
        if slice_dim is not None:
            self._build_slice_index()
        else:
            self._index = list(range(len(self.subjects)))

    # ------------------------------------------------------------------
    def _build_slice_index(self):
        """Flat (subject_idx, slice_idx) index for 2-D mode."""
        self._index = []
        for si, subj in enumerate(self.subjects):
            # Peek at one volume to get depth
            vol = load_nifti(subj["paths"][self.modalities[0]])
            depth = vol.shape[self.slice_dim]
            for sl in range(depth):
                self._index.append((si, sl))

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._index)

    # ------------------------------------------------------------------
    def _load_subject(self, subj: Dict) -> Dict:
        """Load and normalise all modalities for a subject."""
        sid = subj["subject_id"]
        if sid in self._cache:
            return self._cache[sid]

        volumes = []
        for mod in self.modalities:
            vol = load_nifti(subj["paths"][mod])
            if self.target_shape:
                vol = pad_or_crop_to(vol, self.target_shape)
            brain_mask = extract_brain_mask(vol)
            vol = normalise_volume(vol, self.norm_method, mask=brain_mask)
            volumes.append(vol)

        seg = None
        if subj["paths"]["seg"] is not None:
            seg = load_nifti(subj["paths"]["seg"]).astype(np.int64)
            if self.target_shape:
                seg = pad_or_crop_to(seg, self.target_shape)
            seg = remap_brats_labels(seg)
        else:
            seg = np.zeros(volumes[0].shape, dtype=np.int64)

        data = {"volumes": volumes, "seg": seg,
                "grade": subj["grade"], "subject_id": sid}

        if self.cache_data:
            self._cache[sid] = data

        return data

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Dict:
        if self.slice_dim is not None:
            return self._get_slice_item(idx)
        return self._get_volume_item(idx)

    # ------------------------------------------------------------------
    def _get_volume_item(self, idx: int) -> Dict:
        subj_idx = self._index[idx]
        subj = self.subjects[subj_idx]
        data = self._load_subject(subj)

        volumes = [v.copy() for v in data["volumes"]]
        seg = data["seg"].copy()

        # Patch extraction
        if self.patch_size is not None:
            volumes, seg = random_crop_3d(volumes, seg,
                                          self.patch_size, self.fg_ratio)

        # Augmentation
        if self.augmenter is not None:
            volumes, seg = self.augmenter(volumes, seg)

        # Stack modalities along channel dim: (C, H, W, D)
        image = np.stack(volumes, axis=0)

        return {
            "image": torch.from_numpy(image),
            "seg":   torch.from_numpy(seg),
            "grade": torch.tensor(data["grade"], dtype=torch.long),
            "subject_id": data["subject_id"],
        }

    # ------------------------------------------------------------------
    def _get_slice_item(self, idx: int) -> Dict:
        subj_idx, slice_idx = self._index[idx]
        subj = self.subjects[subj_idx]
        data = self._load_subject(subj)

        def take_slice(arr, dim, sl):
            slices = [slice(None)] * 3
            slices[dim] = sl
            return arr[tuple(slices)]

        slices_v = [take_slice(v, self.slice_dim, slice_idx)
                    for v in data["volumes"]]
        slices_s = take_slice(data["seg"], self.slice_dim, slice_idx)

        if self.augmenter is not None:
            # Wrap 2-D arrays as 3-D with depth=1 for augmenter compatibility
            slices_v_3d = [s[..., np.newaxis] for s in slices_v]
            slices_s_3d = slices_s[..., np.newaxis]
            slices_v_3d, slices_s_3d = self.augmenter(slices_v_3d, slices_s_3d)
            slices_v = [s[..., 0] for s in slices_v_3d]
            slices_s = slices_s_3d[..., 0]

        image = np.stack(slices_v, axis=0)   # (C, H, W)

        return {
            "image": torch.from_numpy(image),
            "seg":   torch.from_numpy(slices_s),
            "grade": torch.tensor(data["grade"], dtype=torch.long),
            "subject_id": data["subject_id"],
            "slice_idx": slice_idx,
        }

    # ------------------------------------------------------------------
    def get_class_weights(self) -> torch.Tensor:
        """
        Compute per-class pixel weights for weighted cross-entropy.
        Iterates over all segmentation masks – may be slow for large datasets.
        """
        counts = np.zeros(4, dtype=np.float64)
        for subj in self.subjects:
            if subj["paths"]["seg"] is None:
                continue
            seg = remap_brats_labels(
                load_nifti(subj["paths"]["seg"]).astype(np.int64))
            for c in range(4):
                counts[c] += (seg == c).sum()
        total = counts.sum()
        weights = total / (4 * counts + 1e-8)
        return torch.tensor(weights, dtype=torch.float32)

    def get_sampler(self) -> WeightedRandomSampler:
        """
        Per-sample weights inversely proportional to tumour load.
        Helps balance subjects with tiny vs. large tumours.
        """
        sample_weights = []
        for subj in self.subjects:
            if subj["paths"]["seg"] is None:
                sample_weights.append(1.0)
                continue
            seg = remap_brats_labels(
                load_nifti(subj["paths"]["seg"]).astype(np.int64))
            fg = float((seg > 0).sum()) / seg.size
            # Inverse frequency: subjects with little tumour get higher weight
            w = 1.0 / (fg + 0.01)
            sample_weights.append(w)
        return WeightedRandomSampler(sample_weights,
                                     num_samples=len(sample_weights),
                                     replacement=True)


# ──────────────────────────────────────────────
#  DataLoader factory
# ──────────────────────────────────────────────

def build_loaders(cfg: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders from a config dict.

    Expected keys in cfg:
        data_root, patch_size, norm_method, batch_size,
        num_workers, pin_memory, fg_ratio, cache_data,
        label_file (optional)
    """
    patch = tuple(cfg.get("patch_size", [128, 128, 128]))

    train_ds = MultimodalBraTSDataset(
        data_root=cfg["data_root"],
        split="train",
        patch_size=patch,
        norm_method=cfg.get("norm_method", "z_score"),
        augment=True,
        fg_ratio=cfg.get("fg_ratio", 0.5),
        cache_data=cfg.get("cache_data", False),
        label_file=cfg.get("label_file", None),
    )
    val_ds = MultimodalBraTSDataset(
        data_root=cfg["data_root"],
        split="val",
        patch_size=None,        # full volume for validation
        norm_method=cfg.get("norm_method", "z_score"),
        augment=False,
        cache_data=cfg.get("cache_data", False),
        label_file=cfg.get("label_file", None),
    )
    test_ds = MultimodalBraTSDataset(
        data_root=cfg["data_root"],
        split="test",
        patch_size=None,
        norm_method=cfg.get("norm_method", "z_score"),
        augment=False,
        cache_data=False,
        label_file=cfg.get("label_file", None),
    )

    nw = cfg.get("num_workers", 4)
    bs = cfg.get("batch_size", 2)
    pin = cfg.get("pin_memory", True)

    sampler = train_ds.get_sampler() if cfg.get("use_sampler", False) else None
    shuffle_train = sampler is None

    train_loader = DataLoader(train_ds, batch_size=bs,
                              shuffle=shuffle_train, sampler=sampler,
                              num_workers=nw, pin_memory=pin,
                              drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=1,
                              shuffle=False, num_workers=nw, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=1,
                              shuffle=False, num_workers=nw, pin_memory=pin)

    logger.info(f"Train: {len(train_ds)} subjects | "
                f"Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader