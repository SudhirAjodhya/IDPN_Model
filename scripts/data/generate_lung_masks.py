"""
generate_lung_masks.py
======================
Generate binary lung masks for JSRT + NODE21 training data using
torchxrayvision's PSPNet lung segmenter.
Masks are cached as PNG files in data/lung_masks/ and loaded during training.

Usage:
    python scripts/data/generate_lung_masks.py

Prerequisites:
    pip install torchxrayvision
"""

import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR   = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# ---- Paths ----
JSRT_IMAGE_DIR   = ROOT_DIR / "data_sets" / "JSRT" / "images" / "images"
JSRT_CSV         = ROOT_DIR / "data_sets" / "JSRT" / "jsrt_metadata.csv"
NODE21_IMAGE_DIR = ROOT_DIR / "data_sets" / "node21" / "cxr_images" / "original_data" / "images"
NODE21_CSV       = ROOT_DIR / "data_sets" / "node21" / "cxr_images" / "original_data" / "metadata.csv"
MASK_OUTPUT_DIR  = ROOT_DIR / "data" / "lung_masks"
MASK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MASK_SIZE = 512

# PSPNet lung channel indices (per torchxrayvision docs):
#   4 = Left Lung, 5 = Right Lung
LUNG_CHANNELS = [4, 5]


def load_segmenter():
    try:
        import torchxrayvision as xrv
    except ImportError:
        print("ERROR: torchxrayvision not installed. Run: pip install torchxrayvision")
        sys.exit(1)

    print("Loading torchxrayvision PSPNet lung segmenter...")
    seg = xrv.baseline_models.chestx_det.PSPNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seg.eval().to(device)
    print(f"  Segmenter on {device}")
    return seg, device


def image_to_mask(image_path, seg_model, device, size=MASK_SIZE):
    """Load image, run PSPNet, return binary lung mask resized to `size`."""
    img = cv2.imread(str(image_path), cv2.IMREAD_ANYDEPTH)
    if img is None:
        # Try MHA via SimpleITK (NODE21)
        try:
            import SimpleITK as sitk
            img_sitk = sitk.ReadImage(str(image_path))
            img = sitk.GetArrayFromImage(img_sitk).squeeze()
        except Exception:
            return None

    img_small = cv2.resize(img.astype(np.float32), (224, 224), interpolation=cv2.INTER_AREA)
    mx = img_small.max()
    img_norm = (img_small / (mx + 1e-8)) * 2048 - 1024
    img_tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        seg_logits = seg_model(img_tensor)                    # (1, C, 224, 224)
        lung_logits = seg_logits[0, LUNG_CHANNELS].sum(dim=0) # (224, 224)
        lung_mask = (torch.sigmoid(lung_logits) > 0.5).float().cpu().numpy()

    lung_mask = cv2.resize(lung_mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return lung_mask.astype(np.uint8)


def process_jsrt(seg_model, device):
    if not JSRT_CSV.exists():
        print(f"JSRT CSV not found: {JSRT_CSV}  — skipping JSRT")
        return
    df = pd.read_csv(JSRT_CSV)
    print(f"\nProcessing {len(df)} JSRT images...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="JSRT"):
        img_name = row["study_id"]
        img_path = JSRT_IMAGE_DIR / img_name
        out_name = f"jsrt_{img_name.replace('.', '_')}.png"
        out_path = MASK_OUTPUT_DIR / out_name
        if out_path.exists():
            continue
        mask = image_to_mask(img_path, seg_model, device)
        if mask is not None:
            cv2.imwrite(str(out_path), mask * 255)


def process_node21(seg_model, device):
    if not NODE21_CSV.exists():
        print(f"NODE21 CSV not found: {NODE21_CSV}  — skipping NODE21")
        return
    all_images = [f for f in os.listdir(NODE21_IMAGE_DIR) if f.endswith(".mha")]
    print(f"\nProcessing {len(all_images)} NODE21 images...")
    for img_name in tqdm(all_images, desc="NODE21"):
        img_path = NODE21_IMAGE_DIR / img_name
        out_name = f"node21_{img_name.replace('.', '_')}.png"
        out_path = MASK_OUTPUT_DIR / out_name
        if out_path.exists():
            continue
        mask = image_to_mask(img_path, seg_model, device)
        if mask is not None:
            cv2.imwrite(str(out_path), mask * 255)


def sanity_check():
    all_masks = list(MASK_OUTPUT_DIR.glob("*.png"))
    print(f"\nGenerated {len(all_masks)} lung masks in {MASK_OUTPUT_DIR}")
    print("Sample masks:")
    for p in sorted(all_masks)[:5]:
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        frac = m.mean() / 255.0
        status = "OK" if 0.20 <= frac <= 0.55 else "SUSPICIOUS"
        print(f"  [{status}] {p.name}: shape={m.shape} mean_lung_frac={frac:.3f}")

    fracs = []
    for p in all_masks:
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        fracs.append(m.mean() / 255.0)
    median_frac = float(np.median(fracs)) if fracs else 0.0
    print(f"\nMedian lung fraction across all masks: {median_frac:.3f}")
    if median_frac < 0.05 or median_frac > 0.90:
        print("WARNING: lung fraction out of expected range (0.20–0.55).")
        print("         Segmenter may be failing. Inspect masks and skip talebi_aux if so.")
    else:
        print("Lung fraction looks reasonable. Safe to run talebi_aux.")


if __name__ == "__main__":
    seg_model, device = load_segmenter()
    process_jsrt(seg_model, device)
    process_node21(seg_model, device)
    sanity_check()
