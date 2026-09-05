from pathlib import Path

import numpy as np
import cv2


# ============================================================
# LUCAS - DATA INSPECTION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# FILE PATHS
# ============================================================

TMC_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "tmc2_apollo16"
    / "data"
    / "calibrated"
    / "20210720"
    / "ch2_tmc_ncn_20210720T2333035757_d_img_d32.img"
)

LRO_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "lro_apollo16"
    / "NAC_ROI_APOLLO16LOA_E090S0155_5M.TIF"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. READ TMC-2
# ============================================================

print("\n========== TMC-2 ==========")

tmc_height = 147326
tmc_width = 4000

tmc = np.memmap(
    TMC_PATH,
    dtype="<u2",
    mode="r",
    shape=(tmc_height, tmc_width)
)

print("TMC shape:", tmc.shape)
print("TMC dtype:", tmc.dtype)
print(
    "TMC size:",
    TMC_PATH.stat().st_size / (1024**3),
    "GB"
)


# ============================================================
# 2. FULL TMC PREVIEW
# ============================================================

print("\nCreating TMC preview...")

tmc_preview = np.asarray(
    tmc[::200, ::20]
)

print(
    "Preview shape:",
    tmc_preview.shape
)

low, high = np.percentile(
    tmc_preview,
    [2, 98]
)

tmc_preview_8bit = np.clip(
    (tmc_preview - low)
    / (high - low)
    * 255,
    0,
    255
).astype(np.uint8)

tmc_preview_path = (
    OUTPUT_DIR /
    "tmc2_full_preview.png"
)

cv2.imwrite(
    str(tmc_preview_path),
    tmc_preview_8bit
)

print(
    "Saved:",
    tmc_preview_path
)


# ============================================================
# 3. APOLLO-16 FOCUSED TMC CANDIDATE
# ============================================================

print(
    "\nExtracting focused Apollo-16 candidate..."
)

center_row = 99000
center_col = 1700

crop_height = 8000
crop_width = 3200


start_row = (
    center_row -
    crop_height // 2
)

end_row = (
    center_row +
    crop_height // 2
)

start_col = (
    center_col -
    crop_width // 2
)

end_col = (
    center_col +
    crop_width // 2
)


tmc_crop = np.asarray(
    tmc[
        start_row:end_row,
        start_col:end_col
    ]
)

print(
    "Focused crop shape:",
    tmc_crop.shape
)

print(
    "Rows:",
    start_row,
    "to",
    end_row
)

print(
    "Columns:",
    start_col,
    "to",
    end_col
)


# Contrast normalization

low, high = np.percentile(
    tmc_crop,
    [2, 98]
)

tmc_crop_8bit = np.clip(
    (tmc_crop - low)
    / (high - low)
    * 255,
    0,
    255
).astype(np.uint8)


crop_path = (
    OUTPUT_DIR /
    "tmc2_apollo16_focused.png"
)

cv2.imwrite(
    str(crop_path),
    tmc_crop_8bit
)

print(
    "Saved:",
    crop_path
)


# ============================================================
# 4. READ LRO NAC
# ============================================================

print("\n========== LRO NAC ==========")

lro = cv2.imread(
    str(LRO_PATH),
    cv2.IMREAD_GRAYSCALE
)

if lro is None:
    raise RuntimeError(
        f"Could not read LRO image: {LRO_PATH}"
    )

print(
    "LRO shape:",
    lro.shape
)

print(
    "LRO dtype:",
    lro.dtype
)

print(
    "LRO size:",
    LRO_PATH.stat().st_size / (1024**2),
    "MB"
)


# ============================================================
# 5. LRO PREVIEW
# ============================================================

print("\nCreating LRO preview...")

low, high = np.percentile(
    lro,
    [2, 98]
)

lro_preview_8bit = np.clip(
    (lro - low)
    / (high - low)
    * 255,
    0,
    255
).astype(np.uint8)


lro_preview_path = (
    OUTPUT_DIR /
    "lro_apollo16_preview.png"
)

cv2.imwrite(
    str(lro_preview_path),
    lro_preview_8bit
)

print(
    "Saved:",
    lro_preview_path
)


# ============================================================
# 6. SUMMARY
# ============================================================

print(
    "\n========================================"
)

print(
    "LUCAS DATA INSPECTION COMPLETE"
)

print(
    "========================================"
)

print("\nTMC-2:")
print("  Full shape :", tmc.shape)
print("  dtype      :", tmc.dtype)

print("\nFocused TMC candidate:")
print("  shape      :", tmc_crop.shape)
print(
    "  rows       :",
    start_row,
    "to",
    end_row
)
print(
    "  columns    :",
    start_col,
    "to",
    end_col
)

print("\nLRO NAC:")
print("  shape      :", lro.shape)
print("  dtype      :", lro.dtype)

print("\nGenerated files:")

print(
    "  Full TMC:",
    tmc_preview_path
)

print(
    "  Focused TMC:",
    crop_path
)

print(
    "  LRO:",
    lro_preview_path
)