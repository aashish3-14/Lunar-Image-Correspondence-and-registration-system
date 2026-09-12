import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from ai_engine.preprocessing import preprocess_image
from ai_engine.features import detect_features
from ai_engine.matching import match_features


TMC_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "tmc2_apollo16_focused.png"
)

LRO_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lro_apollo16_geographic_crop.png"
)


print("=" * 50)
print("ORIENTATION CONTROL EXPERIMENT")
print("=" * 50)


# ---------------------------------------------------------
# Load images
# ---------------------------------------------------------

tmc = cv2.imread(
    str(TMC_PATH),
    cv2.IMREAD_GRAYSCALE
)

lro = cv2.imread(
    str(LRO_PATH),
    cv2.IMREAD_GRAYSCALE
)

if tmc is None:
    raise RuntimeError("Could not load TMC image.")

if lro is None:
    raise RuntimeError("Could not load LRO image.")


print(f"\nOriginal TMC: {tmc.shape}")
print(f"LRO crop:     {lro.shape}")


# ---------------------------------------------------------
# Create two TMC orientations
# ---------------------------------------------------------

tmc_original = tmc

tmc_flipped = cv2.flip(
    tmc,
    1
)


# ---------------------------------------------------------
# Preprocess
# ---------------------------------------------------------

print("\nPreprocessing...")

tmc_original = preprocess_image(tmc_original)
tmc_flipped = preprocess_image(tmc_flipped)
lro_processed = preprocess_image(lro)


# ---------------------------------------------------------
# Detect features
# ---------------------------------------------------------

print("\nDetecting features...")

original_kp, original_desc = detect_features(
    tmc_original,
    n_features=5000
)

flipped_kp, flipped_desc = detect_features(
    tmc_flipped,
    n_features=5000
)

lro_kp, lro_desc = detect_features(
    lro_processed,
    n_features=5000
)

print(f"Original TMC keypoints: {len(original_kp)}")
print(f"Flipped TMC keypoints:  {len(flipped_kp)}")
print(f"LRO keypoints:          {len(lro_kp)}")


# ---------------------------------------------------------
# Compare orientations
# ---------------------------------------------------------

for ratio in [0.70, 0.75, 0.80, 0.85, 0.90]:

    original_matches = match_features(
        original_desc,
        lro_desc,
        ratio_threshold=ratio
    )

    flipped_matches = match_features(
        flipped_desc,
        lro_desc,
        ratio_threshold=ratio
    )

    print(
        f"\nRatio {ratio:.2f}"
    )

    print(
        f"  Original TMC -> "
        f"{len(original_matches)} reciprocal matches"
    )

    print(
        f"  Flipped TMC  -> "
        f"{len(flipped_matches)} reciprocal matches"
    )


print("\n" + "=" * 50)
print("CONTROL EXPERIMENT COMPLETE")
print("=" * 50)