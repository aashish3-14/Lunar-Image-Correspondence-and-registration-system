import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from ai_engine.structural import gradient_representation
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


print("=" * 55)
print("LUCAS STRUCTURAL MATCHING EXPERIMENT")
print("=" * 55)


# ---------------------------------------------------------
# Load
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
    raise RuntimeError("Could not load TMC.")

if lro is None:
    raise RuntimeError("Could not load LRO.")


print(f"TMC: {tmc.shape}")
print(f"LRO: {lro.shape}")


# ---------------------------------------------------------
# Structural representation
# ---------------------------------------------------------

print("\nCreating gradient representations...")

tmc_structure = gradient_representation(tmc)
lro_structure = gradient_representation(lro)


# ---------------------------------------------------------
# Feature detection
# ---------------------------------------------------------

print("\nDetecting SIFT features...")

tmc_kp, tmc_desc = detect_features(
    tmc_structure,
    n_features=5000
)

lro_kp, lro_desc = detect_features(
    lro_structure,
    n_features=5000
)

print(f"TMC structural keypoints: {len(tmc_kp)}")
print(f"LRO structural keypoints: {len(lro_kp)}")


# ---------------------------------------------------------
# Matching
# ---------------------------------------------------------

print("\nReciprocal matching diagnostic...")

for ratio in [0.70, 0.75, 0.80, 0.85, 0.90]:

    matches = match_features(
        tmc_desc,
        lro_desc,
        ratio_threshold=ratio
    )

    print(
        f"Ratio {ratio:.2f} -> "
        f"{len(matches)} reciprocal matches"
    )


print("\n" + "=" * 55)
print("STRUCTURAL MATCHING EXPERIMENT COMPLETE")
print("=" * 55)