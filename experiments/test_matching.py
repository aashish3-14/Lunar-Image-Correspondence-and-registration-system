import sys
from pathlib import Path

import cv2


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# LUCAS MODULES
# ============================================================

from ai_engine.preprocessing import preprocess_image
from ai_engine.features import detect_features
from ai_engine.matching import match_features


# ============================================================
# IMAGE PATHS
# ============================================================

TMC_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "tmc2_apollo16_focused.png"
)

LRO_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "lro_apollo16"
    / "NAC_ROI_APOLLO16LOA_E090S0155_5M.TIF"
)


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(path):

    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        raise RuntimeError(
            f"Could not read: {path}"
        )

    return image


print("Loading images...")

tmc = load_image(TMC_PATH)
lro = load_image(LRO_PATH)

print("TMC:", tmc.shape)
print("LRO:", lro.shape)


# ============================================================
# PREPROCESSING
# ============================================================

print("\nPreprocessing...")

tmc_processed = preprocess_image(tmc)
lro_processed = preprocess_image(lro)


# ============================================================
# FEATURE DETECTION
# ============================================================

print("\nDetecting features...")

tmc_keypoints, tmc_descriptors = detect_features(
    tmc_processed
)

lro_keypoints, lro_descriptors = detect_features(
    lro_processed
)

print(
    "TMC keypoints:",
    len(tmc_keypoints)
)

print(
    "LRO keypoints:",
    len(lro_keypoints)
)


# ============================================================
# RECIPROCAL MATCHING DIAGNOSTIC
# ============================================================

print("\nReciprocal matching diagnostic...")

for ratio in [0.70, 0.75, 0.80, 0.85, 0.90]:

    matches = match_features(
        tmc_descriptors,
        lro_descriptors,
        ratio_threshold=ratio,
    )

    print(
        f"Ratio {ratio:.2f} -> "
        f"{len(matches)} reciprocal matches"
    )


print("\n========================================")
print("RECIPROCAL MATCHING TEST COMPLETE")
print("========================================")