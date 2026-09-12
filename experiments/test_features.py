import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt


# Make the LUCAS root importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ai_engine.preprocessing import preprocess_image
from ai_engine.features import detect_features


TMC_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "tmc2_apollo16_candidate.png"
)

LRO_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "lro_apollo16"
    / "NAC_ROI_APOLLO16LOA_E090S0155_5M.TIF"
)


def load_image(path):
    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise RuntimeError(f"Could not read: {path}")

    return image


print("Loading images...")

tmc = load_image(TMC_PATH)
lro = load_image(LRO_PATH)

print("TMC shape:", tmc.shape)
print("LRO shape:", lro.shape)


print("\nPreprocessing...")

tmc_processed = preprocess_image(tmc)
lro_processed = preprocess_image(lro)


print("\nDetecting TMC features...")

tmc_keypoints, tmc_descriptors = detect_features(
    tmc_processed
)

print(
    "TMC keypoints:",
    len(tmc_keypoints)
)

print(
    "TMC descriptor shape:",
    tmc_descriptors.shape
    if tmc_descriptors is not None
    else None
)


print("\nDetecting LRO features...")

lro_keypoints, lro_descriptors = detect_features(
    lro_processed
)

print(
    "LRO keypoints:",
    len(lro_keypoints)
)

print(
    "LRO descriptor shape:",
    lro_descriptors.shape
    if lro_descriptors is not None
    else None
)


# ============================================================
# VISUALIZE KEYPOINTS
# ============================================================

tmc_visual = cv2.drawKeypoints(
    tmc_processed,
    tmc_keypoints,
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

lro_visual = cv2.drawKeypoints(
    lro_processed,
    lro_keypoints,
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)


plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(tmc_visual, cmap="gray")
plt.title(f"TMC-2 — {len(tmc_keypoints)} keypoints")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(lro_visual, cmap="gray")
plt.title(f"LRO — {len(lro_keypoints)} keypoints")
plt.axis("off")

plt.tight_layout()
plt.show()