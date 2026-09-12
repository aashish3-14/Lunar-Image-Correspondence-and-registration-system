import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import matplotlib.pyplot as plt

from ai_engine.preprocessing import preprocess_image

PROJECT_ROOT = Path(__file__).resolve().parent.parent

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


def show_preprocessing(name, image):
    processed = preprocess_image(image)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap="gray")
    plt.title(f"{name} - Before")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(processed, cmap="gray")
    plt.title(f"{name} - After CLAHE")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


print("Loading TMC-2 crop...")
tmc = cv2.imread(
    str(TMC_PATH),
    cv2.IMREAD_GRAYSCALE
)

if tmc is None:
    raise RuntimeError(f"Could not read TMC image: {TMC_PATH}")

print("TMC shape:", tmc.shape)


print("\nLoading LRO...")
lro = cv2.imread(
    str(LRO_PATH),
    cv2.IMREAD_GRAYSCALE
)

if lro is None:
    raise RuntimeError(f"Could not read LRO image: {LRO_PATH}")

print("LRO shape:", lro.shape)


print("\nRunning preprocessing...")

show_preprocessing("TMC-2", tmc)
show_preprocessing("LRO", lro)

print("\nPreprocessing test complete!")