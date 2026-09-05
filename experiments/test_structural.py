import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import matplotlib.pyplot as plt

from ai_engine.structural import gradient_representation


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


print("Loading images...")

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


print("Creating structural representations...")

tmc_structure = gradient_representation(tmc)
lro_structure = gradient_representation(lro)


output = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "structural_comparison.png"
)


plt.figure(figsize=(14, 8))

plt.subplot(1, 2, 1)
plt.imshow(tmc_structure, cmap="gray")
plt.title("TMC-2 Gradient Structure")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(lro_structure, cmap="gray")
plt.title("LRO Gradient Structure")
plt.axis("off")

plt.tight_layout()
plt.savefig(
    str(output),
    dpi=150
)
plt.close()

print(f"Saved: {output}")