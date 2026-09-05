import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2


TMC_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "tmc2_apollo16_focused.png"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "tmc2_apollo16_oriented.png"
)


print("Loading TMC crop...")

tmc = cv2.imread(
    str(TMC_PATH),
    cv2.IMREAD_GRAYSCALE
)

if tmc is None:
    raise RuntimeError("Could not load TMC crop.")


print(f"Original shape: {tmc.shape}")

# Horizontal flip:
# left <-> right
oriented = cv2.flip(tmc, 1)

cv2.imwrite(
    str(OUTPUT_PATH),
    oriented
)

print(f"Oriented shape: {oriented.shape}")
print(f"Saved: {OUTPUT_PATH}")

print("=" * 50)
print("TMC ORIENTATION CORRECTION COMPLETE")
print("=" * 50)