import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import matplotlib.pyplot as plt

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
    / "raw"
    / "lro_apollo16"
    / "NAC_ROI_APOLLO16LOA_E090S0155_5M.TIF"
)


print("Loading images...")

tmc = cv2.imread(str(TMC_PATH), cv2.IMREAD_GRAYSCALE)
lro = cv2.imread(str(LRO_PATH), cv2.IMREAD_GRAYSCALE)

print("Preprocessing...")

tmc_processed = preprocess_image(tmc)
lro_processed = preprocess_image(lro)

print("Detecting features...")

tmc_kp, tmc_desc = detect_features(tmc_processed)
lro_kp, lro_desc = detect_features(lro_processed)

print("Matching...")

matches = match_features(
    tmc_desc,
    lro_desc,
    ratio_threshold=0.90
)

print(f"Reciprocal matches: {len(matches)}")

matched_image = cv2.drawMatches(
    tmc_processed,
    tmc_kp,
    lro_processed,
    lro_kp,
    matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

plt.figure(figsize=(18, 10))
plt.imshow(matched_image, cmap="gray")
plt.title(f"Reciprocal SIFT Matches: {len(matches)}")
plt.axis("off")
plt.tight_layout()

output_path = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "reciprocal_matches.png"
)

plt.savefig(output_path, dpi=150)
plt.close()

print(f"Saved: {output_path}")