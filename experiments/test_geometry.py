import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import matplotlib.pyplot as plt

from ai_engine.preprocessing import preprocess_image
from ai_engine.features import detect_features
from ai_engine.matching import match_features
from ai_engine.geometry import estimate_homography


# =========================================================
# Paths
# =========================================================

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

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "geometry_matches.png"
)


# =========================================================
# Load
# =========================================================

print("=" * 55)
print("LUCAS GEOMETRIC VERIFICATION")
print("=" * 55)

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


print(f"TMC: {tmc.shape}")
print(f"LRO: {lro.shape}")


# =========================================================
# Preprocessing
# =========================================================

print("\nPreprocessing...")

tmc_processed = preprocess_image(tmc)
lro_processed = preprocess_image(lro)


# =========================================================
# Features
# =========================================================

print("\nDetecting features...")

tmc_kp, tmc_desc = detect_features(
    tmc_processed,
    n_features=5000
)

lro_kp, lro_desc = detect_features(
    lro_processed,
    n_features=5000
)

print(f"TMC keypoints: {len(tmc_kp)}")
print(f"LRO keypoints: {len(lro_kp)}")


# =========================================================
# Matching
# =========================================================

print("\nMatching...")

matches = match_features(
    tmc_desc,
    lro_desc,
    ratio_threshold=0.90
)

print(f"Candidate reciprocal matches: {len(matches)}")


# =========================================================
# Need at least 4 points
# =========================================================

if len(matches) < 4:
    print("\nNot enough matches for geometric verification.")
    raise SystemExit


# =========================================================
# Estimate homography
# =========================================================

print("\nEstimating homography with RANSAC...")

H, inlier_mask = estimate_homography(
    tmc_kp,
    lro_kp,
    matches,
    reprojection_threshold=5.0
)


if H is None or inlier_mask is None:
    print("Homography estimation failed.")
    raise SystemExit


inlier_mask = np.asarray(inlier_mask, dtype=bool)

inliers = [
    match
    for match, is_inlier in zip(matches, inlier_mask)
    if is_inlier
]


print(f"RANSAC inliers: {len(inliers)}")

inlier_ratio = len(inliers) / len(matches)

print(f"Inlier ratio: {inlier_ratio:.3f}")


# =========================================================
# Print homography
# =========================================================

print("\nEstimated homography:")

np.set_printoptions(
    precision=6,
    suppress=True
)

print(H)


# =========================================================
# Check homography validity
# =========================================================

if H is not None:

    determinant = np.linalg.det(H)

    matrix_rank = np.linalg.matrix_rank(H)

    print("\nHomography diagnostics:")
    print(f"Determinant: {determinant:.6e}")
    print(f"Matrix rank: {matrix_rank}")


# =========================================================
# Extract point coordinates
# =========================================================

source_points = np.float32([
    tmc_kp[m.queryIdx].pt
    for m in inliers
])

reference_points = np.float32([
    lro_kp[m.trainIdx].pt
    for m in inliers
])


# =========================================================
# Spatial distribution
# =========================================================

if len(inliers) > 0:

    source_x = source_points[:, 0]
    source_y = source_points[:, 1]

    reference_x = reference_points[:, 0]
    reference_y = reference_points[:, 1]

    print("\nTMC inlier distribution:")

    print(
        f"X: {source_x.min():.2f} "
        f"to {source_x.max():.2f}"
    )

    print(
        f"Y: {source_y.min():.2f} "
        f"to {source_y.max():.2f}"
    )

    print("\nLRO inlier distribution:")

    print(
        f"X: {reference_x.min():.2f} "
        f"to {reference_x.max():.2f}"
    )

    print(
        f"Y: {reference_y.min():.2f} "
        f"to {reference_y.max():.2f}"
    )


# =========================================================
# Reprojection error
# =========================================================

projected = cv2.perspectiveTransform(
    source_points.reshape(-1, 1, 2),
    H
).reshape(-1, 2)

errors = np.linalg.norm(
    projected - reference_points,
    axis=1
)

rmse = np.sqrt(
    np.mean(errors ** 2)
)

print(f"\nInlier RMSE: {rmse:.3f} pixels")

if len(errors) > 0:

    print(
        f"Median error: "
        f"{np.median(errors):.3f} pixels"
    )

    print(
        f"Maximum error: "
        f"{errors.max():.3f} pixels"
    )


# =========================================================
# Visualize ONLY RANSAC inliers
# =========================================================

inlier_image = cv2.drawMatches(
    tmc_processed,
    tmc_kp,
    lro_processed,
    lro_kp,
    inliers,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)


plt.figure(figsize=(18, 10))

plt.imshow(
    inlier_image,
    cmap="gray"
)

plt.title(
    f"RANSAC Inliers: {len(inliers)} / "
    f"{len(matches)}"
)

plt.axis("off")

plt.tight_layout()

plt.savefig(
    str(OUTPUT_PATH),
    dpi=150
)

plt.close()

print(f"\nSaved: {OUTPUT_PATH}")

print("\n" + "=" * 55)
print("GEOMETRIC VERIFICATION COMPLETE")
print("=" * 55)