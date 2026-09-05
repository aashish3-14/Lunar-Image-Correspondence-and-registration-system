import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import matplotlib.pyplot as plt

from ai_engine.structural import gradient_representation
from ai_engine.features import detect_features
from ai_engine.matching import match_features
from ai_engine.geometry import estimate_homography


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
    / "structural_geometry_matches.png"
)


print("=" * 55)
print("LUCAS STRUCTURAL GEOMETRIC VERIFICATION")
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
# Gradient representation
# ---------------------------------------------------------

print("\nCreating structural representations...")

tmc_structure = gradient_representation(tmc)
lro_structure = gradient_representation(lro)


# ---------------------------------------------------------
# Features
# ---------------------------------------------------------

print("\nDetecting features...")

tmc_kp, tmc_desc = detect_features(
    tmc_structure,
    n_features=5000
)

lro_kp, lro_desc = detect_features(
    lro_structure,
    n_features=5000
)

print(f"TMC keypoints: {len(tmc_kp)}")
print(f"LRO keypoints: {len(lro_kp)}")


# ---------------------------------------------------------
# Matching
# ---------------------------------------------------------

print("\nMatching...")

matches = match_features(
    tmc_desc,
    lro_desc,
    ratio_threshold=0.90
)

print(f"Candidate reciprocal matches: {len(matches)}")


if len(matches) < 4:
    print("Not enough matches for homography.")
    raise SystemExit


# ---------------------------------------------------------
# RANSAC
# ---------------------------------------------------------

print("\nEstimating homography...")

H, inlier_mask = estimate_homography(
    tmc_kp,
    lro_kp,
    matches,
    reprojection_threshold=5.0
)

if H is None or inlier_mask is None:
    print("Homography estimation failed.")
    raise SystemExit


inlier_mask = np.asarray(
    inlier_mask,
    dtype=bool
)

inliers = [
    match
    for match, is_inlier in zip(matches, inlier_mask)
    if is_inlier
]


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

inlier_count = len(inliers)

inlier_ratio = (
    inlier_count / len(matches)
)

print(f"RANSAC inliers: {inlier_count}")
print(f"Inlier ratio: {inlier_ratio:.3f}")


# ---------------------------------------------------------
# Homography
# ---------------------------------------------------------

print("\nEstimated homography:")

np.set_printoptions(
    precision=6,
    suppress=True
)

print(H)

print("\nHomography diagnostics:")

print(
    f"Determinant: "
    f"{np.linalg.det(H):.6e}"
)

print(
    f"Matrix rank: "
    f"{np.linalg.matrix_rank(H)}"
)


# ---------------------------------------------------------
# Inlier coordinates
# ---------------------------------------------------------

source_points = np.float32([
    tmc_kp[m.queryIdx].pt
    for m in inliers
])

reference_points = np.float32([
    lro_kp[m.trainIdx].pt
    for m in inliers
])


if len(inliers) > 0:

    print("\nTMC inlier distribution:")

    print(
        f"X: {source_points[:,0].min():.2f} "
        f"to {source_points[:,0].max():.2f}"
    )

    print(
        f"Y: {source_points[:,1].min():.2f} "
        f"to {source_points[:,1].max():.2f}"
    )


    print("\nLRO inlier distribution:")

    print(
        f"X: {reference_points[:,0].min():.2f} "
        f"to {reference_points[:,0].max():.2f}"
    )

    print(
        f"Y: {reference_points[:,1].min():.2f} "
        f"to {reference_points[:,1].max():.2f}"
    )


# ---------------------------------------------------------
# Reprojection error
# ---------------------------------------------------------

if len(inliers) >= 4:

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

    print(
        f"\nInlier RMSE: "
        f"{rmse:.3f} pixels"
    )

    print(
        f"Median error: "
        f"{np.median(errors):.3f} pixels"
    )

    print(
        f"Maximum error: "
        f"{errors.max():.3f} pixels"
    )


# ---------------------------------------------------------
# Visualize inliers
# ---------------------------------------------------------

inlier_image = cv2.drawMatches(
    tmc_structure,
    tmc_kp,
    lro_structure,
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
    f"Structural RANSAC Inliers: "
    f"{inlier_count} / {len(matches)}"
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
print("STRUCTURAL GEOMETRIC VERIFICATION COMPLETE")
print("=" * 55)