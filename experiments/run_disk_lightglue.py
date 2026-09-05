import cv2
import torch
import numpy as np

from lightglue import LightGlue, DISK
from lightglue.utils import load_image, rbd


SOURCE_PATH = r"data/processed/tmc2_apollo16_focused.png"
REFERENCE_PATH = r"data/processed/lro_apollo16_overlap.png"


print("Device:", "cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------
# Load images
# ---------------------------------------------------------

print("\nLoading images...")

source = load_image(SOURCE_PATH).to(device)
reference = load_image(REFERENCE_PATH).to(device)

print("Source shape:", tuple(source.shape))
print("Reference shape:", tuple(reference.shape))


# ---------------------------------------------------------
# Load DISK
# ---------------------------------------------------------

print("\nLoading DISK...")

extractor = DISK(
    max_num_keypoints=2048
).eval().to(device)


# ---------------------------------------------------------
# Load LightGlue
# ---------------------------------------------------------

print("Loading LightGlue...")

matcher = LightGlue(
    features="disk"
).eval().to(device)


# ---------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------

print("\nExtracting features...")

with torch.inference_mode():

    feats_source = extractor.extract(source)
    feats_reference = extractor.extract(reference)

    print(
        "Source keypoints:",
        feats_source["keypoints"].shape[1]
    )

    print(
        "Reference keypoints:",
        feats_reference["keypoints"].shape[1]
    )


# ---------------------------------------------------------
# Matching
# ---------------------------------------------------------

print("\nRunning LightGlue...")

with torch.inference_mode():

    matches01 = matcher({
        "image0": feats_source,
        "image1": feats_reference,
    })

feats_source, feats_reference, matches01 = [
    rbd(x) for x in (
        feats_source,
        feats_reference,
        matches01
    )
]

matches = matches01["matches"].cpu().numpy()

print("Matches:", len(matches))


# ---------------------------------------------------------
# Extract matched coordinates
# ---------------------------------------------------------

source_points = (
    feats_source["keypoints"]
    .cpu()
    .numpy()[matches[:, 0]]
)

reference_points = (
    feats_reference["keypoints"]
    .cpu()
    .numpy()[matches[:, 1]]
)


# ---------------------------------------------------------
# RANSAC
# ---------------------------------------------------------

print("\nRunning RANSAC...")

if len(matches) >= 4:

    H, mask = cv2.findHomography(
        source_points,
        reference_points,
        cv2.RANSAC,
        5.0
    )

    if mask is not None:

        mask = mask.ravel().astype(bool)

        inlier_source = source_points[mask]
        inlier_reference = reference_points[mask]

        predicted = cv2.perspectiveTransform(
            inlier_source.reshape(-1, 1, 2).astype(np.float32),
            H
        ).reshape(-1, 2)

        errors = np.linalg.norm(
            predicted - inlier_reference,
            axis=1
        )

        inlier_count = len(errors)
        inlier_ratio = inlier_count / len(matches)

        rmse = np.sqrt(np.mean(errors ** 2))
        median_error = np.median(errors)
        max_error = np.max(errors)

        print("\n========== LUCAS DISK RESULT ==========")
        print(f"Matches:         {len(matches)}")
        print(f"Inliers:         {inlier_count}")
        print(f"Inlier ratio:    {inlier_ratio:.3f}")
        print(f"RMSE:            {rmse:.3f} px")
        print(f"Median error:    {median_error:.3f} px")
        print(f"Max error:       {max_error:.3f} px")

        print("\nHomography:")
        print(H)

        print("========================================")

    else:
        print("RANSAC failed: no homography found.")

else:
    print("Not enough matches for RANSAC.")