from pathlib import Path

import cv2
import numpy as np
import torch

from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image

from ai_engine.quality import evaluate_registration_quality


# =========================================================
# Paths
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

SOURCE_PATH = (
    ROOT
    / "data"
    / "processed"
    / "tmc2_apollo16_focused.png"
)

REFERENCE_PATH = (
    ROOT
    / "data"
    / "processed"
    / "lro_apollo16_overlap.png"
)

RESULTS_DIR = ROOT / "data" / "results"

ALL_MATCHES_PATH = (
    RESULTS_DIR / "lightglue_all_matches.png"
)

INLIER_MATCHES_PATH = (
    RESULTS_DIR / "lightglue_inlier_matches.png"
)


# =========================================================
# Device
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("==========================================")
print("              LUCAS ENGINE")
print("==========================================")

print(f"Device: {device}")

if device.type == "cuda":
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# =========================================================
# Load images
# =========================================================

print("\nLoading images...")

source = load_image(
    str(SOURCE_PATH)
).to(device)

reference = load_image(
    str(REFERENCE_PATH)
).to(device)

print(
    "Source shape:",
    tuple(source.shape)
)

print(
    "Reference shape:",
    tuple(reference.shape)
)


# =========================================================
# Load models
# =========================================================

print("\nLoading SuperPoint...")

extractor = SuperPoint(
    max_num_keypoints=4096
).eval().to(device)


print("Loading LightGlue...")

matcher = LightGlue(
    features="superpoint",
).eval().to(device)


# =========================================================
# Feature extraction
# =========================================================

print("\nExtracting features...")

with torch.inference_mode():

    source_features = extractor.extract(
        source
    )

    reference_features = extractor.extract(
        reference
    )


# =========================================================
# Extract keypoints
# =========================================================

source_keypoints = (
    source_features["keypoints"][0]
    .detach()
    .cpu()
    .numpy()
)

reference_keypoints = (
    reference_features["keypoints"][0]
    .detach()
    .cpu()
    .numpy()
)

print(
    "Source keypoints:",
    len(source_keypoints)
)

print(
    "Reference keypoints:",
    len(reference_keypoints)
)


# =========================================================
# LightGlue matching
# =========================================================

print("\nRunning LightGlue...")

with torch.inference_mode():

    matches01 = matcher(
        {
            "image0": source_features,
            "image1": reference_features,
        }
    )


# =========================================================
# Extract matches
# =========================================================

matches = (
    matches01["matches"][0]
    .detach()
    .cpu()
    .numpy()
)

print(
    "Matches:",
    len(matches)
)


# =========================================================
# Extract matched coordinates
# =========================================================

if len(matches) > 0:

    source_points = (
        source_keypoints[
            matches[:, 0]
        ]
    )

    reference_points = (
        reference_keypoints[
            matches[:, 1]
        ]
    )

else:

    source_points = np.empty(
        (0, 2),
        dtype=np.float32,
    )

    reference_points = np.empty(
        (0, 2),
        dtype=np.float32,
    )


# =========================================================
# Load original images
# =========================================================

source_img = cv2.imread(
    str(SOURCE_PATH),
    cv2.IMREAD_GRAYSCALE,
)

reference_img = cv2.imread(
    str(REFERENCE_PATH),
    cv2.IMREAD_GRAYSCALE,
)


if source_img is None:

    raise FileNotFoundError(
        f"Could not load source image: "
        f"{SOURCE_PATH}"
    )


if reference_img is None:

    raise FileNotFoundError(
        f"Could not load reference image: "
        f"{REFERENCE_PATH}"
    )


# =========================================================
# Correspondence visualization
# =========================================================

def draw_correspondences(
    image0,
    image1,
    points0,
    points1,
    inlier_mask=None,
):

    h0, w0 = image0.shape
    h1, w1 = image1.shape

    canvas_height = max(
        h0,
        h1,
    )

    canvas_width = (
        w0 + w1
    )

    canvas = np.zeros(
        (
            canvas_height,
            canvas_width,
        ),
        dtype=np.uint8,
    )

    # Place images side by side

    canvas[
        :h0,
        :w0
    ] = image0

    canvas[
        :h1,
        w0:w0 + w1
    ] = image1

    canvas = cv2.cvtColor(
        canvas,
        cv2.COLOR_GRAY2BGR,
    )


    # Draw correspondences

    for i, (p0, p1) in enumerate(
        zip(points0, points1)
    ):

        x0, y0 = p0
        x1, y1 = p1

        x0 = int(
            round(x0)
        )

        y0 = int(
            round(y0)
        )

        x1 = int(
            round(x1 + w0)
        )

        y1 = int(
            round(y1)
        )


        # Determine color

        if inlier_mask is not None:

            if inlier_mask[i]:

                # Green = geometric inlier

                color = (
                    0,
                    255,
                    0,
                )

            else:

                # Red = rejected correspondence

                color = (
                    0,
                    0,
                    255,
                )

        else:

            # White = candidate match

            color = (
                255,
                255,
                255,
            )


        # Draw source point

        cv2.circle(
            canvas,
            (x0, y0),
            8,
            color,
            -1,
        )


        # Draw reference point

        cv2.circle(
            canvas,
            (x1, y1),
            8,
            color,
            -1,
        )


        # Draw correspondence line

        cv2.line(
            canvas,
            (x0, y0),
            (x1, y1),
            color,
            2,
        )


    return canvas


# =========================================================
# Create results directory
# =========================================================

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# Save all matches
# =========================================================

if len(matches) > 0:

    all_match_image = (
        draw_correspondences(
            source_img,
            reference_img,
            source_points,
            reference_points,
        )
    )

    cv2.imwrite(
        str(ALL_MATCHES_PATH),
        all_match_image,
    )

    print(
        "\nSaved all matches:"
    )

    print(
        ALL_MATCHES_PATH
    )


# =========================================================
# RANSAC geometric verification
# =========================================================

print("\nRunning RANSAC...")


if len(matches) < 4:

    print(
        "\nNot enough matches "
        "for homography."
    )

    print(
        "Registration: REJECTED"
    )

else:

    H, mask = cv2.findHomography(
        source_points.astype(
            np.float32
        ),
        reference_points.astype(
            np.float32
        ),
        cv2.RANSAC,
        5.0,
    )


    # -----------------------------------------------------
    # Check RANSAC result
    # -----------------------------------------------------

    if H is None or mask is None:

        print(
            "\nRANSAC failed to estimate "
            "a homography."
        )

        print(
            "Registration: REJECTED"
        )

    else:

        inlier_mask = (
            mask
            .ravel()
            .astype(bool)
        )

        inlier_count = int(
            inlier_mask.sum()
        )

        inlier_ratio = (
            inlier_count
            / len(matches)
        )


        # -------------------------------------------------
        # Calculate reprojection errors
        # -------------------------------------------------

        projected = (
            cv2.perspectiveTransform(
                source_points.astype(
                    np.float32
                ).reshape(
                    -1,
                    1,
                    2,
                ),
                H,
            )
            .reshape(-1, 2)
        )


        errors = np.linalg.norm(
            projected
            - reference_points,
            axis=1,
        )


        inlier_errors = (
            errors[inlier_mask]
        )


        # -------------------------------------------------
        # Calculate metrics
        # -------------------------------------------------

        if len(inlier_errors) > 0:

            rmse = float(
                np.sqrt(
                    np.mean(
                        inlier_errors ** 2
                    )
                )
            )

            median_error = float(
                np.median(
                    inlier_errors
                )
            )

            max_error = float(
                np.max(
                    inlier_errors
                )
            )

        else:

            rmse = float("nan")
            median_error = float("nan")
            max_error = float("nan")


        # =================================================
        # LUCAS Quality Gate
        # =================================================

        quality = (
            evaluate_registration_quality(
                total_matches=len(matches),
                inlier_count=inlier_count,
            )
        )


        # =================================================
        # Print complete result
        # =================================================

        print(
            "\n========== LUCAS RESULT =========="
        )

        print(
            f"Matches:         {len(matches)}"
        )

        print(
            f"Inliers:         {inlier_count}"
        )

        print(
            f"Inlier ratio:    "
            f"{inlier_ratio:.3f}"
        )

        print(
            f"RMSE:            "
            f"{rmse:.3f} px"
        )

        print(
            f"Median error:    "
            f"{median_error:.3f} px"
        )

        print(
            f"Max error:       "
            f"{max_error:.3f} px"
        )


        print(
            "\nHomography:"
        )

        print(H)

        print(
            "==================================="
        )


        # =================================================
        # Quality Gate Result
        # =================================================

        print(
            "\n========== LUCAS QUALITY =========="
        )

        if quality["success"]:

            print(
                "Registration: ACCEPTED"
            )

        else:

            print(
                "Registration: REJECTED"
            )


        print(
            "Reason:",
            quality["reason"],
        )

        print(
            "Inliers:",
            quality["inlier_count"],
        )

        print(
            "Inlier ratio:",
            f"{quality['inlier_ratio']:.3f}",
        )

        print(
            "==================================="
        )


        # =================================================
        # Inlier visualization
        # =================================================

        inlier_match_image = (
            draw_correspondences(
                source_img,
                reference_img,
                source_points,
                reference_points,
                inlier_mask,
            )
        )

        cv2.imwrite(
            str(INLIER_MATCHES_PATH),
            inlier_match_image,
        )

        print(
            "\nSaved inlier visualization:"
        )

        print(
            INLIER_MATCHES_PATH
        )


        # =================================================
        # Registration decision
        # =================================================

        if quality["success"]:

            print(
                "\n✓ Geometric verification passed."
            )

            print(
                "✓ Registration may proceed."
            )

        else:

            print(
                "\n⚠ Registration rejected."
            )

            print(
                "⚠ Insufficient geometric evidence."
            )

            print(
                "⚠ No trustworthy registered image "
                "should be produced."
            )