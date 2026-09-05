from __future__ import annotations

from pathlib import Path
import json
import sys

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import cv2
import numpy as np
import torch

from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image

from ai_engine.quality import evaluate_registration_quality
from ai_engine.spatial import analyze_spatial_distribution

SOURCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lro_apollo16_overlap.png"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "data"
    / "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# KNOWN TRANSFORMATION
# ============================================================

ROTATION_DEGREES = 8.0
SCALE = 1.08
TRANSLATION_X = 120.0
TRANSLATION_Y = 80.0


# ============================================================
# CREATE SYNTHETIC TRANSFORMATION
# ============================================================

def create_transformed_image(
    image: np.ndarray,
):
    """
    Create a transformed version of an image using a known
    rotation, scale, and translation.

    Returns
    -------
    transformed:
        Transformed image.

    ground_truth_matrix:
        2x3 affine transformation matrix.
    """

    height, width = image.shape[:2]

    center = (
        width / 2.0,
        height / 2.0,
    )

    # --------------------------------------------------------
    # Rotation + scale
    # --------------------------------------------------------

    matrix = cv2.getRotationMatrix2D(
        center,
        ROTATION_DEGREES,
        SCALE,
    )

    # --------------------------------------------------------
    # Add translation
    # --------------------------------------------------------

    matrix[0, 2] += TRANSLATION_X
    matrix[1, 2] += TRANSLATION_Y

    # --------------------------------------------------------
    # Apply transformation
    # --------------------------------------------------------

    transformed = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return transformed, matrix


# ============================================================
# CONVERT IMAGE FOR LIGHTGLUE
# ============================================================

def prepare_lightglue_image(
    image: np.ndarray,
):
    """
    Convert OpenCV grayscale image to the format expected
    by LightGlue's load_image workflow.
    """

    temp_path = (
        RESULTS_DIR
        / "_validation_temp.png"
    )

    cv2.imwrite(
        str(temp_path),
        image,
    )

    tensor = load_image(
        temp_path
    )

    return tensor


# ============================================================
# MAIN VALIDATION
# ============================================================

def main():

    print()
    print("=" * 60)
    print("        LUCAS CONTROLLED VALIDATION")
    print("=" * 60)

    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(f"Device: {device}")

    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # ========================================================
    # LOAD ORIGINAL IMAGE
    # ========================================================

    print()
    print("Loading original LRO image...")

    image = cv2.imread(
        str(SOURCE_PATH),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        raise FileNotFoundError(
            f"Could not load: {SOURCE_PATH}"
        )

    print(
        f"Image shape: {image.shape}"
    )

    # ========================================================
    # CREATE KNOWN TRANSFORMATION
    # ========================================================

    print()
    print("Creating controlled transformation...")

    transformed, ground_truth_matrix = (
        create_transformed_image(image)
    )

    transformed_path = (
        RESULTS_DIR
        / "validation_transformed.png"
    )

    cv2.imwrite(
        str(transformed_path),
        transformed,
    )

    print(
        f"Rotation:    {ROTATION_DEGREES} degrees"
    )

    print(
        f"Scale:       {SCALE}"
    )

    print(
        f"Translation: "
        f"({TRANSLATION_X}, {TRANSLATION_Y}) px"
    )

    print(
        f"Saved transformed image: "
        f"{transformed_path}"
    )

    # ========================================================
    # SAVE GROUND TRUTH
    # ========================================================

    print()
    print("Ground-truth affine matrix:")

    print(
        ground_truth_matrix
    )

    # ========================================================
    # PREPARE LIGHTGLUE INPUTS
    # ========================================================

    print()
    print("Preparing LightGlue inputs...")

    original_tensor = (
        prepare_lightglue_image(image)
        .to(device)
    )

    transformed_tensor = (
        prepare_lightglue_image(transformed)
        .to(device)
    )

    # ========================================================
    # LOAD MODELS
    # ========================================================

    print()
    print("Loading SuperPoint...")

    extractor = (
        SuperPoint(
            max_num_keypoints=4096
        )
        .eval()
        .to(device)
    )

    print("Loading LightGlue...")

    matcher = (
        LightGlue(
            features="superpoint"
        )
        .eval()
        .to(device)
    )

    # ========================================================
    # FEATURE EXTRACTION
    # ========================================================

    print()
    print("Extracting features...")

    with torch.inference_mode():

        feats0 = extractor.extract(
            original_tensor
        )

        feats1 = extractor.extract(
            transformed_tensor
        )

    keypoints0 = (
        feats0["keypoints"][0]
        .detach()
        .cpu()
        .numpy()
    )

    keypoints1 = (
        feats1["keypoints"][0]
        .detach()
        .cpu()
        .numpy()
    )

    print(
        f"Original keypoints: "
        f"{len(keypoints0)}"
    )

    print(
        f"Transformed keypoints: "
        f"{len(keypoints1)}"
    )

    # ========================================================
    # LIGHTGLUE
    # ========================================================

    print()
    print("Running LightGlue...")

    with torch.inference_mode():

        matches01 = matcher(
            {
                "image0": feats0,
                "image1": feats1,
            }
        )

    matches = (
        matches01["matches"][0]
        .detach()
        .cpu()
        .numpy()
    )

    total_matches = int(
        len(matches)
    )

    print(
        f"Matches: {total_matches}"
    )

    # ========================================================
    # MATCH VISUALIZATION
    # ========================================================

    try:

        from ai_engine.pipeline import (
            save_match_visualization,
        )

        match_visualization_path = (
            RESULTS_DIR
            / "validation_matches.png"
        )

        save_match_visualization(
            image0=original_tensor,
            image1=transformed_tensor,
            keypoints0=keypoints0,
            keypoints1=keypoints1,
            matches=matches,
            output_path=match_visualization_path,
            title=(
                f"Controlled Validation "
                f"Matches: {total_matches}"
            ),
        )

        print(
            f"Saved match visualization: "
            f"{match_visualization_path}"
        )

    except Exception as exc:

        print(
            "Could not create match "
            f"visualization: {exc}"
        )

        match_visualization_path = None

    # ========================================================
    # CHECK MATCH COUNT
    # ========================================================

    if total_matches < 4:

        print()
        print(
            "❌ Validation failed: "
            "fewer than 4 matches."
        )

        return

    # ========================================================
    # CORRESPONDENCE POINTS
    # ========================================================

    source_points = (
        keypoints0[matches[:, 0]]
    )

    transformed_points = (
        keypoints1[matches[:, 1]]
    )

    # ========================================================
    # RANSAC
    # ========================================================

    print()
    print("Running RANSAC...")

    estimated_matrix, mask = (
        cv2.estimateAffinePartial2D(
            source_points,
            transformed_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0,
        )
    )

    # ========================================================
    # RANSAC FAILURE
    # ========================================================

    if mask is None:

        print()
        print(
            "❌ Validation failed: "
            "RANSAC could not estimate "
            "the transformation."
        )

        return

    inlier_mask = (
        mask.ravel()
        .astype(bool)
    )

    inlier_count = int(
        np.sum(inlier_mask)
    )

    inlier_ratio = (
        inlier_count / total_matches
    )

    # ========================================================
    # SPATIAL ANALYSIS
    # ========================================================

    inlier_points = (
        source_points[inlier_mask]
    )

    spatial = (
        analyze_spatial_distribution(
            points=inlier_points,
            image_width=image.shape[1],
            image_height=image.shape[0],
            grid_rows=4,
            grid_cols=4,
        )
    )

    # ========================================================
    # REPROJECTION ERROR
    # ========================================================

    transformed_estimated = cv2.transform(
        source_points.reshape(
            -1,
            1,
            2,
        ).astype(np.float32),
        estimated_matrix,
    ).reshape(-1, 2)

    errors = np.linalg.norm(
        transformed_estimated
        - transformed_points,
        axis=1,
    )

    inlier_errors = (
        errors[inlier_mask]
    )

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

    # ========================================================
    # QUALITY GATE
    # ========================================================

    quality = (
        evaluate_registration_quality(
            total_matches=total_matches,
            inlier_count=inlier_count,
            spatial_coverage=spatial["coverage"],
            min_inliers=10,
            min_inlier_ratio=0.20,
            min_spatial_coverage=0.25,
        )
    )

    # ========================================================
    # COMPARE ESTIMATED TRANSFORMATION
    # ========================================================

    transformation_error = None

    if estimated_matrix is not None:

        transformation_error = float(
            np.linalg.norm(
                estimated_matrix
                - ground_truth_matrix
            )
        )

    # ========================================================
    # RESULTS
    # ========================================================

    result = {

        "experiment": (
            "Controlled geometric validation"
        ),

        "status": (
            "ACCEPTED"
            if quality["success"]
            else "REJECTED"
        ),

        "reason": quality["reason"],

        "quality_reasons": quality["reasons"],

        "matches": total_matches,

        "inliers": inlier_count,

        "inlier_ratio": float(
            inlier_ratio
        ),

        "rmse_px": rmse,

        "median_error_px": median_error,

        "max_error_px": max_error,

        "spatial_coverage": float(
            spatial["coverage"]
        ),

        "spatial_uniformity": float(
            spatial["uniformity"]
        ),

        "spatial_distribution": (
            spatial["distribution"]
        ),

        "occupied_grid_cells": int(
            spatial["occupied_cells"]
        ),

        "total_grid_cells": int(
            spatial["total_cells"]
        ),

        "rotation_degrees": (
            ROTATION_DEGREES
        ),

        "scale": SCALE,

        "translation_x": (
            TRANSLATION_X
        ),

        "translation_y": (
            TRANSLATION_Y
        ),

        "ground_truth_matrix": (
            ground_truth_matrix.tolist()
        ),

        "estimated_matrix": (
            estimated_matrix.tolist()
            if estimated_matrix is not None
            else None
        ),

        "transformation_matrix_error": (
            transformation_error
        ),

        "source_image": str(
            SOURCE_PATH
        ),

        "transformed_image": str(
            transformed_path
        ),

        "match_visualization": (
            str(match_visualization_path)
            if match_visualization_path
            else None
        ),
    }

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print(
        "=================================================="
    )

    print(
        "        CONTROLLED VALIDATION RESULT"
    )

    print(
        "=================================================="
    )

    print(
        f"Matches:             {total_matches}"
    )

    print(
        f"Inliers:             {inlier_count}"
    )

    print(
        f"Inlier ratio:        "
        f"{inlier_ratio:.3f}"
    )

    print(
        f"RMSE:                "
        f"{rmse:.3f} px"
    )

    print(
        f"Median error:        "
        f"{median_error:.3f} px"
    )

    print(
        f"Max error:           "
        f"{max_error:.3f} px"
    )

    print(
        f"Spatial coverage:    "
        f"{spatial['coverage']:.3f}"
    )

    print(
        f"Spatial uniformity:  "
        f"{spatial['uniformity']:.3f}"
    )

    print(
        f"Distribution:        "
        f"{spatial['distribution']}"
    )

    print()

    print(
        "Ground-truth matrix:"
    )

    print(
        ground_truth_matrix
    )

    print()

    print(
        "Estimated matrix:"
    )

    print(
        estimated_matrix
    )

    print()

    print(
        f"Transformation matrix error: "
        f"{transformation_error:.6f}"
        if transformation_error is not None
        else "Transformation matrix error: N/A"
    )

    print()

    print(
        f"QUALITY: "
        f"{'ACCEPTED' if quality['success'] else 'REJECTED'}"
    )

    print(
        f"Reason: {quality['reason']}"
    )

    if quality["reasons"]:

        print()
        print(
            "Failure criteria:"
        )

        for reason in quality["reasons"]:

            print(
                f"  • {reason}"
            )

    print(
        "=================================================="
    )

    # ========================================================
    # SAVE JSON
    # ========================================================

    result_path = (
        RESULTS_DIR
        / "controlled_validation.json"
    )

    with open(
        result_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result,
            f,
            indent=4,
        )

    print()
    print(
        f"Saved validation result: "
        f"{result_path}"
    )

    # ========================================================
    # CLEAN TEMP FILE
    # ========================================================

    temp_path = (
        RESULTS_DIR
        / "_validation_temp.png"
    )

    if temp_path.exists():

        temp_path.unlink()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()