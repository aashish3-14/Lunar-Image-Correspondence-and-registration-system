"""
LUCAS — Controlled Lunar Sub-pixel Failure Analysis

Purpose
-------
Analyze why local sub-pixel refinements are accepted/rejected on real lunar
texture under controlled geometry.

Runs one SuperPoint + LightGlue + RANSAC experiment, then evaluates the
sub-pixel refinement with several maximum-correction thresholds.

IMPORTANT:
This is still a controlled experiment. It does not claim cross-sensor
OHRC/TMC/IIRS accuracy.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image

from ai_engine.subpixel import refine_point


SOURCE_PATH = ROOT / "data" / "processed" / "lro_apollo16_overlap.png"
RESULT_DIR = ROOT / "data" / "results"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------
# Controlled transformation
# ---------------------------------------------------------------------

def make_controlled_transform(
    width: int,
    height: int,
) -> np.ndarray:

    center = (
        width / 2.0,
        height / 2.0,
    )

    angle = 5.0
    scale = 1.035

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        scale,
    )

    matrix[0, 2] += 45.0
    matrix[1, 2] += 35.0

    return matrix.astype(np.float32)


def apply_transform(
    image: np.ndarray,
    matrix: np.ndarray,
) -> np.ndarray:

    h, w = image.shape[:2]

    return cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )


def affine_points(
    points: np.ndarray,
    matrix: np.ndarray,
) -> np.ndarray:

    ones = np.ones(
        (len(points), 1),
        dtype=np.float32,
    )

    homogeneous = np.hstack(
        [
            points.astype(np.float32),
            ones,
        ]
    )

    return (
        homogeneous @ matrix.T
    ).astype(np.float32)


# ---------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------

def load_grayscale(
    path: Path,
) -> np.ndarray:

    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {path}"
        )

    return image


# ---------------------------------------------------------------------
# Single threshold evaluation
# ---------------------------------------------------------------------

def evaluate_threshold(
    source_image: np.ndarray,
    reference_image: np.ndarray,
    source_points: np.ndarray,
    observed_reference_points: np.ndarray,
    true_reference_points: np.ndarray,
    max_refinement: float,
) -> dict:

    refined_points = observed_reference_points.copy()

    reasons = Counter()

    successful = 0

    responses = []

    displacement_values = []

    for i in range(len(source_points)):

        result = refine_point(
            source_image=source_image,
            reference_image=reference_image,
            source_point=source_points[i],
            predicted_reference_point=observed_reference_points[i],
            patch_radius=15,
            max_refinement=max_refinement,
            min_response=0.20,
        )

        reasons[result["reason"]] += 1

        responses.append(
            float(result["response"])
        )

        correction = np.asarray(
            result["correction"],
            dtype=np.float64,
        )

        displacement_values.append(
            float(np.linalg.norm(correction))
        )

        if result["success"]:

            successful += 1

            refined_points[i] = np.asarray(
                result["point"],
                dtype=np.float32,
            )

    # -------------------------------------------------------------
    # Error against known ground truth
    # -------------------------------------------------------------

    before_errors = np.linalg.norm(
        observed_reference_points.astype(np.float64)
        - true_reference_points.astype(np.float64),
        axis=1,
    )

    after_errors = np.linalg.norm(
        refined_points.astype(np.float64)
        - true_reference_points.astype(np.float64),
        axis=1,
    )

    before_rmse = float(
        np.sqrt(
            np.mean(
                before_errors ** 2
            )
        )
    )

    after_rmse = float(
        np.sqrt(
            np.mean(
                after_errors ** 2
            )
        )
    )

    before_mean = float(
        np.mean(before_errors)
    )

    after_mean = float(
        np.mean(after_errors)
    )

    improvement = float(
        (
            before_rmse
            - after_rmse
        )
        / max(before_rmse, 1e-12)
        * 100.0
    )

    return {
        "max_refinement_px": max_refinement,

        "successful": successful,

        "rejected": (
            len(source_points)
            - successful
        ),

        "success_rate": (
            successful
            / len(source_points)
            if len(source_points)
            else 0.0
        ),

        "before_mean_error_px": before_mean,

        "after_mean_error_px": after_mean,

        "before_rmse_px": before_rmse,

        "after_rmse_px": after_rmse,

        "rmse_improvement_percent": improvement,

        "mean_response": float(
            np.mean(responses)
        ),

        "median_response": float(
            np.median(responses)
        ),

        "mean_correction_magnitude_px": float(
            np.mean(displacement_values)
        ),

        "median_correction_magnitude_px": float(
            np.median(displacement_values)
        ),

        "failure_reasons": dict(
            reasons
        ),
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    print("=" * 78)
    print(
        "LUCAS — SUB-PIXEL FAILURE ANALYSIS "
        "AND THRESHOLD SWEEP"
    )
    print("=" * 78)

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Source: {SOURCE_PATH}"
    )

    print()

    if not SOURCE_PATH.exists():

        raise FileNotFoundError(
            f"Missing source image: {SOURCE_PATH}"
        )

    # -------------------------------------------------------------
    # Load actual lunar texture
    # -------------------------------------------------------------

    base = load_grayscale(
        SOURCE_PATH
    )

    # Keep GPU/memory requirements reasonable.
    max_dim = 1800

    h, w = base.shape

    scale = min(
        1.0,
        max_dim / max(h, w),
    )

    if scale < 1.0:

        base = cv2.resize(
            base,
            (
                int(round(w * scale)),
                int(round(h * scale)),
            ),
            interpolation=cv2.INTER_AREA,
        )

    h, w = base.shape

    # -------------------------------------------------------------
    # Generate controlled transformed lunar image
    # -------------------------------------------------------------

    true_matrix = make_controlled_transform(
        w,
        h,
    )

    transformed = apply_transform(
        base,
        true_matrix,
    )

    transformed_path = (
        RESULT_DIR
        / "lunar_subpixel_threshold_test.png"
    )

    cv2.imwrite(
        str(transformed_path),
        transformed,
    )

    print(
        f"Test image shape: {base.shape}"
    )

    print()

    print(
        "Known affine matrix:"
    )

    print(
        true_matrix
    )

    print()

    # -------------------------------------------------------------
    # Temporary files for LightGlue
    # -------------------------------------------------------------

    source_path = (
        RESULT_DIR
        / "_threshold_source.png"
    )

    reference_path = (
        RESULT_DIR
        / "_threshold_reference.png"
    )

    cv2.imwrite(
        str(source_path),
        base,
    )

    cv2.imwrite(
        str(reference_path),
        transformed,
    )

    # -------------------------------------------------------------
    # SuperPoint + LightGlue
    # -------------------------------------------------------------

    print(
        "Running SuperPoint + LightGlue..."
    )

    image0 = load_image(
        source_path
    ).to(DEVICE)

    image1 = load_image(
        reference_path
    ).to(DEVICE)

    extractor = (
        SuperPoint(
            max_num_keypoints=2048
        )
        .eval()
        .to(DEVICE)
    )

    matcher = (
        LightGlue(
            features="superpoint"
        )
        .eval()
        .to(DEVICE)
    )

    with torch.inference_mode():

        feats0 = extractor.extract(
            image0
        )

        feats1 = extractor.extract(
            image1
        )

        matches01 = matcher(
            {
                "image0": feats0,
                "image1": feats1,
            }
        )

    kpts0 = (
        feats0["keypoints"][0]
        .detach()
        .cpu()
        .numpy()
    )

    kpts1 = (
        feats1["keypoints"][0]
        .detach()
        .cpu()
        .numpy()
    )

    matches = (
        matches01["matches"][0]
        .detach()
        .cpu()
        .numpy()
    )

    if len(matches) < 4:

        raise RuntimeError(
            f"Only {len(matches)} matches found."
        )

    source_points = (
        kpts0[matches[:, 0]]
    )

    reference_points = (
        kpts1[matches[:, 1]]
    )

    # -------------------------------------------------------------
    # RANSAC
    # -------------------------------------------------------------

    estimated_matrix, mask = (
        cv2.estimateAffinePartial2D(
            source_points,
            reference_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=5000,
            confidence=0.999,
        )
    )

    if estimated_matrix is None:

        raise RuntimeError(
            "RANSAC failed."
        )

    mask = (
        mask.ravel()
        .astype(bool)
    )

    inlier_source = (
        source_points[mask]
    )

    inlier_reference = (
        reference_points[mask]
    )

    print()
    print(
        f"Keypoints source/reference: "
        f"{len(kpts0)} / {len(kpts1)}"
    )

    print(
        f"Candidate matches: "
        f"{len(matches)}"
    )

    print(
        f"RANSAC inliers: "
        f"{len(inlier_source)}"
    )

    print(
        f"Inlier ratio: "
        f"{len(inlier_source) / len(matches):.3f}"
    )

    print()

    # -------------------------------------------------------------
    # Ground-truth transformed locations
    # -------------------------------------------------------------

    true_reference = affine_points(
        inlier_source,
        true_matrix,
    )

    # -------------------------------------------------------------
    # BEFORE baseline
    # -------------------------------------------------------------

    baseline_errors = np.linalg.norm(
        inlier_reference.astype(np.float64)
        - true_reference.astype(np.float64),
        axis=1,
    )

    baseline_rmse = float(
        np.sqrt(
            np.mean(
                baseline_errors ** 2
            )
        )
    )

    print(
        f"Baseline RMSE: "
        f"{baseline_rmse:.4f} px"
    )

    print()

    # -------------------------------------------------------------
    # Threshold sweep
    # -------------------------------------------------------------

    thresholds = [
        0.50,
        0.75,
        1.00,
        1.25,
        1.50,
        2.00,
    ]

    all_results = []

    print("=" * 78)
    print(
        "THRESHOLD SWEEP"
    )
    print("=" * 78)

    for threshold in thresholds:

        result = evaluate_threshold(
            source_image=base,
            reference_image=transformed,
            source_points=inlier_source,
            observed_reference_points=inlier_reference,
            true_reference_points=true_reference,
            max_refinement=threshold,
        )

        all_results.append(
            result
        )

        print()
        print(
            f"MAX REFINEMENT = "
            f"{threshold:.2f} px"
        )

        print(
            f"  Accepted       : "
            f"{result['successful']}/"
            f"{len(inlier_source)} "
            f"({result['success_rate']:.1%})"
        )

        print(
            f"  RMSE before    : "
            f"{result['before_rmse_px']:.4f} px"
        )

        print(
            f"  RMSE after     : "
            f"{result['after_rmse_px']:.4f} px"
        )

        print(
            f"  Improvement     : "
            f"{result['rmse_improvement_percent']:+.2f}%"
        )

        print(
            f"  Mean response   : "
            f"{result['mean_response']:.4f}"
        )

        print(
            "  Failure reasons:"
        )

        for reason, count in (
            result["failure_reasons"]
            .items()
        ):

            if count > 0:

                print(
                    f"    - {reason}: "
                    f"{count}"
                )

    # -------------------------------------------------------------
    # Pick best threshold
    # -------------------------------------------------------------

    # We prioritize lowest RMSE, not highest acceptance rate.
    best = min(
        all_results,
        key=lambda x: x["after_rmse_px"],
    )

    print()
    print("=" * 78)
    print(
        "BEST THRESHOLD"
    )
    print("=" * 78)

    print(
        f"Max refinement: "
        f"{best['max_refinement_px']:.2f} px"
    )

    print(
        f"Acceptance rate: "
        f"{best['success_rate']:.1%}"
    )

    print(
        f"RMSE: "
        f"{best['after_rmse_px']:.4f} px"
    )

    print(
        f"Improvement: "
        f"{best['rmse_improvement_percent']:+.2f}%"
    )

    # -------------------------------------------------------------
    # Save complete report
    # -------------------------------------------------------------

    report = {
        "experiment": (
            "LUCAS sub-pixel failure analysis "
            "and threshold sweep"
        ),

        "device": DEVICE,

        "source_image": str(
            SOURCE_PATH
        ),

        "image_shape": list(
            base.shape
        ),

        "known_transform": (
            true_matrix.tolist()
        ),

        "estimated_transform": (
            estimated_matrix.tolist()
        ),

        "candidate_matches": int(
            len(matches)
        ),

        "ransac_inliers": int(
            len(inlier_source)
        ),

        "inlier_ratio": float(
            len(inlier_source)
            / len(matches)
        ),

        "baseline_rmse_px": baseline_rmse,

        "threshold_results": all_results,

        "best_threshold": best,
    }

    output = (
        RESULT_DIR
        / "subpixel_failure_analysis.json"
    )

    output.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        f"Saved: {output}"
    )


if __name__ == "__main__":
    main()