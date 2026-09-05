"""
LUCAS - Controlled validation for sub-pixel correspondence refinement.

This experiment uses the same controlled-validation setup as the current
LUCAS baseline, but evaluates localization against the known ground-truth
affine transform.

Important:
    NCC quality alone is NOT treated as positional accuracy.

For every LightGlue/RANSAC inlier:
    1. The known ground-truth transform gives the true reference position.
    2. The original matched reference position is measured against that truth.
    3. Sub-pixel refinement is applied using the current estimated transform
       as the local prediction.
    4. The refined position is measured against the same ground truth.
    5. Before/after localization error is reported.

This is a validation experiment only. It does not modify pipeline.py.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
LIGHTGLUE_ROOT = ROOT / "LightGlue"

for import_path in (ROOT, LIGHTGLUE_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import cv2
import numpy as np
import torch

from lightglue import LightGlue, SuperPoint  # pyright: ignore[reportMissingImports]
from lightglue.utils import load_image  # pyright: ignore[reportMissingImports]

from ai_engine.subpixel import refine_matches



SOURCE_PATH = ROOT / "data" / "processed" / "lro_apollo16_overlap.png"
TRANSFORMED_PATH = ROOT / "data" / "results" / "validation_transformed.png"

OUTPUT_JSON = ROOT / "data" / "results" / "subpixel_validation.json"
OUTPUT_VIS = ROOT / "data" / "results" / "subpixel_validation_comparison.png"


# This is the exact matrix printed by the existing controlled validation.
GROUND_TRUTH = np.array(
    [
        [1.06948951e00, 1.50306949e-01, -5.87180542e02],
        [-1.50306949e-01, 1.06948951e00, 5.60796248e01],
    ],
    dtype=np.float64,
)


def affine_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    ones = np.ones((len(points), 1), dtype=np.float32)
    return np.hstack([points, ones]) @ matrix.T


def error_stats(estimated: np.ndarray, truth: np.ndarray) -> dict:
    errors = np.linalg.norm(
        np.asarray(estimated, dtype=np.float64)
        - np.asarray(truth, dtype=np.float64),
        axis=1,
    )

    if len(errors) == 0:
        return {
            "count": 0,
            "rmse_px": None,
            "median_px": None,
            "max_px": None,
            "mean_px": None,
        }

    return {
        "count": int(len(errors)),
        "rmse_px": float(np.sqrt(np.mean(errors**2))),
        "median_px": float(np.median(errors)),
        "max_px": float(np.max(errors)),
        "mean_px": float(np.mean(errors)),
    }


def main() -> None:
    print("=" * 60)
    print("        LUCAS SUB-PIXEL CONTROLLED VALIDATION")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if not SOURCE_PATH.exists():
        raise FileNotFoundError(f"Missing source image: {SOURCE_PATH}")

    if not TRANSFORMED_PATH.exists():
        raise FileNotFoundError(
            f"Missing controlled transformed image: {TRANSFORMED_PATH}\n"
            "Run experiments/controlled_validation.py first."
        )

    print("\nLoading controlled images...")
    source = load_image(str(SOURCE_PATH))
    transformed = load_image(str(TRANSFORMED_PATH))

    print(f"Source tensor:      {tuple(source.shape)}")
    print(f"Transformed tensor: {tuple(transformed.shape)}")

    print("\nLoading SuperPoint...")
    extractor = SuperPoint(
        max_num_keypoints=4096,
    ).eval().to(device)

    print("Loading LightGlue...")
    matcher = LightGlue(
        features="superpoint",
    ).eval().to(device)

    print("\nExtracting features...")

    with torch.inference_mode():
        feats_source = extractor.extract(source.to(device))
        feats_transformed = extractor.extract(transformed.to(device))

    print(
        f"Source keypoints:      "
        f"{len(feats_source['keypoints'][0])}"
    )
    print(
        f"Transformed keypoints: "
        f"{len(feats_transformed['keypoints'][0])}"
    )

    print("\nRunning LightGlue...")

    with torch.inference_mode():
        matches01 = matcher(
            {
                "image0": feats_source,
                "image1": feats_transformed,
            }
        )

    matches = (
        matches01["matches"][0]
        .detach()
        .cpu()
        .numpy()
    )

    if len(matches) < 4:
        raise RuntimeError(
            f"Only {len(matches)} matches found; cannot run RANSAC."
        )

    source_keypoints = (
        feats_source["keypoints"][0]
        .detach()
        .cpu()
        .numpy()
    )

    transformed_keypoints = (
        feats_transformed["keypoints"][0]
        .detach()
        .cpu()
        .numpy()
    )

    source_points = source_keypoints[matches[:, 0]]
    reference_points = transformed_keypoints[matches[:, 1]]

    print(f"Matches: {len(matches)}")

    print("\nRunning RANSAC...")

    estimated_matrix, inlier_mask = cv2.estimateAffinePartial2D(
        source_points,
        reference_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0,
        maxIters=5000,
        confidence=0.999,
    )

    if estimated_matrix is None or inlier_mask is None:
        raise RuntimeError("RANSAC failed.")

    inlier_mask = inlier_mask.ravel().astype(bool)

    inlier_source = source_points[inlier_mask]
    inlier_reference = reference_points[inlier_mask]

    print(f"Inliers: {len(inlier_source)}")

    # ------------------------------------------------------------------
    # Ground-truth localization baseline
    # ------------------------------------------------------------------

    truth_reference = affine_points(
        inlier_source,
        GROUND_TRUTH,
    )

    before = error_stats(
        inlier_reference,
        truth_reference,
    )

    # ------------------------------------------------------------------
    # Sub-pixel refinement
    # ------------------------------------------------------------------

    print("\nRunning sub-pixel refinement...")
    print("Patch radius: 7 px")
    print("Search radius: 4 px")
    print("Minimum NCC: 0.60")

    refined_reference, success_mask, scores = refine_matches(
        source_image=source.cpu().numpy()[0],
        reference_image=transformed.cpu().numpy()[0],
        source_points=inlier_source,
        reference_points=inlier_reference,
        transform=estimated_matrix,
        patch_radius=7,
        search_radius=4,
        min_score=0.60,
    )

    # Evaluate only successful refinements. This avoids treating a rejected
    # refinement as a positive/negative localization measurement.
    successful = success_mask.astype(bool)

    if np.any(successful):
        after = error_stats(
            refined_reference[successful],
            truth_reference[successful],
        )

        before_same_subset = error_stats(
            inlier_reference[successful],
            truth_reference[successful],
        )

        displacement = np.linalg.norm(
            refined_reference[successful]
            - inlier_reference[successful],
            axis=1,
        )

        successful_scores = scores[successful]
    else:
        after = error_stats(
            np.empty((0, 2)),
            np.empty((0, 2)),
        )
        before_same_subset = after
        displacement = np.empty(0)
        successful_scores = np.empty(0)

    improvement_px = None
    improvement_percent = None

    if before_same_subset["rmse_px"] is not None and after["rmse_px"] is not None:
        improvement_px = (
            before_same_subset["rmse_px"]
            - after["rmse_px"]
        )

        if before_same_subset["rmse_px"] > 1e-12:
            improvement_percent = (
                improvement_px
                / before_same_subset["rmse_px"]
                * 100.0
            )

    # ------------------------------------------------------------------
    # Optional sanity check: compare estimated transform to truth.
    # ------------------------------------------------------------------

    matrix_error = float(
        np.linalg.norm(
            np.asarray(estimated_matrix, dtype=np.float64)
            - GROUND_TRUTH
        )
    )

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    print("\n" + "=" * 50)
    print("        SUB-PIXEL VALIDATION RESULT")
    print("=" * 50)

    print(f"Total matches:                 {len(matches)}")
    print(f"RANSAC inliers:                {len(inlier_source)}")
    print(
        f"Successful refinements:       "
        f"{int(np.sum(successful))}"
    )
    print(
        f"Refinement success rate:      "
        f"{float(np.mean(successful)) if len(successful) else 0.0:.3f}"
    )

    print("\nBEFORE REFINEMENT")
    print(f"RMSE:                          {before['rmse_px']:.4f} px")
    print(f"Median:                        {before['median_px']:.4f} px")
    print(f"Max:                           {before['max_px']:.4f} px")

    if np.any(successful):
        print("\nBEFORE REFINEMENT (same successful subset)")
        print(
            f"RMSE:                          "
            f"{before_same_subset['rmse_px']:.4f} px"
        )
        print(
            f"Median:                        "
            f"{before_same_subset['median_px']:.4f} px"
        )
        print(
            f"Max:                           "
            f"{before_same_subset['max_px']:.4f} px"
        )

        print("\nAFTER SUB-PIXEL REFINEMENT")
        print(f"RMSE:                          {after['rmse_px']:.4f} px")
        print(f"Median:                        {after['median_px']:.4f} px")
        print(f"Max:                           {after['max_px']:.4f} px")

        print("\nIMPROVEMENT")
        print(f"RMSE change:                   {improvement_px:.4f} px")

        if improvement_percent is not None:
            print(
                f"RMSE improvement:              "
                f"{improvement_percent:.2f}%"
            )

        print(
            f"Mean refinement displacement:  "
            f"{float(np.mean(displacement)):.4f} px"
        )
        print(
            f"Median refinement displacement:"
            f" {float(np.median(displacement)):.4f} px"
        )
        print(
            f"Mean NCC score:                "
            f"{float(np.mean(successful_scores)):.4f}"
        )

    print("\nEstimated affine matrix:")
    print(estimated_matrix)

    print("\nGround-truth affine matrix:")
    print(GROUND_TRUTH)

    print(f"\nTransformation matrix error:   {matrix_error:.6f}")

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    vis_source = cv2.imread(str(SOURCE_PATH), cv2.IMREAD_GRAYSCALE)
    vis_reference = cv2.imread(
        str(TRANSFORMED_PATH),
        cv2.IMREAD_GRAYSCALE,
    )

    if vis_source is not None and vis_reference is not None:
        # Keep visualization lightweight.
        canvas = np.hstack(
            [
                cv2.normalize(
                    vis_source,
                    None,
                    0,
                    255,
                    cv2.NORM_MINMAX,
                ),
                cv2.normalize(
                    vis_reference,
                    None,
                    0,
                    255,
                    cv2.NORM_MINMAX,
                ),
            ]
        )

        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

        # Draw up to 100 successful refinements.
        selected_indices = np.flatnonzero(successful)[:100]

        offset_x = vis_source.shape[1]

        for idx in selected_indices:
            old = inlier_reference[idx]
            new = refined_reference[idx]

            old_x = int(round(float(old[0]))) + offset_x
            old_y = int(round(float(old[1])))

            new_x = int(round(float(new[0]))) + offset_x
            new_y = int(round(float(new[1])))

            cv2.circle(
                canvas,
                (old_x, old_y),
                2,
                255,
                -1,
            )

            cv2.circle(
                canvas,
                (new_x, new_y),
                2,
                128,
                -1,
            )

            cv2.line(
                canvas,
                (old_x, old_y),
                (new_x, new_y),
                200,
                1,
            )

        cv2.imwrite(str(OUTPUT_VIS), canvas)
        print(f"\nSaved comparison visualization: {OUTPUT_VIS}")

    result = {
        "device": device,
        "source": str(SOURCE_PATH),
        "transformed": str(TRANSFORMED_PATH),
        "total_matches": int(len(matches)),
        "ransac_inliers": int(len(inlier_source)),
        "successful_refinements": int(np.sum(successful)),
        "refinement_success_rate": (
            float(np.mean(successful))
            if len(successful)
            else 0.0
        ),
        "before": before,
        "before_same_successful_subset": before_same_subset,
        "after": after,
        "rmse_improvement_px": improvement_px,
        "rmse_improvement_percent": improvement_percent,
        "mean_refinement_displacement_px": (
            float(np.mean(displacement))
            if len(displacement)
            else None
        ),
        "median_refinement_displacement_px": (
            float(np.median(displacement))
            if len(displacement)
            else None
        ),
        "mean_ncc_successful": (
            float(np.mean(successful_scores))
            if len(successful_scores)
            else None
        ),
        "estimated_matrix": estimated_matrix.tolist(),
        "ground_truth_matrix": GROUND_TRUTH.tolist(),
        "transformation_matrix_error": matrix_error,
    }

    OUTPUT_JSON.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print(f"Saved validation result: {OUTPUT_JSON}")

    print("\n" + "=" * 60)
    print("SUB-PIXEL VALIDATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
