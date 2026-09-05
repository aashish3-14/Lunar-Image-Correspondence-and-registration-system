from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_engine.subpixel import refine_matches


SOURCE_PATH = ROOT / "data" / "processed" / "lro_apollo16_overlap.png"
TRANSFORMED_PATH = ROOT / "data" / "results" / "validation_transformed.png"

OUTPUT_JSON = ROOT / "data" / "results" / "subpixel_validation_v2.json"
OUTPUT_VIS = ROOT / "data" / "results" / "subpixel_validation_v2.png"

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
    estimated = np.asarray(estimated, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)

    if len(estimated) == 0:
        return {
            "count": 0,
            "rmse_px": None,
            "median_px": None,
            "mean_px": None,
            "max_px": None,
        }

    errors = np.linalg.norm(estimated - truth, axis=1)

    return {
        "count": int(len(errors)),
        "rmse_px": float(np.sqrt(np.mean(errors ** 2))),
        "median_px": float(np.median(errors)),
        "mean_px": float(np.mean(errors)),
        "max_px": float(np.max(errors)),
    }


def main() -> None:
    print("=" * 60)
    print("        LUCAS SUB-PIXEL CONTROLLED VALIDATION V2")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if not SOURCE_PATH.exists():
        raise FileNotFoundError(SOURCE_PATH)

    if not TRANSFORMED_PATH.exists():
        raise FileNotFoundError(
            f"{TRANSFORMED_PATH}\n"
            "Run controlled_validation.py first."
        )

    print("\nLoading controlled images...")
    source = load_image(str(SOURCE_PATH))
    transformed = load_image(str(TRANSFORMED_PATH))

    print(f"Source tensor:      {tuple(source.shape)}")
    print(f"Transformed tensor: {tuple(transformed.shape)}")

    print("\nLoading SuperPoint...")
    extractor = SuperPoint(max_num_keypoints=4096).eval().to(device)

    print("Loading LightGlue...")
    matcher = LightGlue(features="superpoint").eval().to(device)

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
            f"Only {len(matches)} matches found."
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

    truth_reference = affine_points(
        inlier_source,
        GROUND_TRUTH,
    )

    before = error_stats(
        inlier_reference,
        truth_reference,
    )

    print("\nRunning conservative sub-pixel refinement...")
    print("Patch radius:        7 px")
    print("LK window:            15 px")
    print("Maximum displacement: 0.75 px")
    print("Minimum NCC:          0.60")

    source_np = source.cpu().numpy()[0]
    transformed_np = transformed.cpu().numpy()[0]

    refined_reference, success_mask, scores = refine_matches(
        source_image=source_np,
        reference_image=transformed_np,
        source_points=inlier_source,
        reference_points=inlier_reference,
        transform=estimated_matrix,
        patch_radius=7,
        max_displacement=0.75,
        min_ncc=0.60,
        lk_window=15,
        max_iterations=30,
    )

    successful = success_mask.astype(bool)

    # Evaluate ALL inliers with unsuccessful points left unchanged.
    # This measures the actual effect of the module as it would behave
    # inside LUCAS.
    after_all = error_stats(
        refined_reference,
        truth_reference,
    )

    if np.any(successful):
        before_successful = error_stats(
            inlier_reference[successful],
            truth_reference[successful],
        )
        after_successful = error_stats(
            refined_reference[successful],
            truth_reference[successful],
        )

        displacement = np.linalg.norm(
            refined_reference[successful]
            - inlier_reference[successful],
            axis=1,
        )

        successful_scores = scores[successful]
    else:
        before_successful = error_stats(
            np.empty((0, 2)),
            np.empty((0, 2)),
        )
        after_successful = before_successful
        displacement = np.empty(0)
        successful_scores = np.empty(0)

    rmse_change = (
        after_all["rmse_px"] - before["rmse_px"]
        if after_all["rmse_px"] is not None
        and before["rmse_px"] is not None
        else None
    )

    improvement_percent = None
    if (
        before["rmse_px"] is not None
        and after_all["rmse_px"] is not None
        and before["rmse_px"] > 1e-12
    ):
        improvement_percent = (
            (before["rmse_px"] - after_all["rmse_px"])
            / before["rmse_px"]
            * 100.0
        )

    matrix_error = float(
        np.linalg.norm(
            np.asarray(estimated_matrix, dtype=np.float64)
            - GROUND_TRUTH
        )
    )

    print("\n" + "=" * 50)
    print("        SUB-PIXEL VALIDATION V2 RESULT")
    print("=" * 50)

    print(f"Total matches:                 {len(matches)}")
    print(f"RANSAC inliers:                {len(inlier_source)}")
    print(f"Successful refinements:        {int(np.sum(successful))}")
    print(
        f"Refinement success rate:       "
        f"{float(np.mean(successful)) if len(successful) else 0.0:.3f}"
    )

    print("\nBEFORE REFINEMENT")
    print(f"RMSE:                          {before['rmse_px']:.4f} px")
    print(f"Median:                        {before['median_px']:.4f} px")
    print(f"Mean:                          {before['mean_px']:.4f} px")
    print(f"Max:                           {before['max_px']:.4f} px")

    print("\nAFTER REFINEMENT (ALL INLIERS)")
    print(f"RMSE:                          {after_all['rmse_px']:.4f} px")
    print(f"Median:                        {after_all['median_px']:.4f} px")
    print(f"Mean:                          {after_all['mean_px']:.4f} px")
    print(f"Max:                           {after_all['max_px']:.4f} px")

    if np.any(successful):
        print("\nSUCCESSFUL REFINEMENTS ONLY")
        print(
            f"Before RMSE:                   "
            f"{before_successful['rmse_px']:.4f} px"
        )
        print(
            f"After RMSE:                    "
            f"{after_successful['rmse_px']:.4f} px"
        )
        print(
            f"Mean displacement:             "
            f"{float(np.mean(displacement)):.4f} px"
        )
        print(
            f"Median displacement:           "
            f"{float(np.median(displacement)):.4f} px"
        )
        print(
            f"Max displacement:              "
            f"{float(np.max(displacement)):.4f} px"
        )
        print(
            f"Mean NCC:                      "
            f"{float(np.mean(successful_scores)):.4f}"
        )

    print("\nOVERALL CHANGE")
    print(f"RMSE change:                   {rmse_change:.4f} px")

    if improvement_percent is not None:
        print(
            f"RMSE improvement:              "
            f"{improvement_percent:.2f}%"
        )

    print("\nEstimated affine matrix:")
    print(estimated_matrix)

    print("\nGround-truth affine matrix:")
    print(GROUND_TRUTH)

    print(f"\nTransformation matrix error:   {matrix_error:.6f}")

    # Visualization: original and refined positions on transformed image.
    vis = cv2.imread(
        str(TRANSFORMED_PATH),
        cv2.IMREAD_GRAYSCALE,
    )

    if vis is not None:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

        for idx in np.flatnonzero(successful)[:150]:
            old = inlier_reference[idx]
            new = refined_reference[idx]

            old_xy = (
                int(round(float(old[0]))),
                int(round(float(old[1]))),
            )
            new_xy = (
                int(round(float(new[0]))),
                int(round(float(new[1]))),
            )

            cv2.circle(vis, old_xy, 2, 255, -1)
            cv2.circle(vis, new_xy, 2, 128, -1)
            cv2.line(vis, old_xy, new_xy, 200, 1)

        cv2.imwrite(str(OUTPUT_VIS), vis)
        print(f"\nSaved comparison visualization: {OUTPUT_VIS}")

    result = {
        "device": device,
        "total_matches": int(len(matches)),
        "ransac_inliers": int(len(inlier_source)),
        "successful_refinements": int(np.sum(successful)),
        "refinement_success_rate": (
            float(np.mean(successful))
            if len(successful)
            else 0.0
        ),
        "before_all": before,
        "after_all": after_all,
        "before_successful_subset": before_successful,
        "after_successful_subset": after_successful,
        "rmse_change_px": rmse_change,
        "rmse_improvement_percent": improvement_percent,
        "mean_displacement_px": (
            float(np.mean(displacement))
            if len(displacement)
            else None
        ),
        "median_displacement_px": (
            float(np.median(displacement))
            if len(displacement)
            else None
        ),
        "max_displacement_px": (
            float(np.max(displacement))
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
        "parameters": {
            "patch_radius": 7,
            "lk_window": 15,
            "max_displacement": 0.75,
            "min_ncc": 0.60,
            "max_iterations": 30,
        },
    }

    OUTPUT_JSON.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print(f"Saved validation result: {OUTPUT_JSON}")

    print("\n" + "=" * 60)
    print("SUB-PIXEL VALIDATION V2 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
