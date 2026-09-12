"""
LUCAS — Sub-pixel Refinement V3

Conservative local sub-pixel refinement for already-verified correspondences.

Design:
    trusted source point
        ↓
    predicted reference point from geometric model
        ↓
    local source/reference patches
        ↓
    phase correlation
        ↓
    fractional correction
        ↓
    reject unstable / oversized corrections
        ↓
    refined reference point

This module is deliberately independent of the main LUCAS pipeline until
controlled validation proves that it improves correspondence accuracy.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "data" / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def _to_gray_float32(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = image.astype(np.float32)

    if image.size == 0:
        raise ValueError("Empty image")

    return image


def _extract_patch(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    radius: int,
) -> np.ndarray | None:
    """
    Extract a fixed-size integer-centered patch.

    Fractional coordinates are intentionally handled by phase correlation,
    not by silently rounding the final estimate.
    """
    h, w = image.shape[:2]
    cx = int(round(center_x))
    cy = int(round(center_y))

    x0 = cx - radius
    x1 = cx + radius + 1
    y0 = cy - radius
    y1 = cy + radius + 1

    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None

    patch = image[y0:y1, x0:x1]

    if patch.shape != (2 * radius + 1, 2 * radius + 1):
        return None

    return patch.astype(np.float32, copy=False)


def _normalize_patch(patch: np.ndarray) -> np.ndarray:
    patch = patch.astype(np.float32)

    # Remove local brightness/offset differences.
    patch = patch - float(np.mean(patch))
    std = float(np.std(patch))

    if std < 1e-6:
        return np.zeros_like(patch)

    return patch / std


def refine_point(
    source_image: np.ndarray,
    reference_image: np.ndarray,
    source_point: np.ndarray | tuple[float, float],
    predicted_reference_point: np.ndarray | tuple[float, float],
    patch_radius: int = 15,
    max_refinement: float = 1.0,
    min_response: float = 0.20,
) -> dict:
    """
    Refine one trusted correspondence.

    Parameters
    ----------
    source_point:
        (x, y) point in source image.

    predicted_reference_point:
        (x, y) prediction in reference image from the geometric model.

    Returns
    -------
    dict containing refined point, correction, response and success flag.
    """
    src = _to_gray_float32(source_image)
    ref = _to_gray_float32(reference_image)

    sx, sy = map(float, source_point)
    px, py = map(float, predicted_reference_point)

    source_patch = _extract_patch(src, sx, sy, patch_radius)
    reference_patch = _extract_patch(ref, px, py, patch_radius)

    if source_patch is None or reference_patch is None:
        return {
            "success": False,
            "reason": "Patch outside image",
            "point": [px, py],
            "correction": [0.0, 0.0],
            "response": 0.0,
        }

    source_patch = _normalize_patch(source_patch)
    reference_patch = _normalize_patch(reference_patch)

    if (
        float(np.std(source_patch)) < 1e-6
        or float(np.std(reference_patch)) < 1e-6
    ):
        return {
            "success": False,
            "reason": "Low-texture patch",
            "point": [px, py],
            "correction": [0.0, 0.0],
            "response": 0.0,
        }

    # phaseCorrelate(src1, src2) estimates the translation of src2
    # relative to src1. Here reference_patch is the observed patch and
    # source_patch acts as the local template.
    shift, response = cv2.phaseCorrelate(
        source_patch,
        reference_patch,
    )

    dx = float(shift[0])
    dy = float(shift[1])
    response = float(response)

    magnitude = float(np.hypot(dx, dy))

    if not np.isfinite(dx) or not np.isfinite(dy) or not np.isfinite(response):
        return {
            "success": False,
            "reason": "Non-finite phase-correlation result",
            "point": [px, py],
            "correction": [0.0, 0.0],
            "response": 0.0,
        }

    if response < min_response:
        return {
            "success": False,
            "reason": f"Low phase response ({response:.4f})",
            "point": [px, py],
            "correction": [0.0, 0.0],
            "response": response,
        }

    if magnitude > max_refinement:
        return {
            "success": False,
            "reason": (
                f"Correction too large ({magnitude:.4f}px > "
                f"{max_refinement:.4f}px)"
            ),
            "point": [px, py],
            "correction": [0.0, 0.0],
            "response": response,
        }

    refined = np.array(
        [px + dx, py + dy],
        dtype=np.float32,
    )

    return {
        "success": True,
        "reason": "Accepted",
        "point": refined.tolist(),
        "correction": [dx, dy],
        "response": response,
    }


def refine_matches(
    source_image: np.ndarray,
    reference_image: np.ndarray,
    source_points: np.ndarray,
    reference_points: np.ndarray,
    predicted_reference_points: np.ndarray | None = None,
    patch_radius: int = 15,
    max_refinement: float = 1.0,
    min_response: float = 0.20,
) -> dict:
    """
    Refine a set of trusted correspondences.

    If predicted_reference_points is omitted, the supplied reference_points
    are used as the geometric predictions.
    """
    source_points = np.asarray(source_points, dtype=np.float32)
    reference_points = np.asarray(reference_points, dtype=np.float32)

    if source_points.ndim != 2 or source_points.shape[1] != 2:
        raise ValueError("source_points must have shape (N, 2)")

    if reference_points.shape != source_points.shape:
        raise ValueError("reference_points must have shape (N, 2)")

    if predicted_reference_points is None:
        predicted_reference_points = reference_points.copy()
    else:
        predicted_reference_points = np.asarray(
            predicted_reference_points,
            dtype=np.float32,
        )

    if predicted_reference_points.shape != source_points.shape:
        raise ValueError(
            "predicted_reference_points must have shape (N, 2)"
        )

    refined_points = reference_points.copy()
    details = []

    for i in range(len(source_points)):
        result = refine_point(
            source_image=source_image,
            reference_image=reference_image,
            source_point=source_points[i],
            predicted_reference_point=predicted_reference_points[i],
            patch_radius=patch_radius,
            max_refinement=max_refinement,
            min_response=min_response,
        )

        details.append(result)

        if result["success"]:
            refined_points[i] = np.asarray(
                result["point"],
                dtype=np.float32,
            )

    successful = int(sum(d["success"] for d in details))

    return {
        "refined_points": refined_points,
        "details": details,
        "successful": successful,
        "total": int(len(source_points)),
        "success_rate": (
            float(successful / len(source_points))
            if len(source_points)
            else 0.0
        ),
    }


def synthetic_validation() -> dict:
    """
    Independent sanity test.

    A known fractional translation is applied to a textured image.
    The same local patch is used to estimate the fractional correction.
    """
    rng = np.random.default_rng(7)

    size = 512
    base = rng.normal(0, 1, (size, size)).astype(np.float32)
    base = cv2.GaussianBlur(base, (0, 0), 2.0)

    # Add deterministic structures.
    yy, xx = np.mgrid[0:size, 0:size]
    base += 2.0 * np.exp(
        -((xx - 250) ** 2 + (yy - 260) ** 2) / (2 * 30**2)
    )
    base += 1.3 * np.sin(xx / 17.0)

    base -= base.min()
    base /= max(base.max(), 1e-8)
    base *= 255.0

    true_dx = 0.37
    true_dy = -0.42

    shifted = cv2.warpAffine(
        base,
        np.array(
            [[1.0, 0.0, true_dx], [0.0, 1.0, true_dy]],
            dtype=np.float32,
        ),
        (size, size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )

    point = np.array([256.0, 256.0], dtype=np.float32)

    result = refine_point(
        source_image=base,
        reference_image=shifted,
        source_point=point,
        predicted_reference_point=point,
        patch_radius=31,
        max_refinement=1.0,
        min_response=0.20,
    )

    estimated = np.asarray(result["correction"], dtype=np.float64)
    truth = np.array([true_dx, true_dy], dtype=np.float64)

    error = float(np.linalg.norm(estimated - truth))

    output = {
        "true_shift": truth.tolist(),
        "estimated_shift": estimated.tolist(),
        "error_px": error,
        "response": result["response"],
        "success": bool(result["success"]),
        "details": result,
    }

    return output


if __name__ == "__main__":
    result = synthetic_validation()

    print("=" * 72)
    print("LUCAS — SUB-PIXEL REFINEMENT V3 SANITY TEST")
    print("=" * 72)
    print(f"True shift      : {result['true_shift']}")
    print(f"Estimated shift : {result['estimated_shift']}")
    print(f"Error           : {result['error_px']:.4f} px")
    print(f"Response        : {result['response']:.4f}")
    print(f"Accepted        : {result['success']}")
    print()

    output_path = RESULT_DIR / "subpixel_refinement_v3_sanity.json"
    output_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(f"Saved: {output_path}")
