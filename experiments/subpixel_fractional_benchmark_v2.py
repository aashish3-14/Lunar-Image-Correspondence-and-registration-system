"""
LUCAS — Fractional-Pixel Benchmark V2

Purpose
-------
Validate whether OpenCV phase correlation can recover known sub-pixel
translations with the correct sign convention.

V1 showed a systematic sign reversal. V2 intentionally reports the raw
phase-correlation shift and uses it directly as the estimated translation.

The benchmark:
1. Generates a textured synthetic reference image.
2. Applies known fractional-pixel translations.
3. Uses a central ROI to reduce border effects.
4. Estimates translation with cv2.phaseCorrelate.
5. Reports X, Y and Euclidean errors.
6. Tests both positive and negative fractional shifts.
7. Applies a simple pass/fail criterion.

This is a benchmark only. It does NOT claim that the method is ready for
production cross-sensor lunar registration.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "data" / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def make_texture(size: int = 768, seed: int = 42) -> np.ndarray:
    """Create a repeatable, feature-rich grayscale texture."""
    rng = np.random.default_rng(seed)

    image = rng.normal(0, 1, (size, size)).astype(np.float32)

    # Multi-scale smoothing creates realistic spatial structure.
    coarse = cv2.GaussianBlur(image, (0, 0), 6.0)
    fine = cv2.GaussianBlur(image, (0, 0), 1.2)

    texture = 0.65 * coarse + 0.35 * fine

    # Add several deterministic geometric structures.
    yy, xx = np.mgrid[0:size, 0:size]

    for cx, cy, sigma, amp in [
        (180, 210, 28, 2.5),
        (500, 180, 45, -2.0),
        (390, 510, 32, 2.2),
        (650, 600, 55, -1.8),
        (120, 620, 22, 1.7),
    ]:
        texture += amp * np.exp(
            -((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2)
        )

    # Add weak directional structure.
    texture += 0.35 * np.sin(xx / 13.0) + 0.25 * np.cos(yy / 19.0)

    texture -= texture.min()
    texture /= max(texture.max(), 1e-8)
    return (texture * 255.0).astype(np.float32)


def fractional_translate(
    image: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    """Translate an image by a known fractional-pixel amount."""
    matrix = np.array(
        [[1.0, 0.0, dx], [0.0, 1.0, dy]],
        dtype=np.float32,
    )

    return cv2.warpAffine(
        image,
        matrix,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )


def central_roi(image: np.ndarray, margin: int = 128) -> np.ndarray:
    """Remove translated-image borders before phase correlation."""
    h, w = image.shape
    return image[margin : h - margin, margin : w - margin]


def run_single_test(
    reference: np.ndarray,
    dx: float,
    dy: float,
) -> dict:
    """Run one known fractional translation test."""
    shifted = fractional_translate(reference, dx, dy)

    ref_roi = central_roi(reference)
    shifted_roi = central_roi(shifted)

    # OpenCV phaseCorrelate returns the shift of src2 relative to src1.
    raw_shift, response = cv2.phaseCorrelate(
        ref_roi.astype(np.float32),
        shifted_roi.astype(np.float32),
    )

    estimated_dx = float(raw_shift[0])
    estimated_dy = float(raw_shift[1])

    error_x = estimated_dx - dx
    error_y = estimated_dy - dy
    euclidean_error = float(np.hypot(error_x, error_y))

    return {
        "ground_truth_dx": float(dx),
        "ground_truth_dy": float(dy),
        "raw_phase_shift_x": float(raw_shift[0]),
        "raw_phase_shift_y": float(raw_shift[1]),
        "estimated_dx": estimated_dx,
        "estimated_dy": estimated_dy,
        "error_x": float(error_x),
        "error_y": float(error_y),
        "euclidean_error": euclidean_error,
        "response": float(response),
    }


def main() -> None:
    print("=" * 72)
    print("LUCAS — FRACTIONAL-PIXEL BENCHMARK V2")
    print("=" * 72)
    print()
    print("Convention check:")
    print("  reference -> shifted image")
    print("  estimated shift = raw cv2.phaseCorrelate output")
    print("  (V1 incorrectly negated this value)")
    print()

    reference = make_texture()

    tests = [
        (0.10, 0.15),
        (0.25, 0.30),
        (0.37, -0.42),
        (-0.45, 0.28),
        (0.61, -0.57),
        (-0.68, -0.34),
        (0.74, 0.72),
        (-0.72, 0.65),
    ]

    results: list[dict] = []

    for i, (dx, dy) in enumerate(tests, start=1):
        result = run_single_test(reference, dx, dy)
        result["test"] = i
        results.append(result)

        print(f"TEST {i}")
        print(
            f"  Ground truth : "
            f"{result['ground_truth_dx']:+.3f}, "
            f"{result['ground_truth_dy']:+.3f}"
        )
        print(
            f"  Raw shift    : "
            f"{result['raw_phase_shift_x']:+.3f}, "
            f"{result['raw_phase_shift_y']:+.3f}"
        )
        print(
            f"  Estimated    : "
            f"{result['estimated_dx']:+.3f}, "
            f"{result['estimated_dy']:+.3f}"
        )
        print(
            f"  Error X/Y    : "
            f"{result['error_x']:+.4f}, "
            f"{result['error_y']:+.4f}"
        )
        print(f"  Euclidean    : {result['euclidean_error']:.4f} px")
        print(f"  Response     : {result['response']:.4f}")
        print()

    errors = np.array([r["euclidean_error"] for r in results], dtype=np.float64)
    abs_x = np.abs([r["error_x"] for r in results])
    abs_y = np.abs([r["error_y"] for r in results])

    mean_error = float(errors.mean())
    median_error = float(np.median(errors))
    max_error = float(errors.max())
    mean_abs_x = float(np.mean(abs_x))
    mean_abs_y = float(np.mean(abs_y))

    # Target for this benchmark. These are validation thresholds, not
    # guaranteed production accuracy requirements.
    mean_threshold = 0.25
    max_threshold = 0.50

    passed = mean_error < mean_threshold and max_error < max_threshold

    summary = {
        "benchmark": "LUCAS fractional-pixel benchmark V2",
        "method": "OpenCV phase correlation",
        "sign_convention": "raw phaseCorrelate shift used directly",
        "num_tests": len(results),
        "mean_euclidean_error_px": mean_error,
        "median_euclidean_error_px": median_error,
        "max_euclidean_error_px": max_error,
        "mean_abs_error_x_px": mean_abs_x,
        "mean_abs_error_y_px": mean_abs_y,
        "mean_error_threshold_px": mean_threshold,
        "max_error_threshold_px": max_threshold,
        "passed": bool(passed),
        "tests": results,
    }

    output = RESULT_DIR / "subpixel_fractional_benchmark_v2.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"Mean Euclidean error : {mean_error:.4f} px")
    print(f"Median error         : {median_error:.4f} px")
    print(f"Maximum error        : {max_error:.4f} px")
    print(f"Mean |X| error       : {mean_abs_x:.4f} px")
    print(f"Mean |Y| error       : {mean_abs_y:.4f} px")
    print()
    print(f"Criterion: mean < {mean_threshold:.2f}px")
    print(f"           max  < {max_threshold:.2f}px")
    print()
    print("RESULT:", "PASS" if passed else "FAIL")
    print()
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
