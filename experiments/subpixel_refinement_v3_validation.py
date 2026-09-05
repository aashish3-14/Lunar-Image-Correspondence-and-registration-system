"""
LUCAS — Sub-pixel Refinement V3 Controlled Validation

Runs multiple known fractional translations and evaluates the V3 local
phase-correlation refinement against exact ground truth.

This is an independent validation experiment. It does not modify the
production LUCAS pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_engine.subpixel import refine_point


RESULT_DIR = ROOT / "data" / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def make_texture(size: int = 768, seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)

    noise = rng.normal(0, 1, (size, size)).astype(np.float32)
    texture = (
        0.55 * cv2.GaussianBlur(noise, (0, 0), 2.0)
        + 0.45 * cv2.GaussianBlur(noise, (0, 0), 6.0)
    )

    yy, xx = np.mgrid[0:size, 0:size]

    structures = [
        (180, 180, 25, 2.2),
        (420, 230, 40, -1.8),
        (300, 470, 35, 2.0),
        (610, 560, 48, -2.2),
        (120, 620, 20, 1.7),
    ]

    for cx, cy, sigma, amp in structures:
        texture += amp * np.exp(
            -((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2)
        )

    texture += 0.25 * np.sin(xx / 11.0)
    texture += 0.20 * np.cos(yy / 17.0)

    texture -= texture.min()
    texture /= max(texture.max(), 1e-8)
    return (texture * 255).astype(np.float32)


def translate(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
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


def main() -> None:
    print("=" * 76)
    print("LUCAS — SUB-PIXEL REFINEMENT V3 CONTROLLED VALIDATION")
    print("=" * 76)
    print()

    reference = make_texture()

    # Keep the point far from boundaries.
    point = np.array([384.0, 384.0], dtype=np.float32)

    tests = [
        (0.10, 0.15),
        (0.25, 0.30),
        (0.37, -0.42),
        (-0.45, 0.28),
        (0.50, -0.50),
        (0.61, -0.57),
        (-0.68, -0.34),
        (0.74, 0.72),
        (-0.72, 0.65),
    ]

    results = []

    for index, (dx, dy) in enumerate(tests, start=1):
        shifted = translate(reference, dx, dy)

        result = refine_point(
            source_image=reference,
            reference_image=shifted,
            source_point=point,
            predicted_reference_point=point,
            patch_radius=31,
            max_refinement=1.0,
            min_response=0.20,
        )

        estimated = np.asarray(result["correction"], dtype=np.float64)
        truth = np.array([dx, dy], dtype=np.float64)

        error_vector = estimated - truth

        record = {
            "test": index,
            "ground_truth_dx": float(dx),
            "ground_truth_dy": float(dy),
            "estimated_dx": float(estimated[0]),
            "estimated_dy": float(estimated[1]),
            "error_x_px": float(error_vector[0]),
            "error_y_px": float(error_vector[1]),
            "euclidean_error_px": float(np.linalg.norm(error_vector)),
            "response": float(result["response"]),
            "success": bool(result["success"]),
            "reason": result["reason"],
        }

        results.append(record)

        print(f"TEST {index}")
        print(f"  Truth      : {dx:+.3f}, {dy:+.3f}")
        print(f"  Estimated  : {estimated[0]:+.3f}, {estimated[1]:+.3f}")
        print(
            f"  Error X/Y  : "
            f"{error_vector[0]:+.4f}, {error_vector[1]:+.4f}"
        )
        print(f"  Euclidean  : {record['euclidean_error_px']:.4f} px")
        print(f"  Response   : {record['response']:.4f}")
        print(f"  Accepted   : {record['success']}")
        print()

    errors = np.array(
        [r["euclidean_error_px"] for r in results],
        dtype=np.float64,
    )

    successful = [r for r in results if r["success"]]

    mean_error = float(errors.mean())
    median_error = float(np.median(errors))
    max_error = float(errors.max())
    success_rate = float(len(successful) / len(results))

    mean_threshold = 0.25
    max_threshold = 0.50
    success_threshold = 0.80

    passed = (
        mean_error < mean_threshold
        and max_error < max_threshold
        and success_rate >= success_threshold
    )

    summary = {
        "benchmark": "LUCAS sub-pixel refinement V3 controlled validation",
        "method": "local phase correlation",
        "num_tests": len(results),
        "successful_tests": len(successful),
        "success_rate": success_rate,
        "mean_error_px": mean_error,
        "median_error_px": median_error,
        "max_error_px": max_error,
        "mean_error_threshold_px": mean_threshold,
        "max_error_threshold_px": max_threshold,
        "success_rate_threshold": success_threshold,
        "passed": bool(passed),
        "tests": results,
    }

    output = RESULT_DIR / "subpixel_refinement_v3_validation.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 76)
    print("SUMMARY")
    print("=" * 76)
    print(f"Successful refinements : {len(successful)}/{len(results)}")
    print(f"Success rate            : {success_rate:.3f}")
    print(f"Mean error              : {mean_error:.4f} px")
    print(f"Median error            : {median_error:.4f} px")
    print(f"Maximum error           : {max_error:.4f} px")
    print()
    print(f"Criterion: mean < {mean_threshold:.2f}px")
    print(f"           max  < {max_threshold:.2f}px")
    print(f"           success >= {success_threshold:.0%}")
    print()
    print("RESULT:", "PASS" if passed else "FAIL")
    print()
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
