from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
OUTPUT_JSON = RESULTS / "subpixel_fractional_benchmark.json"
OUTPUT_VIS = RESULTS / "subpixel_fractional_benchmark.png"

TEST_SHIFTS = [
    (0.25, 0.30),
    (0.37, -0.42),
    (-0.45, 0.28),
    (0.61, -0.57),
    (-0.68, -0.34),
]


def make_textured_image(width=512, height=512, seed=42):
    rng = np.random.default_rng(seed)
    image = rng.normal(100.0, 30.0, (height, width)).astype(np.float32)
    image = cv2.GaussianBlur(image, (0, 0), 1.2)
    coarse = rng.normal(0.0, 18.0, (height, width)).astype(np.float32)
    image += cv2.GaussianBlur(coarse, (0, 0), 7.0)

    for _ in range(100):
        x = int(rng.integers(20, width - 20))
        y = int(rng.integers(20, height - 20))
        radius = int(rng.integers(2, 12))
        value = float(rng.uniform(-60.0, 80.0))
        cv2.circle(image, (x, y), radius, value, -1)

    for _ in range(30):
        x1 = int(rng.integers(10, width - 30))
        y1 = int(rng.integers(10, height - 30))
        x2 = x1 + int(rng.integers(5, 50))
        y2 = y1 + int(rng.integers(5, 50))
        cv2.rectangle(
            image, (x1, y1), (x2, y2),
            float(rng.uniform(-50.0, 70.0)), -1
        )

    image = cv2.GaussianBlur(image, (0, 0), 0.8)
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)


def fractional_translate(image, dx, dy):
    matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    return cv2.warpAffine(
        image, matrix, (image.shape[1], image.shape[0]),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101
    )


def phase_estimate(reference, shifted):
    a = reference.astype(np.float32)
    b = shifted.astype(np.float32)
    window = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    shift, response = cv2.phaseCorrelate(a, b, window)
    return -float(shift[0]), -float(shift[1]), float(response)


def run_test(reference, dx, dy):
    shifted = fractional_translate(reference, dx, dy)
    ex, ey, response = phase_estimate(reference, shifted)
    return {
        "ground_truth_dx": float(dx),
        "ground_truth_dy": float(dy),
        "estimated_dx": ex,
        "estimated_dy": ey,
        "error_dx": ex - dx,
        "error_dy": ey - dy,
        "euclidean_error_px": float(np.hypot(ex - dx, ey - dy)),
        "response": response,
    }


def main():
    print("=" * 60)
    print("      LUCAS FRACTIONAL-PIXEL BENCHMARK")
    print("=" * 60)
    RESULTS.mkdir(parents=True, exist_ok=True)

    reference = make_textured_image()
    results = []

    for i, (dx, dy) in enumerate(TEST_SHIFTS, 1):
        r = run_test(reference, dx, dy)
        results.append(r)
        print(f"\nTEST {i}")
        print(f"Ground truth:  dx={dx:+.3f}, dy={dy:+.3f}")
        print(f"Estimated:     dx={r['estimated_dx']:+.3f}, dy={r['estimated_dy']:+.3f}")
        print(f"Error:         {r['euclidean_error_px']:.4f} px")
        print(f"Response:      {r['response']:.4f}")

    errors = np.array([r["euclidean_error_px"] for r in results])
    mean_error = float(np.mean(errors))
    median_error = float(np.median(errors))
    max_error = float(np.max(errors))
    passed = bool(mean_error < 0.25 and max_error < 0.50)

    print("\n" + "=" * 50)
    print("        BENCHMARK SUMMARY")
    print("=" * 50)
    print(f"Tests:             {len(results)}")
    print(f"Mean error:        {mean_error:.4f} px")
    print(f"Median error:      {median_error:.4f} px")
    print(f"Maximum error:     {max_error:.4f} px")
    print("Criterion:         mean < 0.25 px, max < 0.50 px")
    print(f"RESULT:            {'PASS' if passed else 'FAIL'}")

    shifted_demo = fractional_translate(reference, *TEST_SHIFTS[1])
    ref_u8 = cv2.normalize(reference, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    demo_u8 = cv2.normalize(shifted_demo, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(str(OUTPUT_VIS), np.hstack([ref_u8, demo_u8]))

    payload = {
        "method": "OpenCV phase correlation",
        "tests": results,
        "mean_error_px": mean_error,
        "median_error_px": median_error,
        "max_error_px": max_error,
        "criterion": {"mean_error_less_than_px": 0.25, "max_error_less_than_px": 0.50},
        "passed": passed,
    }
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved JSON: {OUTPUT_JSON}")
    print(f"Saved visualization: {OUTPUT_VIS}")
    print("\n" + "=" * 60)
    print("FRACTIONAL-PIXEL BENCHMARK COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
