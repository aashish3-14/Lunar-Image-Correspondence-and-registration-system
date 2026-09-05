from pathlib import Path
import json
import sys

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lightglue import LightGlue, SuperPoint
from ai_engine.preprocessing import (
    load_grayscale_image,
    preprocess_image,
    structural_representation,
)
from ai_engine.spatial import analyze_spatial_distribution

SOURCE = ROOT / "data" / "processed" / "tmc2_apollo16_focused.png"
REFERENCE = ROOT / "data" / "processed" / "lro_apollo16_overlap.png"
RESULTS = ROOT / "data" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def to_tensor(image, device):
    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    tensor = torch.from_numpy(
        rgb.astype(np.float32) / 255.0
    ).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def evaluate(source, reference, extractor, matcher, device, name):
    image0 = to_tensor(source, device)
    image1 = to_tensor(reference, device)

    with torch.inference_mode():
        f0 = extractor.extract(image0)
        f1 = extractor.extract(image1)
        m01 = matcher({"image0": f0, "image1": f1})

    k0 = f0["keypoints"][0].detach().cpu().numpy()
    k1 = f1["keypoints"][0].detach().cpu().numpy()
    matches = m01["matches"][0].detach().cpu().numpy()

    result = {
        "representation": name,
        "source_keypoints": int(len(k0)),
        "reference_keypoints": int(len(k1)),
        "matches": int(len(matches)),
        "inliers": 0,
        "inlier_ratio": 0.0,
        "rmse_px": None,
        "spatial_coverage": 0.0,
        "spatial_uniformity": 0.0,
        "spatial_distribution": "EMPTY",
    }

    if len(matches) < 4:
        return result

    p0 = k0[matches[:, 0]]
    p1 = k1[matches[:, 1]]

    H, mask = cv2.findHomography(
        p0, p1, cv2.RANSAC, 5.0
    )

    if H is None or mask is None:
        return result

    inliers = mask.ravel().astype(bool)
    count = int(inliers.sum())
    ratio = count / len(matches)

    projected = cv2.perspectiveTransform(
        p0.reshape(-1, 1, 2).astype(np.float32), H
    ).reshape(-1, 2)

    errors = np.linalg.norm(projected - p1, axis=1)
    ie = errors[inliers]

    spatial = analyze_spatial_distribution(
        p0[inliers],
        image_width=source.shape[1],
        image_height=source.shape[0],
        grid_rows=4,
        grid_cols=4,
    )

    result.update({
        "inliers": count,
        "inlier_ratio": float(ratio),
        "rmse_px": float(np.sqrt(np.mean(ie ** 2))) if len(ie) else None,
        "spatial_coverage": float(spatial["coverage"]),
        "spatial_uniformity": float(spatial["uniformity"]),
        "spatial_distribution": spatial["distribution"],
    })
    return result


def main():
    print("=" * 70)
    print("LUCAS | STEP 1 | REAL TMC-2 ↔ LRO TEST")
    print("=" * 70)

    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    if not REFERENCE.exists():
        raise FileNotFoundError(REFERENCE)

    source_raw = load_grayscale_image(SOURCE)
    reference_raw = load_grayscale_image(REFERENCE)

    print(f"Source:    {source_raw.shape}")
    print(f"Reference: {reference_raw.shape}")

    print("\nBuilding existing representation...")
    source_a = preprocess_image(source_raw)
    reference_a = preprocess_image(reference_raw)

    print("Building illumination-robust structural representation...")
    source_b = structural_representation(source_raw)
    reference_b = structural_representation(reference_raw)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\nLoading SuperPoint...")
    extractor = SuperPoint(max_num_keypoints=4096).eval().to(device)

    print("Loading LightGlue...")
    matcher = LightGlue(features="superpoint").eval().to(device)

    print("\nTEST A — EXISTING NORMALIZE + CLAHE")
    a = evaluate(
        source_a, reference_a, extractor, matcher,
        device, "existing_normalize_clahe"
    )
    print_result(a)

    print("\nTEST B — ILLUMINATION-ROBUST STRUCTURAL")
    b = evaluate(
        source_b, reference_b, extractor, matcher,
        device, "illumination_robust_structural"
    )
    print_result(b)

    output = RESULTS / "illumination_validation.json"
    with open(output, "w", encoding="utf-8") as f:
        json.dump({"existing": a, "structural": b}, f, indent=4)

    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"{'Metric':24s}{'Existing':>16s}{'Structural':>16s}")
    print("-" * 56)

    for label, key in [
        ("Source keypoints", "source_keypoints"),
        ("Matches", "matches"),
        ("Inliers", "inliers"),
        ("Inlier ratio", "inlier_ratio"),
        ("RMSE px", "rmse_px"),
        ("Spatial coverage", "spatial_coverage"),
        ("Spatial uniformity", "spatial_uniformity"),
    ]:
        av = a[key]
        bv = b[key]
        av = "N/A" if av is None else f"{av:.3f}" if isinstance(av, float) else str(av)
        bv = "N/A" if bv is None else f"{bv:.3f}" if isinstance(bv, float) else str(bv)
        print(f"{label:24s}{av:>16s}{bv:>16s}")

    print(f"\nSaved: {output}")
    print("STEP 1 REAL-DATA TEST COMPLETE")


def print_result(r):
    print(f"  Keypoints: {r['source_keypoints']} / {r['reference_keypoints']}")
    print(f"  Matches: {r['matches']}")
    print(f"  Inliers: {r['inliers']}")
    print(f"  Inlier ratio: {r['inlier_ratio']:.3f}")
    print(f"  RMSE: {r['rmse_px']}")
    print(f"  Spatial coverage: {r['spatial_coverage']:.3f}")
    print(f"  Spatial uniformity: {r['spatial_uniformity']:.3f}")
    print(f"  Distribution: {r['spatial_distribution']}")


if __name__ == "__main__":
    main()
