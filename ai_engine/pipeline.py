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


# ============================================================
# IMPORTS
# ============================================================

import cv2
import numpy as np
import torch

from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image

from ai_engine.quality import evaluate_registration_quality
from ai_engine.spatial import (
    analyze_spatial_distribution,
    select_uniform_matches,
)
from ai_engine.subpixel import refine_matches
from ai_engine.registration import homography_registration


# ============================================================
# PATHS
# ============================================================

RESULTS_DIR = PROJECT_ROOT / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# IMAGE UTILITIES
# ============================================================

def tensor_to_uint8_image(image_tensor):
    """
    Convert [3,H,W] tensor in [0,1]
    to OpenCV BGR uint8 [H,W,3].
    """

    if isinstance(image_tensor, torch.Tensor):
        image = image_tensor.detach().cpu().numpy()
    else:
        image = np.asarray(image_tensor)

    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = np.transpose(image, (1, 2, 0))

    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]

    if image.dtype != np.uint8:
        if np.max(image) <= 1.5:
            image = image * 255.0

        image = np.clip(
            image,
            0,
            255,
        ).astype(np.uint8)

    if image.ndim == 2:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR,
        )
    elif image.shape[2] == 3:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR,
        )

    return image


# ============================================================
# SIFT FEATURE EXTRACTION
# ============================================================

def _extract_sift_features(
    image,
    max_features=5000,
):
    """
    Extract SIFT features for the classical branch.
    """

    if image is None:
        raise ValueError(
            "Image cannot be None."
        )

    image = np.asarray(image)

    if image.ndim == 3:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )
    else:
        gray = image

    gray = np.asarray(
        gray,
        dtype=np.uint8,
    )

    sift = cv2.SIFT_create(
        nfeatures=int(max_features),
        contrastThreshold=0.01,
        edgeThreshold=10,
        sigma=1.6,
    )

    keypoints, descriptors = (
        sift.detectAndCompute(
            gray,
            None,
        )
    )

    if keypoints is None:
        keypoints = []

    if descriptors is None:
        descriptors = np.empty(
            (0, 128),
            dtype=np.float32,
        )

    return keypoints, descriptors


# ============================================================
# RECIPROCAL SIFT MATCHING
# ============================================================

def _match_sift_reciprocal(
    source_descriptors,
    reference_descriptors,
    ratio_threshold=0.90,
):
    """
    Reciprocal Lowe-ratio SIFT matching.
    """

    if (
        source_descriptors is None
        or reference_descriptors is None
        or len(source_descriptors) == 0
        or len(reference_descriptors) == 0
    ):
        return []

    matcher = cv2.BFMatcher(
        cv2.NORM_L2,
        crossCheck=False,
    )

    forward = matcher.knnMatch(
        source_descriptors.astype(np.float32),
        reference_descriptors.astype(np.float32),
        k=2,
    )

    backward = matcher.knnMatch(
        reference_descriptors.astype(np.float32),
        source_descriptors.astype(np.float32),
        k=2,
    )

    forward_good = {}

    for pair in forward:

        if len(pair) < 2:
            continue

        m, n = pair

        if (
            m.distance
            <
            ratio_threshold * n.distance
        ):
            forward_good[m.queryIdx] = m

    backward_good = {}

    for pair in backward:

        if len(pair) < 2:
            continue

        m, n = pair

        if (
            m.distance
            <
            ratio_threshold * n.distance
        ):
            backward_good[m.queryIdx] = m

    reciprocal = []

    for source_idx, match in (
        forward_good.items()
    ):

        reference_idx = match.trainIdx

        reverse = backward_good.get(
            reference_idx
        )

        if reverse is None:
            continue

        if reverse.trainIdx != source_idx:
            continue

        reciprocal.append(match)

    reciprocal.sort(
        key=lambda m: float(m.distance)
    )

    return reciprocal


# ============================================================
# GEOMETRY-GUIDED SIFT
# ============================================================

def _geometry_guided_sift(
    source_image,
    reference_image,
    scale_ratio=1.156,
    source_center=None,
    reference_center=None,
    rotation_degrees=(0.0, -5.0, 5.0),
    ratio_threshold=0.92,
):
    """Generate SIFT candidates using an approximate geometry prior."""
    source = np.asarray(source_image)
    reference = np.asarray(reference_image)

    source_gray = (
        cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        if source.ndim == 3 else source.copy()
    )
    reference_gray = (
        cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        if reference.ndim == 3 else reference.copy()
    )

    source_gray = source_gray.astype(np.uint8)
    reference_gray = reference_gray.astype(np.uint8)

    sh, sw = source_gray.shape
    rh, rw = reference_gray.shape

    if source_center is None:
        source_center = (sw / 2.0, sh / 2.0)
    if reference_center is None:
        reference_center = (rw / 2.0, rh / 2.0)

    sx, sy = source_center
    rx, ry = reference_center

    sift = cv2.SIFT_create(
        nfeatures=6000,
        contrastThreshold=0.008,
        edgeThreshold=10,
        sigma=1.6,
    )
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    all_source_points = []
    all_reference_points = []
    all_scores = []

    print()
    print("========== GEOMETRY-GUIDED SIFT ==========")
    print(f"Scale prior: {scale_ratio:.4f}")
    print(f"Source center: ({sx:.1f}, {sy:.1f})")
    print(f"Reference center: ({rx:.1f}, {ry:.1f})")

    kp1, des1 = sift.detectAndCompute(reference_gray, None)
    if des1 is None or len(kp1) < 4:
        print("Reference SIFT features insufficient.")
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    for angle in rotation_degrees:
        theta = np.deg2rad(angle)
        c, s = np.cos(theta), np.sin(theta)

        a11 = scale_ratio * c
        a12 = -scale_ratio * s
        a21 = scale_ratio * s
        a22 = scale_ratio * c
        tx = rx - a11 * sx - a12 * sy
        ty = ry - a21 * sx - a22 * sy

        M = np.array(
            [[a11, a12, tx], [a21, a22, ty]],
            dtype=np.float32,
        )

        warped_source = cv2.warpAffine(
            source_gray,
            M,
            (rw, rh),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        kp0, des0 = sift.detectAndCompute(warped_source, None)
        if des0 is None or len(kp0) < 4:
            print(f"Angle {angle:+.1f}°: 0 guided matches")
            continue

        forward = bf.knnMatch(des0, des1, k=2)
        backward = bf.knnMatch(des1, des0, k=2)

        forward_good = {}
        for pair in forward:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < ratio_threshold * n.distance:
                forward_good[m.queryIdx] = m

        backward_good = {}
        for pair in backward:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < ratio_threshold * n.distance:
                backward_good[m.queryIdx] = m

        M3 = np.vstack([M, [0.0, 0.0, 1.0]])
        M_inv = np.linalg.inv(M3)

        angle_source = []
        angle_reference = []
        angle_scores = []

        for source_idx, match in forward_good.items():
            reverse = backward_good.get(match.trainIdx)
            if reverse is None or reverse.trainIdx != source_idx:
                continue

            p = np.array(
                [kp0[source_idx].pt[0], kp0[source_idx].pt[1], 1.0],
                dtype=np.float64,
            )
            original = M_inv @ p
            if abs(original[2]) < 1e-12:
                continue
            original /= original[2]

            source_point = original[:2].astype(np.float32)
            reference_point = np.asarray(
                kp1[match.trainIdx].pt,
                dtype=np.float32,
            )

            if not (
                0 <= source_point[0] < sw
                and 0 <= source_point[1] < sh
            ):
                continue

            predicted = M @ np.array(
                [source_point[0], source_point[1], 1.0],
                dtype=np.float32,
            )
            residual = float(
                np.linalg.norm(predicted[:2] - reference_point)
            )

            if residual > 80.0:
                continue

            angle_source.append(source_point)
            angle_reference.append(reference_point)
            angle_scores.append(1.0 / (1.0 + float(match.distance)))

        print(f"Angle {angle:+.1f}°: {len(angle_source)} guided matches")
        all_source_points.extend(angle_source)
        all_reference_points.extend(angle_reference)
        all_scores.extend(angle_scores)

    if not all_source_points:
        print("Geometry-guided SIFT: 0 matches")
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    source_points = np.asarray(all_source_points, dtype=np.float32)
    reference_points = np.asarray(all_reference_points, dtype=np.float32)
    scores = np.asarray(all_scores, dtype=np.float32)

    keep = []
    for i in range(len(source_points)):
        duplicate = False
        for j in keep:
            if (
                np.linalg.norm(source_points[i] - source_points[j]) < 2.0
                and np.linalg.norm(reference_points[i] - reference_points[j]) < 2.0
            ):
                duplicate = True
                break
        if not duplicate:
            keep.append(i)

    keep = np.asarray(keep, dtype=np.int64)
    source_points = source_points[keep]
    reference_points = reference_points[keep]
    scores = scores[keep]

    print(f"Guided SIFT final candidates: {len(source_points)}")
    print("==========================================")

    return source_points, reference_points, scores


# ============================================================
# POINT MATCH VISUALIZATION
# ============================================================

def save_point_match_visualization(
    image0,
    image1,
    source_points,
    reference_points,
    output_path,
    title="LUCAS Correspondence",
):
    """
    Visualize arbitrary source/reference point pairs.
    """

    img0 = tensor_to_uint8_image(image0)
    img1 = tensor_to_uint8_image(image1)

    max_height = 900

    scale0 = min(
        1.0,
        max_height /
        max(img0.shape[0], 1),
    )

    scale1 = min(
        1.0,
        max_height /
        max(img1.shape[0], 1),
    )

    if scale0 < 1.0:
        img0 = cv2.resize(
            img0,
            None,
            fx=scale0,
            fy=scale0,
            interpolation=cv2.INTER_AREA,
        )

    if scale1 < 1.0:
        img1 = cv2.resize(
            img1,
            None,
            fx=scale1,
            fy=scale1,
            interpolation=cv2.INTER_AREA,
        )

    original_h0, original_w0 = (
        image0.shape[-2:]
        if isinstance(image0, torch.Tensor)
        else image0.shape[:2]
    )

    original_h1, original_w1 = (
        image1.shape[-2:]
        if isinstance(image1, torch.Tensor)
        else image1.shape[:2]
    )

    sx0 = (
        img0.shape[1] /
        max(float(original_w0), 1.0)
    )

    sy0 = (
        img0.shape[0] /
        max(float(original_h0), 1.0)
    )

    sx1 = (
        img1.shape[1] /
        max(float(original_w1), 1.0)
    )

    sy1 = (
        img1.shape[0] /
        max(float(original_h1), 1.0)
    )

    canvas_height = max(
        img0.shape[0],
        img1.shape[0],
    )

    canvas_width = (
        img0.shape[1] +
        img1.shape[1]
    )

    canvas = np.zeros(
        (
            canvas_height,
            canvas_width,
            3,
        ),
        dtype=np.uint8,
    )

    canvas[
        :img0.shape[0],
        :img0.shape[1],
    ] = img0

    canvas[
        :img1.shape[0],
        img0.shape[1]:
    ] = img1

    source_points = np.asarray(
        source_points,
        dtype=np.float32,
    )

    reference_points = np.asarray(
        reference_points,
        dtype=np.float32,
    )

    offset = img0.shape[1]

    for p0, p1 in zip(
        source_points,
        reference_points,
    ):

        x0 = (
            float(p0[0]) * sx0
        )

        y0 = (
            float(p0[1]) * sy0
        )

        x1 = (
            float(p1[0]) * sx1
            + offset
        )

        y1 = (
            float(p1[1]) * sy1
        )

        q0 = (
            int(round(x0)),
            int(round(y0)),
        )

        q1 = (
            int(round(x1)),
            int(round(y1)),
        )

        cv2.line(
            canvas,
            q0,
            q1,
            (0, 180, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.circle(
            canvas,
            q0,
            3,
            (0, 255, 0),
            -1,
            cv2.LINE_AA,
        )

        cv2.circle(
            canvas,
            q1,
            3,
            (0, 255, 0),
            -1,
            cv2.LINE_AA,
        )

    cv2.putText(
        canvas,
        "SOURCE / MOVING",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        "REFERENCE / FIXED",
        (
            img0.shape[1] + 20,
            40,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        title,
        (
            20,
            canvas_height - 20,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.imwrite(
        str(output_path),
        canvas,
    )


# ============================================================
# LIGHTGLUE VISUALIZATION
# ============================================================

def save_lightglue_visualization(
    image0,
    image1,
    keypoints0,
    keypoints1,
    matches,
    output_path,
):
    """
    Save LightGlue-specific correspondence visualization.
    """

    source_points = keypoints0[
        matches[:, 0]
    ] if len(matches) else np.empty(
        (0, 2),
        dtype=np.float32,
    )

    reference_points = keypoints1[
        matches[:, 1]
    ] if len(matches) else np.empty(
        (0, 2),
        dtype=np.float32,
    )

    save_point_match_visualization(
        image0,
        image1,
        source_points,
        reference_points,
        output_path,
        title=(
            f"LUCAS LightGlue Matches: "
            f"{len(matches)}"
        ),
    )


# ============================================================
# GEOMETRIC MODEL EVALUATION
# ============================================================

def evaluate_geometric_model(
    source_points,
    reference_points,
    model,
    mask,
    model_name,
):
    """
    Evaluate affine or homography model.
    """

    if model is None or mask is None:
        return {
            "model_name": model_name,
            "model": None,
            "mask": None,
            "inlier_count": 0,
            "inlier_ratio": 0.0,
            "rmse_px": None,
            "median_error_px": None,
            "max_error_px": None,
        }

    mask = (
        np.asarray(mask)
        .ravel()
        .astype(bool)
    )

    if len(mask) != len(source_points):
        return {
            "model_name": model_name,
            "model": None,
            "mask": None,
            "inlier_count": 0,
            "inlier_ratio": 0.0,
            "rmse_px": None,
            "median_error_px": None,
            "max_error_px": None,
        }

    try:

        source_points = np.asarray(
            source_points,
            dtype=np.float32,
        )

        reference_points = np.asarray(
            reference_points,
            dtype=np.float32,
        )

        if model_name == "Affine":

            projected = cv2.transform(
                source_points.reshape(
                    -1,
                    1,
                    2,
                ),
                model,
            ).reshape(
                -1,
                2,
            )

        else:

            projected = cv2.perspectiveTransform(
                source_points.reshape(
                    -1,
                    1,
                    2,
                ),
                model,
            ).reshape(
                -1,
                2,
            )

        errors = np.linalg.norm(
            projected -
            reference_points,
            axis=1,
        )

        inlier_errors = errors[
            mask
        ]

    except cv2.error:

        return {
            "model_name": model_name,
            "model": None,
            "mask": None,
            "inlier_count": 0,
            "inlier_ratio": 0.0,
            "rmse_px": None,
            "median_error_px": None,
            "max_error_px": None,
        }

    inlier_count = int(
        np.sum(mask)
    )

    if len(inlier_errors) == 0:

        rmse = None
        median_error = None
        max_error = None

    else:

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

    return {
        "model_name": model_name,
        "model": model,
        "mask": mask,
        "inlier_count": inlier_count,
        "inlier_ratio": (
            inlier_count /
            max(len(source_points), 1)
        ),
        "rmse_px": rmse,
        "median_error_px": median_error,
        "max_error_px": max_error,
    }


# ============================================================
# GEOMETRY → HOMOGRAPHY
# ============================================================

def affine_to_homography(
    affine_matrix,
):
    """
    Convert 2x3 affine matrix to 3x3 homography.
    """

    if affine_matrix is None:
        return None

    return np.vstack(
        [
            np.asarray(
                affine_matrix,
                dtype=np.float64,
            ),
            np.array(
                [[0.0, 0.0, 1.0]],
                dtype=np.float64,
            ),
        ]
    )


# ============================================================
# JSON
# ============================================================

def save_result_json(
    result,
    output_path,
):
    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            result,
            f,
            indent=4,
        )

    print(
        f"Saved result JSON: "
        f"{output_path}"
    )


# ============================================================
# MAIN LUCAS ENGINE
# ============================================================

def run_lucas(
    source_path,
    reference_path,
    max_num_keypoints=4096,
    ransac_threshold=5.0,
    show_visualization=False,
):
    """
    LUCAS:

        Source
          ↓
        SuperPoint
          ↓
        LightGlue
          ↓
        SIFT fallback
          ↓
        Candidate fusion
          ↓
        Affine + Homography RANSAC
          ↓
        Geometric model selection
          ↓
        Subpixel refinement V3
          ↓
        Final geometric verification
          ↓
        Spatial selection
          ↓
        Quality gate
          ↓
        Registration / rejection
    """

    source_path = Path(
        source_path
    )

    reference_path = Path(
        reference_path
    )

    print()
    print("=" * 50)
    print(
        "              LUCAS ENGINE"
    )
    print("=" * 50)

    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    if device.type == "cuda":

        print(
            "GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # ========================================================
    # VALIDATION
    # ========================================================

    if not source_path.exists():

        raise FileNotFoundError(
            f"Source image not found: "
            f"{source_path}"
        )

    if not reference_path.exists():

        raise FileNotFoundError(
            f"Reference image not found: "
            f"{reference_path}"
        )

    # ========================================================
    # LOAD IMAGES
    # ========================================================

    print()
    print(
        "Loading images..."
    )

    image0 = load_image(
        source_path
    )

    image1 = load_image(
        reference_path
    )

    print(
        f"Source shape: "
        f"{tuple(image0.shape)}"
    )

    print(
        f"Reference shape: "
        f"{tuple(image1.shape)}"
    )

    image0 = image0.to(
        device
    )

    image1 = image1.to(
        device
    )

    source_uint8 = (
        tensor_to_uint8_image(
            image0
        )
    )

    reference_uint8 = (
        tensor_to_uint8_image(
            image1
        )
    )

    # ========================================================
    # SUPERPOINT
    # ========================================================

    print()
    print(
        "Loading SuperPoint..."
    )

    extractor = (
        SuperPoint(
            max_num_keypoints=
            max_num_keypoints
        )
        .eval()
        .to(device)
    )

    # ========================================================
    # LIGHTGLUE
    # ========================================================

    print(
        "Loading LightGlue..."
    )

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
    print(
        "Extracting features..."
    )

    with torch.inference_mode():

        feats0 = extractor.extract(
            image0
        )

        feats1 = extractor.extract(
            image1
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
        f"Source keypoints: "
        f"{len(keypoints0)}"
    )

    print(
        f"Reference keypoints: "
        f"{len(keypoints1)}"
    )

    # ========================================================
    # LIGHTGLUE
    # ========================================================

    print()
    print(
        "Running LightGlue..."
    )

    with torch.inference_mode():

        matches01 = matcher(
            {
                "image0": feats0,
                "image1": feats1,
            }
        )

    lightglue_matches = (
        matches01["matches"][0]
        .detach()
        .cpu()
        .numpy()
    )

    lightglue_scores = (
        matches01["scores"][0]
        .detach()
        .cpu()
        .numpy()
    )

    total_lightglue_matches = int(
        len(lightglue_matches)
    )

    print(
        f"LightGlue matches: "
        f"{total_lightglue_matches}"
    )

    lightglue_visualization = (
        RESULTS_DIR /
        "lucas_matches.png"
    )

    save_lightglue_visualization(
        image0,
        image1,
        keypoints0,
        keypoints1,
        lightglue_matches,
        lightglue_visualization,
    )

    print(
        f"Saved match visualization: "
        f"{lightglue_visualization}"
    )

    # ========================================================
    # SIFT FALLBACK
    # ========================================================

    print()
    print(
        "========== SIFT FALLBACK =========="
    )

    print(
        "Extracting SIFT features..."
    )

    sift_keypoints0, sift_desc0 = (
        _extract_sift_features(
            source_uint8,
            max_features=5000,
        )
    )

    sift_keypoints1, sift_desc1 = (
        _extract_sift_features(
            reference_uint8,
            max_features=5000,
        )
    )

    print(
        f"SIFT source keypoints: "
        f"{len(sift_keypoints0)}"
    )

    print(
        f"SIFT reference keypoints: "
        f"{len(sift_keypoints1)}"
    )

    print(
        "Running reciprocal SIFT matching..."
    )

    sift_matches = (
        _match_sift_reciprocal(
            sift_desc0,
            sift_desc1,
            ratio_threshold=0.90,
        )
    )

    total_sift_matches = int(
        len(sift_matches)
    )

    print(
        f"SIFT matches: "
        f"{total_sift_matches}"
    )

    # ========================================================
    # GEOMETRY-GUIDED SIFT
    # ========================================================

    guided_source_points, guided_reference_points, guided_scores = (
        _geometry_guided_sift(
            source_image=source_uint8,
            reference_image=reference_uint8,
            scale_ratio=5.78 / 5.00,
            source_center=(1600.0, 4000.0),
            reference_center=(1997.4, 3893.9),
            rotation_degrees=(0.0, -5.0, 5.0),
            ratio_threshold=0.92,
        )
    )

    # ========================================================
    # LIGHTGLUE → POINTS
    # ========================================================

    if total_lightglue_matches > 0:

        learned_source_points = (
            keypoints0[
                lightglue_matches[:, 0]
            ].astype(np.float32)
        )

        learned_reference_points = (
            keypoints1[
                lightglue_matches[:, 1]
            ].astype(np.float32)
        )

    else:

        learned_source_points = np.empty(
            (0, 2),
            dtype=np.float32,
        )

        learned_reference_points = np.empty(
            (0, 2),
            dtype=np.float32,
        )

    # ========================================================
    # SIFT → POINTS
    # ========================================================

    if total_sift_matches > 0:

        sift_source_points = np.asarray(
            [
                sift_keypoints0[
                    m.queryIdx
                ].pt
                for m in sift_matches
            ],
            dtype=np.float32,
        )

        sift_reference_points = np.asarray(
            [
                sift_keypoints1[
                    m.trainIdx
                ].pt
                for m in sift_matches
            ],
            dtype=np.float32,
        )

    else:

        sift_source_points = np.empty(
            (0, 2),
            dtype=np.float32,
        )

        sift_reference_points = np.empty(
            (0, 2),
            dtype=np.float32,
        )

    # ========================================================
    # CANDIDATE FUSION
    # ========================================================

    print()
    print(
        "========== CANDIDATE FUSION =========="
    )

    fused_source_points = np.vstack(
        [
            learned_source_points,
            sift_source_points,
            guided_source_points,
        ]
    ).astype(np.float32)

    fused_reference_points = np.vstack(
        [
            learned_reference_points,
            sift_reference_points,
            guided_reference_points,
        ]
    ).astype(np.float32)

    # --------------------------------------------------------
    # MATCH SCORES
    # --------------------------------------------------------

    learned_scores = np.asarray(
        lightglue_scores,
        dtype=np.float32,
    )

    if total_sift_matches > 0:

        sift_distances = np.asarray(
            [
                float(m.distance)
                for m in sift_matches
            ],
            dtype=np.float32,
        )

        sift_scores = (
            1.0 /
            (1.0 + sift_distances)
        )

        max_sift_score = float(
            np.max(sift_scores)
        )

        if max_sift_score > 1e-8:

            sift_scores = (
                sift_scores /
                max_sift_score
            )

    else:

        sift_scores = np.empty(
            (0,),
            dtype=np.float32,
        )

    fused_scores = np.concatenate(
        [
            learned_scores,
            sift_scores,
            guided_scores,
        ]
    ).astype(np.float32)

    # --------------------------------------------------------
    # DUPLICATE SUPPRESSION
    # --------------------------------------------------------

    if len(fused_source_points) > 0:

        keep = []

        duplicate_radius = 1.5
        radius_sq = (
            duplicate_radius ** 2
        )

        for i in range(
            len(fused_source_points)
        ):

            is_duplicate = False

            for j in keep:

                source_delta = (
                    fused_source_points[i]
                    -
                    fused_source_points[j]
                )

                reference_delta = (
                    fused_reference_points[i]
                    -
                    fused_reference_points[j]
                )

                source_distance_sq = float(
                    np.dot(
                        source_delta,
                        source_delta,
                    )
                )

                reference_distance_sq = float(
                    np.dot(
                        reference_delta,
                        reference_delta,
                    )
                )

                if (
                    source_distance_sq
                    <= radius_sq
                    and
                    reference_distance_sq
                    <= radius_sq
                ):

                    is_duplicate = True
                    break

            if not is_duplicate:
                keep.append(i)

        keep = np.asarray(
            keep,
            dtype=np.int64,
        )

        fused_source_points = (
            fused_source_points[keep]
        )

        fused_reference_points = (
            fused_reference_points[keep]
        )

        fused_scores = (
            fused_scores[keep]
        )

    total_matches = int(
        len(fused_source_points)
    )

    print(
        f"LightGlue candidates: "
        f"{total_lightglue_matches}"
    )

    print(
        f"SIFT candidates:      "
        f"{total_sift_matches}"
    )

    print(
        f"Guided SIFT:          "
        f"{len(guided_source_points)}"
    )

    print(
        f"Fused candidates:     "
        f"{total_matches}"
    )

    print(
        "========================================"
    )

    # ========================================================
    # HYBRID VISUALIZATION
    # ========================================================

    hybrid_visualization = (
        RESULTS_DIR /
        "lucas_hybrid_matches.png"
    )

    save_point_match_visualization(
        image0,
        image1,
        fused_source_points,
        fused_reference_points,
        hybrid_visualization,
        title=(
            f"LUCAS Hybrid Candidates: "
            f"{total_matches}"
        ),
    )

    print(
        f"Saved hybrid match visualization: "
        f"{hybrid_visualization}"
    )

    # ========================================================
    # NOT ENOUGH CANDIDATES
    # ========================================================

    if total_matches < 4:

        result = {
            "status": "REJECTED",
            "reason": (
                "Fewer than 4 fused matches"
            ),
            "quality_reasons": [
                "Fewer than 4 fused matches"
            ],
            "matches": total_matches,
            "lightglue_matches":
                total_lightglue_matches,
            "sift_matches":
                total_sift_matches,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "rmse_px": None,
            "median_error_px": None,
            "max_error_px": None,
            "spatial_coverage": 0.0,
            "spatial_uniformity": 0.0,
            "spatial_distribution": "EMPTY",
            "occupied_grid_cells": 0,
            "total_grid_cells": 16,
            "matcher":
                "SuperPoint + LightGlue + SIFT + Geometry-Guided SIFT",
            "registration_performed": False,
            "registered_image": None,
        }

        save_result_json(
            result,
            RESULTS_DIR /
            "lucas_result.json",
        )

        return result

    # ========================================================
    # GEOMETRIC VERIFICATION
    # ========================================================

    source_points = (
        fused_source_points
    )

    reference_points = (
        fused_reference_points
    )

    print()
    print(
        "========== GEOMETRIC VERIFICATION =========="
    )

    # ========================================================
    # AFFINE RANSAC
    # ========================================================

    print()
    print(
        "Testing affine model..."
    )

    affine_matrix, affine_mask = (
        cv2.estimateAffinePartial2D(
            source_points,
            reference_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=
                ransac_threshold,
            maxIters=5000,
            confidence=0.999,
            refineIters=10,
        )
    )

    affine_result = (
        evaluate_geometric_model(
            source_points,
            reference_points,
            affine_matrix,
            affine_mask,
            "Affine",
        )
    )

    # ========================================================
    # HOMOGRAPHY RANSAC
    # ========================================================

    print()
    print(
        "Testing homography model..."
    )

    homography, homography_mask = (
        cv2.findHomography(
            source_points,
            reference_points,
            cv2.RANSAC,
            ransac_threshold,
            maxIters=5000,
            confidence=0.999,
        )
    )

    homography_result = (
        evaluate_geometric_model(
            source_points,
            reference_points,
            homography,
            homography_mask,
            "Homography",
        )
    )

    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    print()
    print(
        "---------- MODEL COMPARISON ----------"
    )

    print(
        "Affine:"
    )

    print(
        f"  Inliers: "
        f"{affine_result['inlier_count']}"
    )

    print(
        f"  Ratio:   "
        f"{affine_result['inlier_ratio']:.3f}"
    )

    print(
        f"  RMSE:    "
        f"{affine_result['rmse_px']}"
    )

    print(
        "Homography:"
    )

    print(
        f"  Inliers: "
        f"{homography_result['inlier_count']}"
    )

    print(
        f"  Ratio:   "
        f"{homography_result['inlier_ratio']:.3f}"
    )

    print(
        f"  RMSE:    "
        f"{homography_result['rmse_px']}"
    )

    print(
        "---------------------------------------"
    )

    # ========================================================
    # MODEL SELECTION
    # ========================================================

    def geometry_score(result):

        if result["model"] is None:
            return (
                -1,
                -1.0,
                float("inf"),
            )

        rmse = (
            result["rmse_px"]
            if result["rmse_px"] is not None
            else float("inf")
        )

        return (
            result["inlier_count"],
            result["inlier_ratio"],
            -rmse,
        )

    if (
        geometry_score(affine_result)
        >=
        geometry_score(homography_result)
    ):

        selected_geometry = (
            affine_result
        )

    else:

        selected_geometry = (
            homography_result
        )

    selected_model = (
        selected_geometry["model"]
    )

    selected_model_name = (
        selected_geometry["model_name"]
    )

    inlier_mask = (
        selected_geometry["mask"]
    )

    inlier_count = int(
        selected_geometry["inlier_count"]
    )

    inlier_ratio = float(
        selected_geometry["inlier_ratio"]
    )

    print()
    print(
        f"✓ Selected model: "
        f"{selected_model_name}"
    )

    print(
        f"✓ Initial inliers: "
        f"{inlier_count}"
    )

    print(
        f"✓ Initial ratio: "
        f"{inlier_ratio:.3f}"
    )

    print(
        "=========================================="
    )

    # ========================================================
    # GEOMETRIC FAILURE
    # ========================================================

    if (
        selected_model is None
        or inlier_mask is None
        or inlier_count < 4
    ):

        print()
        print(
            "✗ Geometric verification failed."
        )

        result = {
            "status": "REJECTED",
            "reason": (
                "No valid geometric model"
            ),
            "quality_reasons": [
                "No valid geometric model"
            ],
            "matches": total_matches,
            "lightglue_matches":
                total_lightglue_matches,
            "sift_matches":
                total_sift_matches,
            "inliers": inlier_count,
            "inlier_ratio": inlier_ratio,
            "rmse_px":
                selected_geometry[
                    "rmse_px"
                ],
            "median_error_px":
                selected_geometry[
                    "median_error_px"
                ],
            "max_error_px":
                selected_geometry[
                    "max_error_px"
                ],
            "matcher":
                "SuperPoint + LightGlue + SIFT + Geometry-Guided SIFT",
            "registration_performed": False,
            "registered_image": None,
        }

        save_result_json(
            result,
            RESULTS_DIR /
            "lucas_result.json",
        )

        return result

    # ========================================================
    # TRUSTED INITIAL INLIERS
    # ========================================================

    inlier_source_points = (
        source_points[
            inlier_mask
        ]
    )

    inlier_reference_points = (
        reference_points[
            inlier_mask
        ]
    )

    inlier_scores = (
        fused_scores[
            inlier_mask
        ]
    )

    # ========================================================
    # SUBPIXEL REFINEMENT V3
    # ========================================================

    print()
    print(
        "========== SUB-PIXEL REFINEMENT =========="
    )

    subpixel_result = refine_matches(
        source_image=source_uint8,
        reference_image=reference_uint8,
        source_points=inlier_source_points,
        reference_points=inlier_reference_points,
        predicted_reference_points=
            inlier_reference_points,
        patch_radius=15,
        max_refinement=2.0,
        min_response=0.20,
    )

    refined_reference_points = (
        np.asarray(
            subpixel_result[
                "refined_points"
            ],
            dtype=np.float32,
        )
    )

    subpixel_successful = int(
        subpixel_result[
            "successful"
        ]
    )

    subpixel_total = int(
        subpixel_result[
            "total"
        ]
    )

    subpixel_success_rate = float(
        subpixel_result[
            "success_rate"
        ]
    )

    refinement_displacements = (
        np.linalg.norm(
            refined_reference_points.astype(
                np.float64
            )
            -
            inlier_reference_points.astype(
                np.float64
            ),
            axis=1,
        )
    )

    subpixel_mean_displacement = (
        float(
            np.mean(
                refinement_displacements
            )
        )
        if len(
            refinement_displacements
        )
        else 0.0
    )

    subpixel_median_displacement = (
        float(
            np.median(
                refinement_displacements
            )
        )
        if len(
            refinement_displacements
        )
        else 0.0
    )

    print(
        f"Trusted inliers:       "
        f"{inlier_count}"
    )

    print(
        f"Successful refinement: "
        f"{subpixel_successful}/"
        f"{subpixel_total} "
        f"({subpixel_success_rate:.1%})"
    )

    print(
        f"Mean displacement:     "
        f"{subpixel_mean_displacement:.4f} px"
    )

    print(
        f"Median displacement:   "
        f"{subpixel_median_displacement:.4f} px"
    )

    print(
        "============================================"
    )

    # ========================================================
    # FINAL GEOMETRY
    # ========================================================

    final_geometry = (
        selected_model
    )

    final_geometry_name = (
        selected_model_name
    )

    final_geometry_mask = (
        inlier_mask.copy()
    )

    final_reference_points = (
        reference_points.copy()
    )

    refined_geometry = None
    refined_local_mask = None

    # --------------------------------------------------------
    # ONLY RE-FIT WHEN SUBPIXEL ACTUALLY PRODUCED
    # ENOUGH VALID REFINEMENTS
    # --------------------------------------------------------

    MIN_SUBPIXEL_REFINEMENTS = 4

    if (
        subpixel_successful
        >=
        MIN_SUBPIXEL_REFINEMENTS
    ):

        print()
        print(
            "Subpixel refinement sufficient "
            "for geometric refit."
        )

        if selected_model_name == "Affine":

            (
                refined_geometry,
                refined_local_mask,
            ) = cv2.estimateAffinePartial2D(
                inlier_source_points,
                refined_reference_points,
                method=cv2.RANSAC,
                ransacReprojThreshold=
                    ransac_threshold,
                maxIters=5000,
                confidence=0.999,
                refineIters=10,
            )

        else:

            (
                refined_geometry,
                refined_local_mask,
            ) = cv2.findHomography(
                inlier_source_points,
                refined_reference_points,
                cv2.RANSAC,
                ransac_threshold,
                maxIters=5000,
                confidence=0.999,
            )

        if refined_geometry is not None:

            refined_geometry_eval = (
                evaluate_geometric_model(
                    inlier_source_points,
                    refined_reference_points,
                    refined_geometry,
                    refined_local_mask,
                    selected_model_name,
                )
            )

            # ------------------------------------------------
            # Accept refinement only if it does not destroy
            # geometric support.
            # ------------------------------------------------

            original_local_count = (
                len(inlier_source_points)
            )

            refined_local_count = (
                refined_geometry_eval[
                    "inlier_count"
                ]
            )

            if (
                refined_local_count
                >=
                max(
                    4,
                    int(
                        0.5 *
                        original_local_count
                    ),
                )
            ):

                final_geometry = (
                    refined_geometry
                )

                final_reference_points = (
                    reference_points.copy()
                )

                final_reference_points[
                    inlier_mask
                ] = refined_reference_points

                # Map local refined mask back to
                # the complete candidate set.
                final_geometry_mask = (
                    np.zeros(
                        total_matches,
                        dtype=bool,
                    )
                )

                original_indices = (
                    np.flatnonzero(
                        inlier_mask
                    )
                )

                refined_local_mask = (
                    np.asarray(
                        refined_local_mask
                    )
                    .ravel()
                    .astype(bool)
                )

                final_geometry_mask[
                    original_indices[
                        refined_local_mask
                    ]
                ] = True

                print(
                    "✓ Refined geometry accepted."
                )

                print(
                    f"✓ Final local inliers: "
                    f"{refined_local_count}"
                )

            else:

                print(
                    "⚠ Refined geometry rejected."
                )

                print(
                    "⚠ Geometric support degraded."
                )

        else:

            print(
                "⚠ Refined geometry failed."
            )

    else:

        print()
        print(
            "⚠ Subpixel refinement insufficient."
        )

        print(
            f"⚠ Successful refinements: "
            f"{subpixel_successful}"
        )

        print(
            "⚠ Keeping original verified geometry."
        )

    # ========================================================
    # FINAL HOMOGRAPHY FORM
    # ========================================================

    if final_geometry_name == "Affine":

        final_homography = (
            affine_to_homography(
                final_geometry
            )
        )

    else:

        final_homography = (
            final_geometry
        )

    # ========================================================
    # FINAL GEOMETRIC VERIFICATION
    # ========================================================

    print()
    print(
        "========== FINAL VERIFICATION =========="
    )

    final_inlier_count = int(
        np.sum(
            final_geometry_mask
        )
    )

    final_inlier_ratio = (
        final_inlier_count /
        max(total_matches, 1)
    )

    # --------------------------------------------------------
    # Final reprojection using final geometry and the
    # appropriate reference coordinates.
    # --------------------------------------------------------

    try:

        final_projected_points = (
            cv2.perspectiveTransform(
                source_points.reshape(
                    -1,
                    1,
                    2,
                ).astype(np.float32),
                final_homography,
            ).reshape(
                -1,
                2,
            )
        )

        final_errors = np.linalg.norm(
            final_projected_points
            -
            final_reference_points,
            axis=1,
        )

        final_inlier_errors = (
            final_errors[
                final_geometry_mask
            ]
        )

    except cv2.error:

        final_inlier_errors = (
            np.empty(
                (0,),
                dtype=np.float64,
            )
        )

    if len(final_inlier_errors) > 0:

        final_rmse = float(
            np.sqrt(
                np.mean(
                    final_inlier_errors ** 2
                )
            )
        )

        final_median_error = float(
            np.median(
                final_inlier_errors
            )
        )

        final_max_error = float(
            np.max(
                final_inlier_errors
            )
        )

    else:

        final_rmse = None
        final_median_error = None
        final_max_error = None

    # --------------------------------------------------------
    # Original geometry error for comparison
    # --------------------------------------------------------

    try:

        original_homography = (
            affine_to_homography(
                selected_model
            )
            if selected_model_name == "Affine"
            else selected_model
        )

        original_projected = (
            cv2.perspectiveTransform(
                source_points.reshape(
                    -1,
                    1,
                    2,
                ).astype(np.float32),
                original_homography,
            ).reshape(
                -1,
                2,
            )
        )

        original_errors = np.linalg.norm(
            original_projected
            -
            reference_points,
            axis=1,
        )

        original_inlier_errors = (
            original_errors[
                inlier_mask
            ]
        )

        if len(
            original_inlier_errors
        ) > 0:

            rmse_before_subpixel = float(
                np.sqrt(
                    np.mean(
                        original_inlier_errors
                        ** 2
                    )
                )
            )

        else:

            rmse_before_subpixel = None

    except cv2.error:

        rmse_before_subpixel = None

    if (
        rmse_before_subpixel is not None
        and
        final_rmse is not None
    ):

        subpixel_rmse_improvement = float(
            (
                rmse_before_subpixel
                -
                final_rmse
            )
            /
            max(
                rmse_before_subpixel,
                1e-12,
            )
            *
            100.0
        )

    else:

        subpixel_rmse_improvement = None

    print(
        f"Final model:       "
        f"{final_geometry_name}"
    )

    print(
        f"Final inliers:     "
        f"{final_inlier_count}"
    )

    print(
        f"Final ratio:       "
        f"{final_inlier_ratio:.3f}"
    )

    if final_rmse is not None:

        print(
            f"Final RMSE:        "
            f"{final_rmse:.3f} px"
        )

        print(
            f"Final median:      "
            f"{final_median_error:.3f} px"
        )

        print(
            f"Final max:         "
            f"{final_max_error:.3f} px"
        )

    else:

        print(
            "Final RMSE:        N/A"
        )

    print(
        "=========================================="
    )

    # ========================================================
    # SPATIAL MATCH SELECTION
    # ========================================================

    source_height, source_width = (
        image0.shape[-2:]
    )

    final_inlier_source_points = (
        source_points[
            final_geometry_mask
        ]
    )

    final_inlier_reference_points = (
        final_reference_points[
            final_geometry_mask
        ]
    )

    final_inlier_scores = (
        fused_scores[
            final_geometry_mask
        ]
    )

    (
        selected_source_points,
        selected_reference_points,
        selected_indices,
    ) = select_uniform_matches(
        source_points=
            final_inlier_source_points,
        reference_points=
            final_inlier_reference_points,
        match_scores=
            final_inlier_scores,
        image_width=
            source_width,
        image_height=
            source_height,
        grid_rows=4,
        grid_cols=4,
        max_per_cell=8,
    )

    selected_match_count = int(
        len(selected_indices)
    )

    print()
    print(
        "===== SPATIAL MATCH SELECTION ====="
    )

    print(
        f"Final inliers:      "
        f"{final_inlier_count}"
    )

    print(
        f"Uniform matches:    "
        f"{selected_match_count}"
    )

    print(
        "===================================="
    )

    # ========================================================
    # SPATIAL ANALYSIS
    # ========================================================

    spatial = (
        analyze_spatial_distribution(
            points=
                selected_source_points,
            image_width=
                source_width,
            image_height=
                source_height,
            grid_rows=4,
            grid_cols=4,
        )
    )

    spatial_coverage = float(
        spatial["coverage"]
    )

    spatial_uniformity = float(
        spatial["uniformity"]
    )

    spatial_distribution = (
        spatial["distribution"]
    )

    occupied_grid_cells = int(
        spatial["occupied_cells"]
    )

    total_grid_cells = int(
        spatial["total_cells"]
    )

    print()
    print(
        "========== SPATIAL QUALITY =========="
    )

    print(
        f"Occupied cells: "
        f"{occupied_grid_cells}/"
        f"{total_grid_cells}"
    )

    print(
        f"Spatial coverage: "
        f"{spatial_coverage:.3f}"
    )

    print(
        f"Spatial uniformity: "
        f"{spatial_uniformity:.3f}"
    )

    print(
        f"Distribution: "
        f"{spatial_distribution}"
    )

    print(
        "======================================"
    )

    # ========================================================
    # INLIER VISUALIZATION
    # ========================================================

    inlier_visualization = (
        RESULTS_DIR /
        "lucas_inliers.png"
    )

    save_point_match_visualization(
        image0,
        image1,
        final_inlier_source_points,
        final_inlier_reference_points,
        inlier_visualization,
        title=(
            f"LUCAS Final Inliers: "
            f"{final_inlier_count}"
        ),
    )

    print(
        f"Saved inlier visualization: "
        f"{inlier_visualization}"
    )

    # ========================================================
    # QUALITY GATE
    # ========================================================

    quality = (
        evaluate_registration_quality(
            total_matches=
                total_matches,
            inlier_count=
                final_inlier_count,
            spatial_coverage=
                spatial_coverage,
            min_spatial_coverage=
                0.25,
        )
    )

    # ========================================================
    # RESULT
    # ========================================================

    result = {

        "status": (
            "ACCEPTED"
            if quality["success"]
            else "REJECTED"
        ),

        "reason":
            quality["reason"],

        "quality_reasons":
            quality["reasons"],

        "matches":
            total_matches,

        "lightglue_matches":
            total_lightglue_matches,

        "sift_matches":
            total_sift_matches,

        "inliers":
            final_inlier_count,

        "uniform_matches":
            selected_match_count,

        "inlier_ratio":
            float(final_inlier_ratio),

        "geometry": {
            "selected_model":
                final_geometry_name,

            "initial_model":
                selected_model_name,

            "initial_inliers":
                inlier_count,

            "initial_inlier_ratio":
                inlier_ratio,

            "final_inliers":
                final_inlier_count,

            "final_inlier_ratio":
                final_inlier_ratio,
        },

        "subpixel": {

            "enabled":
                True,

            "method":
                "local phase correlation V3",

            "successful":
                subpixel_successful,

            "total":
                subpixel_total,

            "success_rate":
                subpixel_success_rate,

            "patch_radius":
                15,

            "max_refinement_px":
                2.0,

            "min_response":
                0.20,

            "mean_displacement_px":
                subpixel_mean_displacement,

            "median_displacement_px":
                subpixel_median_displacement,

            "rmse_before_px":
                rmse_before_subpixel,

            "rmse_after_px":
                final_rmse,

            "rmse_improvement_percent":
                subpixel_rmse_improvement,
        },

        "rmse_px":
            final_rmse,

        "median_error_px":
            final_median_error,

        "max_error_px":
            final_max_error,

        "spatial_coverage":
            spatial_coverage,

        "spatial_uniformity":
            spatial_uniformity,

        "spatial_distribution":
            spatial_distribution,

        "occupied_grid_cells":
            occupied_grid_cells,

        "total_grid_cells":
            total_grid_cells,

        "source_keypoints":
            int(len(keypoints0)),

        "reference_keypoints":
            int(len(keypoints1)),

        "matcher":
            "SuperPoint + LightGlue + SIFT + Geometry-Guided SIFT",

        "ransac_threshold_px":
            float(ransac_threshold),

        "homography":
            (
                original_homography.tolist()
                if original_homography is not None
                else None
            ),

        "final_homography":
            (
                final_homography.tolist()
                if final_homography is not None
                else None
            ),

        "source_image":
            str(source_path),

        "reference_image":
            str(reference_path),

        "match_visualization":
            str(lightglue_visualization),

        "hybrid_match_visualization":
            str(hybrid_visualization),

        "inlier_visualization":
            str(inlier_visualization),

        "registration_performed":
            False,

        "registered_image":
            None,
    }

    # ========================================================
    # CONSOLE RESULT
    # ========================================================

    print()
    print(
        "========== LUCAS RESULT =========="
    )

    print(
        f"Matches:         "
        f"{total_matches}"
    )

    print(
        f"  LightGlue:     "
        f"{total_lightglue_matches}"
    )

    print(
        f"  SIFT:          "
        f"{total_sift_matches}"
    )

    print(
        f"Inliers:         "
        f"{final_inlier_count}"
    )

    print(
        f"Inlier ratio:    "
        f"{final_inlier_ratio:.3f}"
    )

    if final_rmse is not None:

        print(
            f"RMSE:            "
            f"{final_rmse:.3f} px"
        )

        print(
            f"Median error:    "
            f"{final_median_error:.3f} px"
        )

        print(
            f"Max error:       "
            f"{final_max_error:.3f} px"
        )

    else:

        print(
            "RMSE:            N/A"
        )

    print()
    print(
        f"Geometry model:   "
        f"{final_geometry_name}"
    )

    print()
    print(
        "Final homography:"
    )

    if final_homography is not None:
        print(
            final_homography
        )
    else:
        print(
            "None"
        )

    print(
        "==================================="
    )

    # ========================================================
    # QUALITY OUTPUT
    # ========================================================

    print()
    print(
        "========== LUCAS QUALITY =========="
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
        f"Reason: "
        f"{quality['reason']}"
    )

    print()
    print(
        "Quality criteria:"
    )

    if quality["reasons"]:

        for reason in (
            quality["reasons"]
        ):

            print(
                f"  • {reason}"
            )

    else:

        print(
            "  • All quality criteria passed"
        )

    print()
    print(
        f"Inliers: "
        f"{final_inlier_count}"
    )

    print(
        f"Inlier ratio: "
        f"{final_inlier_ratio:.3f}"
    )

    print(
        f"Spatial coverage: "
        f"{spatial_coverage:.3f}"
    )

    print(
        f"Spatial uniformity: "
        f"{spatial_uniformity:.3f}"
    )

    print(
        f"Distribution: "
        f"{spatial_distribution}"
    )

    print(
        "==================================="
    )

    # ========================================================
    # REGISTRATION
    # ========================================================

    if quality["success"]:

        print()
        print(
            "========== LUCAS REGISTRATION =========="
        )

        print(
            "✓ Quality gate passed."
        )

        print(
            "✓ Applying verified geometry..."
        )

        registered_image_path = (
            RESULTS_DIR /
            "lucas_registered.png"
        )

        registered_image = (
            homography_registration(
                moving=
                    source_uint8,
                fixed=
                    reference_uint8,
                homography=
                    final_homography,
            )
        )

        registration_saved = (
            cv2.imwrite(
                str(
                    registered_image_path
                ),
                registered_image,
            )
        )

        if not registration_saved:

            raise IOError(
                "Failed to save registered "
                f"image: "
                f"{registered_image_path}"
            )

        result[
            "registration_performed"
        ] = True

        result[
            "registered_image"
        ] = str(
            registered_image_path
        )

        print(
            f"Registered image saved: "
            f"{registered_image_path}"
        )

        print(
            "Registration status: SUCCESS"
        )

        print(
            "========================================"
        )

    else:

        print()
        print(
            "========== LUCAS REGISTRATION =========="
        )

        print(
            "✗ Quality gate failed."
        )

        print(
            "✗ Registration skipped."
        )

        print(
            "✗ No registered image will be produced."
        )

        print(
            "========================================"
        )

    # ========================================================
    # SAFETY
    # ========================================================

    if quality["success"]:

        print()
        print(
            "✓ Geometric verification passed."
        )

        print(
            "✓ Spatial quality passed."
        )

        print(
            "✓ Registration completed."
        )

    else:

        print()
        print(
            "⚠ Registration rejected."
        )

        print(
            "⚠ Insufficient correspondence quality."
        )

        print(
            "⚠ No trustworthy registered image "
            "should be produced."
        )

    # ========================================================
    # SAVE RESULT
    # ========================================================

    result_json_path = (
        RESULTS_DIR /
        "lucas_result.json"
    )

    save_result_json(
        result,
        result_json_path,
    )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    print()
    print(
        "========== FINAL LUCAS STATUS =========="
    )

    print(
        f"Status: "
        f"{result['status']}"
    )

    print(
        f"Reason: "
        f"{result['reason']}"
    )

    if result[
        "registration_performed"
    ]:

        print(
            "Registered image: "
            f"{result['registered_image']}"
        )

    else:

        print(
            "Registered image: "
            "NOT PRODUCED"
        )

    print(
        "========================================"
    )

    return result


# ============================================================
# STANDALONE EXECUTION
# ============================================================

if __name__ == "__main__":

    source = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "tmc2_apollo16_focused.png"
    )

    reference = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "lro_apollo16_overlap.png"
    )

    run_lucas(
        source_path=source,
        reference_path=reference,
        max_num_keypoints=4096,
        ransac_threshold=5.0,
        show_visualization=False,
    )