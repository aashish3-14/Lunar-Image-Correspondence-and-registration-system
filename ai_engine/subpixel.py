from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


# ============================================================
# IMAGE UTILITIES
# ============================================================

def _as_gray_float32(image: np.ndarray) -> np.ndarray:
    """
    Convert an image to grayscale float32.
    """

    if image is None:
        raise ValueError("Image cannot be None.")

    image = np.asarray(image)

    if image.ndim == 2:
        gray = image

    elif image.ndim == 3:

        if image.shape[2] == 1:
            gray = image[..., 0]

        else:
            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY,
            )

    else:
        raise ValueError(
            "Image must be 2-D or 3-D."
        )

    gray = np.asarray(
        gray,
        dtype=np.float32,
    )

    return np.nan_to_num(
        gray,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


# ============================================================
# GEOMETRIC TRANSFORMATION
# ============================================================

def apply_transform(
    points: np.ndarray,
    transform: np.ndarray,
) -> np.ndarray:
    """
    Apply a 2x3 affine transformation or
    3x3 homography to Nx2 points.
    """

    points = np.asarray(
        points,
        dtype=np.float32,
    )

    transform = np.asarray(
        transform,
        dtype=np.float64,
    )

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(
            "points must have shape (N, 2)."
        )

    ones = np.ones(
        (len(points), 1),
        dtype=np.float32,
    )

    homogeneous = np.hstack(
        [
            points,
            ones,
        ]
    )

    # --------------------------------------------------------
    # Affine
    # --------------------------------------------------------

    if transform.shape == (2, 3):

        return (
            homogeneous @ transform.T
        ).astype(np.float32)

    # --------------------------------------------------------
    # Homography
    # --------------------------------------------------------

    if transform.shape == (3, 3):

        projected = (
            homogeneous @ transform.T
        )

        w = projected[:, 2:3]

        safe_w = np.where(
            np.abs(w) < 1e-10,
            1e-10,
            w,
        )

        return (
            projected[:, :2] / safe_w
        ).astype(np.float32)

    raise ValueError(
        "transform must have shape "
        "(2, 3) or (3, 3)."
    )


# ============================================================
# IMAGE BOUNDARY CHECK
# ============================================================

def _inside(
    point: np.ndarray,
    width: int,
    height: int,
    radius: int,
) -> bool:

    x = float(point[0])
    y = float(point[1])

    return (
        radius + 1 <= x < width - radius - 1
        and
        radius + 1 <= y < height - radius - 1
    )


# ============================================================
# PATCH EXTRACTION
# ============================================================

def _patch(
    image: np.ndarray,
    point: np.ndarray,
    radius: int,
) -> Optional[np.ndarray]:
    """
    Extract an integer-centered local patch.

    The patch is centered around the nearest pixel.
    """

    x = int(
        round(float(point[0]))
    )

    y = int(
        round(float(point[1]))
    )

    r = int(radius)

    if (
        x - r < 0
        or
        y - r < 0
        or
        x + r + 1 > image.shape[1]
        or
        y + r + 1 > image.shape[0]
    ):
        return None

    return image[
        y - r:y + r + 1,
        x - r:x + r + 1,
    ]


# ============================================================
# LOCAL NORMALIZED CROSS CORRELATION
# ============================================================

def _local_ncc(
    source_patch: np.ndarray,
    reference_patch: np.ndarray,
) -> float:

    a = source_patch.astype(
        np.float32
    )

    b = reference_patch.astype(
        np.float32
    )

    a = a - float(
        np.mean(a)
    )

    b = b - float(
        np.mean(b)
    )

    denom = float(
        np.linalg.norm(a)
        *
        np.linalg.norm(b)
    )

    if denom < 1e-8:
        return -1.0

    return float(
        np.sum(a * b) / denom
    )


# ============================================================
# PATCH NORMALIZATION
# ============================================================

def _normalize_patch(
    patch: np.ndarray,
) -> np.ndarray:

    patch = patch.astype(
        np.float32
    )

    patch = (
        patch
        -
        float(np.mean(patch))
    )

    std = float(
        np.std(patch)
    )

    if std < 1e-6:
        return np.zeros_like(
            patch
        )

    return patch / std


# ============================================================
# SUB-PIXEL REFINEMENT
# ============================================================

def refine_point(
    source_image: np.ndarray,
    reference_image: np.ndarray,
    source_point: np.ndarray,
    predicted_reference_point: np.ndarray,
    patch_radius: int = 15,
    max_refinement: float = 2.0,
    min_response: float = 0.20,
) -> dict:
    """
    Refine one trusted correspondence using
    local phase correlation.

    The original correspondence is preserved whenever
    the refinement is considered unreliable.
    """

    source = _as_gray_float32(
        source_image
    )

    reference = _as_gray_float32(
        reference_image
    )

    source_point = np.asarray(
        source_point,
        dtype=np.float32,
    ).reshape(2)

    predicted = np.asarray(
        predicted_reference_point,
        dtype=np.float32,
    ).reshape(2)

    px = float(predicted[0])
    py = float(predicted[1])

    # --------------------------------------------------------
    # Boundary check
    # --------------------------------------------------------

    source_patch = _patch(
        source,
        source_point,
        patch_radius,
    )

    reference_patch = _patch(
        reference,
        predicted,
        patch_radius,
    )

    if (
        source_patch is None
        or
        reference_patch is None
    ):

        return {
            "success": False,
            "reason": "Patch outside image",
            "point": [px, py],
            "correction": [0.0, 0.0],
            "response": 0.0,
        }

    # --------------------------------------------------------
    # Normalize patches
    # --------------------------------------------------------

    source_patch = _normalize_patch(
        source_patch
    )

    reference_patch = _normalize_patch(
        reference_patch
    )

    if (
        float(np.std(source_patch)) < 1e-6
        or
        float(np.std(reference_patch)) < 1e-6
    ):

        return {
            "success": False,
            "reason": "Low-texture patch",
            "point": [px, py],
            "correction": [0.0, 0.0],
            "response": 0.0,
        }

    # --------------------------------------------------------
    # Phase correlation
    # --------------------------------------------------------

    try:

        shift, response = cv2.phaseCorrelate(
            source_patch,
            reference_patch,
        )

    except cv2.error:

        return {
            "success": False,
            "reason": "Phase-correlation failed",
            "point": [px, py],
            "correction": [0.0, 0.0],
            "response": 0.0,
        }

    dx = float(
        shift[0]
    )

    dy = float(
        shift[1]
    )

    response = float(
        response
    )

    magnitude = float(
        np.hypot(
            dx,
            dy,
        )
    )

    # --------------------------------------------------------
    # Numerical validation
    # --------------------------------------------------------

    if (
        not np.isfinite(dx)
        or
        not np.isfinite(dy)
        or
        not np.isfinite(response)
    ):

        return {
            "success": False,
            "reason": (
                "Non-finite "
                "phase-correlation result"
            ),
            "point": [px, py],
            "correction": [0.0, 0.0],
            "response": 0.0,
        }

    # --------------------------------------------------------
    # Response threshold
    # --------------------------------------------------------

    if response < float(
        min_response
    ):

        return {
            "success": False,
            "reason": (
                f"Low phase response "
                f"({response:.4f})"
            ),
            "point": [px, py],
            "correction": [0.0, 0.0],
            "response": response,
        }

    # --------------------------------------------------------
    # Maximum correction safety gate
    # --------------------------------------------------------

    if magnitude > float(
        max_refinement
    ):

        return {
            "success": False,
            "reason": (
                f"Correction too large "
                f"({magnitude:.4f}px > "
                f"{float(max_refinement):.4f}px)"
            ),
            "point": [px, py],
            "correction": [0.0, 0.0],
            "response": response,
        }

    # --------------------------------------------------------
    # Refined sub-pixel coordinate
    # --------------------------------------------------------

    refined_x = px + dx
    refined_y = py + dy

    return {
        "success": True,
        "reason": "Accepted",
        "point": [
            refined_x,
            refined_y,
        ],
        "correction": [
            dx,
            dy,
        ],
        "response": response,
    }


# ============================================================
# BATCH SUB-PIXEL REFINEMENT
# ============================================================

def refine_matches(
    source_image: np.ndarray,
    reference_image: np.ndarray,
    source_points: np.ndarray,
    reference_points: np.ndarray,
    predicted_reference_points: Optional[np.ndarray] = None,
    patch_radius: int = 15,
    max_refinement: float = 2.0,
    min_response: float = 0.20,
) -> dict:
    """
    Refine a collection of trusted correspondences.

    Parameters
    ----------
    source_image:
        Moving/source image.

    reference_image:
        Fixed/reference image.

    source_points:
        Source coordinates, shape (N, 2).

    reference_points:
        Original reference coordinates, shape (N, 2).

    predicted_reference_points:
        Geometrically predicted reference coordinates.

        If None, the original reference coordinates are used.

    patch_radius:
        Radius of the local phase-correlation patch.

    max_refinement:
        Maximum accepted correction in pixels.

    min_response:
        Minimum phase-correlation response.

    Returns
    -------
    dict
        refined_points
        success
        responses
        details
        successful
        total
        success_rate
    """

    # --------------------------------------------------------
    # Convert arrays
    # --------------------------------------------------------

    source_points = np.asarray(
        source_points,
        dtype=np.float32,
    )

    reference_points = np.asarray(
        reference_points,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Validate source points
    # --------------------------------------------------------

    if (
        source_points.ndim != 2
        or
        source_points.shape[1] != 2
    ):

        raise ValueError(
            "source_points must have "
            "shape (N, 2)."
        )

    # --------------------------------------------------------
    # Validate reference points
    # --------------------------------------------------------

    if (
        reference_points.shape
        !=
        source_points.shape
    ):

        raise ValueError(
            "reference_points must have "
            "the same shape as "
            "source_points."
        )

    # --------------------------------------------------------
    # Predicted locations
    # --------------------------------------------------------

    if predicted_reference_points is None:

        predicted = (
            reference_points.copy()
        )

    else:

        predicted = np.asarray(
            predicted_reference_points,
            dtype=np.float32,
        )

        if (
            predicted.shape
            !=
            reference_points.shape
        ):

            raise ValueError(
                "predicted_reference_points "
                "must have the same shape "
                "as source_points and "
                "reference_points."
            )

    # --------------------------------------------------------
    # Output containers
    # --------------------------------------------------------

    refined_points = (
        reference_points.copy()
    )

    success = np.zeros(
        len(source_points),
        dtype=bool,
    )

    responses = np.zeros(
        len(source_points),
        dtype=np.float32,
    )

    details = []

    # --------------------------------------------------------
    # Refine every correspondence
    # --------------------------------------------------------

    for i in range(
        len(source_points)
    ):

        result = refine_point(
            source_image=source_image,

            reference_image=reference_image,

            source_point=source_points[i],

            predicted_reference_point=predicted[i],

            patch_radius=patch_radius,

            max_refinement=max_refinement,

            min_response=min_response,
        )

        ok = bool(
            result.get(
                "success",
                False,
            )
        )

        success[i] = ok

        responses[i] = float(
            result.get(
                "response",
                0.0,
            )
        )

        # ----------------------------------------------------
        # Accept successful refinement
        # ----------------------------------------------------

        if ok:

            refined_points[i] = np.asarray(
                result["point"],
                dtype=np.float32,
            )

        # ----------------------------------------------------
        # Safety fallback
        # ----------------------------------------------------

        else:

            refined_points[i] = (
                reference_points[i]
            )

        details.append(
            result
        )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    successful = int(
        np.count_nonzero(
            success
        )
    )

    total = int(
        len(source_points)
    )

    if total > 0:

        success_rate = float(
            successful / total
        )

    else:

        success_rate = 0.0

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    return {
        "refined_points": refined_points,

        "success": success,

        "responses": responses,

        "details": details,

        "successful": successful,

        "total": total,

        "success_rate": success_rate,
    }


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print(
        "LUCAS SUB-PIXEL REFINEMENT V3"
    )
    print("=" * 60)

    print()
    print(
        "Running synthetic sub-pixel test..."
    )

    # --------------------------------------------------------
    # Generate textured image
    # --------------------------------------------------------

    rng = np.random.default_rng(
        42
    )

    base = rng.normal(
        100.0,
        35.0,
        (256, 256),
    ).astype(
        np.float32
    )

    base = cv2.GaussianBlur(
        base,
        (0, 0),
        1.5,
    )

    # --------------------------------------------------------
    # Known fractional translation
    # --------------------------------------------------------

    true_shift = np.array(
        [
            0.35,
            -0.42,
        ],
        dtype=np.float32,
    )

    matrix = np.array(
        [
            [
                1.0,
                0.0,
                true_shift[0],
            ],
            [
                0.0,
                1.0,
                true_shift[1],
            ],
        ],
        dtype=np.float32,
    )

    moved = cv2.warpAffine(
        base,
        matrix,
        (256, 256),
        flags=cv2.INTER_LINEAR,
    )

    # --------------------------------------------------------
    # Test point
    # --------------------------------------------------------

    point = np.array(
        [
            128.0,
            128.0,
        ],
        dtype=np.float32,
    )

    expected = (
        point
        +
        true_shift
    )

    # --------------------------------------------------------
    # Run refinement
    # --------------------------------------------------------

    result = refine_matches(
        source_image=base,

        reference_image=moved,

        source_points=point.reshape(
            1,
            2,
        ),

        reference_points=expected.reshape(
            1,
            2,
        ),

        predicted_reference_points=(
            expected.reshape(
                1,
                2,
            )
        ),

        patch_radius=15,

        max_refinement=2.0,

        min_response=0.20,
    )

    refined = result[
        "refined_points"
    ][0]

    success = bool(
        result["success"][0]
    )

    response = float(
        result["responses"][0]
    )

    error = float(
        np.linalg.norm(
            refined
            -
            expected
        )
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()
    print(
        f"True reference point : "
        f"{expected}"
    )

    print(
        f"Refined point        : "
        f"{refined}"
    )

    print(
        f"Success              : "
        f"{success}"
    )

    print(
        f"Phase response       : "
        f"{response:.6f}"
    )

    print(
        f"Absolute error       : "
        f"{error:.6f} px"
    )

    print(
        f"Successful points    : "
        f"{result['successful']}/"
        f"{result['total']}"
    )

    print(
        f"Success rate         : "
        f"{result['success_rate']:.2%}"
    )

    print()

    if success:
        print(
            "STATUS: SUB-PIXEL TEST PASSED"
        )
    else:
        print(
            "STATUS: SUB-PIXEL TEST FAILED"
        )

    print("=" * 60)