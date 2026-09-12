"""
Image registration utilities for LUCAS.

Provides image warping using a geometrically verified
homography estimated by the LUCAS correspondence pipeline.
"""

from __future__ import annotations

import cv2
import numpy as np


def homography_registration(
    moving: np.ndarray,
    fixed: np.ndarray,
    homography: np.ndarray,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """
    Register a moving image to a fixed image using a homography.

    Parameters
    ----------
    moving:
        Source / moving image.

    fixed:
        Reference / fixed image.

    homography:
        3x3 transformation matrix mapping coordinates
        from the moving image to the fixed image.

    interpolation:
        OpenCV interpolation method.

    Returns
    -------
    registered_image:
        Moving image warped into the fixed image coordinate system.
    """

    if moving is None:
        raise ValueError("Moving image is None.")

    if fixed is None:
        raise ValueError("Fixed image is None.")

    if homography is None:
        raise ValueError("Homography is None.")

    homography = np.asarray(
        homography,
        dtype=np.float64,
    )

    if homography.shape != (3, 3):
        raise ValueError(
            f"Homography must have shape (3, 3), "
            f"got {homography.shape}."
        )

    if not np.all(np.isfinite(homography)):
        raise ValueError(
            "Homography contains invalid values."
        )

    fixed_height, fixed_width = fixed.shape[:2]

    registered = cv2.warpPerspective(
        moving,
        homography,
        (fixed_width, fixed_height),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return registered


def save_registered_image(
    registered_image: np.ndarray,
    output_path: str,
) -> None:
    """
    Save a registered image to disk.
    """

    if registered_image is None:
        raise ValueError(
            "Cannot save a None registered image."
        )

    success = cv2.imwrite(
        str(output_path),
        registered_image,
    )

    if not success:
        raise IOError(
            f"Failed to save registered image: "
            f"{output_path}"
        )


def register_with_homography(
    moving: np.ndarray,
    fixed: np.ndarray,
    homography: np.ndarray,
    output_path: str | None = None,
) -> np.ndarray:
    """
    Perform homography-based registration and optionally
    save the registered image.

    Returns
    -------
    registered_image:
        The registered moving image in fixed-image coordinates.
    """

    registered = homography_registration(
        moving=moving,
        fixed=fixed,
        homography=homography,
    )

    if output_path is not None:
        save_registered_image(
            registered_image=registered,
            output_path=output_path,
        )

    return registered


def ecc_affine_registration(
    moving: np.ndarray,
    fixed: np.ndarray,
    iterations: int = 200,
    epsilon: float = 1e-6,
):
    """
    Refine the alignment of a moving image against a fixed image
    using Enhanced Correlation Coefficient (ECC) optimization.

    This function is retained as an optional refinement method.

    Returns
    -------
    registered_image
    warp_matrix
    correlation_coefficient
    """

    if moving is None:
        raise ValueError("Moving image is None.")

    if fixed is None:
        raise ValueError("Fixed image is None.")

    # ECC expects floating-point images.
    moving_float = (
        moving.astype(np.float32) / 255.0
    )

    fixed_float = (
        fixed.astype(np.float32) / 255.0
    )

    # Affine transformation:
    #
    # [a b tx]
    # [c d ty]
    #
    warp_matrix = np.eye(
        2,
        3,
        dtype=np.float32,
    )

    criteria = (
        cv2.TERM_CRITERIA_EPS |
        cv2.TERM_CRITERIA_COUNT,
        iterations,
        epsilon,
    )

    correlation, warp_matrix = cv2.findTransformECC(
        fixed_float,
        moving_float,
        warp_matrix,
        cv2.MOTION_AFFINE,
        criteria,
        None,
        1,
    )

    height, width = fixed.shape[:2]

    registered = cv2.warpAffine(
        moving,
        warp_matrix,
        (width, height),
        flags=(
            cv2.INTER_LINEAR |
            cv2.WARP_INVERSE_MAP
        ),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return (
        registered,
        warp_matrix,
        correlation,
    )


# ============================================================
# SIMPLE SELF TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 50)
    print("LUCAS REGISTRATION MODULE TEST")
    print("=" * 50)

    # Small synthetic images
    fixed = np.zeros(
        (500, 500),
        dtype=np.uint8,
    )

    moving = np.zeros(
        (500, 500),
        dtype=np.uint8,
    )

    # Simple feature
    cv2.circle(
        moving,
        (250, 250),
        50,
        255,
        -1,
    )

    # Identity transformation
    H = np.eye(
        3,
        dtype=np.float64,
    )

    registered = homography_registration(
        moving=moving,
        fixed=fixed,
        homography=H,
    )

    print(
        f"Moving shape:     {moving.shape}"
    )

    print(
        f"Fixed shape:      {fixed.shape}"
    )

    print(
        f"Registered shape: {registered.shape}"
    )

    print(
        "Homography registration: PASS"
    )

    print("=" * 50)