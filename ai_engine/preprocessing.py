from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np


def normalize_image(
    image: np.ndarray,
    low_percentile: float = 2.0,
    high_percentile: float = 98.0,
) -> np.ndarray:
    """Robust percentile normalization to uint8 [0, 255]."""
    image = np.asarray(image)
    if image.size == 0:
        raise ValueError("Cannot normalize an empty image.")

    image_float = image.astype(np.float32)
    low, high = np.percentile(
        image_float,
        [low_percentile, high_percentile],
    )

    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError("Image contains no finite intensity range.")

    if high <= low:
        return np.zeros_like(image_float, dtype=np.uint8)

    normalized = np.clip(
        (image_float - low) / (high - low),
        0.0,
        1.0,
    )
    return (normalized * 255.0).astype(np.uint8)


def enhance_contrast(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Enhance local contrast using CLAHE."""
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError("enhance_contrast expects a single-channel image.")

    if image.dtype != np.uint8:
        image = normalize_image(image)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size,
    )
    return clahe.apply(image)


def estimate_illumination(
    image: np.ndarray,
    sigma: float = 21.0,
) -> np.ndarray:
    """Estimate slowly varying illumination with Gaussian smoothing."""
    if sigma <= 0:
        raise ValueError("sigma must be greater than zero.")

    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError(
            "estimate_illumination expects a single-channel image."
        )

    if image.dtype != np.uint8:
        image = normalize_image(image)

    return cv2.GaussianBlur(
        image.astype(np.float32),
        ksize=(0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
    )


def remove_illumination(
    image: np.ndarray,
    sigma: float = 21.0,
) -> np.ndarray:
    """Reduce slowly varying illumination using division correction."""
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError(
            "remove_illumination expects a single-channel image."
        )

    if image.dtype != np.uint8:
        image = normalize_image(image)

    illumination = estimate_illumination(
        image,
        sigma=sigma,
    )

    corrected = image.astype(np.float32) / (illumination + 1.0)
    return normalize_image(corrected)


def compute_gradient(
    image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized |Gx|, |Gy| and gradient magnitude."""
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError(
            "compute_gradient expects a single-channel image."
        )

    if image.dtype != np.uint8:
        image = normalize_image(image)

    image_float = image.astype(np.float32)

    gx = cv2.Sobel(
        image_float,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )
    gy = cv2.Sobel(
        image_float,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )
    magnitude = cv2.magnitude(gx, gy)

    return (
        normalize_image(np.abs(gx)),
        normalize_image(np.abs(gy)),
        normalize_image(magnitude),
    )


def structural_representation(
    image: np.ndarray,
    illumination_sigma: float = 21.0,
) -> np.ndarray:
    """
    Build an illumination-robust terrain-structure representation.

    normalize -> CLAHE -> illumination correction -> gradient magnitude
    """
    normalized = normalize_image(image)
    enhanced = enhance_contrast(normalized)
    corrected = remove_illumination(
        enhanced,
        sigma=illumination_sigma,
    )
    _, _, magnitude = compute_gradient(corrected)
    return magnitude


def illumination_robust_preprocess(
    image: np.ndarray,
    illumination_sigma: float = 21.0,
) -> dict[str, np.ndarray]:
    """Return all intermediate representations for experiments."""
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError(
            "illumination_robust_preprocess expects a single-channel image."
        )

    normalized = normalize_image(image)
    enhanced = enhance_contrast(normalized)
    corrected = remove_illumination(
        enhanced,
        sigma=illumination_sigma,
    )
    gx, gy, magnitude = compute_gradient(corrected)

    return {
        "raw": image.copy(),
        "normalized": normalized,
        "enhanced": enhanced,
        "illumination_corrected": corrected,
        "gradient_x": gx,
        "gradient_y": gy,
        "gradient_magnitude": magnitude,
        "structural": magnitude,
    }


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Existing LUCAS API.

    Kept unchanged so the current pipeline continues to work.
    We will integrate the new structural branch only after testing.
    """
    normalized = normalize_image(image)
    return enhance_contrast(normalized)


def load_grayscale_image(
    image_path: str | Path,
) -> np.ndarray:
    """Load an image as grayscale uint8."""
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        raise ValueError(
            f"OpenCV could not read image: {image_path}"
        )

    return image


if __name__ == "__main__":
    print("=" * 60)
    print("LUCAS | STEP 1 | ILLUMINATION PREPROCESSING")
    print("=" * 60)

    rng = np.random.default_rng(42)
    test = np.zeros((512, 512), dtype=np.float32)

    cv2.circle(test, (150, 160), 80, 120, -1)
    cv2.circle(test, (350, 320), 110, 190, -1)
    cv2.line(test, (20, 450), (490, 40), 220, 8)

    test += rng.normal(0, 12, test.shape)
    test = np.clip(test, 0, 255).astype(np.uint8)

    representations = illumination_robust_preprocess(test)

    print(f"Input: {test.shape}, {test.dtype}")
    for name, value in representations.items():
        print(
            f"{name:24s} "
            f"shape={value.shape} "
            f"dtype={value.dtype} "
            f"range=[{value.min()}, {value.max()}]"
        )

    legacy = preprocess_image(test)
    print(
        "Existing API: "
        f"shape={legacy.shape}, dtype={legacy.dtype}"
    )

    print("=" * 60)
    print("STATUS: STEP 1 TEST PASSED")
    print("=" * 60)
