"""Structural image representations for LUCAS."""

import cv2
import numpy as np


def gradient_representation(image: np.ndarray) -> np.ndarray:
    """
    Convert an image into a local gradient-magnitude representation.

    The goal is to emphasize terrain boundaries and local structure
    while reducing dependence on absolute image intensity.
    """

    image = image.astype(np.float32)

    # Horizontal and vertical intensity gradients
    gx = cv2.Sobel(
        image,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(
        image,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    # Gradient magnitude
    magnitude = cv2.magnitude(gx, gy)

    # Robust normalization
    low, high = np.percentile(
        magnitude,
        [2, 98]
    )

    if high <= low:
        return np.zeros_like(
            magnitude,
            dtype=np.uint8
        )

    normalized = (
        magnitude - low
    ) / (
        high - low
    )

    normalized = np.clip(
        normalized,
        0.0,
        1.0
    )

    return (
        normalized * 255
    ).astype(np.uint8)