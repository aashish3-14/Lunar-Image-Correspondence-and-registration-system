"""Feature detection utilities for LUCAS."""

from __future__ import annotations

import cv2
import numpy as np


def detect_features(
    image: np.ndarray,
    n_features: int = 5000,
):
    """
    Detect SIFT keypoints and compute their descriptors.

    Parameters
    ----------
    image:
        Grayscale, preprocessed image.

    n_features:
        Maximum number of features to retain.

    Returns
    -------
    keypoints:
        Detected SIFT keypoints.

    descriptors:
        Numerical descriptor for each keypoint.
    """

    sift = cv2.SIFT_create(
        nfeatures=n_features
    )

    keypoints, descriptors = sift.detectAndCompute(
        image,
        None
    )

    return keypoints, descriptors