from __future__ import annotations

import cv2
import numpy as np


# ============================================================
# SIFT FEATURE EXTRACTION
# ============================================================

def extract_sift_features(
    image: np.ndarray,
    max_features: int = 5000,
):
    """
    Extract SIFT keypoints and descriptors.

    Returns
    -------
    keypoints
    descriptors
    """

    if image is None:
        raise ValueError("Image cannot be None.")

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

    keypoints, descriptors = sift.detectAndCompute(
        gray,
        None,
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
# SIFT RECIPROCAL RATIO MATCHING
# ============================================================

def match_sift_reciprocal(
    source_descriptors: np.ndarray,
    reference_descriptors: np.ndarray,
    ratio_threshold: float = 0.90,
):
    """
    Perform reciprocal Lowe-ratio SIFT matching.

    A match is retained only when:

        1. source -> reference passes ratio test
        2. reference -> source returns the same source index

    Returns
    -------
    matches:
        List of cv2.DMatch objects.
    """

    if (
        source_descriptors is None
        or reference_descriptors is None
        or len(source_descriptors) == 0
        or len(reference_descriptors) == 0
    ):
        return []

    source_descriptors = np.asarray(
        source_descriptors,
        dtype=np.float32,
    )

    reference_descriptors = np.asarray(
        reference_descriptors,
        dtype=np.float32,
    )

    matcher = cv2.BFMatcher(
        cv2.NORM_L2,
        crossCheck=False,
    )

    forward = matcher.knnMatch(
        source_descriptors,
        reference_descriptors,
        k=2,
    )

    backward = matcher.knnMatch(
        reference_descriptors,
        source_descriptors,
        k=2,
    )

    # --------------------------------------------------------
    # Forward ratio test
    # --------------------------------------------------------

    forward_good = {}

    for pair in forward:

        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < ratio_threshold * n.distance:

            forward_good[m.queryIdx] = m

    # --------------------------------------------------------
    # Backward ratio test
    # --------------------------------------------------------

    backward_good = {}

    for pair in backward:

        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < ratio_threshold * n.distance:

            backward_good[m.queryIdx] = m

    # --------------------------------------------------------
    # Reciprocal consistency
    # --------------------------------------------------------

    matches = []

    for source_idx, forward_match in forward_good.items():

        reference_idx = forward_match.trainIdx

        reverse_match = backward_good.get(
            reference_idx
        )

        if reverse_match is None:
            continue

        if reverse_match.trainIdx != source_idx:
            continue

        matches.append(
            forward_match
        )

    # Strongest matches first.
    matches.sort(
        key=lambda m: float(m.distance)
    )

    return matches


# ============================================================
# CONVERT MATCHES TO POINTS
# ============================================================

def matches_to_points(
    source_keypoints,
    reference_keypoints,
    matches,
):
    """
    Convert cv2.DMatch objects into Nx2 source/reference arrays.
    """

    if len(matches) == 0:

        empty = np.empty(
            (0, 2),
            dtype=np.float32,
        )

        return empty.copy(), empty.copy()

    source_points = np.asarray(
        [
            source_keypoints[m.queryIdx].pt
            for m in matches
        ],
        dtype=np.float32,
    )

    reference_points = np.asarray(
        [
            reference_keypoints[m.trainIdx].pt
            for m in matches
        ],
        dtype=np.float32,
    )

    return (
        source_points,
        reference_points,
    )


# ============================================================
# DUPLICATE-SAFE CANDIDATE FUSION
# ============================================================

def fuse_candidate_matches(
    source_points_a: np.ndarray,
    reference_points_a: np.ndarray,
    source_points_b: np.ndarray,
    reference_points_b: np.ndarray,
    deduplication_radius: float = 1.5,
):
    """
    Fuse candidate correspondences from two independent
    matching branches.

    Branch A is kept first.

    Branch B candidates are added only when they are not
    effectively duplicates of an existing correspondence.

    This function does NOT perform geometric verification.
    RANSAC remains responsible for geometric consistency.
    """

    source_points_a = np.asarray(
        source_points_a,
        dtype=np.float32,
    )

    reference_points_a = np.asarray(
        reference_points_a,
        dtype=np.float32,
    )

    source_points_b = np.asarray(
        source_points_b,
        dtype=np.float32,
    )

    reference_points_b = np.asarray(
        reference_points_b,
        dtype=np.float32,
    )

    if (
        source_points_a.ndim != 2
        or source_points_a.shape[1] != 2
    ):
        raise ValueError(
            "source_points_a must have shape (N, 2)."
        )

    if (
        reference_points_a.shape
        != source_points_a.shape
    ):
        raise ValueError(
            "Branch A point arrays must match."
        )

    if (
        source_points_b.ndim != 2
        or source_points_b.shape[1] != 2
    ):
        raise ValueError(
            "source_points_b must have shape (N, 2)."
        )

    if (
        reference_points_b.shape
        != source_points_b.shape
    ):
        raise ValueError(
            "Branch B point arrays must match."
        )

    # --------------------------------------------------------
    # Start with branch A
    # --------------------------------------------------------

    fused_source = []

    fused_reference = []

    for s, r in zip(
        source_points_a,
        reference_points_a,
    ):
        fused_source.append(
            s
        )
        fused_reference.append(
            r
        )

    radius_sq = float(
        deduplication_radius
    ) ** 2

    # --------------------------------------------------------
    # Add branch B candidates
    # --------------------------------------------------------

    for s, r in zip(
        source_points_b,
        reference_points_b,
    ):

        duplicate = False

        for existing_s, existing_r in zip(
            fused_source,
            fused_reference,
        ):

            source_distance_sq = float(
                np.sum(
                    (s - existing_s) ** 2
                )
            )

            reference_distance_sq = float(
                np.sum(
                    (r - existing_r) ** 2
                )
            )

            if (
                source_distance_sq <= radius_sq
                and
                reference_distance_sq <= radius_sq
            ):

                duplicate = True
                break

        if not duplicate:

            fused_source.append(
                s
            )

            fused_reference.append(
                r
            )

    if len(fused_source) == 0:

        empty = np.empty(
            (0, 2),
            dtype=np.float32,
        )

        return (
            empty.copy(),
            empty.copy(),
        )

    return (
        np.asarray(
            fused_source,
            dtype=np.float32,
        ),
        np.asarray(
            fused_reference,
            dtype=np.float32,
        ),
    )