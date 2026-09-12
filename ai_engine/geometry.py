import cv2
import numpy as np


def estimate_homography(
    source_keypoints,
    reference_keypoints,
    matches,
    reprojection_threshold: float = 5.0,
):
    if len(matches) < 4:
        return None, None

    source_points = np.float32([
        source_keypoints[m.queryIdx].pt for m in matches
    ]).reshape(-1, 1, 2)

    reference_points = np.float32([
        reference_keypoints[m.trainIdx].pt for m in matches
    ]).reshape(-1, 1, 2)

    homography, mask = cv2.findHomography(
        source_points,
        reference_points,
        cv2.RANSAC,
        reprojection_threshold,
    )

    if mask is None:
        return homography, None

    return homography, mask.ravel().astype(bool)