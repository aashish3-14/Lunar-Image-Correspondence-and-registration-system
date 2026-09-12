from pathlib import Path

import cv2
import numpy as np
import torch

from lightglue import LightGlue, SuperPoint


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "tmc2_apollo16_focused.png"
)

REFERENCE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lro_apollo16_overlap.png"
)


# ============================================================
# CENTER PRIOR
# ============================================================

CENTER_X = 1997.4
CENTER_Y = 3893.9


# Large local geographic ROI.
#
# We deliberately make this generous.
#
ROI_WIDTH = 2800
ROI_HEIGHT = 6500


MAX_KEYPOINTS = 4096


# ============================================================
# LOAD
# ============================================================

def load_gray(path):

    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:

        raise FileNotFoundError(
            path
        )

    return image


# ============================================================
# ROI
# ============================================================

def extract_roi(
    image,
    center_x,
    center_y,
    width,
    height,
):

    h, w = image.shape

    x0 = int(
        center_x
        - width / 2
    )

    y0 = int(
        center_y
        - height / 2
    )

    x1 = int(
        center_x
        + width / 2
    )

    y1 = int(
        center_y
        + height / 2
    )

    x0 = max(
        0,
        x0,
    )

    y0 = max(
        0,
        y0,
    )

    x1 = min(
        w,
        x1,
    )

    y1 = min(
        h,
        y1,
    )

    return (
        image[
            y0:y1,
            x0:x1,
        ],
        (
            x0,
            y0,
            x1,
            y1,
        ),
    )


# ============================================================
# TENSOR
# ============================================================

def make_tensor(
    image,
    device,
):

    image = (
        image.astype(
            np.float32
        )
        /
        255.0
    )

    image = np.stack(
        [
            image,
            image,
            image,
        ],
        axis=0,
    )

    return (
        torch.from_numpy(
            image
        )
        .to(device)
    )


# ============================================================
# MAIN
# ============================================================

def run():

    print()
    print("=" * 65)
    print(
        "LUCAS GEOGRAPHIC ROI LIGHTGLUE EXPERIMENT"
    )
    print("=" * 65)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

    source = load_gray(
        SOURCE
    )

    reference = load_gray(
        REFERENCE
    )

    print(
        "Source:",
        source.shape,
    )

    print(
        "Reference:",
        reference.shape,
    )

    # --------------------------------------------------------
    # Extract reference ROI
    # --------------------------------------------------------

    reference_roi, bounds = (
        extract_roi(
            reference,
            CENTER_X,
            CENTER_Y,
            ROI_WIDTH,
            ROI_HEIGHT,
        )
    )

    print()
    print(
        "Geographic center prior:"
    )

    print(
        f"({CENTER_X:.1f}, "
        f"{CENTER_Y:.1f})"
    )

    print(
        "Reference ROI bounds:"
    )

    print(
        bounds
    )

    print(
        "Reference ROI shape:",
        reference_roi.shape,
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    print()
    print(
        "Loading SuperPoint..."
    )

    extractor = (
        SuperPoint(
            max_num_keypoints=MAX_KEYPOINTS
        )
        .eval()
        .to(device)
    )

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

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    tensor0 = make_tensor(
        source,
        device,
    )

    tensor1 = make_tensor(
        reference_roi,
        device,
    )

    print()
    print(
        "Extracting features..."
    )

    with torch.inference_mode():

        feats0 = extractor.extract(
            tensor0
        )

        feats1 = extractor.extract(
            tensor1
        )

        result = matcher(
            {
                "image0": feats0,
                "image1": feats1,
            }
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

    matches = (
        result["matches"][0]
        .detach()
        .cpu()
        .numpy()
    )

    print()
    print(
        "=" * 65
    )

    print(
        "MATCHING RESULT"
    )

    print(
        "=" * 65
    )

    print(
        "Source keypoints:",
        len(keypoints0),
    )

    print(
        "Reference ROI keypoints:",
        len(keypoints1),
    )

    print(
        "LightGlue matches:",
        len(matches),
    )

    # --------------------------------------------------------
    # Geometry
    # --------------------------------------------------------

    if len(matches) < 4:

        print(
            "Not enough matches."
        )

        return

    points0 = (
        keypoints0[
            matches[:, 0]
        ]
    )

    points1 = (
        keypoints1[
            matches[:, 1]
        ]
    )

    # Convert ROI coordinates
    # into full LRO coordinates.

    points1_global = (
        points1.copy()
    )

    points1_global[:, 0] += bounds[0]
    points1_global[:, 1] += bounds[1]

    H, mask = cv2.findHomography(
        points0,
        points1_global,
        cv2.RANSAC,
        5.0,
    )

    print()
    print(
        "=" * 65
    )

    print(
        "GEOMETRIC VERIFICATION"
    )

    print(
        "=" * 65
    )

    if mask is None:

        print(
            "RANSAC failed."
        )

        return

    mask = (
        mask.ravel()
        .astype(bool)
    )

    inliers = int(
        np.sum(mask)
    )

    ratio = (
        inliers
        /
        len(matches)
    )

    src_inliers = (
        points0[mask]
    )

    ref_inliers = (
        points1_global[mask]
    )

    projected = cv2.perspectiveTransform(
        src_inliers.reshape(
            -1,
            1,
            2,
        ),
        H,
    ).reshape(
        -1,
        2,
    )

    errors = np.linalg.norm(
        projected
        -
        ref_inliers,
        axis=1,
    )

    rmse = float(
        np.sqrt(
            np.mean(
                errors ** 2
            )
        )
    )

    print(
        "Total matches:",
        len(matches),
    )

    print(
        "Inliers:",
        inliers,
    )

    print(
        f"Inlier ratio: "
        f"{ratio:.3f}"
    )

    print(
        f"RMSE: "
        f"{rmse:.4f} px"
    )

    # --------------------------------------------------------
    # Save ROI
    # --------------------------------------------------------

    output = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "lro_geographic_roi.png"
    )

    cv2.imwrite(
        str(output),
        reference_roi,
    )

    print()
    print(
        "Saved reference ROI:"
    )

    print(
        output
    )

    print()
    print("=" * 65)
    print(
        "EXPERIMENT COMPLETE"
    )
    print("=" * 65)


if __name__ == "__main__":

    run()