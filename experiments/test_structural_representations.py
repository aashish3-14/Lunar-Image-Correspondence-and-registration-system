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


MAX_KEYPOINTS = 2048
RANSAC_THRESHOLD = 5.0


# ============================================================
# LOAD
# ============================================================

def load_gray(path):

    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        raise FileNotFoundError(path)

    return image


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_uint8(image):

    image = image.astype(
        np.float32
    )

    low, high = np.percentile(
        image,
        [2, 98],
    )

    if high <= low:

        return np.zeros_like(
            image,
            dtype=np.uint8,
        )

    image = (
        image - low
    ) / (
        high - low
    )

    image = np.clip(
        image,
        0,
        1,
    )

    return (
        image * 255
    ).astype(
        np.uint8
    )


# ============================================================
# REPRESENTATIONS
# ============================================================

def raw(image):

    return normalize_uint8(
        image
    )


def clahe(image):

    image = normalize_uint8(
        image
    )

    enhancer = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    return enhancer.apply(
        image
    )


def gradient(image):

    image = normalize_uint8(
        image
    )

    gx = cv2.Sobel(
        image,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    gy = cv2.Sobel(
        image,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    magnitude = cv2.magnitude(
        gx,
        gy,
    )

    return normalize_uint8(
        magnitude
    )


def laplacian(image):

    image = normalize_uint8(
        image
    )

    lap = cv2.Laplacian(
        image,
        cv2.CV_32F,
        ksize=3,
    )

    return normalize_uint8(
        np.abs(lap)
    )


def local_normalized(image):

    image = image.astype(
        np.float32
    )

    mean = cv2.GaussianBlur(
        image,
        (0, 0),
        15,
    )

    sq_mean = cv2.GaussianBlur(
        image * image,
        (0, 0),
        15,
    )

    variance = (
        sq_mean
        -
        mean * mean
    )

    std = np.sqrt(
        np.maximum(
            variance,
            1e-6,
        )
    )

    result = (
        image - mean
    ) / std

    return normalize_uint8(
        result
    )


def intensity_gradient_fusion(
    image
):

    intensity = normalize_uint8(
        image
    ).astype(
        np.float32
    )

    gx = cv2.Sobel(
        intensity,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    gy = cv2.Sobel(
        intensity,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    magnitude = cv2.magnitude(
        gx,
        gy,
    )

    magnitude = normalize_uint8(
        magnitude
    ).astype(
        np.float32
    )

    fused = (
        0.65 * intensity
        +
        0.35 * magnitude
    )

    return np.clip(
        fused,
        0,
        255,
    ).astype(
        np.uint8
    )


def gradient_orientation(
    image
):

    image = normalize_uint8(
        image
    )

    gx = cv2.Sobel(
        image,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    gy = cv2.Sobel(
        image,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    angle = np.arctan2(
        gy,
        gx,
    )

    # Orientation is periodic.
    #
    # Encode angle into [0, 255].

    encoded = (
        (
            angle
            +
            np.pi
        )
        /
        (2 * np.pi)
        *
        255.0
    )

    return encoded.astype(
        np.uint8
    )


# ============================================================
# THREE CHANNEL
# ============================================================

def make_three_channel(
    image
):

    return np.stack(
        [
            image,
            image,
            image,
        ],
        axis=0,
    )


# ============================================================
# LIGHTGLUE
# ============================================================

def run_match(
    source,
    reference,
    extractor,
    matcher,
    device,
):

    tensor0 = (
        torch.from_numpy(
            make_three_channel(
                source
            )
        ).float()
        / 255.0
    )

    tensor1 = (
        torch.from_numpy(
            make_three_channel(
                reference
            )
        ).float()
        / 255.0
    )

    tensor0 = tensor0.to(
        device
    )

    tensor1 = tensor1.to(
        device
    )

    with torch.inference_mode():

        feats0 = extractor.extract(
            tensor0
        )

        feats1 = extractor.extract(
            tensor1
        )

        output = matcher(
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
        output["matches"][0]
        .detach()
        .cpu()
        .numpy()
    )

    if len(matches) < 4:

        return {
            "keypoints0": len(
                keypoints0
            ),
            "keypoints1": len(
                keypoints1
            ),
            "matches": len(matches),
            "inliers": 0,
            "ratio": 0.0,
            "rmse": None,
        }

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

    H, mask = cv2.findHomography(
        points0,
        points1,
        cv2.RANSAC,
        RANSAC_THRESHOLD,
    )

    if mask is None:

        return {
            "keypoints0": len(
                keypoints0
            ),
            "keypoints1": len(
                keypoints1
            ),
            "matches": len(matches),
            "inliers": 0,
            "ratio": 0.0,
            "rmse": None,
        }

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

    if inliers >= 4:

        p0 = points0[mask]
        p1 = points1[mask]

        projected = (
            cv2.perspectiveTransform(
                p0.reshape(
                    -1,
                    1,
                    2,
                ),
                H,
            ).reshape(
                -1,
                2,
            )
        )

        errors = np.linalg.norm(
            projected - p1,
            axis=1,
        )

        rmse = float(
            np.sqrt(
                np.mean(
                    errors ** 2
                )
            )
        )

    else:

        rmse = None

    return {
        "keypoints0": len(
            keypoints0
        ),
        "keypoints1": len(
            keypoints1
        ),
        "matches": len(matches),
        "inliers": inliers,
        "ratio": ratio,
        "rmse": rmse,
    }


# ============================================================
# MAIN
# ============================================================

def run():

    print()
    print("=" * 70)

    print(
        "LUCAS STRUCTURAL REPRESENTATION BENCHMARK"
    )

    print("=" * 70)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    source = load_gray(
        SOURCE
    )

    reference = load_gray(
        REFERENCE
    )

    source_representations = {
        "raw": raw(source),
        "clahe": clahe(source),
        "gradient": gradient(source),
        "laplacian": laplacian(source),
        "local_normalized":
            local_normalized(source),
        "intensity_gradient":
            intensity_gradient_fusion(
                source
            ),
        "gradient_orientation":
            gradient_orientation(
                source
            ),
    }

    reference_representations = {
        "raw": raw(reference),
        "clahe": clahe(reference),
        "gradient": gradient(reference),
        "laplacian": laplacian(reference),
        "local_normalized":
            local_normalized(reference),
        "intensity_gradient":
            intensity_gradient_fusion(
                reference
            ),
        "gradient_orientation":
            gradient_orientation(
                reference
            ),
    }

    print()
    print(
        "Loading SuperPoint..."
    )

    extractor = (
        SuperPoint(
            max_num_keypoints=
                MAX_KEYPOINTS
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

    names = list(
        source_representations.keys()
    )

    results = []

    # --------------------------------------------------------
    # SAME REPRESENTATION
    # --------------------------------------------------------

    for name in names:

        print()
        print(
            "-" * 70
        )

        print(
            f"TEST: {name} → {name}"
        )

        result = run_match(
            source_representations[name],
            reference_representations[name],
            extractor,
            matcher,
            device,
        )

        results.append(
            (
                name,
                name,
                result,
            )
        )

        print(
            f"Keypoints: "
            f"{result['keypoints0']} / "
            f"{result['keypoints1']}"
        )

        print(
            f"Matches: "
            f"{result['matches']}"
        )

        print(
            f"Inliers: "
            f"{result['inliers']}"
        )

        print(
            f"Ratio: "
            f"{result['ratio']:.3f}"
        )

        print(
            f"RMSE: "
            f"{result['rmse']}"
        )

    # --------------------------------------------------------
    # CROSS REPRESENTATION
    # --------------------------------------------------------

    tests = [
        ("raw", "gradient"),
        ("gradient", "raw"),
        ("raw", "intensity_gradient"),
        ("intensity_gradient", "raw"),
        ("gradient", "intensity_gradient"),
        ("intensity_gradient", "gradient"),
        ("local_normalized", "raw"),
        ("raw", "local_normalized"),
        ("local_normalized", "local_normalized"),
        ("gradient_orientation", "gradient_orientation"),
    ]

    for name0, name1 in tests:

        print()
        print(
            "-" * 70
        )

        print(
            f"TEST: {name0} → {name1}"
        )

        result = run_match(
            source_representations[name0],
            reference_representations[name1],
            extractor,
            matcher,
            device,
        )

        results.append(
            (
                name0,
                name1,
                result,
            )
        )

        print(
            f"Keypoints: "
            f"{result['keypoints0']} / "
            f"{result['keypoints1']}"
        )

        print(
            f"Matches: "
            f"{result['matches']}"
        )

        print(
            f"Inliers: "
            f"{result['inliers']}"
        )

        print(
            f"Ratio: "
            f"{result['ratio']:.3f}"
        )

        print(
            f"RMSE: "
            f"{result['rmse']}"
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 70)

    print(
        "BENCHMARK SUMMARY"
    )

    print("=" * 70)

    for name0, name1, result in results:

        print(
            f"{name0:22s} → "
            f"{name1:22s} | "
            f"matches={result['matches']:3d} | "
            f"inliers={result['inliers']:3d} | "
            f"ratio={result['ratio']:.3f}"
        )

    print()
    print(
        "=" * 70
    )

    print(
        "EXPERIMENT COMPLETE"
    )

    print("=" * 70)


if __name__ == "__main__":

    run()