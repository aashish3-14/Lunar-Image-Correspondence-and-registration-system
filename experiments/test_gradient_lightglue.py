from pathlib import Path

import cv2
import numpy as np
import torch

from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image


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


def to_gray_uint8(image):

    if isinstance(image, torch.Tensor):

        image = (
            image
            .detach()
            .cpu()
            .numpy()
        )

        image = np.transpose(
            image,
            (1, 2, 0),
        )

        image = np.clip(
            image * 255.0,
            0,
            255,
        ).astype(np.uint8)

        image = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY,
        )

    else:

        image = np.asarray(
            image,
            dtype=np.uint8,
        )

        if image.ndim == 3:

            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY,
            )

    return image


def gradient_representation(image):

    gray = to_gray_uint8(
        image
    )

    gx = cv2.Sobel(
        gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    gy = cv2.Sobel(
        gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    magnitude = cv2.magnitude(
        gx,
        gy,
    )

    magnitude = cv2.normalize(
        magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    )

    return magnitude.astype(
        np.uint8
    )


def clahe_representation(image):

    gray = to_gray_uint8(
        image
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    return clahe.apply(
        gray
    )


def make_three_channel(gray):

    return np.stack(
        [
            gray,
            gray,
            gray,
        ],
        axis=0,
    )


def run():

    print()
    print("=" * 55)
    print(
        "LUCAS GRADIENT LIGHTGLUE EXPERIMENT"
    )
    print("=" * 55)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    # --------------------------------------------------------
    # Load original images
    # --------------------------------------------------------

    source = cv2.imread(
        str(SOURCE),
        cv2.IMREAD_GRAYSCALE,
    )

    reference = cv2.imread(
        str(REFERENCE),
        cv2.IMREAD_GRAYSCALE,
    )

    if source is None:
        raise FileNotFoundError(
            SOURCE
        )

    if reference is None:
        raise FileNotFoundError(
            REFERENCE
        )

    # --------------------------------------------------------
    # Build representations
    # --------------------------------------------------------

    representations = {

        "original":
            source,

        "clahe":
            clahe_representation(
                source
            ),

        "gradient":
            gradient_representation(
                source
            ),
    }

    reference_representations = {

        "original":
            reference,

        "clahe":
            clahe_representation(
                reference
            ),

        "gradient":
            gradient_representation(
                reference
            ),
    }

    extractor = (
        SuperPoint(
            max_num_keypoints=4096
        )
        .eval()
        .to(device)
    )

    matcher = (
        LightGlue(
            features="superpoint"
        )
        .eval()
        .to(device)
    )

    # --------------------------------------------------------
    # Test every representation pair
    # --------------------------------------------------------

    for name0, img0 in representations.items():

        for name1, img1 in (
            reference_representations.items()
        ):

            print()
            print(
                "-" * 55
            )

            print(
                f"TEST: "
                f"{name0} → {name1}"
            )

            tensor0 = torch.from_numpy(
                make_three_channel(
                    img0
                )
            ).float() / 255.0

            tensor1 = torch.from_numpy(
                make_three_channel(
                    img1
                )
            ).float() / 255.0

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

                matches = matcher(
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

            match_array = (
                matches["matches"][0]
                .detach()
                .cpu()
                .numpy()
            )

            print(
                f"Keypoints source: "
                f"{len(keypoints0)}"
            )

            print(
                f"Keypoints reference: "
                f"{len(keypoints1)}"
            )

            print(
                f"LightGlue matches: "
                f"{len(match_array)}"
            )

            # ------------------------------------------------
            # RANSAC
            # ------------------------------------------------

            if len(match_array) >= 4:

                points0 = (
                    keypoints0[
                        match_array[:, 0]
                    ]
                )

                points1 = (
                    keypoints1[
                        match_array[:, 1]
                    ]
                )

                H, mask = cv2.findHomography(
                    points0,
                    points1,
                    cv2.RANSAC,
                    5.0,
                )

                if mask is not None:

                    mask = (
                        mask.ravel()
                        .astype(bool)
                    )

                    inliers = int(
                        np.sum(mask)
                    )

                    ratio = (
                        inliers /
                        len(match_array)
                    )

                    print(
                        f"RANSAC inliers: "
                        f"{inliers}"
                    )

                    print(
                        f"Inlier ratio: "
                        f"{ratio:.3f}"
                    )

                else:

                    print(
                        "RANSAC failed."
                    )

            else:

                print(
                    "Not enough matches "
                    "for RANSAC."
                )

    print()
    print("=" * 55)
    print(
        "EXPERIMENT COMPLETE"
    )
    print("=" * 55)


if __name__ == "__main__":
    run()