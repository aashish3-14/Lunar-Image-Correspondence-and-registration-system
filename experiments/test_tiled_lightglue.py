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
# CONFIGURATION
# ============================================================

GRID_ROWS = 4
GRID_COLS = 4

TILE_OVERLAP = 0.25

MAX_KEYPOINTS = 1024

RANSAC_THRESHOLD = 5.0

MIN_TILE_SIZE = 256


# ============================================================
# IMAGE HELPERS
# ============================================================

def load_gray(path):

    image = cv2.imread(
        str(path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        raise FileNotFoundError(path)

    return image


def make_tensor(image):

    image = image.astype(
        np.float32
    ) / 255.0

    tensor = np.stack(
        [
            image,
            image,
            image,
        ],
        axis=0,
    )

    return torch.from_numpy(
        tensor
    )


# ============================================================
# TILE GENERATION
# ============================================================

def make_tiles(
    image,
    rows,
    cols,
    overlap,
):

    height, width = image.shape

    base_h = height / rows
    base_w = width / cols

    tiles = []

    for r in range(rows):

        for c in range(cols):

            center_y = (
                (r + 0.5)
                * base_h
            )

            center_x = (
                (c + 0.5)
                * base_w
            )

            tile_h = int(
                base_h
                * (1.0 + overlap)
            )

            tile_w = int(
                base_w
                * (1.0 + overlap)
            )

            y0 = int(
                center_y
                - tile_h / 2
            )

            y1 = int(
                center_y
                + tile_h / 2
            )

            x0 = int(
                center_x
                - tile_w / 2
            )

            x1 = int(
                center_x
                + tile_w / 2
            )

            y0 = max(
                0,
                y0,
            )

            x0 = max(
                0,
                x0,
            )

            y1 = min(
                height,
                y1,
            )

            x1 = min(
                width,
                x1,
            )

            if (
                y1 - y0
                < MIN_TILE_SIZE
                or
                x1 - x0
                < MIN_TILE_SIZE
            ):
                continue

            tiles.append(
                {
                    "row": r,
                    "col": c,
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                }
            )

    return tiles


# ============================================================
# GLOBAL COORDINATE CONVERSION
# ============================================================

def local_to_global(
    points,
    tile,
):

    points = np.asarray(
        points,
        dtype=np.float32,
    ).copy()

    points[:, 0] += tile["x0"]
    points[:, 1] += tile["y0"]

    return points


# ============================================================
# TILE PAIRING
# ============================================================

def corresponding_reference_tile(
    source_tile,
    source_shape,
    reference_shape,
):

    sh, sw = source_shape
    rh, rw = reference_shape

    sx = (
        source_tile["x0"]
        + source_tile["x1"]
    ) / 2.0

    sy = (
        source_tile["y0"]
        + source_tile["y1"]
    ) / 2.0

    # --------------------------------------------------------
    # Approximate normalized coordinate mapping.
    #
    # This intentionally uses only image coordinates.
    # We are NOT claiming this is spacecraft geometry.
    # --------------------------------------------------------

    nx = sx / sw
    ny = sy / sh

    rx = nx * rw
    ry = ny * rh

    tile_w = (
        source_tile["x1"]
        - source_tile["x0"]
    )

    tile_h = (
        source_tile["y1"]
        - source_tile["y0"]
    )

    # Allow a generous local search region.
    ref_w = int(
        tile_w * 1.75
    )

    ref_h = int(
        tile_h * 1.75
    )

    x0 = int(
        rx - ref_w / 2
    )

    y0 = int(
        ry - ref_h / 2
    )

    x1 = int(
        rx + ref_w / 2
    )

    y1 = int(
        ry + ref_h / 2
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
        rw,
        x1,
    )

    y1 = min(
        rh,
        y1,
    )

    return {
        "x0": x0,
        "y0": y0,
        "x1": x1,
        "y1": y1,
    }


# ============================================================
# MATCHING
# ============================================================

def match_tile(
    source_tile_image,
    reference_tile_image,
    extractor,
    matcher,
    device,
):

    tensor0 = make_tensor(
        source_tile_image
    ).to(device)

    tensor1 = make_tensor(
        reference_tile_image
    ).to(device)

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

    if len(matches) == 0:

        return (
            np.empty(
                (0, 2),
                dtype=np.float32,
            ),
            np.empty(
                (0, 2),
                dtype=np.float32,
            ),
        )

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

    return (
        points0.astype(
            np.float32
        ),
        points1.astype(
            np.float32
        ),
    )


# ============================================================
# DUPLICATE REMOVAL
# ============================================================

def remove_duplicates(
    source_points,
    reference_points,
    radius=2.0,
):

    if len(source_points) == 0:

        return (
            source_points,
            reference_points,
        )

    keep = []

    radius_sq = (
        radius * radius
    )

    for i in range(
        len(source_points)
    ):

        duplicate = False

        for j in keep:

            ds = (
                source_points[i]
                -
                source_points[j]
            )

            dr = (
                reference_points[i]
                -
                reference_points[j]
            )

            if (
                float(
                    np.dot(ds, ds)
                )
                <= radius_sq
                and
                float(
                    np.dot(dr, dr)
                )
                <= radius_sq
            ):

                duplicate = True
                break

        if not duplicate:

            keep.append(i)

    keep = np.asarray(
        keep,
        dtype=np.int64,
    )

    return (
        source_points[keep],
        reference_points[keep],
    )


# ============================================================
# GEOMETRIC VERIFICATION
# ============================================================

def verify_geometry(
    source_points,
    reference_points,
):

    if len(source_points) < 4:

        return None

    H, mask = cv2.findHomography(
        source_points,
        reference_points,
        cv2.RANSAC,
        RANSAC_THRESHOLD,
    )

    if mask is None:

        return None

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
        len(source_points)
    )

    if inliers > 0:

        src_inliers = (
            source_points[mask]
        )

        ref_inliers = (
            reference_points[mask]
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

    else:

        rmse = float("inf")

    return {
        "H": H,
        "mask": mask,
        "inliers": inliers,
        "ratio": ratio,
        "rmse": rmse,
    }


# ============================================================
# MAIN
# ============================================================

def run():

    print()
    print("=" * 65)
    print(
        "LUCAS SPATIALLY CONSTRAINED "
        "TILED LIGHTGLUE EXPERIMENT"
    )
    print("=" * 65)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    print()
    print(
        "Loading images..."
    )

    source = load_gray(
        SOURCE
    )

    reference = load_gray(
        REFERENCE
    )

    print(
        "Source shape:",
        source.shape,
    )

    print(
        "Reference shape:",
        reference.shape,
    )

    # --------------------------------------------------------
    # MODELS
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
    # TILES
    # --------------------------------------------------------

    source_tiles = make_tiles(
        source,
        GRID_ROWS,
        GRID_COLS,
        TILE_OVERLAP,
    )

    print()
    print(
        f"Source tiles: "
        f"{len(source_tiles)}"
    )

    # --------------------------------------------------------
    # MATCH ALL TILES
    # --------------------------------------------------------

    all_source_points = []
    all_reference_points = []

    for index, tile in enumerate(
        source_tiles
    ):

        print()
        print(
            "-" * 65
        )

        print(
            f"TILE "
            f"{index + 1}/"
            f"{len(source_tiles)}"
            f" "
            f"[row={tile['row']}, "
            f"col={tile['col']}]"
        )

        source_crop = source[
            tile["y0"]:tile["y1"],
            tile["x0"]:tile["x1"],
        ]

        reference_tile = (
            corresponding_reference_tile(
                tile,
                source.shape,
                reference.shape,
            )
        )

        reference_crop = reference[
            reference_tile["y0"]:
            reference_tile["y1"],
            reference_tile["x0"]:
            reference_tile["x1"],
        ]

        print(
            "Source tile:",
            source_crop.shape,
        )

        print(
            "Reference search:",
            reference_crop.shape,
        )

        local_source, local_reference = (
            match_tile(
                source_crop,
                reference_crop,
                extractor,
                matcher,
                device,
            )
        )

        print(
            "Local LightGlue matches:",
            len(local_source),
        )

        if len(local_source) == 0:
            continue

        # Convert local coordinates
        # back to full-image coordinates.

        global_source = (
            local_to_global(
                local_source,
                tile,
            )
        )

        reference_tile_origin = {
            "x0": reference_tile["x0"],
            "y0": reference_tile["y0"],
        }

        global_reference = (
            local_to_global(
                local_reference,
                reference_tile_origin,
            )
        )

        all_source_points.append(
            global_source
        )

        all_reference_points.append(
            global_reference
        )

    # --------------------------------------------------------
    # COMBINE
    # --------------------------------------------------------

    print()
    print("=" * 65)
    print(
        "COMBINING TILE MATCHES"
    )
    print("=" * 65)

    if len(all_source_points) == 0:

        print(
            "No tile matches found."
        )

        return

    source_points = np.vstack(
        all_source_points
    ).astype(
        np.float32
    )

    reference_points = np.vstack(
        all_reference_points
    ).astype(
        np.float32
    )

    print(
        "Raw tiled matches:",
        len(source_points),
    )

    # --------------------------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------------------------

    source_points, reference_points = (
        remove_duplicates(
            source_points,
            reference_points,
            radius=2.0,
        )
    )

    print(
        "After duplicate removal:",
        len(source_points),
    )

    # --------------------------------------------------------
    # GLOBAL GEOMETRY
    # --------------------------------------------------------

    result = verify_geometry(
        source_points,
        reference_points,
    )

    print()
    print("=" * 65)
    print(
        "TILED GLOBAL GEOMETRIC VERIFICATION"
    )
    print("=" * 65)

    if result is None:

        print(
            "RANSAC failed."
        )

        return

    print(
        f"Total matches: "
        f"{len(source_points)}"
    )

    print(
        f"Inliers: "
        f"{result['inliers']}"
    )

    print(
        f"Inlier ratio: "
        f"{result['ratio']:.3f}"
    )

    print(
        f"RMSE: "
        f"{result['rmse']:.4f} px"
    )

    # --------------------------------------------------------
    # SPATIAL DISTRIBUTION
    # --------------------------------------------------------

    mask = result["mask"]

    inlier_source = (
        source_points[mask]
    )

    height, width = source.shape

    occupied = set()

    for point in inlier_source:

        x, y = point

        col = min(
            GRID_COLS - 1,
            int(
                x
                /
                width
                *
                GRID_COLS
            ),
        )

        row = min(
            GRID_ROWS - 1,
            int(
                y
                /
                height
                *
                GRID_ROWS
            ),
        )

        occupied.add(
            (row, col)
        )

    coverage = (
        len(occupied)
        /
        (GRID_ROWS * GRID_COLS)
    )

    print()
    print(
        "=" * 65
    )
    print(
        "SPATIAL DISTRIBUTION"
    )
    print(
        "=" * 65
    )

    print(
        f"Occupied cells: "
        f"{len(occupied)}/"
        f"{GRID_ROWS * GRID_COLS}"
    )

    print(
        f"Coverage: "
        f"{coverage:.3f}"
    )

    print()
    print(
        "Occupied cells:"
    )

    print(
        sorted(
            occupied
        )
    )

    # --------------------------------------------------------
    # VISUALIZATION
    # --------------------------------------------------------

    output = (
        PROJECT_ROOT
        / "data"
        / "results"
        / "lucas_tiled_matches.png"
    )

    vis0 = cv2.cvtColor(
        source,
        cv2.COLOR_GRAY2BGR,
    )

    vis1 = cv2.cvtColor(
        reference,
        cv2.COLOR_GRAY2BGR,
    )

    # Resize for visualization.

    max_height = 1200

    scale0 = min(
        1.0,
        max_height / vis0.shape[0],
    )

    scale1 = min(
        1.0,
        max_height / vis1.shape[0],
    )

    vis0 = cv2.resize(
        vis0,
        None,
        fx=scale0,
        fy=scale0,
    )

    vis1 = cv2.resize(
        vis1,
        None,
        fx=scale1,
        fy=scale1,
    )

    canvas_height = max(
        vis0.shape[0],
        vis1.shape[0],
    )

    canvas_width = (
        vis0.shape[1]
        +
        vis1.shape[1]
    )

    canvas = np.zeros(
        (
            canvas_height,
            canvas_width,
            3,
        ),
        dtype=np.uint8,
    )

    canvas[
        :vis0.shape[0],
        :vis0.shape[1],
    ] = vis0

    canvas[
        :vis1.shape[0],
        vis0.shape[1]:
        vis0.shape[1] + vis1.shape[1],
    ] = vis1

    offset_x = vis0.shape[1]

    rng = np.random.default_rng(
        42
    )

    for p0, p1, is_inlier in zip(
        source_points,
        reference_points,
        mask,
    ):

        x0 = int(
            p0[0] * scale0
        )

        y0 = int(
            p0[1] * scale0
        )

        x1 = int(
            p1[0] * scale1
        ) + offset_x

        y1 = int(
            p1[1] * scale1
        )

        if is_inlier:

            cv2.line(
                canvas,
                (x0, y0),
                (x1, y1),
                (0, 255, 0),
                1,
            )

            cv2.circle(
                canvas,
                (x0, y0),
                4,
                (0, 255, 0),
                -1,
            )

            cv2.circle(
                canvas,
                (x1, y1),
                4,
                (0, 255, 0),
                -1,
            )

        else:

            cv2.circle(
                canvas,
                (x0, y0),
                2,
                (120, 120, 120),
                -1,
            )

    cv2.imwrite(
        str(output),
        canvas,
    )

    print()
    print(
        "Saved:",
        output,
    )

    print()
    print("=" * 65)
    print(
        "EXPERIMENT COMPLETE"
    )
    print("=" * 65)


if __name__ == "__main__":
    run()