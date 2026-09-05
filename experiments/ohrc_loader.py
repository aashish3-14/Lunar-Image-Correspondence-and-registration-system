from __future__ import annotations

from pathlib import Path
import json

import cv2
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OHRC_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ohrc"
    / "ch2_ohr_ncp_20210405T1606537227_d_img_d18.img"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# OHRC PRODUCT DIMENSIONS
# ============================================================

HEIGHT = 93692
WIDTH = 12000
DTYPE = np.uint8

EXPECTED_BYTES = (
    HEIGHT
    * WIDTH
    * np.dtype(DTYPE).itemsize
)


# ============================================================
# OHRC LOADER
# ============================================================

def load_ohrc_memmap() -> np.memmap:
    """
    Open the calibrated OHRC image using memory mapping.

    The complete ~1.12 GB image is NOT loaded into RAM.
    """

    if not OHRC_PATH.exists():
        raise FileNotFoundError(
            f"OHRC file not found:\n{OHRC_PATH}"
        )

    actual_bytes = OHRC_PATH.stat().st_size

    if actual_bytes != EXPECTED_BYTES:
        raise ValueError(
            "Unexpected OHRC file size:\n"
            f"Actual:   {actual_bytes:,} bytes\n"
            f"Expected: {EXPECTED_BYTES:,} bytes"
        )

    return np.memmap(
        OHRC_PATH,
        dtype=DTYPE,
        mode="r",
        shape=(HEIGHT, WIDTH),
        order="C",
    )


# ============================================================
# CONTRAST NORMALIZATION
# ============================================================

def robust_uint8_preview(
    image: np.ndarray,
) -> np.ndarray:
    """
    Contrast-stretch an image using robust 1st/99th percentiles.
    """

    arr = np.asarray(
        image,
        dtype=np.float32,
    )

    low, high = np.percentile(
        arr,
        [1, 99],
    )

    if high <= low:
        return np.zeros(
            arr.shape,
            dtype=np.uint8,
        )

    normalized = (
        arr - low
    ) / (
        high - low
    )

    normalized = np.clip(
        normalized,
        0.0,
        1.0,
    )

    return (
        normalized
        * 255.0
    ).astype(np.uint8)


# ============================================================
# ASPECT-RATIO-PRESERVING RESIZE
# ============================================================

def resize_preserve_aspect(
    image: np.ndarray,
    max_width: int = 1600,
    max_height: int = 1600,
) -> np.ndarray:
    """
    Resize while preserving the original aspect ratio.
    """

    h, w = image.shape[:2]

    scale = min(
        max_width / max(w, 1),
        max_height / max(h, 1),
        1.0,
    )

    target_w = max(
        1,
        int(round(w * scale)),
    )

    target_h = max(
        1,
        int(round(h * scale)),
    )

    return cv2.resize(
        image,
        (target_w, target_h),
        interpolation=cv2.INTER_AREA,
    )


# ============================================================
# EFFICIENT FULL-SCENE PREVIEW
# ============================================================

def make_thumbnail(
    image: np.ndarray,
    max_width: int = 1600,
    max_height: int = 1600,
) -> np.ndarray:
    """
    Create a correctly proportioned full-scene preview.

    Only a sampled portion of the memmapped image is loaded.
    """

    height, width = image.shape

    # Determine the desired preview dimensions.
    scale = min(
        max_width / width,
        max_height / height,
        1.0,
    )

    target_width = max(
        1,
        int(round(width * scale)),
    )

    target_height = max(
        1,
        int(round(height * scale)),
    )

    # Sample approximately enough pixels for the target preview.
    row_step = max(
        1,
        int(np.ceil(height / target_height)),
    )

    col_step = max(
        1,
        int(np.ceil(width / target_width)),
    )

    sampled = np.asarray(
        image[
            ::row_step,
            ::col_step,
        ]
    )

    # Resize once more to exactly preserve the target dimensions.
    preview = cv2.resize(
        sampled,
        (
            target_width,
            target_height,
        ),
        interpolation=cv2.INTER_AREA,
    )

    return robust_uint8_preview(
        preview
    )


# ============================================================
# ROTATED INSPECTION PREVIEW
# ============================================================

def make_inspection_preview(
    image: np.ndarray,
    max_width: int = 1800,
    max_height: int = 900,
) -> np.ndarray:
    """
    Create a wide, human-friendly inspection preview.

    OHRC is a very long push-broom style scene. Rotating the
    full-scene preview makes the lunar terrain much easier to inspect.
    """

    # First make a compact preview of the original scene.
    preview = make_thumbnail(
        image,
        max_width=max_height,
        max_height=max_width,
    )

    # Rotate 90 degrees clockwise.
    rotated = cv2.rotate(
        preview,
        cv2.ROTATE_90_CLOCKWISE,
    )

    return resize_preserve_aspect(
        rotated,
        max_width=max_width,
        max_height=max_height,
    )


# ============================================================
# SAVE PREVIEWS
# ============================================================

def save_preview(
    preview: np.ndarray,
) -> Path:

    output = (
        OUTPUT_DIR
        / "ohrc_d18_full_preview.png"
    )

    if not cv2.imwrite(
        str(output),
        preview,
    ):
        raise IOError(
            f"Could not write preview:\n{output}"
        )

    return output


def save_inspection_preview(
    preview: np.ndarray,
) -> Path:

    output = (
        OUTPUT_DIR
        / "ohrc_d18_inspection_preview.png"
    )

    if not cv2.imwrite(
        str(output),
        preview,
    ):
        raise IOError(
            "Could not write inspection preview:\n"
            f"{output}"
        )

    return output


# ============================================================
# TILE GENERATION
# ============================================================

def make_tiles(
    image: np.ndarray,
    rows: int = 4,
    cols: int = 4,
    tile_max_width: int = 900,
    tile_max_height: int = 450,
) -> list[Path]:
    """
    Create a 4x4 set of aspect-ratio-preserving OHRC scene tiles.

    Tiles are processed one at a time so the entire OHRC scene
    is never loaded into memory.
    """

    height, width = image.shape
    paths = []

    for r in range(rows):

        y0 = (
            r * height
        ) // rows

        y1 = (
            (r + 1) * height
        ) // rows

        for c in range(cols):

            x0 = (
                c * width
            ) // cols

            x1 = (
                (c + 1) * width
            ) // cols

            tile = np.asarray(
                image[
                    y0:y1,
                    x0:x1,
                ]
            )

            tile = robust_uint8_preview(
                tile
            )

            tile = resize_preserve_aspect(
                tile,
                max_width=tile_max_width,
                max_height=tile_max_height,
            )

            path = (
                OUTPUT_DIR
                / (
                    f"ohrc_d18_tile_"
                    f"r{r + 1}_c{c + 1}.png"
                )
            )

            if not cv2.imwrite(
                str(path),
                tile,
            ):
                raise IOError(
                    f"Could not write tile:\n{path}"
                )

            paths.append(path)

            print(
                f"  Tile "
                f"{r + 1},{c + 1} "
                f"saved"
            )

    return paths


# ============================================================
# CONTACT SHEET
# ============================================================

def make_contact_sheet(
    tile_paths: list[Path],
    rows: int = 4,
    cols: int = 4,
    cell_width: int = 450,
    cell_height: int = 280,
) -> Path:
    """
    Create a labeled 4x4 contact sheet for quickly inspecting
    the OHRC scene and identifying candidate regions.
    """

    sheet = np.zeros(
        (
            rows * cell_height,
            cols * cell_width,
            3,
        ),
        dtype=np.uint8,
    )

    for index, path in enumerate(
        tile_paths
    ):

        tile = cv2.imread(
            str(path),
            cv2.IMREAD_GRAYSCALE,
        )

        if tile is None:
            continue

        tile = resize_preserve_aspect(
            tile,
            max_width=cell_width - 20,
            max_height=cell_height - 50,
        )

        tile_bgr = cv2.cvtColor(
            tile,
            cv2.COLOR_GRAY2BGR,
        )

        r = index // cols
        c = index % cols

        y0 = (
            r * cell_height
            + 35
        )

        x0 = (
            c * cell_width
            + 10
        )

        available_h = (
            cell_height - 45
        )

        available_w = (
            cell_width - 20
        )

        h, w = tile_bgr.shape[:2]

        y1 = min(
            y0 + h,
            (r + 1) * cell_height,
        )

        x1 = min(
            x0 + w,
            (c + 1) * cell_width,
        )

        crop_h = y1 - y0
        crop_w = x1 - x0

        sheet[
            y0:y1,
            x0:x1,
        ] = tile_bgr[
            :crop_h,
            :crop_w,
        ]

        label = (
            f"R{r + 1} C{c + 1}"
        )

        cv2.putText(
            sheet,
            label,
            (
                c * cell_width + 10,
                r * cell_height + 25,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    output = (
        OUTPUT_DIR
        / "ohrc_d18_contact_sheet.png"
    )

    if not cv2.imwrite(
        str(output),
        sheet,
    ):
        raise IOError(
            f"Could not write contact sheet:\n{output}"
        )

    return output


# ============================================================
# SAMPLE STATISTICS
# ============================================================

def compute_statistics(
    image: np.ndarray,
) -> dict:
    """
    Compute statistics from an evenly spaced sample.

    The complete 1.12 GB image is not loaded.
    """

    rows = np.linspace(
        0,
        HEIGHT - 1,
        1000,
        dtype=np.int64,
    )

    cols = np.linspace(
        0,
        WIDTH - 1,
        500,
        dtype=np.int64,
    )

    sample = np.asarray(
        image[
            np.ix_(
                rows,
                cols,
            )
        ]
    )

    values = np.percentile(
        sample,
        [
            0,
            1,
            5,
            25,
            50,
            75,
            95,
            99,
            100,
        ],
    )

    return {
        "shape": [
            HEIGHT,
            WIDTH,
        ],
        "dtype": str(
            np.dtype(DTYPE)
        ),
        "file_size_bytes": (
            OHRC_PATH.stat().st_size
        ),
        "sample_size": list(
            sample.shape
        ),
        "percentiles": {
            "p0": float(values[0]),
            "p1": float(values[1]),
            "p5": float(values[2]),
            "p25": float(values[3]),
            "p50": float(values[4]),
            "p75": float(values[5]),
            "p95": float(values[6]),
            "p99": float(values[7]),
            "p100": float(values[8]),
        },
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 72)
    print(
        "LUCAS | OHRC LOADER / SCENE INSPECTION"
    )
    print("=" * 72)

    # --------------------------------------------------------
    # Validate file
    # --------------------------------------------------------

    if not OHRC_PATH.exists():
        raise FileNotFoundError(
            f"OHRC file not found:\n{OHRC_PATH}"
        )

    actual_size = (
        OHRC_PATH.stat().st_size
    )

    print(
        f"OHRC: {OHRC_PATH}"
    )

    print(
        f"Shape: "
        f"{HEIGHT} x {WIDTH}"
    )

    print(
        f"Dtype: "
        f"{DTYPE}"
    )

    print(
        f"File size: "
        f"{actual_size:,} bytes"
    )

    print(
        f"Expected size: "
        f"{EXPECTED_BYTES:,} bytes"
    )

    # --------------------------------------------------------
    # Memory map
    # --------------------------------------------------------

    print()
    print(
        "Opening memory map..."
    )

    image = load_ohrc_memmap()

    print(
        "Memory map: OK"
    )

    # --------------------------------------------------------
    # Full scene preview
    # --------------------------------------------------------

    print()
    print(
        "Creating full-scene preview..."
    )

    thumbnail = make_thumbnail(
        image
    )

    preview_path = save_preview(
        thumbnail
    )

    print(
        f"Preview saved: "
        f"{preview_path}"
    )

    print(
        f"Preview shape: "
        f"{thumbnail.shape}"
    )

    # --------------------------------------------------------
    # Rotated inspection preview
    # --------------------------------------------------------

    print()
    print(
        "Creating rotated inspection preview..."
    )

    inspection = make_inspection_preview(
        image
    )

    inspection_path = (
        save_inspection_preview(
            inspection
        )
    )

    print(
        f"Inspection preview saved: "
        f"{inspection_path}"
    )

    print(
        f"Inspection shape: "
        f"{inspection.shape}"
    )

    # --------------------------------------------------------
    # Tiles
    # --------------------------------------------------------

    print()
    print(
        "Creating 4 x 4 scene tiles..."
    )

    tile_paths = make_tiles(
        image,
        rows=4,
        cols=4,
    )

    print(
        f"Saved {len(tile_paths)} tiles."
    )

    # --------------------------------------------------------
    # Contact sheet
    # --------------------------------------------------------

    print()
    print(
        "Creating contact sheet..."
    )

    contact_sheet_path = (
        make_contact_sheet(
            tile_paths
        )
    )

    print(
        f"Contact sheet saved: "
        f"{contact_sheet_path}"
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print()
    print(
        "Computing sampled statistics..."
    )

    stats = compute_statistics(
        image
    )

    stats_path = (
        OUTPUT_DIR
        / "ohrc_d18_inspection.json"
    )

    stats_path.write_text(
        json.dumps(
            stats,
            indent=4,
        ),
        encoding="utf-8",
    )

    print(
        f"Statistics saved: "
        f"{stats_path}"
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "OHRC INSPECTION COMPLETE"
    )
    print("=" * 72)

    print()
    print("Inspect these files:")
    print(
        f"  Full preview:      "
        f"{preview_path}"
    )
    print(
        f"  Rotated preview:   "
        f"{inspection_path}"
    )
    print(
        f"  Contact sheet:     "
        f"{contact_sheet_path}"
    )
    print(
        f"  Statistics:        "
        f"{stats_path}"
    )

    print()
    print(
        "The contact sheet is the most useful "
        "file for selecting candidate terrain."
    )


if __name__ == "__main__":
    main()
