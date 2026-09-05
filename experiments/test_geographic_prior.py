from pathlib import Path

import cv2
import numpy as np


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
# TMC FULL IMAGE METADATA
# ============================================================

TMC_HEIGHT = 147326
TMC_WIDTH = 4000


TMC_CORNERS = {
    "UL": (7.054831, 15.970056),
    "UR": (7.055161, 15.199975),
    "LL": (-16.840721, 15.844733),
    "LR": (-16.838414, 15.041161),
}


# Focused crop inside TMC
TMC_CROP = {
    "row0": 95000,
    "row1": 103000,
    "col0": 100,
    "col1": 3300,
}


# ============================================================
# LRO PUBLISHED FOOTPRINT
# ============================================================

LRO_LAT_MIN = -9.76
LRO_LAT_MAX = -8.26

LRO_LON_MIN = 15.17
LRO_LON_MAX = 15.80


# ============================================================
# LOAD
# ============================================================

def load_image(path):

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
# BILINEAR CORNER INTERPOLATION
# ============================================================

def interpolate_corners(
    u,
    v,
):

    """
    Bilinear interpolation over
    the four TMC geographic corners.

    u = horizontal normalized coordinate
    v = vertical normalized coordinate
    """

    ul_lat, ul_lon = TMC_CORNERS["UL"]
    ur_lat, ur_lon = TMC_CORNERS["UR"]
    ll_lat, ll_lon = TMC_CORNERS["LL"]
    lr_lat, lr_lon = TMC_CORNERS["LR"]

    lat = (
        (1 - u) * (1 - v) * ul_lat
        + u * (1 - v) * ur_lat
        + (1 - u) * v * ll_lat
        + u * v * lr_lat
    )

    lon = (
        (1 - u) * (1 - v) * ul_lon
        + u * (1 - v) * ur_lon
        + (1 - u) * v * ll_lon
        + u * v * lr_lon
    )

    return lat, lon


# ============================================================
# TMC PIXEL → LAT/LON
# ============================================================

def tmc_pixel_to_latlon(
    row,
    col,
):

    u = (
        col
        /
        (TMC_WIDTH - 1)
    )

    v = (
        row
        /
        (TMC_HEIGHT - 1)
    )

    return interpolate_corners(
        u,
        v,
    )


# ============================================================
# LRO LAT/LON → PIXEL
# ============================================================

def latlon_to_lro_pixel(
    lat,
    lon,
    image_shape,
):

    height, width = image_shape

    u = (
        (lon - LRO_LON_MIN)
        /
        (
            LRO_LON_MAX
            -
            LRO_LON_MIN
        )
    )

    v = (
        (LRO_LAT_MAX - lat)
        /
        (
            LRO_LAT_MAX
            -
            LRO_LAT_MIN
        )
    )

    x = (
        u
        *
        (width - 1)
    )

    y = (
        v
        *
        (height - 1)
    )

    return x, y
# ============================================================
# CROP CORNERS
# ============================================================

def print_crop_geography():

    print()
    print(
        "=" * 65
    )

    print(
        "TMC FOCUSED CROP GEOGRAPHIC PRIOR"
    )

    print(
        "=" * 65
    )

    points = {
        "UL": (
            TMC_CROP["row0"],
            TMC_CROP["col0"],
        ),

        "UR": (
            TMC_CROP["row0"],
            TMC_CROP["col1"] - 1,
        ),

        "LL": (
            TMC_CROP["row1"] - 1,
            TMC_CROP["col0"],
        ),

        "LR": (
            TMC_CROP["row1"] - 1,
            TMC_CROP["col1"] - 1,
        ),

        "CENTER": (
            (
                TMC_CROP["row0"]
                +
                TMC_CROP["row1"]
            ) // 2,

            (
                TMC_CROP["col0"]
                +
                TMC_CROP["col1"]
            ) // 2,
        ),
    }

    for name, (
        row,
        col,
    ) in points.items():

        lat, lon = (
            tmc_pixel_to_latlon(
                row,
                col,
            )
        )

        print(
            f"{name:>6}: "
            f"pixel=({col}, {row}) "
            f"lat={lat:.6f} "
            f"lon={lon:.6f}"
        )


# ============================================================
# MAP CROP TO LRO
# ============================================================

def map_crop_to_lro(
    reference,
):

    print()
    print(
        "=" * 65
    )

    print(
        "PROJECTED LRO POSITIONS"
    )

    print(
        "=" * 65
    )

    points = {
        "UL": (
            TMC_CROP["row0"],
            TMC_CROP["col0"],
        ),

        "UR": (
            TMC_CROP["row0"],
            TMC_CROP["col1"] - 1,
        ),

        "LL": (
            TMC_CROP["row1"] - 1,
            TMC_CROP["col0"],
        ),

        "LR": (
            TMC_CROP["row1"] - 1,
            TMC_CROP["col1"] - 1,
        ),

        "CENTER": (
            (
                TMC_CROP["row0"]
                +
                TMC_CROP["row1"]
            ) // 2,

            (
                TMC_CROP["col0"]
                +
                TMC_CROP["col1"]
            ) // 2,
        ),
    }

    for name, (
        row,
        col,
    ) in points.items():

        lat, lon = (
            tmc_pixel_to_latlon(
                row,
                col,
            )
        )

        x, y = (
            latlon_to_lro_pixel(
                lat,
                lon,
                reference.shape,
            )
        )

        print(
            f"{name:>6}: "
            f"lat={lat:.6f} "
            f"lon={lon:.6f} "
            f"→ LRO "
            f"pixel=({x:.1f}, {y:.1f})"
        )


# ============================================================
# COVERAGE CHECK
# ============================================================

def check_overlap():

    print()
    print(
        "=" * 65
    )

    print(
        "OVERLAP CHECK"
    )

    print(
        "=" * 65
    )

    lats = []
    lons = []

    for row, col in [
        (
            TMC_CROP["row0"],
            TMC_CROP["col0"],
        ),
        (
            TMC_CROP["row0"],
            TMC_CROP["col1"] - 1,
        ),
        (
            TMC_CROP["row1"] - 1,
            TMC_CROP["col0"],
        ),
        (
            TMC_CROP["row1"] - 1,
            TMC_CROP["col1"] - 1,
        ),
    ]:

        lat, lon = (
            tmc_pixel_to_latlon(
                row,
                col,
            )
        )

        lats.append(lat)
        lons.append(lon)

    print(
        f"TMC crop latitude range: "
        f"{min(lats):.6f} "
        f"to "
        f"{max(lats):.6f}"
    )

    print(
        f"TMC crop longitude range: "
        f"{min(lons):.6f} "
        f"to "
        f"{max(lons):.6f}"
    )

    print()

    lat_overlap = (
        min(lats) <= LRO_LAT_MAX
        and
        max(lats) >= LRO_LAT_MIN
    )

    lon_overlap = (
        min(lons) <= LRO_LON_MAX
        and
        max(lons) >= LRO_LON_MIN
    )

    print(
        "Latitude overlap:",
        lat_overlap,
    )

    print(
        "Longitude overlap:",
        lon_overlap,
    )

    print(
        "Geographic overlap:",
        lat_overlap and lon_overlap,
    )


# ============================================================
# MAIN
# ============================================================

def run():

    print()
    print(
        "LUCAS GEOGRAPHIC PRIOR EXPERIMENT"
    )

    source = load_image(
        SOURCE
    )

    reference = load_image(
        REFERENCE
    )

    print()
    print(
        "Source:",
        source.shape,
    )

    print(
        "Reference:",
        reference.shape,
    )

    print_crop_geography()

    map_crop_to_lro(
        reference
    )

    check_overlap()

    print()
    print(
        "=" * 65
    )

    print(
        "EXPERIMENT COMPLETE"
    )

    print(
        "=" * 65
    )


if __name__ == "__main__":

    run()