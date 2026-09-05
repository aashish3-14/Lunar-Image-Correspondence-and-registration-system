import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# TMC-2 metadata
# ---------------------------------------------------------

TMC_ROWS = 147326
TMC_COLS = 4000

# Order:
# UL, UR, LL, LR
TMC_CORNERS = np.array([
    [7.054831, 15.970056],
    [7.055161, 15.199975],
    [-16.840721, 15.844733],
    [-16.838414, 15.041161],
])


# ---------------------------------------------------------
# LRO Apollo 16 5m mosaic
# ---------------------------------------------------------

LRO_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "lro_apollo16"
    / "NAC_ROI_APOLLO16LOA_E090S0155_5M.TIF"
)

lro = cv2.imread(str(LRO_PATH), cv2.IMREAD_GRAYSCALE)

if lro is None:
    raise RuntimeError("Could not load LRO image.")

LRO_H, LRO_W = lro.shape

print("=" * 50)
print("GEOGRAPHIC CORRESPONDENCE CHECK")
print("=" * 50)

print(f"LRO shape: {lro.shape}")


# ---------------------------------------------------------
# Bilinear geographic interpolation
# ---------------------------------------------------------

def tmc_pixel_to_latlon(row, col):
    """
    Approximate TMC pixel -> latitude/longitude
    using the four metadata corner coordinates.

    This is intentionally a coarse geographic prior,
    NOT a final spacecraft imaging model.
    """

    u = col / (TMC_COLS - 1)
    v = row / (TMC_ROWS - 1)

    ul = TMC_CORNERS[0]
    ur = TMC_CORNERS[1]
    ll = TMC_CORNERS[2]
    lr = TMC_CORNERS[3]

    top = (1 - u) * ul + u * ur
    bottom = (1 - u) * ll + u * lr

    point = (1 - v) * top + v * bottom

    return point[0], point[1]


# ---------------------------------------------------------
# Inspect our focused crop
# ---------------------------------------------------------

crop_start_row = 95000
crop_end_row = 103000

crop_start_col = 100
crop_end_col = 3300

points = [
    ("top-left", crop_start_row, crop_start_col),
    ("top-right", crop_start_row, crop_end_col),
    ("bottom-left", crop_end_row, crop_start_col),
    ("bottom-right", crop_end_row, crop_end_col),
]


print("\nFocused TMC crop geographic extent:")

for name, row, col in points:
    lat, lon = tmc_pixel_to_latlon(row, col)

    print(
        f"{name:12s} "
        f"row={row:6d} col={col:4d} "
        f"lat={lat:9.5f} "
        f"lon={lon:9.5f}"
    )


# ---------------------------------------------------------
# LRO geographic footprint
# ---------------------------------------------------------

# Apollo 16 LRO mosaic approximate footprint
LRO_LAT_MIN = -9.76
LRO_LAT_MAX = -8.26

LRO_LON_MIN = 15.17
LRO_LON_MAX = 15.80


print("\nLRO Apollo-16 footprint:")
print(f"Latitude : {LRO_LAT_MIN} to {LRO_LAT_MAX}")
print(f"Longitude: {LRO_LON_MIN} to {LRO_LON_MAX}")


# ---------------------------------------------------------
# Check overlap
# ---------------------------------------------------------

crop_lats = []
crop_lons = []

for _, row, col in points:
    lat, lon = tmc_pixel_to_latlon(row, col)
    crop_lats.append(lat)
    crop_lons.append(lon)

crop_lat_min = min(crop_lats)
crop_lat_max = max(crop_lats)

crop_lon_min = min(crop_lons)
crop_lon_max = max(crop_lons)

lat_overlap = (
    max(crop_lat_min, LRO_LAT_MIN)
    <= min(crop_lat_max, LRO_LAT_MAX)
)

lon_overlap = (
    max(crop_lon_min, LRO_LON_MIN)
    <= min(crop_lon_max, LRO_LON_MAX)
)

print("\nOverlap test:")
print(f"Latitude overlap : {lat_overlap}")
print(f"Longitude overlap: {lon_overlap}")

if lat_overlap and lon_overlap:
    print("\nRESULT: Geographic overlap exists.")
else:
    print("\nRESULT: No geographic overlap.")


# ---------------------------------------------------------
# Save simple visualization
# ---------------------------------------------------------

plt.figure(figsize=(8, 10))
plt.imshow(lro, cmap="gray")

plt.title(
    "LRO Apollo-16 Reference\n"
    "Geographic footprint available"
)

plt.axis("off")

output = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lro_geographic_check.png"
)

plt.tight_layout()
plt.savefig(output, dpi=150)
plt.close()

print(f"\nSaved: {output}")
print("=" * 50)