import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# Paths
# =========================================================

LRO_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "lro_apollo16"
    / "NAC_ROI_APOLLO16LOA_E090S0155_5M.TIF"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lro_apollo16_geographic_crop.png"
)


# =========================================================
# Load LRO
# =========================================================

print("Loading LRO image...")

lro = cv2.imread(str(LRO_PATH), cv2.IMREAD_GRAYSCALE)

if lro is None:
    raise RuntimeError("Could not load LRO image.")

height, width = lro.shape

print(f"LRO shape: {lro.shape}")


# =========================================================
# LRO geographic footprint
# =========================================================
#
# Approximate footprint from the LRO Apollo-16 product.
#
# Latitude:
#   -9.76 to -8.26
#
# Longitude:
#   15.17 to 15.80
#
# We assume the image is approximately:
#
#   top    = north (-8.26)
#   bottom = south (-9.76)
#   left   = west (15.17)
#   right  = east (15.80)
#
# This is ONLY a coarse geographic mapping.
# =========================================================

LRO_LAT_MIN = -9.76
LRO_LAT_MAX = -8.26

LRO_LON_MIN = 15.17
LRO_LON_MAX = 15.80


# =========================================================
# Geographic extent of our TMC focused crop
# =========================================================

TMC_LAT_MIN = -9.65133
TMC_LAT_MAX = -8.35248

TMC_LON_MIN = 15.22764
TMC_LON_MAX = 15.86945


# =========================================================
# Calculate geographic intersection
# =========================================================

overlap_lat_min = max(TMC_LAT_MIN, LRO_LAT_MIN)
overlap_lat_max = min(TMC_LAT_MAX, LRO_LAT_MAX)

overlap_lon_min = max(TMC_LON_MIN, LRO_LON_MIN)
overlap_lon_max = min(TMC_LON_MAX, LRO_LON_MAX)

print("\nGeographic intersection:")
print(f"Latitude : {overlap_lat_min:.5f} to {overlap_lat_max:.5f}")
print(f"Longitude: {overlap_lon_min:.5f} to {overlap_lon_max:.5f}")


# =========================================================
# Convert geographic coordinates -> LRO pixels
# =========================================================

def lon_to_x(lon):
    fraction = (
        (lon - LRO_LON_MIN)
        / (LRO_LON_MAX - LRO_LON_MIN)
    )
    return fraction * (width - 1)


def lat_to_y(lat):
    # North is at the top of the image.
    fraction = (
        (LRO_LAT_MAX - lat)
        / (LRO_LAT_MAX - LRO_LAT_MIN)
    )
    return fraction * (height - 1)


x1 = lon_to_x(overlap_lon_min)
x2 = lon_to_x(overlap_lon_max)

y1 = lat_to_y(overlap_lat_max)
y2 = lat_to_y(overlap_lat_min)


x_min = max(0, int(np.floor(min(x1, x2))))
x_max = min(width, int(np.ceil(max(x1, x2))))

y_min = max(0, int(np.floor(min(y1, y2))))
y_max = min(height, int(np.ceil(max(y1, y2))))


print("\nLRO pixel crop:")
print(f"x: {x_min} to {x_max}")
print(f"y: {y_min} to {y_max}")


# =========================================================
# Extract crop
# =========================================================

lro_crop = lro[y_min:y_max, x_min:x_max]

print(f"\nCrop shape: {lro_crop.shape}")


# =========================================================
# Save
# =========================================================

cv2.imwrite(str(OUTPUT_PATH), lro_crop)

print(f"\nSaved raw crop:")
print(OUTPUT_PATH)


# =========================================================
# Visualization
# =========================================================

plt.figure(figsize=(10, 10))

plt.imshow(lro_crop, cmap="gray")
plt.title(
    "LRO Apollo-16\n"
    "Geographically Corresponding Region"
)

plt.axis("off")
plt.tight_layout()

plt.savefig(
    str(
        PROJECT_ROOT
        / "data"
        / "processed"
        / "lro_apollo16_geographic_crop_preview.png"
    ),
    dpi=150
)

plt.close()

print("Saved preview.")

print("\n" + "=" * 50)
print("GEOGRAPHIC LRO CROP COMPLETE")
print("=" * 50)