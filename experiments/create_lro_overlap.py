import cv2
import os

LRO_PATH = r"data/raw/lro_apollo16/NAC_ROI_APOLLO16LOA_E090S0155_5M.TIF"
OUT_PATH = r"data/processed/lro_apollo16_overlap.png"

img = cv2.imread(LRO_PATH, cv2.IMREAD_GRAYSCALE)

# Approximate overlap from the published footprint
x1 = 335
x2 = 3657
y1 = 560
y2 = 8434

crop = img[y1:y2, x1:x2]

cv2.imwrite(OUT_PATH, crop)

print("Original:", img.shape)
print("Crop:", crop.shape)
print("Saved:", OUT_PATH)