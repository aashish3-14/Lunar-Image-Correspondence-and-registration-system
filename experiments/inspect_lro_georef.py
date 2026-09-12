from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

LRO_PATH = (
    ROOT
    / "data"
    / "raw"
    / "lro_apollo16"
    / "NAC_ROI_APOLLO16LOA_E090S0155_5M.TIF"
)

print("LRO:", LRO_PATH)
print()

with Image.open(LRO_PATH) as img:

    print("Image size:", img.size)
    print("Image mode:", img.mode)
    print()

    print("TIFF tags:")
    print("--------------------------------")

    for tag_id, value in img.tag_v2.items():

        print(
            f"Tag {tag_id}: {value}"
        )