from PIL import Image
from PIL.TiffTags import TAGS

path = r"data/raw/lro_apollo16/NAC_ROI_APOLLO16LOA_E090S0155_5M.TIF"

img = Image.open(path)

print("GeoTIFF relevant tags")
print("=" * 70)

for tag_id, value in img.tag_v2.items():

    if tag_id in [33550, 33922, 34735, 34736, 34737]:
        print(f"\nTag {tag_id} ({TAGS.get(tag_id, 'Unknown')})")
        print(value)