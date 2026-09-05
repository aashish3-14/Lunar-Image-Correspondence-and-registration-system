import numpy as np

# LRO raster
width = 3658
height = 9093

# Published LRO footprint
lon_west = 15.17
lon_east = 15.80

lat_south = -9.76
lat_north = -8.26

# TMC focused crop corners
corners = {
    "UL": (-8.35377, 15.86945),
    "UR": (-8.35248, 15.23595),
    "LL": (-9.65133, 15.86260),
    "LR": (-9.64996, 15.22764),
}

print("Approximate LRO pixel mapping from published footprint")
print("=" * 65)

for name, (lat, lon) in corners.items():

    x = (lon - lon_west) / (lon_east - lon_west) * (width - 1)

    # latitude decreases as raster row increases
    y = (lat_north - lat) / (lat_north - lat_south) * (height - 1)

    print(f"\n{name}")
    print(f"  lat/lon : {lat:.6f}, {lon:.6f}")
    print(f"  pixel   : x={x:.1f}, y={y:.1f}")
