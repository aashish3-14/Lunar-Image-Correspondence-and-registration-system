from pyproj import CRS, Transformer

# Lunar geographic CRS
moon_geographic = CRS.from_proj4(
    "+proj=longlat "
    "+R=1737400 "
    "+no_defs"
)

# LRO Equirectangular projection
lro_projection = CRS.from_proj4(
    "+proj=eqc "
    "+R=1737400 "
    "+lat_ts=-9.01 "
    "+lon_0=15.47 "
    "+x_0=0 "
    "+y_0=0"
)

transformer = Transformer.from_crs(
    moon_geographic,
    lro_projection,
    always_xy=True,
)

# LRO GeoTIFF georeferencing
tie_x = 706695.0
tie_y = -250500.0
pixel_size = 5.0

# TMC focused crop corners
corners = {
    "UL": (-8.35377, 15.86945),
    "UR": (-8.35248, 15.23595),
    "LL": (-9.65133, 15.86260),
    "LR": (-9.64996, 15.22764),
}

print("TMC → LRO pixel mapping")
print("=" * 60)

for name, (lat, lon) in corners.items():

    x, y = transformer.transform(lon, lat)

    pixel_x = (x - tie_x) / pixel_size
    pixel_y = (y - tie_y) / pixel_size

    print(f"\n{name}")
    print(f"  lat/lon       : {lat:.6f}, {lon:.6f}")
    print(f"  projected     : {x:.3f}, {y:.3f}")
    print(f"  LRO pixel     : {pixel_x:.3f}, {pixel_y:.3f}")