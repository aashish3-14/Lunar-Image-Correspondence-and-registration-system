from pyproj import CRS, Transformer

moon_geographic = CRS.from_proj4(
    "+proj=longlat "
    "+R=1737400 "
    "+no_defs"
)

lro_projection = CRS.from_proj4(
    "+proj=eqc "
    "+R=1737400 "
    "+lat_ts=15.47 "
    "+lon_0=-9.01 "
    "+x_0=0 "
    "+y_0=0"
)

transformer = Transformer.from_crs(
    moon_geographic,
    lro_projection,
    always_xy=True,
)

lon = 15.47
lat = -9.01

x, y = transformer.transform(lon, lat)

print("Projected:")
print("x =", x)
print("y =", y)

print("\nTiepoint:")
print("x =", 706695.0)
print("y =", -250500.0)

print("\nDifference:")
print("dx =", x - 706695.0)
print("dy =", y - (-250500.0))