from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OHRC_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ohrc"
    / "ch2_ohr_ncp_20210405T1606537227_d_img_d18.img"
)

H = 93692
W = 12000

img = np.memmap(
    OHRC_PATH,
    dtype=np.uint8,
    mode="r",
    shape=(H, W),
)

print("=" * 70)
print("LUCAS | OHRC DATA DIAGNOSTIC")
print("=" * 70)

print(f"Shape: {img.shape}")
print(f"Dtype: {img.dtype}")
print(f"File size: {OHRC_PATH.stat().st_size:,}")

# ------------------------------------------------------------
# Sample the image
# ------------------------------------------------------------

rows = np.linspace(0, H - 1, 1000, dtype=np.int64)
cols = np.linspace(0, W - 1, 1200, dtype=np.int64)

sample = np.asarray(img[np.ix_(rows, cols)])

print("\nGLOBAL SAMPLE STATISTICS")
print("-" * 70)

print(f"Min:    {sample.min()}")
print(f"Max:    {sample.max()}")
print(f"Mean:   {sample.mean():.3f}")
print(f"Median: {np.median(sample):.3f}")

print("\nPercentiles:")
for p in [0, 0.1, 1, 5, 25, 50, 75, 95, 99, 99.9, 100]:
    print(f"  P{p:5}: {np.percentile(sample, p):.3f}")

# ------------------------------------------------------------
# Non-zero analysis
# ------------------------------------------------------------

nonzero = sample > 0

print("\nNON-ZERO ANALYSIS")
print("-" * 70)

print(
    f"Non-zero pixels: "
    f"{nonzero.sum():,} / {sample.size:,}"
)

print(
    f"Non-zero fraction: "
    f"{nonzero.mean():.6f}"
)

# ------------------------------------------------------------
# Column activity
# ------------------------------------------------------------

column_activity = np.mean(sample > 0, axis=0)

active_columns = np.where(column_activity > 0.01)[0]

print("\nCOLUMN ACTIVITY")
print("-" * 70)

print(
    f"Columns with >1% non-zero pixels: "
    f"{len(active_columns)}"
)

if len(active_columns) > 0:
    print(
        f"Sampled active-column range: "
        f"{active_columns.min()} → {active_columns.max()}"
    )

    # Convert sampled column indices back approximately
    scale = (W - 1) / (len(cols) - 1)

    print(
        f"Approximate original range: "
        f"{int(active_columns.min() * scale)} → "
        f"{int(active_columns.max() * scale)}"
    )

# ------------------------------------------------------------
# Row activity
# ------------------------------------------------------------

row_activity = np.mean(sample > 0, axis=1)

active_rows = np.where(row_activity > 0.01)[0]

print("\nROW ACTIVITY")
print("-" * 70)

print(
    f"Rows with >1% non-zero pixels: "
    f"{len(active_rows)}"
)

if len(active_rows) > 0:
    print(
        f"Sampled active-row range: "
        f"{active_rows.min()} → {active_rows.max()}"
    )

# ------------------------------------------------------------
# Per-region statistics
# ------------------------------------------------------------

print("\n4 × 4 REGION NON-ZERO FRACTION")
print("-" * 70)

for r in range(4):
    values = []

    y0 = (r * H) // 4
    y1 = ((r + 1) * H) // 4

    sampled_rows = np.linspace(
        y0,
        y1 - 1,
        100,
        dtype=np.int64,
    )

    for c in range(4):
        x0 = (c * W) // 4
        x1 = ((c + 1) * W) // 4

        sampled_cols = np.linspace(
            x0,
            x1 - 1,
            100,
            dtype=np.int64,
        )

        region = np.asarray(
            img[np.ix_(sampled_rows, sampled_cols)]
        )

        values.append(
            f"{np.mean(region > 0):.3f}"
        )

    print(
        f"Row {r + 1}: "
        + " | ".join(values)
    )

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)