import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import matplotlib.pyplot as plt

from ai_engine.structural import gradient_representation
from ai_engine.registration import ecc_affine_registration


TMC_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "tmc2_apollo16_focused.png"
)

LRO_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "lro_apollo16_geographic_crop.png"
)


OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ecc_registration.png"
)


print("=" * 55)
print("LUCAS COARSE-TO-FINE REGISTRATION")
print("=" * 55)


# ---------------------------------------------------------
# Load
# ---------------------------------------------------------

tmc = cv2.imread(
    str(TMC_PATH),
    cv2.IMREAD_GRAYSCALE
)

lro = cv2.imread(
    str(LRO_PATH),
    cv2.IMREAD_GRAYSCALE
)

if tmc is None:
    raise RuntimeError("Could not load TMC.")

if lro is None:
    raise RuntimeError("Could not load LRO.")


print(f"TMC: {tmc.shape}")
print(f"LRO: {lro.shape}")


# ---------------------------------------------------------
# Structural representations
# ---------------------------------------------------------

print("\nCreating structural representations...")

tmc_structure = gradient_representation(tmc)
lro_structure = gradient_representation(lro)


# ---------------------------------------------------------
# Resize moving image to fixed-image size
# ---------------------------------------------------------

print("\nPreparing images for ECC...")

tmc_resized = cv2.resize(
    tmc_structure,
    (lro_structure.shape[1], lro_structure.shape[0]),
    interpolation=cv2.INTER_AREA
)


# ---------------------------------------------------------
# ECC registration
# ---------------------------------------------------------

print("\nRunning ECC affine registration...")

try:

    registered, warp_matrix, correlation = (
        ecc_affine_registration(
            tmc_resized,
            lro_structure
        )
    )

except cv2.error as error:

    print("\nECC registration failed:")
    print(error)

    raise SystemExit


print("\nECC result:")
print(f"Correlation coefficient: {correlation:.6f}")

print("\nAffine transformation:")
print(warp_matrix)


# ---------------------------------------------------------
# Save registered image
# ---------------------------------------------------------

registered_path = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "tmc2_ecc_registered.png"
)

cv2.imwrite(
    str(registered_path),
    registered
)

print(
    f"\nSaved registered image:\n"
    f"{registered_path}"
)


# ---------------------------------------------------------
# Visualization
# ---------------------------------------------------------

plt.figure(figsize=(18, 8))

plt.subplot(1, 3, 1)

plt.imshow(
    tmc_resized,
    cmap="gray"
)

plt.title("TMC Structural")

plt.axis("off")


plt.subplot(1, 3, 2)

plt.imshow(
    lro_structure,
    cmap="gray"
)

plt.title("LRO Structural")

plt.axis("off")


plt.subplot(1, 3, 3)

plt.imshow(
    registered,
    cmap="gray"
)

plt.title(
    f"ECC Registered\n"
    f"Correlation = {correlation:.4f}"
)

plt.axis("off")


plt.tight_layout()

plt.savefig(
    str(OUTPUT_PATH),
    dpi=150
)

plt.close()


print(
    f"\nSaved visualization:\n"
    f"{OUTPUT_PATH}"
)


print("\n" + "=" * 55)
print("ECC REGISTRATION COMPLETE")
print("=" * 55)