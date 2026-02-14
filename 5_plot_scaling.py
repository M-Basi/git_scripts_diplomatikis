# plot_scaling.py
# ------------------------------------------------------------
# Plot 1: Scaling vs Localization Accuracy
# Uses: dert_scaling.csv
# Outputs: scaling_ap75_aps.png
# ------------------------------------------------------------

import os
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Paths
# =========================
CSV_PATH = "D:/Desktop/project/PaddleDetection/dert_scaling.csv"
OUT_DIR = "D:/Desktop/project/figures"
OUT_FILE = "scaling_ap75_aps.png"

os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# Load data
# =========================
df = pd.read_csv(CSV_PATH)

# Ensure sorting by image size
df = df.sort_values("Image_Size")

sizes = df["Image_Size"]
ap75 = df["AP75"]
ap_s = df["AP_S"]

# =========================
# Plot
# =========================
plt.figure(figsize=(7, 5))

plt.plot(
    sizes,
    ap75,
    marker="o",
    linewidth=2,
    label="AP75 (Localization accuracy)"
)

plt.plot(
    sizes,
    ap_s,
    marker="s",
    linewidth=2,
    label="AP$_S$ (Small objects)"
)

plt.xlabel("Input Image Resolution")
plt.ylabel("Average Precision")
plt.xticks(sizes)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()

# ✅ TITLE (this is what was missing)
plt.title(
    f"Effect of Input Resolution on Localization and Small-Object Accuracy)",
    fontsize=12
)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, OUT_FILE), dpi=300)
plt.close()

print("✅ Saved figure to:")
print(os.path.join(OUT_DIR, OUT_FILE))
