# plot_blur_invariance.py
# ------------------------------------------------------------
# Plot 3: Blur invariance (low-pass) sensitivity
# Uses: dert_blur.csv
# Outputs:
#   - blur_invariance_640_titled.png
#   - blur_invariance_800_titled.png
# ------------------------------------------------------------

import os
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Paths
# =========================
CSV_PATH = "D:/Desktop/project/PaddleDetection/dert_blur1.csv"
OUT_DIR = "D:/Desktop/project/figures"

os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# Load data
# =========================
df = pd.read_csv(CSV_PATH)

# Required columns
required = ["Image_Size", "BlurSigma", "AP75", "AP_S"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise KeyError(
        f"Missing columns in CSV: {missing}\nAvailable columns: {df.columns.tolist()}"
    )

# Ensure numeric sigma
df["BlurSigma"] = df["BlurSigma"].astype(float)

# We plot for these two resolutions
sizes_to_plot = [640, 800]

def make_plot(size: int):
    sub = df[df["Image_Size"] == size].copy()
    if sub.empty:
        raise ValueError(f"No rows found for Image_Size={size} in {CSV_PATH}")

    # Sort by sigma so line is ordered
    sub = sub.sort_values("BlurSigma")

    sigmas = sub["BlurSigma"].to_list()
    ap75 = sub["AP75"].to_list()
    ap_s = sub["AP_S"].to_list()

    plt.figure(figsize=(7, 5))

    plt.plot(
        sigmas, ap75,
        marker="o", linewidth=2,
        label="AP75 (Localization accuracy)"
    )
    plt.plot(
        sigmas, ap_s,
        marker="s", linewidth=2,
        label="AP$_S$ (Small objects)"
    )

    plt.xlabel("Gaussian Blur Sigma (low-pass strength)")
    plt.ylabel("Average Precision")
    plt.xticks(sigmas)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower left")

    # ✅ TITLE (this is what was missing)
    plt.title(
        f"Blur Invariance Analysis (Input Resolution: {size}×{size})",
        fontsize=12
    )

    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, f"blur_invariance_{size}_titled.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print("✅ Saved:", out_path)

# Generate plots
for s in sizes_to_plot:
    make_plot(s)

print("\nDone.")
