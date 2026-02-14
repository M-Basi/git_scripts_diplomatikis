# ------------------------------------------------------------
# Plot: Blur Invariance (AP75 & AP_S) for 640 and 800
# ------------------------------------------------------------

import os
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "D:/Desktop/project/PaddleDetection/dert_blur1.csv"
OUT_DIR = "D:/Desktop/project/figures"
OUT_FILE = "blur_ap75_aps_640_800.png"

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)
df = df[df["Image_Size"].isin([640, 800])]
df = df.sort_values(["Image_Size", "BlurSigma"])

df_640 = df[df["Image_Size"] == 640]
df_800 = df[df["Image_Size"] == 800]

plt.figure(figsize=(8, 5))

# AP75
plt.plot(df_640["BlurSigma"], df_640["AP75"], "o-", label="640 AP75")
plt.plot(df_800["BlurSigma"], df_800["AP75"], "s-", label="800 AP75")

# AP_S
plt.plot(df_640["BlurSigma"], df_640["AP_S"], "o--", label="640 AP_S")
plt.plot(df_800["BlurSigma"], df_800["AP_S"], "s--", label="800 AP_S")

plt.xlabel("Gaussian Blur σ (low-pass strength)")
plt.ylabel("Average Precision")
plt.xticks([0, 1, 2 , 3, 4, 5])
plt.grid(True, linestyle="--", alpha=0.5)

plt.title(
    "Effect of Gaussian Blur on Localization and Small-Object Accuracy",
    fontsize=12
)

plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, OUT_FILE), dpi=300)
plt.close()

print("✅ Saved figure to:")
print(os.path.join(OUT_DIR, OUT_FILE))
