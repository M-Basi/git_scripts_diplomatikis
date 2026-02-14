import os
import pandas as pd
import matplotlib.pyplot as plt

IN_CSV = r"D:\Desktop\project\workspace\results\lpips_blur_severity.csv"
OUT_PNG = r"D:\Desktop\project\workspace\results\X5b_LPIPS_vs_sigma.png"

def main():
    df = pd.read_csv(IN_CSV)

    # Plot mean (and optionally median)
    plt.figure()
    for res in sorted(df["Resolution"].unique()):
        d = df[df["Resolution"] == res].sort_values("Sigma")
        plt.plot(d["Sigma"], d["LPIPS_mean"], marker="o", label=f"{res}×{res} (mean)")
        # uncomment if you want median too:
        # plt.plot(d["Sigma"], d["LPIPS_median"], marker="x", linestyle="--", label=f"{res}×{res} (median)")

    plt.xlabel("Gaussian blur σ")
    plt.ylabel("LPIPS (perceptual distance)")
    plt.title("LPIPS ως συνάρτηση του σ (Clean vs Blur)")
    plt.legend()
    plt.grid(True)

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {OUT_PNG}")

if __name__ == "__main__":
    main()
