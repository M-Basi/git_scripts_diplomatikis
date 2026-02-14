# 10_plot_lpips_vs_metrics_FIXED_PATHS.py
# ------------------------------------------------------------
# Reads:
#   - D:\Desktop\project\workspace\results\dert_blur.csv
#   - D:\Desktop\project\workspace\results\lpips_blur_severity.csv
# Produces:
#   - plots_lpips\LPIPS_vs_AP75.png
#   - plots_lpips\LPIPS_vs_AP_S.png
#   - plots_lpips\LPIPS_vs_AR100.png
#   - plots_lpips\merged_lpips_metrics.csv
# ------------------------------------------------------------

import os
import pandas as pd
import matplotlib.pyplot as plt


METRICS_CSV = r"D:\Desktop\project\workspace\results\dert_blur.csv"
LPIPS_CSV   = r"D:\Desktop\project\workspace\results\lpips_blur_severity.csv"
OUT_DIR     = r"D:\Desktop\project\workspace\results\plots_lpips"


def _guess_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of these columns exist: {candidates}. Found: {list(df.columns)}")


def load_lpips_csv(path: str) -> pd.DataFrame:
    lp = pd.read_csv(path)

    col_size  = _guess_col(lp, ["Image_Size", "Resolution", "Res", "image_size"])
    col_sigma = _guess_col(lp, ["BlurSigma", "sigma", "Sigma", "blur_sigma"])
    col_mean  = _guess_col(lp, ["LPIPS_mean", "mean", "lpips_mean", "LPIPSMean"])

    # median optional
    col_median = None
    for c in ["LPIPS_median", "median", "lpips_median", "LPIPSMedian"]:
        if c in lp.columns:
            col_median = c
            break

    out = lp.rename(columns={
        col_size:  "Image_Size",
        col_sigma: "BlurSigma",
        col_mean:  "LPIPS_mean",
    }).copy()

    if col_median:
        out = out.rename(columns={col_median: "LPIPS_median"})

    out["Image_Size"] = out["Image_Size"].astype(int)
    out["BlurSigma"] = out["BlurSigma"].astype(float)
    out["LPIPS_mean"] = out["LPIPS_mean"].astype(float)
    if "LPIPS_median" in out.columns:
        out["LPIPS_median"] = out["LPIPS_median"].astype(float)

    return out


def load_metrics_csv(path: str) -> pd.DataFrame:
    m = pd.read_csv(path)

    col_size  = _guess_col(m, ["Image_Size", "image_size", "Size", "Resolution"])
    col_sigma = _guess_col(m, ["BlurSigma", "sigma", "Sigma", "blur_sigma"])
    col_ap75  = _guess_col(m, ["AP75", "ap75"])
    col_aps   = _guess_col(m, ["AP_S", "APs", "ap_s", "AP_small"])
    col_ar100 = _guess_col(m, ["AR@100", "AR100", "ar@100", "ar100"])

    out = m.rename(columns={
        col_size:  "Image_Size",
        col_sigma: "BlurSigma",
        col_ap75:  "AP75",
        col_aps:   "AP_S",
        col_ar100: "AR@100",
    }).copy()

    out["Image_Size"] = out["Image_Size"].astype(int)
    out["BlurSigma"] = out["BlurSigma"].astype(float)

    keep = ["Image_Size", "BlurSigma", "AP75", "AP_S", "AR@100"]
    # keep extras if exist (won't hurt)
    for extra in ["AP50", "AP50-95", "AP_M", "AP_L", "Condition", "TotalTime_s", "Inference_ms_per_img_e2e", "FPS_e2e", "VRAM_used_GB", "Params_M"]:
        if extra in out.columns:
            keep.append(extra)

    return out[keep]


def plot_lpips_vs_metric(df: pd.DataFrame, metric: str, out_png: str, title: str):
    plt.figure(figsize=(8.5, 5.5))

    dfp = df[df["BlurSigma"] > 0].copy()  # LPIPS is defined clean vs blur
    dfp = dfp.sort_values(["Image_Size", "BlurSigma"])

    for res, g in dfp.groupby("Image_Size"):
        x = g["LPIPS_mean"].values
        y = g[metric].values
        plt.plot(x, y, marker="o", label=f"{res}×{res}")

        for xi, yi, si in zip(x, y, g["BlurSigma"].values):
            plt.text(xi, yi, f"σ={int(si)}", fontsize=8, ha="left", va="bottom")

    plt.xlabel("LPIPS (mean) — clean vs blur")
    plt.ylabel(metric)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    if not os.path.exists(METRICS_CSV):
        raise FileNotFoundError(f"Metrics CSV not found: {METRICS_CSV}")
    if not os.path.exists(LPIPS_CSV):
        raise FileNotFoundError(f"LPIPS CSV not found: {LPIPS_CSV}")

    lp = load_lpips_csv(LPIPS_CSV)
    met = load_metrics_csv(METRICS_CSV)

    df = met.merge(lp, on=["Image_Size", "BlurSigma"], how="inner")
    if df.empty:
        raise RuntimeError(
            "Merge returned empty. Check that both CSVs share matching Image_Size and BlurSigma values."
        )

    os.makedirs(OUT_DIR, exist_ok=True)

    # Save merged table
    merged_csv = os.path.join(OUT_DIR, "merged_lpips_metrics.csv")
    df.sort_values(["Image_Size", "BlurSigma"]).to_csv(merged_csv, index=False)

    # Make plots
    plot_lpips_vs_metric(df, "AP75",  os.path.join(OUT_DIR, "LPIPS_vs_AP75.png"),  "LPIPS vs AP75 (localisation-sensitive)")
    plot_lpips_vs_metric(df, "AP_S",  os.path.join(OUT_DIR, "LPIPS_vs_AP_S.png"),  "LPIPS vs AP_S (small objects — localisation-sensitive)")
    plot_lpips_vs_metric(df, "AR@100", os.path.join(OUT_DIR, "LPIPS_vs_AR100.png"), "LPIPS vs AR@100 (recall-like)")

    print("✅ Done. Saved:")
    print(" -", merged_csv)
    print(" -", os.path.join(OUT_DIR, "LPIPS_vs_AP75.png"))
    print(" -", os.path.join(OUT_DIR, "LPIPS_vs_AP_S.png"))
    print(" -", os.path.join(OUT_DIR, "LPIPS_vs_AR100.png"))


if __name__ == "__main__":
    main()
