import os
import json
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image


import torch
import lpips  # provided by PerceptualSimilarity (pip install -e .)

# ----------------------------
# CONFIG (edit these)
# ----------------------------
COCO_VAL_IMAGES_DIR = "D:/Desktop/project/workspace/data/coco2017/images/val2017"  # folder with 000000xxxxxx.jpg
COCO_VAL_ANN_JSON   = "D:/Desktop/project/workspace/data/coco2017/annotations/instances_val2017.json"

OUT_CSV = "D:/Desktop/project/workspace/results/lpips_blur_severity.csv"

# We match your experiment sizes:
RESOLUTIONS = [640, 800]
SIGMAS = [1, 2, 3, 4, 5]

# LPIPS backbone: alex/vgg/squeeze (alex is common & fast)
LPIPS_NET = "alex"

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_coco_image_filenames(ann_json_path):
    with open(ann_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # COCO val2017 has 5000 images
    return [img["file_name"] for img in data["images"]]


def gaussian_blur_uint8(im_uint8_bgr, sigma):
    # sigma=0 not used here; we compare clean vs blurred
    return cv2.GaussianBlur(im_uint8_bgr, ksize=(0, 0), sigmaX=float(sigma), sigmaY=float(sigma))


def preprocess_for_lpips(im_uint8_bgr, target_size):
    """
    LPIPS expects torch tensors in [-1, 1], shape [1,3,H,W], RGB.
    We also enforce a fixed resolution (e.g., 640 or 800) to match your eval.
    """
    im = cv2.cvtColor(im_uint8_bgr, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    im = im.astype(np.float32) / 255.0  # [0,1]
    im = (im * 2.0) - 1.0               # [-1,1]
    im = np.transpose(im, (2, 0, 1))    # [3,H,W]
    im = torch.from_numpy(im).unsqueeze(0)  # [1,3,H,W]
    return im


@torch.no_grad()
def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    # Init LPIPS
    loss_fn = lpips.LPIPS(net=LPIPS_NET).to(DEVICE)
    loss_fn.eval()

    file_names = load_coco_image_filenames(COCO_VAL_ANN_JSON)

    rows = []

    for res in RESOLUTIONS:
        for sigma in SIGMAS:
            lpips_vals = []

            print(f"\n[LPIPS] res={res} sigma={sigma} | images={len(file_names)}")

            for fn in tqdm(file_names, desc=f"res={res}, sigma={sigma}"):
                path = os.path.join(COCO_VAL_IMAGES_DIR, fn)
                im = cv2.imread(path, cv2.IMREAD_COLOR)
                if im is None:
                    continue

                im_blur = gaussian_blur_uint8(im, sigma=sigma)

                t_clean = preprocess_for_lpips(im, target_size=res).to(DEVICE)
                t_blur  = preprocess_for_lpips(im_blur, target_size=res).to(DEVICE)

                d = loss_fn(t_clean, t_blur)  # scalar tensor [1,1,1,1] or [1]
                d_val = float(d.mean().item())
                lpips_vals.append(d_val)

            lpips_vals = np.array(lpips_vals, dtype=np.float32)
            row = {
                "Resolution": res,
                "Sigma": float(sigma),
                "LPIPS_mean": float(lpips_vals.mean()),
                "LPIPS_median": float(np.median(lpips_vals)),
                "LPIPS_std": float(lpips_vals.std(ddof=0)),
                "N_images_used": int(lpips_vals.shape[0]),
                "LPIPS_net": LPIPS_NET,
            }
            rows.append(row)

    df = pd.DataFrame(rows).sort_values(["Resolution", "Sigma"])
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"\n Saved: {OUT_CSV}")
    print(df)


if __name__ == "__main__":
    main()
