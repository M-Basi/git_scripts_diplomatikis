import os
import json
import random
import cv2
import numpy as np

import torch
import lpips

# ----------------------------
# CONFIG (edit these)
# ----------------------------
COCO_VAL_IMAGES_DIR = "D:/Desktop/project/workspace/data/coco2017/images/val2017" 
COCO_VAL_ANN_JSON   = "D:/Desktop/project/workspace/data/coco2017/annotations/instances_val2017.json"

OUT_DIR = r"D:\Desktop\project\workspace\results\lpips_qualitative"

RES = 800  # choose 640 or 800
SIGMAS = [1, 2, 3, 4, 5]
N_EXAMPLES = 8

LPIPS_NET = "alex"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_coco_image_filenames(ann_json_path):
    with open(ann_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [img["file_name"] for img in data["images"]]


def gaussian_blur_uint8(im_uint8_bgr, sigma):
    return cv2.GaussianBlur(im_uint8_bgr, ksize=(0, 0), sigmaX=float(sigma), sigmaY=float(sigma))


def preprocess_for_lpips(im_uint8_bgr, target_size):
    im = cv2.cvtColor(im_uint8_bgr, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    im = im.astype(np.float32) / 255.0
    im = (im * 2.0) - 1.0
    im = np.transpose(im, (2, 0, 1))
    t = torch.from_numpy(im).unsqueeze(0)
    return t


def put_text(im_bgr, text, y=30):
    cv2.putText(im_bgr, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 4, cv2.LINE_AA)
    cv2.putText(im_bgr, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)


@torch.no_grad()
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    loss_fn = lpips.LPIPS(net=LPIPS_NET).to(DEVICE).eval()

    fns = load_coco_image_filenames(COCO_VAL_ANN_JSON)
    chosen = random.sample(fns, k=min(N_EXAMPLES, len(fns)))

    for fn in chosen:
        path = os.path.join(COCO_VAL_IMAGES_DIR, fn)
        im = cv2.imread(path, cv2.IMREAD_COLOR)
        if im is None:
            continue

        # Resize for display consistency
        im_disp = cv2.resize(im, (RES, RES), interpolation=cv2.INTER_LINEAR)

        # Clean tensor
        t_clean = preprocess_for_lpips(im, RES).to(DEVICE)

        # Build a horizontal strip: clean | blur1 | blur2 | ...
        tiles = []
        clean_tile = im_disp.copy()
        put_text(clean_tile, f"clean ({RES}x{RES})", y=30)
        tiles.append(clean_tile)

        for sigma in SIGMAS:
            im_blur = gaussian_blur_uint8(im, sigma)
            im_blur_disp = cv2.resize(im_blur, (RES, RES), interpolation=cv2.INTER_LINEAR)

            t_blur = preprocess_for_lpips(im_blur, RES).to(DEVICE)
            d = float(loss_fn(t_clean, t_blur).mean().item())

            tile = im_blur_disp.copy()
            put_text(tile, f"blur = {sigma}", y=30)
            put_text(tile, f"LPIPS={d:.4f}", y=65)
            tiles.append(tile)

        strip = np.concatenate(tiles, axis=1)
        out_path = os.path.join(OUT_DIR, f"qual_{os.path.splitext(fn)[0]}_res{RES}.png")
        cv2.imwrite(out_path, strip)
        print(" Saved:", out_path)


if __name__ == "__main__":
    main()
