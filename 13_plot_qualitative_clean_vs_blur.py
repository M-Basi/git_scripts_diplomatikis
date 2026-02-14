# plot_qualitative_clean_vs_blur.py
# ------------------------------------------------------------
# Qualitative figure: clean vs blur (σ=2) on the SAME image/object
# - loads COCO val2017 image + instances_val2017.json
# - loads predictions JSON for two conditions (clean / blur)
# - class-aware greedy matching per image/category
# - selects a "representative" example automatically (prefer small objects)
# - draws GT + Pred boxes and prints IoU + |Δx| |Δy| |Δw| |Δh|
#
# Output:
#   qualitative_clean_vs_blur_sigma2.png (and also a *_debug.txt)
# ------------------------------------------------------------

import os
import json
import random
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter


# =========================
# CONFIG (EDIT THESE PATHS)
# =========================
GT_JSON = r"D:/Desktop/project/workspace/data/coco2017/annotations/instances_val2017.json"
VAL_IMG_DIR = r"D:/Desktop/project/workspace/data/coco2017/images/val2017"  # folder with COCO val images (jpg)

PRED_CLEAN_JSON = r"D:/Desktop/project/workspace/results/rtdetrv2_blur_runs/bbox_img800_bs4_wk4_clean.json"
PRED_BLUR_JSON  = r"D:/Desktop/project/workspace/results/rtdetrv2_blur_runs/bbox_img800_bs4_wk4_blur2.json"

# figure labels
LABEL_CLEAN = "800×800 clean"
LABEL_BLUR  = "800×800 blur (σ=2)"

# If you want the right panel to show an actually blurred image (visual only)
APPLY_VISUAL_BLUR = True
VISUAL_BLUR_SIGMA = 2.0  # for display only (PIL radius ≈ sigma)

# Prediction filtering
SCORE_THR = 0.05
TOPK_PER_IMG_CAT = 200

# Example selection
PREFER_BUCKET = "S"   # "S" / "M" / "L" / None
RANDOM_SEED = 123

# Output
OUT_DIR = r"D:/Desktop/project/figures"
OUT_NAME = "qualitative_clean_vs_blur_sigma2.png"


# =========================
# Geometry helpers
# =========================
def xywh_to_xyxy(b):
    x, y, w, h = map(float, b)
    return x, y, x + w, y + h

def iou_xywh(b1, b2):
    x1a, y1a, x2a, y2a = xywh_to_xyxy(b1)
    x1b, y1b, x2b, y2b = xywh_to_xyxy(b2)

    ix1 = max(x1a, x1b)
    iy1 = max(y1a, y1b)
    ix2 = min(x2a, x2b)
    iy2 = min(y2a, y2b)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    a1 = max(0.0, x2a - x1a) * max(0.0, y2a - y1a)
    a2 = max(0.0, x2b - x1b) * max(0.0, y2b - y1b)
    union = a1 + a2 - inter

    return 0.0 if union <= 0 else inter / union

def center_xywh(b):
    x, y, w, h = map(float, b)
    return x + w / 2.0, y + h / 2.0, w, h

def coco_bucket(area):
    if area < 32 * 32:
        return "S"
    if area < 96 * 96:
        return "M"
    return "L"

def norm_errors(gt_bbox, pr_bbox):
    gcx, gcy, gw, gh = center_xywh(gt_bbox)
    pcx, pcy, pw, ph = center_xywh(pr_bbox)

    dx = abs((pcx - gcx) / max(gw, 1e-6))
    dy = abs((pcy - gcy) / max(gh, 1e-6))
    dw = abs((pw - gw) / max(gw, 1e-6))
    dh = abs((ph - gh) / max(gh, 1e-6))
    return dx, dy, dw, dh


# =========================
# Loaders
# =========================
def load_gt(gt_json_path):
    with open(gt_json_path, "r") as f:
        gt = json.load(f)

    img_id_to_file = {im["id"]: im["file_name"] for im in gt["images"]}
    cat_id_to_name = {c["id"]: c["name"] for c in gt["categories"]}

    gt_by_image = defaultdict(list)
    for ann in gt["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        gt_by_image[ann["image_id"]].append({
            "ann_id": ann["id"],
            "category_id": ann["category_id"],
            "bbox": ann["bbox"],
            "area": ann["area"],
        })

    return img_id_to_file, cat_id_to_name, gt_by_image

def load_preds(pred_json_path):
    with open(pred_json_path, "r") as f:
        preds = json.load(f)

    pred_by_img_cat = defaultdict(list)
    for p in preds:
        if p.get("score", 0.0) < SCORE_THR:
            continue
        key = (p["image_id"], p["category_id"])
        pred_by_img_cat[key].append({
            "bbox": p["bbox"],
            "score": p["score"]
        })

    for k in pred_by_img_cat:
        pred_by_img_cat[k].sort(key=lambda x: x["score"], reverse=True)
        pred_by_img_cat[k] = pred_by_img_cat[k][:TOPK_PER_IMG_CAT]

    return pred_by_img_cat


# =========================
# Matching
# =========================
def greedy_match(gt_list, pred_list):
    pairs = []
    for gi, g in enumerate(gt_list):
        for pi, p in enumerate(pred_list):
            pairs.append((iou_xywh(g["bbox"], p["bbox"]), gi, pi))
    pairs.sort(key=lambda x: x[0], reverse=True)

    used_g, used_p = set(), set()
    matches = []
    for iou, gi, pi in pairs:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        matches.append((gt_list[gi], pred_list[pi], iou))
    return matches

def build_matches(gt_by_image, preds_by_img_cat):
    out = {}
    for image_id, gt_list_all in gt_by_image.items():
        gt_by_cat = defaultdict(list)
        for g in gt_list_all:
            gt_by_cat[g["category_id"]].append(g)

        for cat_id, gt_list in gt_by_cat.items():
            pred_list = preds_by_img_cat.get((image_id, cat_id), [])
            if not pred_list:
                continue

            matches = greedy_match(gt_list, pred_list)
            for g, p, iou in matches:
                dx, dy, dw, dh = norm_errors(g["bbox"], p["bbox"])
                out[g["ann_id"]] = {
                    "image_id": image_id,
                    "category_id": cat_id,
                    "gt_bbox": g["bbox"],
                    "bucket": coco_bucket(g["area"]),
                    "pred_bbox": p["bbox"],
                    "score": p["score"],
                    "iou": float(iou),
                    "dx": float(dx),
                    "dy": float(dy),
                    "dw": float(dw),
                    "dh": float(dh),
                }
    return out


# =========================
# Selection logic
# =========================
def select_representative(clean_matches, blur_matches, prefer_bucket="S"):
    common = sorted(set(clean_matches.keys()) & set(blur_matches.keys()))
    if not common:
        raise RuntimeError("No common matched GT annotations between clean and blur. Try lowering SCORE_THR.")

    if prefer_bucket is not None:
        common_bucket = [aid for aid in common if clean_matches[aid]["bucket"] == prefer_bucket]
        if common_bucket:
            common = common_bucket

    ious_clean = np.array([clean_matches[aid]["iou"] for aid in common], dtype=float)
    ious_blur  = np.array([blur_matches[aid]["iou"]  for aid in common], dtype=float)
    med_clean = float(np.median(ious_clean))
    med_blur  = float(np.median(ious_blur))

    best_aid, best_score = None, 1e9
    for aid in common:
        c = clean_matches[aid]
        b = blur_matches[aid]

        # keep visually meaningful cases
        if c["iou"] < 0.40 or b["iou"] < 0.30:
            continue

        s = abs(c["iou"] - med_clean) + abs(b["iou"] - med_blur)
        if s < best_score:
            best_score = s
            best_aid = aid

    if best_aid is None:
        random.seed(RANDOM_SEED)
        best_aid = random.choice(common)

    return best_aid, med_clean, med_blur


# =========================
# Visualization
# =========================
def draw_box(ax, bbox_xywh, color, label, lw=2.0):
    x, y, w, h = map(float, bbox_xywh)
    rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=lw)
    ax.add_patch(rect)
    ax.text(x, max(0, y - 4), label, color=color, fontsize=10, va="bottom")

def metrics_block(info):
    return (
        f"IoU={info['iou']:.4f}  score={info['score']:.3f}\n"
        f"|Δx|={info['dx']:.4f}  |Δy|={info['dy']:.4f}\n"
        f"|Δw|={info['dw']:.4f}  |Δh|={info['dh']:.4f}"
    )

def make_figure(img_path, clean_info, blur_info, cat_name, out_path, debug_path):
    img = Image.open(img_path).convert("RGB")
    img_blur = img.filter(ImageFilter.GaussianBlur(radius=VISUAL_BLUR_SIGMA)) if APPLY_VISUAL_BLUR else img

    fig = plt.figure(figsize=(12, 6))

    # LEFT: clean
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(img)
    ax1.set_title(f"{LABEL_CLEAN}\nclass: {cat_name}", fontsize=12)

    draw_box(ax1, clean_info["gt_bbox"], color="lime", label="GT")
    draw_box(ax1, clean_info["pred_bbox"], color="red", label="Pred")
    ax1.text(
        10, 10, metrics_block(clean_info),
        fontsize=10, color="white",
        bbox=dict(facecolor="black", alpha=0.55, pad=6),
        va="top"
    )
    ax1.axis("off")

    # RIGHT: blur
    ax2 = plt.subplot(1, 2, 2)
    ax2.imshow(img_blur)
    ax2.set_title(f"{LABEL_BLUR}\nclass: {cat_name}", fontsize=12)

    draw_box(ax2, blur_info["gt_bbox"], color="lime", label="GT")
    draw_box(ax2, blur_info["pred_bbox"], color="red", label="Pred")
    ax2.text(
        10, 10, metrics_block(blur_info),
        fontsize=10, color="white",
        bbox=dict(facecolor="black", alpha=0.55, pad=6),
        va="top"
    )
    ax2.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    # Debug text (so you can reference the exact sample later)
    with open(debug_path, "w", encoding="utf-8") as f:
        f.write("QUALITATIVE SAMPLE DETAILS\n")
        f.write(f"Image path: {img_path}\n")
        f.write(f"Category: {cat_name}\n")
        f.write(f"Condition clean: {LABEL_CLEAN}\n")
        f.write(f"Condition blur : {LABEL_BLUR}\n\n")
        f.write("CLEAN:\n")
        f.write(json.dumps(clean_info, indent=2) + "\n\n")
        f.write("BLUR:\n")
        f.write(json.dumps(blur_info, indent=2) + "\n")


# =========================
# MAIN
# =========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading GT...")
    img_id_to_file, cat_id_to_name, gt_by_image = load_gt(GT_JSON)

    print("Loading predictions...")
    preds_clean = load_preds(PRED_CLEAN_JSON)
    preds_blur  = load_preds(PRED_BLUR_JSON)

    print("Matching (class-aware) for clean...")
    clean_matches = build_matches(gt_by_image, preds_clean)

    print("Matching (class-aware) for blur...")
    blur_matches = build_matches(gt_by_image, preds_blur)

    print("Selecting representative sample...")
    ann_id, med_clean, med_blur = select_representative(clean_matches, blur_matches, prefer_bucket=PREFER_BUCKET)

    cinfo = clean_matches[ann_id]
    binfo = blur_matches[ann_id]

    image_id = cinfo["image_id"]
    cat_id = cinfo["category_id"]
    cat_name = cat_id_to_name.get(cat_id, str(cat_id))

    file_name = img_id_to_file[image_id]
    img_path = os.path.join(VAL_IMG_DIR, file_name)
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"COCO image not found: {img_path}")

    out_path = os.path.join(OUT_DIR, OUT_NAME)
    debug_path = os.path.join(OUT_DIR, OUT_NAME.replace(".png", "_debug.txt"))

    print("Rendering figure...")
    make_figure(img_path, cinfo, binfo, cat_name, out_path, debug_path)

    print("\nSaved qualitative figure:")
    print(out_path)
    print("Saved debug file:")
    print(debug_path)
    print("\nMedians (for reference):")
    print(f"median IoU (clean): {med_clean:.4f}")
    print(f"median IoU (blur) : {med_blur:.4f}")


if __name__ == "__main__":
    main()
