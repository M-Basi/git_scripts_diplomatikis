# box_error_breakdown.py
# ------------------------------------------------------------
# Category D: Box error breakdown (class-aware)
# Prebuilt version (no CLI arguments)
# ------------------------------------------------------------

import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict


# =============================================================================
# PATHS (PREBUILT)
# =============================================================================

GT_JSON = "D:/Desktop/project/workspace/data/coco2017/annotations/instances_val2017.json"

PREDICTIONS = {
    "640_clean":  "D:/Desktop/project/workspace/results/rtdetrv2_blur_runs/bbox_img640_bs4_wk4_clean.json",
    "800_clean":  "D:/Desktop/project/workspace/results/rtdetrv2_blur_runs/bbox_img800_bs4_wk4_clean.json",
    "800_blur2":  "D:/Desktop/project/workspace/results/rtdetrv2_blur_runs/bbox_img800_bs4_wk4_blur2.json",
}

OUT_DIR = "D:/Desktop/project/workspace/results/box_breakdown"

SCORE_THR = 0.0
TOPK_PER_IMG_CAT = 300


# =============================================================================
# GEOMETRY HELPERS
# =============================================================================

def xywh_to_xyxy(b):
    x, y, w, h = b
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


def coco_size_bucket(area):
    if area < 32 * 32:
        return "S"
    if area < 96 * 96:
        return "M"
    return "L"


# =============================================================================
# LOADERS
# =============================================================================

def load_gt(gt_path):
    with open(gt_path, "r") as f:
        gt = json.load(f)

    gt_map = defaultdict(list)
    for ann in gt["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        key = (ann["image_id"], ann["category_id"])
        gt_map[key].append({
            "bbox": ann["bbox"],
            "area": ann["area"]
        })
    return gt_map


def load_preds(pred_path):
    with open(pred_path, "r") as f:
        preds = json.load(f)

    pred_map = defaultdict(list)
    for p in preds:
        if p["score"] < SCORE_THR:
            continue
        key = (p["image_id"], p["category_id"])
        pred_map[key].append({
            "bbox": p["bbox"],
            "score": p["score"]
        })

    for k in pred_map:
        pred_map[k].sort(key=lambda x: x["score"], reverse=True)
        pred_map[k] = pred_map[k][:TOPK_PER_IMG_CAT]

    return pred_map


# =============================================================================
# MATCHING + ERRORS
# =============================================================================

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


def collect_errors(label, gt_map, pred_map):
    rows = []
    for key, gt_list in gt_map.items():
        preds = pred_map.get(key, [])
        if not preds:
            continue
        matches = greedy_match(gt_list, preds)
        for g, p, iou in matches:
            gcx, gcy, gw, gh = center_xywh(g["bbox"])
            pcx, pcy, pw, ph = center_xywh(p["bbox"])

            dx = (pcx - gcx) / max(gw, 1e-6)
            dy = (pcy - gcy) / max(gh, 1e-6)
            dw = (pw - gw) / max(gw, 1e-6)
            dh = (ph - gh) / max(gh, 1e-6)

            rows.append({
                "Condition": label,
                "Bucket": coco_size_bucket(g["area"]),
                "IoU": iou,
                "abs_dx": abs(dx),
                "abs_dy": abs(dy),
                "abs_dw": abs(dw),
                "abs_dh": abs(dh),
            })
    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    gt_map = load_gt(GT_JSON)

    all_dfs = []
    for label, pred_path in PREDICTIONS.items():
        print(f"Processing: {label}")
        pred_map = load_preds(pred_path)
        df = collect_errors(label, gt_map, pred_map)
        df.to_csv(os.path.join(OUT_DIR, f"{label}_per_match.csv"), index=False)
        all_dfs.append(df)

    full_df = pd.concat(all_dfs, ignore_index=True)

    # Summary
    summary = full_df.groupby("Condition").agg(
        IoU_median=("IoU", "median"),
        dx_median=("abs_dx", "median"),
        dy_median=("abs_dy", "median"),
        dw_median=("abs_dw", "median"),
        dh_median=("abs_dh", "median"),
    ).reset_index()

    summary.to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)

    # By bucket
    summary_bucket = full_df.groupby(["Condition", "Bucket"]).agg(
        IoU_median=("IoU", "median"),
        dx_median=("abs_dx", "median"),
        dy_median=("abs_dy", "median"),
        dw_median=("abs_dw", "median"),
        dh_median=("abs_dh", "median"),
    ).reset_index()

    summary_bucket.to_csv(os.path.join(OUT_DIR, "summary_by_bucket.csv"), index=False)

    print("✅ Box error breakdown completed.")
    print(f"Results saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
