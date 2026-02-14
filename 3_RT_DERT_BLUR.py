# rtdetrv2_blur_invariance.py
# ------------------------------------------------------------
# Category C: Low-pass (Gaussian blur) invariance experiments
# Runs:
#   - 800 clean vs 800 blur (sigma=1,2)
#   - optionally 640 clean vs 640 blur (sigma=1,2)
# Saves:
#   - COCO metrics to CSV
#   - per-run predictions JSON (bbox_*.json) so nothing gets overwritten
#
# IMPORTANT (PaddleDetection):
# - EvalReader.sample_transforms expects dict transforms (with .items()).
# - Custom transform name must exist inside ppdet.data.transform namespace.
#   We do a safe local monkey-patch:
#       import ppdet.data.transform as T
#       T.GaussianBlurEval = GaussianBlurEval
# ------------------------------------------------------------

import os
import glob
import json
import time
import GPUtil
import numpy as np
import pandas as pd

import paddle
from ppdet.core.workspace import load_config, merge_config
from ppdet.engine import Trainer


# =============================================================================
# Custom Transform: Gaussian Blur (Low-pass) for Eval
# =============================================================================
try:
    import cv2
except ImportError:
    raise ImportError(
        "opencv-python is required for blur. Install it in your conda env:\n"
        "   pip install opencv-python\n"
    )


class GaussianBlurEval(object):
    """
    Gaussian blur applied at eval-time only.
    Expects sample dict with key 'image' (H,W,C) uint8.
    """
    def __init__(self, sigma=1.0):
        self.sigma = float(sigma)

    def __call__(self, sample):
        im = sample["image"]
        if self.sigma > 0:
            im = cv2.GaussianBlur(im, ksize=(0, 0), sigmaX=self.sigma, sigmaY=self.sigma)
        sample["image"] = im
        return sample


# Make PaddleDetection able to resolve {"GaussianBlurEval": {...}}
# as if it were a built-in transform.
import ppdet.data.transform as T
T.GaussianBlurEval = GaussianBlurEval


# =============================================================================
# HELPERS
# =============================================================================

def get_gpu_memory_used_gb():
    """Returns current GPU used memory in GB (approx)."""
    gpus = GPUtil.getGPUs()
    if not gpus:
        return 0.0
    return float(gpus[0].memoryUsed) / 1024.0  # GB


def count_coco_images(gt_json_path: str) -> int:
    """Reliable count of images from COCO annotation json."""
    try:
        from pycocotools.coco import COCO
        coco_gt = COCO(gt_json_path)
        return len(coco_gt.imgs)
    except Exception:
        # fallback (should be 5000 for val2017)
        return len(glob.glob(os.path.join(os.path.dirname(os.path.dirname(gt_json_path)), "images", "val2017", "*.jpg")))


def extract_coco_metrics(bbox_json_path, gt_json_path):
    """
    Extract COCO metrics from predictions json:
    - AP50, AP75, AP50-95
    - AP small/medium/large
    - AR@1, AR@10, AR@100
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(gt_json_path)
    coco_dt = coco_gt.loadRes(bbox_json_path)

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    AP50_95 = float(coco_eval.stats[0])
    AP50 = float(coco_eval.stats[1])
    AP75 = float(coco_eval.stats[2])

    AP_small = float(coco_eval.stats[3])
    AP_medium = float(coco_eval.stats[4])
    AP_large = float(coco_eval.stats[5])

    AR1 = float(coco_eval.stats[6])
    AR10 = float(coco_eval.stats[7])
    AR100 = float(coco_eval.stats[8])

    return AP50, AP75, AP50_95, AP_small, AP_medium, AP_large, AR1, AR10, AR100


def save_to_csv_row(row_dict, csv_file):
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    write_header = not os.path.exists(csv_file)
    header = list(row_dict.keys())

    with open(csv_file, "a", encoding="utf-8") as f:
        if write_header:
            f.write(",".join(header) + "\n")
        f.write(",".join(str(v) for v in row_dict.values()) + "\n")


def set_rtdetrv2_dynamic_eval(trainer):
    """
    Fix RT-DETRv2 for arbitrary image sizes by disabling fixed eval_size
    (affects both HybridEncoder pos embeddings and Transformer valid mask).
    """
    neck = getattr(trainer.model, "neck", None)
    if neck is None and hasattr(trainer.model, "detector"):
        neck = getattr(trainer.model.detector, "neck", None)
    if neck is not None and hasattr(neck, "eval_size"):
        neck.eval_size = None

    transformer = getattr(trainer.model, "transformer", None)
    if transformer is None and hasattr(trainer.model, "detector"):
        transformer = getattr(trainer.model.detector, "transformer", None)
    if transformer is not None and hasattr(transformer, "eval_size"):
        transformer.eval_size = None


def set_eval_resize(cfg, img_size: int):
    """Find Resize in EvalReader.sample_transforms and set target_size."""
    st = cfg.get("EvalReader", {}).get("sample_transforms", [])
    for t in st:
        if isinstance(t, dict) and "Resize" in t:
            t["Resize"]["target_size"] = [img_size, img_size]
            return
    raise KeyError("Could not find a 'Resize' transform inside EvalReader.sample_transforms.")


def insert_blur_after_resize(cfg, sigma: float):
    """
    Insert GaussianBlurEval right after Resize in EvalReader.sample_transforms.
    If sigma <= 0, do nothing (clean).
    """
    sigma = float(sigma)
    if sigma <= 0:
        return

    transforms = cfg["EvalReader"]["sample_transforms"]
    resize_idx = None
    for i, t in enumerate(transforms):
        if isinstance(t, dict) and "Resize" in t:
            resize_idx = i
            break
    if resize_idx is None:
        raise KeyError("Resize not found; cannot insert blur.")

    # Must be dict so PaddleDetection parser can call .items()
    transforms.insert(resize_idx + 1, {"GaussianBlurEval": {"sigma": sigma}})


# =============================================================================
# MAIN
# =============================================================================

def run_rtdetrv2_val_blur(
    cfg_path,
    weight_path,
    output_dir,
    output_csv_main,
    output_csv_secondary,
    img_size=800,
    blur_sigma=0.0,
    workers=4,
    eval_batch=4,
    model_name="RT-DETRv2-R50",
    seed=42,
):
    os.makedirs(output_dir, exist_ok=True)

    tag = "clean" if blur_sigma <= 0 else f"blur{blur_sigma:g}"
    print(f"\n🚀 RT-DETRv2 Eval | size={img_size} | {tag} | batch={eval_batch} | workers={workers}")

    np.random.seed(seed)
    paddle.seed(seed)

    cfg_path = "D:/Desktop/project/PaddleDetection/configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml"
    cfg = load_config(cfg_path)
    merge_config(cfg)

    dataset_dir = cfg["EvalDataset"]["dataset_dir"]
    anno_rel = cfg["EvalDataset"]["anno_path"]
    val_json = os.path.join(dataset_dir, anno_rel)
    if not os.path.exists(val_json):
        raise FileNotFoundError(f"COCO annotation not found: {val_json}")

    # Fix comparability knobs
    set_eval_resize(cfg, img_size)
    cfg["EvalReader"]["batch_size"] = int(eval_batch)
    cfg["EvalReader"]["num_workers"] = int(workers)

    # Insert blur transform AFTER Resize (low-pass)
    insert_blur_after_resize(cfg, blur_sigma)

    # Quick sanity print
    # Resize is at index 1 for your config; still safe to print it directly
    print("✅ Resize target_size =", cfg["EvalReader"]["sample_transforms"][1]["Resize"]["target_size"])
    if blur_sigma > 0:
        # blur should now be right after Resize
        print("✅ Inserted blur transform:", cfg["EvalReader"]["sample_transforms"][2])

    paddle.set_device("gpu")

    trainer = Trainer(cfg, mode="eval")
    trainer.load_weights(weight_path)
    set_rtdetrv2_dynamic_eval(trainer)

    # Avoid stale bbox.json
    default_bbox = "bbox.json"
    if os.path.exists(default_bbox):
        try:
            os.remove(default_bbox)
        except Exception:
            pass

    start = time.time()
    trainer.evaluate()
    end = time.time()

    if not os.path.exists(default_bbox):
        raise FileNotFoundError("bbox.json not found after evaluation.")

    run_bbox_path = os.path.join(
        output_dir,
        f"bbox_img{img_size}_bs{eval_batch}_wk{workers}_{tag}.json"
    )
    if os.path.exists(run_bbox_path):
        os.remove(run_bbox_path)
    os.replace(default_bbox, run_bbox_path)

    n_images = count_coco_images(val_json)

    total_time = max(end - start, 1e-9)
    e2e_fps = n_images / total_time
    latency_ms_per_image = (total_time * 1000.0) / max(n_images, 1)

    AP50, AP75, AP50_95, AP_S, AP_M, AP_L, _, _, AR100 = extract_coco_metrics(run_bbox_path, val_json)

    params_m = float(sum(p.numel() for p in trainer.model.parameters()) / 1e6)
    vram_used_gb = get_gpu_memory_used_gb()

    results = {
        "Model": model_name,
        "Images": n_images,
        "Image_Size": int(img_size),
        "Workers": int(workers),
        "Eval_Batch": int(eval_batch),

        # Condition
        "BlurSigma": float(blur_sigma),
        "Condition": tag,

        # COCO metrics
        "AP50": round(AP50, 4),
        "AP75": round(AP75, 4),
        "AP50-95": round(AP50_95, 4),
        "AP_S": round(AP_S, 4),
        "AP_M": round(AP_M, 4),
        "AP_L": round(AP_L, 4),
        "AR@100": round(AR100, 4),

        # Speed/resource
        "TotalTime_s": round(total_time, 3),
        "Inference_ms_per_img_e2e": round(latency_ms_per_image, 4),
        "FPS_e2e": round(e2e_fps, 2),
        "VRAM_used_GB": round(vram_used_gb, 3),
        "Params_M": round(params_m, 2),

        # Artifact
        "Predictions_JSON": run_bbox_path,
    }

    save_to_csv_row(results, output_csv_main)
    save_to_csv_row(results, output_csv_secondary)

    print("\n✅ DONE — RESULTS:")
    print(json.dumps(results, indent=4, ensure_ascii=False))
    print(f"\n📁 Saved rows to:")
    print(f" - {output_csv_main}")
    print(f" - {output_csv_secondary}")
    print(f"📦 Saved predictions to:")
    print(f" - {run_bbox_path}\n")

    try:
        paddle.device.cuda.empty_cache()
    except Exception:
        pass

    return results


# =============================================================================
# RUN EXPERIMENTS (Category C)
# =============================================================================
if __name__ == "__main__":
    import multiprocessing

    cfg_path = "D:/Desktop/project/PaddleDetection/configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml"
    weight_path = "D:/Desktop/project/PaddleDetection/weights/rtdetrv2_r50vd_6x_coco.pdparams"

    output_dir = "D:/Desktop/project/workspace/results/rtdetrv2_blur_runs"
    output_csv_main = "D:/Desktop/project/workspace/results/dert_blur.csv"
    output_csv_secondary = "D:/Desktop/project/workspace/results/metrics_blur.csv"

    workers = 4
    eval_batch = 4
    seed = 42

    # Core invariance experiment:
    #   800 clean vs 800 blur(sigma=1,2)
    # Optional:
    #   640 clean vs 640 blur(sigma=1,2)
    sizes = [800]
    include_640 = True
    if include_640:
        sizes = [640, 800]

    # Include clean as sigma=0.0 for proper comparison
    blur_sigmas = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    multiprocessing.freeze_support()

    for sz in sizes:
        for sigma in blur_sigmas:
            run_rtdetrv2_val_blur(
                cfg_path=cfg_path,
                weight_path=weight_path,
                output_dir=output_dir,
                output_csv_main=output_csv_main,
                output_csv_secondary=output_csv_secondary,
                img_size=sz,
                blur_sigma=sigma,
                workers=workers,
                eval_batch=eval_batch,
                model_name="RT-DETRv2-R50",
                seed=seed,
            )

