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
        # fallback: count files (should be 5000 for val2017)
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
    # 1) Hybrid Encoder positional embeddings
    neck = getattr(trainer.model, "neck", None)
    if neck is None and hasattr(trainer.model, "detector"):
        neck = getattr(trainer.model.detector, "neck", None)
    if neck is not None and hasattr(neck, "eval_size"):
        neck.eval_size = None

    # 2) Transformer valid mask generation
    transformer = getattr(trainer.model, "transformer", None)
    if transformer is None and hasattr(trainer.model, "detector"):
        transformer = getattr(trainer.model.detector, "transformer", None)
    if transformer is not None and hasattr(transformer, "eval_size"):
        transformer.eval_size = None


def set_eval_resize(cfg, img_size: int):
    """
    Robustly find the Resize transform in EvalReader.sample_transforms
    and set target_size.
    """
    st = cfg.get("EvalReader", {}).get("sample_transforms", [])
    found = False
    for t in st:
        if isinstance(t, dict) and "Resize" in t:
            t["Resize"]["target_size"] = [img_size, img_size]
            found = True
            break
    if not found:
        raise KeyError("Could not find a 'Resize' transform inside EvalReader.sample_transforms.")


# =============================================================================
# MAIN BENCHMARK FUNCTION
# =============================================================================

def run_rtdetrv2_val(
    cfg_path,
    weight_path,
    output_dir,
    output_csv_main,
    output_csv_secondary,
    img_size=640,
    workers=4,
    model_name="RT-DETRv2-R50",
    batch_size=1,
    seed=42,
):
    """
    Runs PaddleDetection eval for RT-DETRv2 with controlled, comparable settings.
    - Fixes EvalReader resize
    - Fixes EvalReader batch_size and num_workers
    - Saves bbox json per run (no overwrite confusion)
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nRunning Paddle RT-DETRv2 Eval: {model_name} | img_size={img_size} | eval_batch={batch_size} | workers={workers}")

    # Reproducibility (eval is mostly deterministic, but keep consistent)
    np.random.seed(seed)
    paddle.seed(seed)

    cfg = load_config(cfg_path)
    
    merge_config(cfg)

    # Dataset paths (COCO)
    dataset_dir = cfg["EvalDataset"]["dataset_dir"]
    anno_rel = cfg["EvalDataset"]["anno_path"]
    val_json = os.path.join(dataset_dir, anno_rel)
    if not os.path.exists(val_json):
        raise FileNotFoundError(f"COCO annotation not found: {val_json}")

    # Force eval image size + batch size + workers (for comparability)
    set_eval_resize(cfg, img_size)
    cfg["EvalReader"]["batch_size"] = int(batch_size)
    cfg["EvalReader"]["num_workers"] = int(workers)
    print("✅ Applied Resize target_size =", cfg["EvalReader"]["sample_transforms"][1]["Resize"]["target_size"])


    # Ensure we run in eval mode on GPU (your env is already gpu:0)
    paddle.set_device("gpu")

    # Create trainer
    trainer = Trainer(cfg, mode="eval")
    trainer.load_weights(weight_path)

    # Allow arbitrary sizes (important for 320/640/800/960/1024)
    set_rtdetrv2_dynamic_eval(trainer)

    # Clean previous bbox.json if exists (avoid reading stale file)
    default_bbox = "bbox.json"
    if os.path.exists(default_bbox):
        try:
            os.remove(default_bbox)
        except Exception:
            pass

    # Run evaluation + time measurement (end-to-end timing)
    start = time.time()
    trainer.evaluate()
    end = time.time()

    # Locate bbox.json produced by PaddleDetection
    if not os.path.exists(default_bbox):
        raise FileNotFoundError("bbox.json not found after evaluation. Check PaddleDetection eval output settings/logs.")

    # Move bbox.json to a per-run file (so runs are comparable & traceable)
    run_bbox_path = os.path.join(output_dir, f"bbox_img{img_size}_bs{batch_size}_wk{workers}.json")
    if os.path.exists(run_bbox_path):
        os.remove(run_bbox_path)
    os.replace(default_bbox, run_bbox_path)

    # Count images from COCO annotation (stable, not filesystem dependent)
    n_images = count_coco_images(val_json)

    total_time = max(end - start, 1e-9)
    e2e_fps = n_images / total_time
    latency_ms_per_image = (total_time * 1000.0) / max(n_images, 1)

    # PaddleDetection doesn't split preprocess/infer/postprocess in this eval call;
    # keep them as "end-to-end per-image" for comparability across your runs.
    preprocess_ms = 0.0
    inference_ms = latency_ms_per_image
    postprocess_ms = 0.0

    fps_model = 1000.0 / inference_ms if inference_ms > 0 else 0.0
    fps_overall = fps_model
    fps_system = round(e2e_fps, 2)

    # Extract COCO metrics from our per-run bbox json
    AP50, AP75, AP50_95, AP_S, AP_M, AP_L, AR1, AR10, AR100 = extract_coco_metrics(run_bbox_path, val_json)

    # Params + VRAM (used)
    params_m = float(sum(p.numel() for p in trainer.model.parameters()) / 1e6)
    vram_used_gb = get_gpu_memory_used_gb()

    # Efficiency score (optional)
    mAP = round(AP50_95, 4)
    efficiency = round(mAP * fps_overall, 4)

    results = {
        "Model": model_name,
        "Images": n_images,
        "Image_Size": img_size,
        "Workers": int(workers),
        "Eval_Batch": int(cfg["EvalReader"]["batch_size"]),

        # Accuracy (COCO)
        "AP50": round(AP50, 4),
        "AP75": round(AP75, 4),
        "AP50-95": round(AP50_95, 4),
        "AP_S": round(AP_S, 4),
        "AP_M": round(AP_M, 4),
        "AP_L": round(AP_L, 4),

        # Recall (COCO)
        "AR@1": round(AR1, 4),
        "AR@10": round(AR10, 4),
        "AR@100": round(AR100, 4),

        # Timing (end-to-end)
        "TotalTime_s": round(total_time, 3),
        "Inference_ms_per_img_e2e": round(inference_ms, 4),

        # FPS
        "FPS_e2e": fps_system,

        # Optional efficiency scalar
        "EffScore_AP50-95_x_FPS": efficiency,

        # Resources
        "VRAM_used_GB": round(vram_used_gb, 3),
        "Params_M": round(params_m, 2),

    }

    # Save results
    save_to_csv_row(results, output_csv_main)
    save_to_csv_row(results, output_csv_secondary)

    print("\n✅ DONE — RESULTS:")
    print(json.dumps(results, indent=4, ensure_ascii=False))
    print(f"\n📁 Saved rows to:")
    print(f" - {output_csv_main}")
    print(f" - {output_csv_secondary}")
    print(f"📦 Saved predictions to:")
    print(f" - {run_bbox_path}\n")

    # Free up memory between runs (helpful on Windows/WDDM)
    try:
        paddle.device.cuda.empty_cache()
    except Exception:
        pass

    return results


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    import multiprocessing

    cfg_path = "D:/Desktop/project/PaddleDetection/configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml"
    weight_path = "D:/Desktop/project/PaddleDetection/weights/rtdetrv2_r50vd_6x_coco.pdparams"

    # Output
    output_dir = "D:/Desktop/project/workspace/results/rtdetrv2_eval_runs"
    output_csv_main = "D:/Desktop/project/PaddleDetection/dert_scaling.csv"
    output_csv_secondary = "D:/Desktop/project/workspace/results/metrics_scaling.csv"

    # Fixed settings for comparability
    workers = 4
    eval_batch_size = 4
    seed = 42

    # Scaling sweep (adjust if 1024 is too slow)
    image_sizes = [320, 640, 800, 960, 1024] 
    

    multiprocessing.freeze_support()

    for sz in image_sizes:
        run_rtdetrv2_val(
            cfg_path=cfg_path,
            weight_path=weight_path,
            output_dir=output_dir,
            output_csv_main=output_csv_main,
            output_csv_secondary=output_csv_secondary,
            img_size=sz,
            workers=workers,
            model_name="RT-DETRv2-R50",
            batch_size=eval_batch_size,
            seed=seed,
        )

