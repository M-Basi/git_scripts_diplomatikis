import os
import glob
import json
import time
import pandas as pd
import GPUtil
import numpy as np

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


def extract_coco_metrics(bbox_json_path, gt_json_path):
    """
    Extract COCO metrics from PaddleDetection bbox.json:
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


# =============================================================================
# MAIN BENCHMARK FUNCTION
# =============================================================================

def run_rtdetrv2_val(
    cfg_path,
    weight_path,
    img_root,
    output_csv_main,
    output_csv_secondary,
    img_size=640,
    workers=4,
    model_name="RT-DETRv2-R50",
    batch_size=16,
):
    print(f"\n🚀 Running Paddle RT-DETRv2 Benchmark: {model_name} | size={img_size} | batch={batch_size}")

    cfg = load_config(cfg_path)
    merge_config(cfg)

    # Dataset paths
    dataset_dir = cfg["EvalDataset"]["dataset_dir"]
    anno_rel = cfg["EvalDataset"]["anno_path"]
    val_json = os.path.join(dataset_dir, anno_rel)

    if not os.path.exists(val_json):
        raise FileNotFoundError(f"❌ Dataset not found: {val_json}")

    # Force eval image size + batch size
    cfg["EvalReader"]["sample_transforms"][1]["Resize"]["target_size"] = [img_size, img_size]
    cfg["EvalReader"]["batch_size"] = int(batch_size)

    # Create trainer
    trainer = Trainer(cfg, mode="eval")
    trainer.load_weights(weight_path)

    # IMPORTANT: allow arbitrary sizes (320/640/1024 etc.)
    set_rtdetrv2_dynamic_eval(trainer)

    # -------------------------------------------------------------------------
    # Run evaluation + time measurement (end-to-end timing)
    # -------------------------------------------------------------------------
    start = time.time()
    trainer.evaluate()
    end = time.time()

    # Count images
    n_images = len(glob.glob(os.path.join(img_root, "*.jpg")))
    total_time = max(end - start, 1e-9)

    # End-to-end throughput
    e2e_fps = n_images / total_time
    latency_ms_per_image = (total_time * 1000.0) / max(n_images, 1)

    # We keep columns compatible with your YOLO-style table, but clarify meaning:
    preprocess_ms = 0.0
    inference_ms = latency_ms_per_image  # end-to-end per image (PaddleDetection doesn't split it)
    postprocess_ms = 0.0

    fps_model = 1000.0 / inference_ms if inference_ms > 0 else 0.0
    fps_overall = fps_model  # same given our approximation
    fps_system = round(e2e_fps, 2)

    # -------------------------------------------------------------------------
    # Load bbox predictions
    # -------------------------------------------------------------------------
    bbox_path = "bbox.json"
    if not os.path.exists(bbox_path):
        raise FileNotFoundError("❌ bbox.json not found after evaluation!")

    # Extract COCO metrics
    AP50, AP75, AP50_95, AP_S, AP_M, AP_L, AR1, AR10, AR100 = extract_coco_metrics(bbox_path, val_json)

    # Params + VRAM (used)
    params_m = float(sum(p.numel() for p in trainer.model.parameters()) / 1e6)
    vram_used_gb = get_gpu_memory_used_gb()

    # Efficiency score
    mAP = round(AP50_95, 4)
    efficiency = round(mAP * fps_overall, 4)

    results = {
        "Model": model_name,
        "Images": n_images,
        "Image_Size": img_size,
        "Workers": workers,
        "Batch": int(cfg["EvalReader"]["batch_size"]),

        # Accuracy (COCO)
        "mAP50": round(AP50, 4),
        "mAP75": round(AP75, 4),
        "mAP50-95": round(AP50_95, 4),
        "mAP50-95_S": round(AP_S, 4),
        "mAP50-95_M": round(AP_M, 4),
        "mAP50-95_L": round(AP_L, 4),

        # Recall (COCO)
        "AR@1": round(AR1, 4),
        "AR@10": round(AR10, 4),
        "AR@100": round(AR100, 4),

        # Speed (end-to-end approximation)
        "Preprocess_ms": round(preprocess_ms, 4),
        "Inference_ms": round(inference_ms, 4),
        "Postprocess_ms (NMS)": round(postprocess_ms, 4),

        # FPS
        "FPS_model": round(fps_model, 2),
        "FPS_overall": round(fps_overall, 2),
        "FPS_system": fps_system,

        # Efficiency
        "Eff_Score 50-95": efficiency,

        # Resources
        "VRAM_used(GB)": round(vram_used_gb, 3),
        "Params(M)": round(params_m, 2),
    }

    # Save results to both CSVs
    save_to_csv_row(results, output_csv_main)
    save_to_csv_row(results, output_csv_secondary)

    print("\n✅ DONE — RESULTS:")
    print(json.dumps(results, indent=4))
    print(f"\n📁 Saved to: {output_csv_main}")
    print(f"📁 Also saved to: {output_csv_secondary}\n")

    return results


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    import multiprocessing

    cfg_path = "D:/Desktop/project/PaddleDetection/configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml"
    weight_path = "D:/Desktop/project/PaddleDetection/weights/rtdetrv2_r50vd_6x_coco.pdparams"
    img_root = "D:/Desktop/project/workspace/data/coco2017/images/val2017"

    # Your two output CSVs
    output_csv_main = "D:/Desktop/project/PaddleDetection/dert_starter.csv"
    output_csv_secondary = "D:/Desktop/project/workspace/results/metrics_summary_full.csv"

    # Single run settings (you can later extend to loops for size/batch grids)
    workers = 4
    img_size = 640
    batch_size = 16

    multiprocessing.freeze_support()

    run_rtdetrv2_val(
        cfg_path=cfg_path,
        weight_path=weight_path,
        img_root=img_root,
        output_csv_main=output_csv_main,
        output_csv_secondary=output_csv_secondary,
        img_size=img_size,
        workers=workers,
        model_name="RT-DETRv2-R50",
        batch_size=batch_size,
    )



# import os, glob, pandas as pd
# import json
# import time
# import yaml
# import paddle
# import subprocess
# import GPUtil
# import numpy as np
# from ppdet.core.workspace import load_config, merge_config
# from ppdet.engine import Trainer


# # =============================================================================
# # HELPERS
# # =============================================================================

# def get_gpu_memory():
#     gpus = GPUtil.getGPUs()
#     if not gpus:
#         return 0.0
#     return gpus[0].memoryUsed / 1024  # GB


# def extract_coco_metrics(result_file, gt_json):
#     """
#     Διαβάζει COCO AP metrics από bbox.json.
#     """
#     from pycocotools.coco import COCO
#     from pycocotools.cocoeval import COCOeval

#     coco_gt = COCO(gt_json)
#     coco_dt = coco_gt.loadRes(result_file)

#     coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
#     coco_eval.evaluate()
#     coco_eval.accumulate()
#     coco_eval.summarize()

#     AP50_95 = coco_eval.stats[0]
#     AP50 = coco_eval.stats[1]
#     AP_small = coco_eval.stats[3]
#     AP_medium = coco_eval.stats[4]
#     AP_large = coco_eval.stats[5]

#     return AP50, AP50_95, AP_small, AP_medium, AP_large


# def compute_precision_recall(result_file, gt_json, iou_threshold=0.5):
#     """
#     Υπολογισμός Global precision/recall (YOLO-style) από bbox.json και COCO GT.
#     """
#     from pycocotools.coco import COCO
#     from pycocotools.cocoeval import COCOeval

#     coco_gt = COCO(gt_json)
#     coco_dt = coco_gt.loadRes(result_file)

#     coco_eval = COCOeval(coco_gt, coco_dt, "bbox")

#     # Set IoU threshold like YOLO PR curves
#     coco_eval.params.iouThrs = np.array([iou_threshold])

#     coco_eval.evaluate()
#     coco_eval.accumulate()

#     precision = coco_eval.eval['precision']  # [T,R,K,A,M]
#     recall = coco_eval.eval['recall']        # [T,K,A,M]

#     # GLOBAL PRECISION
#     valid_precision = precision[precision > -1]
#     global_precision = float(np.mean(valid_precision)) if len(valid_precision) > 0 else 0.0

#     # GLOBAL RECALL
#     valid_recall = recall[recall > -1]
#     global_recall = float(np.mean(valid_recall)) if len(valid_recall) > 0 else 0.0

#     return global_precision, global_recall


# def save_to_csv(data, csv_file):
#     os.makedirs(os.path.dirname(csv_file), exist_ok=True)
#     header = list(data.keys())

#     write_header = not os.path.exists(csv_file)

#     with open(csv_file, "a", encoding="utf-8") as f:
#         if write_header:
#             f.write(",".join(header) + "\n")
#         f.write(",".join(str(v) for v in data.values()) + "\n")


# # =============================================================================
# # MAIN BENCHMARK FUNCTION
# # =============================================================================

# def run_rt_dert_val(cfg_path, weight_path, img_root, output_csv, output_csv_dert,
#                     img_size=640, workers=4, model_name="RT-DETRv2",
#                     batch_size=16):
#     print(f"\n🚀 Running Paddle RT-DETRv2 Benchmark: {model_name}")

#     cfg = load_config(cfg_path)
#     merge_config(cfg)

#     # Dataset paths
#     dataset_dir = cfg["EvalDataset"]["dataset_dir"]
#     anno_rel = cfg["EvalDataset"]["anno_path"]
#     val_json = os.path.join(dataset_dir, anno_rel)

#     if not os.path.exists(val_json):
#         raise FileNotFoundError(f"❌ Dataset not found: {val_json}")

#     # Force eval image size
#     cfg["EvalReader"]["sample_transforms"][1]["Resize"]["target_size"] = [img_size, img_size]
#     cfg["EvalReader"]["batch_size"] = batch_size

#     trainer = Trainer(cfg, mode="eval")
#     trainer.load_weights(weight_path)

#     # -------------------------------------------------------------------------
#     # Run evaluation + time measurement (SYSTEM-LEVEL TIMING)
#     # -------------------------------------------------------------------------
#     start = time.time()
#     trainer.evaluate()
#     end = time.time()

#     n_images = len(glob.glob(os.path.join(img_root, "*.jpg")))
#     total_time = max(end - start, 1e-6)
#     fps_system = n_images / total_time

#     # Για να έχουμε ίδια columns με το YOLO script:
#     # PaddleDetection δεν δίνει ξεχωριστά preprocess/inference/postprocess times.
#     # Εδώ προσεγγίζουμε:
#     #   - όλο το latency per image ως "Inference_ms"
#     #   - Preprocess_ms = 0, Postprocess_ms = 0 (δεν είναι διαθέσιμα από το API)
#     total_ms_per_image = total_time * 1000.0 / n_images

#     preprocess_ms = 0.0
#     inference_ms = total_ms_per_image
#     postprocess_ms = 0.0

#     # FPS μόνο από inference_ms (ίδιο με fps_system στην προσέγγιση αυτή)
#     FPS_model = 1000.0 / inference_ms if inference_ms > 0 else 0.0

#     # FPS με βάση inference + postprocess (εδώ postprocess=0, άρα ίδιο)
#     total_latency_ms = inference_ms + postprocess_ms
#     FPS_overall = 1000.0 / total_latency_ms if total_latency_ms > 0 else 0.0

#     FPS_system = round(fps_system, 2)

#     # -------------------------------------------------------------------------
#     # Load bbox predictions
#     # -------------------------------------------------------------------------
#     bbox_path = "bbox.json"
#     if not os.path.exists(bbox_path):
#         raise FileNotFoundError("❌ bbox.json not found after evaluation!")

#     # Extract COCO AP metrics
#     AP50, AP50_95, AP_S, AP_M, AP_L = extract_coco_metrics(bbox_path, val_json)

#     # Extract global P/R
#     PREC, REC = compute_precision_recall(bbox_path, val_json)

#     # Params
#     params = sum(p.numel() for p in trainer.model.parameters()) / 1e6
#     params = float(params)
#     vram = get_gpu_memory()

#     # --- Efficiency metric (χρησιμοποιούμε το FPS_overall, όπως στα YOLO)
#     mAP = round(AP50_95, 4)
#     efficiency = round(mAP * FPS_overall, 4)

#     results = {
#         "Model": model_name,
#         "Images": n_images,
#         "Image_Size": img_size,
#         "Workers": workers,
#         "Batch": cfg["EvalReader"]["batch_size"],

#         # Accuracy Metrics
#         "mAP50": round(AP50, 4),
#         "mAP50-95": round(AP50_95, 4),
#         "Precision": round(PREC, 4),
#         "Recall": round(REC, 4),
#         "mAP50-95_S": round(AP_S, 4),
#         "mAP50-95_M": round(AP_M, 4),
#         "mAP50-95_L": round(AP_L, 4),

#         # SPEED METRICS (προσεγγιστικά λόγω PaddleDetection API)
#         "Preprocess_ms": round(preprocess_ms, 4),
#         "Inference_ms": round(inference_ms, 4),
#         "Postprocess_ms (NMS)": round(postprocess_ms, 4),

#         # FPS Metrics
#         "FPS_model": round(FPS_model, 2),
#         "FPS_overall": round(FPS_overall, 2),
#         "FPS_system": FPS_system,

#         # Efficiency Metric
#         "Eff_Score 50-95": efficiency,

#         # Resources
#         "VRAM(GB)": round(vram, 3),
#         "Params(M)": round(params, 2),
#     }

#     # Save results
#     save_to_csv(results, output_csv)
#     save_to_csv(results, output_csv_dert)

#     print("\n✅ DONE — RESULTS:")
#     print(json.dumps(results, indent=4))
#     print(f"\n📁 Saved to: {output_csv}\n")


# # =============================================================================
# # RUN
# # =============================================================================

# if __name__ == "__main__":
#     import multiprocessing
#     cfg_path = "D:/Desktop/project/PaddleDetection/configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml"
#     weight_path = "D:/Desktop/project/PaddleDetection/weights/rtdetrv2_r50vd_6x_coco.pdparams"
#     img_root = "D:/Desktop/project/workspace/data/coco2017/images/val2017"
#     output_csv = "D:/Desktop/project/PaddleDetection/dert_starter.csv"
#     output_csv_dert = "D:/Desktop/project/workspace/results/metrics_summary_full.csv"
#     workers = 4
#     img_size = 640
#     batch = 16

#     multiprocessing.freeze_support()
#     run_rt_dert_val(
#         cfg_path=cfg_path,
#         weight_path=weight_path,
#         img_root=img_root,
#         output_csv=output_csv,
#         output_csv_dert = output_csv_dert,
#         img_size=img_size,
#         workers=workers,
#         model_name="RT-DETRv2-R50",
#         batch_size=batch
#     )





# import os, glob, pandas as pd
# import json
# import time
# import yaml
# import paddle
# import subprocess
# import GPUtil
# import numpy as np
# from ppdet.core.workspace import load_config, merge_config
# from ppdet.engine import Trainer


# # =============================================================================
# # HELPERS
# # =============================================================================

# def get_gpu_memory():
#     gpus = GPUtil.getGPUs()
#     if not gpus:
#         return 0.0
#     return gpus[0].memoryUsed / 1024  # GB


# def extract_coco_metrics(result_file, gt_json):
#     """
#     Διαβάζει COCO AP metrics από bbox.json.
#     """
#     from pycocotools.coco import COCO
#     from pycocotools.cocoeval import COCOeval

#     coco_gt = COCO(gt_json)
#     coco_dt = coco_gt.loadRes(result_file)

#     coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
#     coco_eval.evaluate()
#     coco_eval.accumulate()
#     coco_eval.summarize()

#     AP50_95 = coco_eval.stats[0]
#     AP50 = coco_eval.stats[1]
#     AP_small = coco_eval.stats[3]
#     AP_medium = coco_eval.stats[4]
#     AP_large = coco_eval.stats[5]

#     return AP50, AP50_95, AP_small, AP_medium, AP_large


# def compute_precision_recall(result_file, gt_json, iou_threshold=0.5):
#     """
#     Υπολογισμός Global precision/recall (YOLO-style) από bbox.json και COCO GT.
#     """
#     from pycocotools.coco import COCO
#     from pycocotools.cocoeval import COCOeval

#     coco_gt = COCO(gt_json)
#     coco_dt = coco_gt.loadRes(result_file)

#     coco_eval = COCOeval(coco_gt, coco_dt, "bbox")

#     # Set IoU threshold like YOLO PR curves
#     coco_eval.params.iouThrs = np.array([iou_threshold])

#     coco_eval.evaluate()
#     coco_eval.accumulate()

#     precision = coco_eval.eval['precision']  # [T,R,K,A,M]
#     recall = coco_eval.eval['recall']        # [T,K,A,M]

#     # GLOBAL PRECISION
#     valid_precision = precision[precision > -1]
#     global_precision = float(np.mean(valid_precision)) if len(valid_precision) > 0 else 0.0

#     # GLOBAL RECALL
#     valid_recall = recall[recall > -1]
#     global_recall = float(np.mean(valid_recall)) if len(valid_recall) > 0 else 0.0

#     return global_precision, global_recall


# def save_to_csv(data, csv_file):
#     os.makedirs(os.path.dirname(csv_file), exist_ok=True)
#     header = list(data.keys())

#     write_header = not os.path.exists(csv_file)

#     with open(csv_file, "a", encoding="utf-8") as f:
#         if write_header:
#             f.write(",".join(header) + "\n")
#         f.write(",".join(str(v) for v in data.values()) + "\n")


# # =============================================================================
# # MAIN BENCHMARK FUNCTION
# # =============================================================================

# def run_rt_dert_val(cfg_path, weight_path, img_root, output_csv, img_size=640, workers=2, model_name="RT-DETRv2", batch_size=4):
#     print(f"\n🚀 Running Paddle RT-DETRv2 Benchmark: {model_name}")

#     cfg = load_config(cfg_path)
#     merge_config(cfg)

#     # Dataset paths
#     dataset_dir = cfg["EvalDataset"]["dataset_dir"]
#     anno_rel = cfg["EvalDataset"]["anno_path"]
#     val_json = os.path.join(dataset_dir, anno_rel)

#     if not os.path.exists(val_json):
#         raise FileNotFoundError(f"❌ Dataset not found: {val_json}")

#     # Force eval image size
#     cfg["EvalReader"]["sample_transforms"][1]["Resize"]["target_size"] = [img_size, img_size]
#     cfg["EvalReader"]["batch_size"] = batch_size

#     trainer = Trainer(cfg, mode="eval")
#     trainer.load_weights(weight_path)

#     # Run evaluation + time measurement
#     start = time.time()
#     trainer.evaluate()
#     end = time.time()

#     n_images = len(glob.glob(os.path.join(img_root, "*.jpg")))
#     fps = n_images / (end - start)

#     # Load bbox predictions
#     bbox_path = "bbox.json"
#     if not os.path.exists(bbox_path):
#         raise FileNotFoundError("❌ bbox.json not found after evaluation!")

#     # Extract COCO AP metrics
#     AP50, AP50_95, AP_S, AP_M, AP_L = extract_coco_metrics(bbox_path, val_json)

#     # Extract global P/R
#     PREC, REC = compute_precision_recall(bbox_path, val_json)

#     # Params
#     params = sum(p.numel() for p in trainer.model.parameters()) / 1e6
#     params = float(params)
#     vram = get_gpu_memory()

#     # --- Efficiency metric
#     mAP = round(AP50_95, 4)
#     efficiency = round(mAP * fps, 4)

#     results = {
#         "Model": model_name,
#         "Images": n_images,
#         "Image_Size": img_size,
#         "Workers": workers,
#         "Batch": cfg["EvalReader"]["batch_size"],
#         "mAP50": round(AP50, 4),
#         "mAP50-95": round(AP50_95, 4),
#         "Precision": round(PREC, 4),
#         "Recall": round(REC, 4),
#         "mAP50-95_S": round(AP_S, 4),
#         "mAP50-95_M": round(AP_M, 4),
#         "mAP50-95_L": round(AP_L, 4),
#         "FPS": round(fps, 2),
#         "Eff_Score 50-95": efficiency,
#         "VRAM(GB)": round(vram, 3),
#         "Params(M)": round(params, 2),
#     }

#     # Save results
#     # output_csv = "D:/Desktop/project/workspace/results/metrics_summary_full.csv"
#     save_to_csv(results, output_csv)

#     print("\n✅ DONE — RESULTS:")
#     print(json.dumps(results, indent=4))
#     print(f"\n📁 Saved to: {output_csv}\n")


# # =============================================================================
# # RUN
# # =============================================================================

# if __name__ == "__main__":
#     import multiprocessing
#     cfg_path = "D:/Desktop/project/PaddleDetection/configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml"
#     weight_path = "D:/Desktop/project/PaddleDetection/weights/rtdetrv2_r50vd_6x_coco.pdparams"
#     img_root = "D:/Desktop/project/workspace/data/coco2017/images/val2017"
#     output_csv = "D:/Desktop/project/workspace/results/metrics_summary_full.csv"
#     workers = 4
#     img_size=640
#     batch = 16

    
#     multiprocessing.freeze_support()
#     run_rt_dert_val(
#         cfg_path=cfg_path,
#         weight_path=weight_path,
#         img_root=img_root,
#         output_csv = output_csv,
#         img_size=img_size,
#         workers=workers,
#         model_name="RT-DETRv2-R50",
#         batch_size=batch
#     )
