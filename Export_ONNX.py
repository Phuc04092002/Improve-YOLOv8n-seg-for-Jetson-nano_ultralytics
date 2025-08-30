#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export YOLOv8-seg .pt model into ONNX
with bounding boxes, class scores, masks included.
"""
import sys
from pathlib import Path
from ultralytics import YOLO

# ---------------- CONFIG ----------------
FULL_PT = r"runs/segment/yolov8_custom_train5/weights/best.pt"  # Full YOLOv8 wrapper (.pt)
EXPORT_DIR = r"onnx_export"
OPSET = 13
IMGSZ = 640
BATCH = 1
TASK = "segment"  # task: 'detect', 'segment', 'classify', etc.
# ----------------------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def main():
    export_dir = ensure_dir(Path(EXPORT_DIR))

    # Load full PyTorch YOLOv8 wrapper
    print(f"[INFO] Loading YOLO model from: {FULL_PT}")
    yolo = YOLO(FULL_PT)
    yolo.task = TASK  # đảm bảo task đúng

    # Export ONNX
    export_path = export_dir / "model_export.onnx"
    print(f"[INFO] Exporting ONNX to: {export_path}")
    try:
        yolo.export(
            format="onnx",
            opset=OPSET,
            imgsz=IMGSZ,
            batch=BATCH,
            dynamic=True,     # ONNX dynamic shape
            optimize=True,    # fuse conv + bn nếu có
            simplify=True     # simplify graph
        )
        print("[✅ DONE] ONNX export completed.")
    except Exception as e:
        print("[ERROR] ONNX export failed:", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
