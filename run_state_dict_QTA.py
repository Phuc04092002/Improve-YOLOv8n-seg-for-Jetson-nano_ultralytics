import torch
from ultralytics import YOLO

# load kiến trúc YOLOv8
model = YOLO("yolov8n-seg.yaml")  # hoặc custom.yaml nếu bạn sửa backbone/neck/head
model.load_state_dict(torch.load("qat_model_fixed.pt"))
