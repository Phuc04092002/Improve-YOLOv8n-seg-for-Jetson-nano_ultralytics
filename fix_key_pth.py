# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch

from ultralytics import YOLO

# Load state_dict gốc
state_dict = torch.load("qat_outputs_fixed_v3/model_qat_int8.pth", weights_only=True)

# Thêm prefix "model.model." vào key
new_state_dict = {}
for k, v in state_dict.items():
    new_key = f"model.model.{k}"  # thêm tiền tố
    new_state_dict[new_key] = v

# Load kiến trúc YOLO
model = YOLO("yolov8-seg.yaml", verbose=True)  # hoặc custom.yaml của bạn
model.model.load_state_dict(new_state_dict, strict=False)

# Save lại thành YOLO model hợp lệ
model.save("qat_model_fixed.pt")
