# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
import torch.quantization

from ultralytics import YOLO

# ==== Cấu hình ====
qat_ckpt = "qat_outputs_fixed_v3/model_qat_int8.pth"  # checkpoint QAT đã train
onnx_output = "yolov8n_seg_qat_int8.onnx"
imgsz = 640  # size input (HxW)

# ==== 1. Load mô hình YOLO gốc ====
# Nếu bạn đã dùng custom yaml, thay "yolov8n-seg.yaml" bằng file yaml của bạn
model = YOLO("yolov8-seg.yaml").model

# ==== 2. Load weight QAT checkpoint ====
checkpoint = torch.load(qat_ckpt, map_location="cpu")
if "model" in checkpoint:  # nếu bạn lưu full object
    model.load_state_dict(checkpoint["model"].state_dict(), strict=False)
else:  # nếu bạn chỉ lưu state_dict
    model.load_state_dict(checkpoint, strict=False)

print("✅ Đã load xong checkpoint QAT")

# ==== 3. Convert sang INT8 thật sự ====
model.eval()
model.cpu()
model = torch.quantization.convert(model)

print("✅ Đã convert sang INT8 thực sự")

# ==== 4. Export sang ONNX ====
dummy_input = torch.randn(1, 3, imgsz, imgsz)
torch.onnx.export(
    model,
    dummy_input,
    onnx_output,
    input_names=["images"],
    output_names=["output"],
    opset_version=13,
    do_constant_folding=True,
    dynamic_axes={
        "images": {0: "batch"},  # batch dynamic
        "output": {0: "batch"},  # output dynamic
    },
)

print(f"🎉 Export thành công -> {onnx_output}")
