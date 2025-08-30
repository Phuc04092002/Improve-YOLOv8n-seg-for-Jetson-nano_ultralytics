# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic

# ==== 1. Đặt đường dẫn file ====
onnx_fp32 = "runs/segment/yolov8_custom_train5/weights/best.onnx"  # file bạn vừa export từ PyTorch
onnx_int8 = "yolov8n_seg_qat_int8_best.onnx"  # file INT8 sau khi quantize

# ==== 2. Kiểm tra model gốc ====
model = onnx.load(onnx_fp32)
onnx.checker.check_model(model)
print(f"✅ Model {onnx_fp32} load OK")

# ==== 3. Quantize về INT8 ====
quantize_dynamic(
    model_input=onnx_fp32,
    model_output=onnx_int8,
    weight_type=QuantType.QInt8,  # hoặc QuantType.QUInt8
)

print(f"🎉 Đã tạo file ONNX INT8: {onnx_int8}")
