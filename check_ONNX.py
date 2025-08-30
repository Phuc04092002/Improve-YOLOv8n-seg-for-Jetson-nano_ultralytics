import onnx

onnx_model_path = "yolov8n_seg_qat_int8_real.onnx"
model = onnx.load(onnx_model_path)

# Kiểm tra tính hợp lệ
onnx.checker.check_model(model)
print("✅ ONNX model is valid")

# Liệt kê các node và ops
for node in model.graph.node:
    print(f"{node.op_type} -> {node.name}")
