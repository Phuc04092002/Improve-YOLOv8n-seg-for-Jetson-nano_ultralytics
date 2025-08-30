import onnxruntime as ort
import numpy as np

ONNX_MODEL_PATH = r"runs/segment/yolov8_custom_train5/weights/best.onnx"
session = ort.InferenceSession(ONNX_MODEL_PATH)
input_name = session.get_inputs()[0].name

# Tạo dummy input
dummy_input = np.random.rand(1, 3, 640, 640).astype(np.float32)
ort_outputs = session.run(None, {input_name: dummy_input})

print(f"Shape of output tensor 1 (boxes): {ort_outputs[0].shape}")
print(f"Shape of output tensor 2 (masks): {ort_outputs[1].shape}")