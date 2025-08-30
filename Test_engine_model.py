# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import time

import cv2
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt

# ====== Config ======
engine_path = "qat_model_fixed.engine"
img_path = "Test_file/test_image2.jpg"
img_size = 640

# ====== Load TensorRT engine ======
logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)

with open(engine_path, "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

# ====== Prepare input image ======
img0 = cv2.imread(img_path)
img = cv2.resize(img0, (img_size, img_size))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

# ====== Allocate GPU memory ======
inputs, outputs, bindings = [], [], []
stream = cuda.Stream()

for binding in engine:
    shape = context.get_binding_shape(binding)
    dtype = trt.nptype(engine.get_binding_dtype(binding))
    size = trt.volume(shape) * engine.max_batch_size
    host_mem = cuda.pagelocked_empty(size, dtype)
    device_mem = cuda.mem_alloc(host_mem.nbytes)
    bindings.append(int(device_mem))
    if engine.binding_is_input(binding):
        inputs.append({"host": host_mem, "device": device_mem, "shape": shape})
    else:
        outputs.append({"host": host_mem, "device": device_mem, "shape": shape})

# Copy input to GPU
np.copyto(inputs[0]["host"], img.ravel())
cuda.memcpy_htod_async(inputs[0]["device"], inputs[0]["host"], stream)

# ====== Run inference ======
t1 = time.time()
context.execute_v2(bindings)
stream.synchronize()
t2 = time.time()

print(f"Inference time: {(t2 - t1) * 1000:.2f} ms")

# Copy outputs back
for out in outputs:
    cuda.memcpy_dtoh_async(out["host"], out["device"], stream)
stream.synchronize()

# ====== Reshape outputs ======
out0 = outputs[0]["host"].reshape(outputs[0]["shape"])  # detections
out1 = outputs[1]["host"].reshape(outputs[1]["shape"])  # proto (segmentation)

print("Detections shape:", out0.shape)
print("Proto shape:", out1.shape)
