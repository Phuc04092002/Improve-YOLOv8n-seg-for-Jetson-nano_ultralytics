# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os

import tensorrt as trt

# Thêm thư mục lib trước khi import TensorRT
os.add_dll_directory(r"C:\Program Files\TensorRT-8.5.3.1\bin")
os.add_dll_directory(r"C:\Program Files\TensorRT-8.5.3.1\lib")

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

onnx_model_path = "yolov8n_seg_qat_int8_real.onnx"
trt_engine_path = "yolov8n_seg_qat_int8.trt"

with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
    1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
    # ==== BuilderConfig mới ====
    config = builder.create_builder_config()
    # Thay vì max_workspace_size dùng set_memory_pool_limit
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

    # Bật INT8
    config.set_flag(trt.BuilderFlag.INT8)

    # Load ONNX
    with open(onnx_model_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise ValueError("Failed to parse ONNX model")

    # Build engine
    engine = builder.build_engine(network, config)

    # Lưu engine
    with open(trt_engine_path, "wb") as f:
        f.write(engine.serialize())

print("✅ TensorRT engine built successfully")
