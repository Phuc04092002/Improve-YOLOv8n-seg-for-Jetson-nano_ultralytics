YOLOv8n-seg Optimization for Jetson Nano

This project focuses on improving and optimizing the YOLOv8n-seg model for real-time deployment on edge devices with limited hardware resources, specifically the NVIDIA Jetson Nano.

📌 Overview

Deep learning models such as YOLOv8 deliver state-of-the-art performance in object detection and instance segmentation. However, deploying them on embedded devices like Jetson Nano is challenging due to limited memory (4GB RAM) and GPU resources (128-core Maxwell CUDA).

This project introduces:

Model architecture modifications (replacing Conv & C2f modules with GhostConv & C3Ghost).

Custom dataset preparation (balanced subset of COCO2017 with 5 classes: person, bicycle, car, motorcycle, traffic light).

Quantization Aware Training (QAT) to convert the model to INT8 while preserving accuracy.

ONNX & TensorRT conversion for deployment on Jetson Nano.

The main goal is to achieve real-time inference (≥15 FPS) while keeping accuracy at an acceptable level.

🚀 Features

Lightweight YOLOv8n-seg custom architecture.

Balanced dataset (6000 images, 5 target classes).

Pruning + QAT for smaller model size.

Export to ONNX and TensorRT engine.

Benchmarked speed and accuracy trade-offs.

📂 Dataset

Source: COCO2017

Classes used:

Person (0)

Bicycle (1)

Car (2)

Motorcycle (3)

Traffic Light (4)

Total: 6000 balanced images.

Annotations converted to YOLO-Seg format using pycocotools.

🏗️ Model Architecture

Base: YOLOv8n-seg (Ultralytics)

Replacements:

Conv → GhostConv

C2f → C3Ghost

Optimization:

Keep original early layers (retain low-level features).

Replace deeper backbone & neck layers for reduced FLOPs & params.

⚙️ Training

Epochs: 100

Batch size: 8

Image size: 640×640

Optimizer: SGD/Adam (default in Ultralytics)

Validation: mAP, AP@0.5, AP@0.5:0.95

📉 Results
Model	Size	Params	FLOPs	mAP@0.5	Speed (ms/img)
YOLOv8n-seg (original)	~13 MB	3.4M	12.8 GFLOPs	0.627	~80 ms
YOLOv8n-seg (custom)	~8.5 MB	2.2M	10.6 GFLOPs	0.585	~62 ms
YOLOv8n-seg (custom QAT INT8)	~2.5 MB	2.2M	10.6 GFLOPs	~0.58	~61 ms

Achieved ~16 FPS on Jetson Nano (INT8, TensorRT).

Acceptable accuracy drop while gaining real-time performance.

🔧 Deployment on Jetson Nano

Export PyTorch model → ONNX.

Quantize with QAT → INT8 ONNX.

Convert ONNX → TensorRT engine (.engine).

Run inference with TensorRT runtime.

📌 Limitations & Future Work

TensorRT conversion was tested on Jetson Nano Linux environment (not supported directly on Windows).

Small-object segmentation (e.g., bicycle, traffic light) still needs improvement.

Training dataset limited by hardware → larger dataset & stronger GPU could enhance accuracy.

📚 References

Ultralytics YOLOv8

NVIDIA TensorRT

COCO Dataset
