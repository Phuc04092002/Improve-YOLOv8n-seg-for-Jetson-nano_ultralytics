from ultralytics import YOLO
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == "__main__":
    # Load model từ kiến trúc YAML cá nhân
    # Ví dụ: "custom_model.yaml" là kiến trúc mà bạn đã refactor
    model = YOLO("ultralytics/cfg/models/v8/yolov8n-seg.yaml")  # hoặc yolov8n.yaml, yolov8n-seg.yaml...

    # Train model
    model.train(
        data="ultralytics/cfg/datasets/coco-seg2.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        workers=4,  # Windows: nên để 0 để tránh lỗi multiprocessing
        device=0,
        name="yolov8_original_train",
        save=True,
        save_period=1  # Lưu sau mỗi epoch
    )

# Sau khi train xong có thể test hoặc predict
#

