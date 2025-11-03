# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import time

import cv2

from ultralytics import YOLO

# Load model
model = YOLO("runs/segment/yolov8_custom_train5/weights/best.pt")

# Mở camera (0 = webcam mặc định, có thể đổi thành 1,2... nếu nhiều cam)
cap = cv2.VideoCapture(0)

# Thiết lập kích thước khung hình (nếu muốn)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()

    # Predict (trả về kết quả YOLO)
    results = model.predict(frame, imgsz=640, conf=0.5, verbose=False)

    # Vẽ kết quả trực tiếp lên frame
    annotated_frame = results[0].plot()

    # Tính FPS và tốc độ
    end_time = time.time()
    inference_time = (end_time - start_time) * 1000  # ms
    fps = 1 / (end_time - start_time)

    # Hiển thị tốc độ & FPS lên ảnh
    cv2.putText(
        annotated_frame, f"Inference: {inference_time:.1f} ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
    )
    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Hiển thị khung hình
    cv2.imshow("YOLOv8 Segmentation - Camera", annotated_frame)

    # Nhấn ESC để thoát
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
