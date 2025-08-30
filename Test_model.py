# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import cv2

from ultralytics import YOLO

img = cv2.imread("Test_file/test_image6.jpg")
new_size = (640, 640)
resized_img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)

model = YOLO("runs/segment/yolov8_custom_train5/weights/best.pt")
model.predict(resized_img, save=True, imgsz=640, conf=0.3)

# import cv2
# import os
# import numpy as np
# from ultralytics import YOLO
# from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
#
# def load_yolo_mask(label_path, img_shape):
#     h, w = img_shape[:2]
#     mask = np.zeros((h, w), dtype=np.uint8)
#
#     if not os.path.exists(label_path):
#         return mask  # không có label thì trả về mask rỗng
#
#     with open(label_path, "r") as f:
#         for line in f.readlines():
#             parts = line.strip().split()
#             cls = int(parts[0])
#             coords = list(map(float, parts[1:]))
#
#             # Chuyển về pixel
#             points = np.array([(coords[i] * w, coords[i+1] * h) for i in range(0, len(coords), 2)], dtype=np.int32)
#             cv2.fillPoly(mask, [points], 1)
#
#     return mask
#
# img = cv2.imread("Test_file/test_image2.jpg")
# new_size = (640,640)
# resized_img = cv2.resize(img, new_size, interpolation = cv2.INTER_LINEAR)
#
# # Load model và predict
# model = YOLO("runs/segment/yolov8_custom_train5/weights/best.pt")
# results = model.predict(resized_img, save = True, imgsz=640, conf=0.5)
#
# # Lấy mask dự đoán đầu tiên
# pred_mask = results[0].masks.data[0].cpu().numpy()  # (h,w) giá trị 0/1
# pred_mask = (pred_mask > 0.5).astype(np.uint8)
#
# # Load mask ground truth
# label_file = "D:/coco/labels/val2017/000000026204.txt"
# gt_mask = load_yolo_mask(label_file, resized_img.shape)
#
# # Flatten để tính metric
# y_true = gt_mask.flatten()
# y_pred = pred_mask.flatten()
#
# # Confusion matrix
# tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
#
# # Metrics
# accuracy = accuracy_score(y_true, y_pred)
# precision = precision_score(y_true, y_pred)
# recall = recall_score(y_true, y_pred)   # sensitivity
# specificity = tn / (tn + fp)
# f1 = f1_score(y_true, y_pred)
#
# print(f"Accuracy: {accuracy:.4f}")
# print(f"Precision: {precision:.4f}")
# print(f"Recall (Sensitivity): {recall:.4f}")
# print(f"Specificity: {specificity:.4f}")
# print(f"F1-score: {f1:.4f}")
