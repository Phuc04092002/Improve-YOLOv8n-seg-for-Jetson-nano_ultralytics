import cv2
import numpy as np
import onnxruntime as ort
import os

# ============ CONFIG ============
onnx_path = "runs/segment/yolov8_custom_train5/weights/best.onnx"
test_folder = "D:/new_dataset1/images/test"
gt_folder = "D:/new_dataset1/labels/test"  # folder chứa GT txt
conf_thres = 0.3
iou_thres = 0.5

# Class names (chỉnh theo dataset của bạn)
class_names = ['person', 'bicycle', 'car', 'motorcycle','traffic light']
# colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(len(class_names))]
colors = [
    (0, 255, 0),    # person
    (255, 0, 0),    # bicycle
    (0, 0, 255),    # car
    (255, 255, 0),  # motorcycle
    (0, 255, 255)   # traffic light
]

# color = colors[class_names % len(colors)]


# ============ UTILS ============
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize + pad để giữ nguyên tỉ lệ giống YOLOv8"""
    shape = im.shape[:2]  # h,w
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = dw / 2, dh / 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)

def scale_boxes(boxes, r, dwdh, img0_shape):
    """Scale boxes từ letterbox về ảnh gốc"""
    boxes[:, [0, 2]] -= dwdh[0]  # x padding
    boxes[:, [1, 3]] -= dwdh[1]  # y padding
    boxes[:, :4] /= r
    boxes[:, 0::2] = boxes[:, 0::2].clip(0, img0_shape[1])
    boxes[:, 1::2] = boxes[:, 1::2].clip(0, img0_shape[0])
    return boxes

def process_mask(proto, masks_in, boxes, img_shape):
    c, mh, mw = proto.shape
    masks = masks_in @ proto.reshape(c, -1)      # (n, mh*mw)
    masks = masks.reshape(-1, mh, mw)            # (n, mh, mw)
    # resize mask về ảnh letterbox
    masks = np.array([cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR) for m in masks])
    # (n,h,w)

    final_masks = []
    for m, box in zip(masks, boxes.astype(int)):
        x1, y1, x2, y2 = box
        crop_mask = np.zeros_like(m, dtype=bool)
        crop_mask[y1:y2, x1:x2] = m[y1:y2, x1:x2] > 0.5
        final_masks.append(crop_mask)
    return np.stack(final_masks, axis=0)

# ============ LOAD MODEL ============
session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

# ============ LOAD IMAGE ============
save_folder = "resultsPicture"
os.makedirs(save_folder, exist_ok=True)

# ============ PROCESS 1 LOẠT ẢNH ============
for img_name in os.listdir(test_folder):
    if not img_name.lower().endswith((".jpg", ".png")):
        continue

    img_path = os.path.join(test_folder, img_name)
    img0 = cv2.imread(img_path)
    h0, w0 = img0.shape[:2]

    # Letterbox + input
    img, r, dwdh = letterbox(img0, (640, 640))
    img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_in = img_in.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

    # Inference
    outputs = session.run(None, {session.get_inputs()[0].name: img_in})
    detections, proto = outputs
    proto = proto[0]

    # Parse detections
    boxes_scaled, classes, scores, mask_coef = [], [], [], []
    for det in detections[0]:
        score = det[4]
        if score < conf_thres:
            continue
        boxes_scaled.append(det[:4])
        classes.append(int(det[5]))
        scores.append(score)
        mask_coef.append(det[6:])

    if boxes_scaled:
        boxes_scaled = np.array(boxes_scaled)
        mask_coef = np.array(mask_coef).reshape(len(boxes_scaled), proto.shape[0])
    else:
        boxes_scaled = np.zeros((0, 4))
        mask_coef = np.zeros((0, proto.shape[0]))
        classes = []
        scores = []

    # Vẽ lên ảnh letterbox 640x640
    img_out = img.copy()
    if len(boxes_scaled) > 0:
        masks = process_mask(proto, mask_coef, boxes_scaled, (img.shape[0], img.shape[1]))
        for box, cls, score, mask in zip(boxes_scaled, classes, scores, masks):
            color = colors[cls]
            x1, y1, x2, y2 = box.astype(int)

            # Box
            cv2.rectangle(img_out, (x1, y1), (x2, y2), color, 2)

            # Label
            label = f"{class_names[cls]} {score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_y = y1 - th - 4
            if label_y < 0:
                label_y = y2 + 4
            cv2.rectangle(img_out, (x1, label_y), (x1 + tw + 4, label_y + th), color, -1)
            cv2.putText(img_out, label, (x1 + 2, label_y + th - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Mask
            colored_mask = np.zeros_like(img_out, dtype=np.uint8)
            for i in range(3):
                colored_mask[:, :, i] = mask * color[i]
            img_out = cv2.addWeighted(img_out, 1.0, colored_mask, 0.6, 0)

    # Lưu kết quả
    save_path = os.path.join(save_folder, img_name)
    cv2.imwrite(save_path, img_out)
    print(f"✅ Saved {save_path}")


# import os
# import cv2
# import numpy as np
# import onnxruntime as ort
# import time
# from glob import glob
#
# # ===============================
# # CONFIG
# # ===============================
# onnx_path = "runs/segment/yolov8_custom_train5/weights/best.onnx"
# img_dir = "D:/new_dataset1/images/test"
# label_dir = "D:/new_dataset1/labels/test"
# class_names = ['person', 'bicycle', 'car', 'motorcycle','traffic light']
# conf_thres = 0.25
# iou_thres = 0.5
# input_size = 640
#
# # ===============================
# # ONNX session
# # ===============================
# session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
# input_name = session.get_inputs()[0].name
#
# # ===============================
# # Helper
# # ===============================
# def preprocess(img, new_shape=(640, 640)):
#     h, w = img.shape[:2]
#     r = min(new_shape[0] / h, new_shape[1] / w)
#     new_unpad = (int(round(w * r)), int(round(h * r)))
#     dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
#     dw /= 2
#     dh /= 2
#
#     img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
#     top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
#     left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
#     img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114,114,114))
#
#     img_out = img_padded[:, :, ::-1].transpose(2, 0, 1)
#     img_out = np.ascontiguousarray(img_out, dtype=np.float32) / 255.0
#     return img_out, r, dw, dh
#
# def mask_iou(mask1, mask2):
#     inter = np.logical_and(mask1, mask2).sum()
#     union = np.logical_or(mask1, mask2).sum()
#     return inter / union if union > 0 else 0
#
# def load_gt_mask(label_path, h, w):
#     masks, boxes, classes = [], [], []
#     with open(label_path, "r") as f:
#         for line in f:
#             vals = list(map(float, line.strip().split()))
#             cls, x, y, bw, bh = vals[:5]
#             poly = vals[5:]
#             if len(poly) > 0:
#                 pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
#                 pts[:, 0] *= w
#                 pts[:, 1] *= h
#                 mask = np.zeros((h, w), dtype=np.uint8)
#                 cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
#                 masks.append(mask.astype(bool))
#             # bbox GT (pixel)
#             x1 = (x - bw/2) * w
#             y1 = (y - bh/2) * h
#             x2 = (x + bw/2) * w
#             y2 = (y + bh/2) * h
#             boxes.append([x1,y1,x2,y2])
#             classes.append(int(cls))
#     return boxes, masks, classes
#
# # ===============================
# # Evaluation loop
# # ===============================
# tp, fp, fn = 0, 0, 0
# total_iou = 0
# times = []
#
# img_files = glob(os.path.join(img_dir, "*.jpg"))
#
# for img_path in img_files:
#     img_name = os.path.splitext(os.path.basename(img_path))[0]
#     label_path = os.path.join(label_dir, img_name + ".txt")
#
#     img = cv2.imread(img_path)
#     h0, w0 = img.shape[:2]
#     inp, r, dw, dh = preprocess(img)
#     inp = np.expand_dims(inp, axis=0)
#
#     # Run inference
#     t1 = time.time()
#     outputs = session.run(None, {input_name: inp})
#     t2 = time.time()
#     times.append((t2 - t1) * 1000)
#
#     # Parse ONNX output (YOLOv8-seg: det + mask)
#     dets, proto = outputs[0][0], outputs[1][0]  # [num, 6+nmask], [mask_dim,h,w]
#     dets = dets[dets[:,4] > conf_thres]
#
#     preds_masks = []
#     for det in dets:
#         x1,y1,x2,y2,conf,cls,*mask_coefs = det
#         mask = np.dot(mask_coefs, proto.reshape(proto.shape[0], -1))
#         mask = mask.reshape(proto.shape[1], proto.shape[2])
#         mask = (mask > 0.5).astype(np.uint8)
#         mask = cv2.resize(mask, (input_size, input_size))
#         # undo padding/scale
#         mask = mask[int(dh):int(dh+h0*r), int(dw):int(dw+w0*r)]
#         mask = cv2.resize(mask, (w0, h0))
#         preds_masks.append(mask.astype(bool))
#
#     # Load GT
#     if not os.path.exists(label_path):
#         continue
#     gt_boxes, gt_masks, gt_classes = load_gt_mask(label_path, h0, w0)
#
#     matched = set()
#     for gmask in gt_masks:
#         best_iou, best_pred = 0, -1
#         for i, pmask in enumerate(preds_masks):
#             iou = mask_iou(gmask, pmask)
#             if iou > best_iou:
#                 best_iou, best_pred = iou, i
#         if best_iou > iou_thres:
#             tp += 1
#             total_iou += best_iou
#             matched.add(best_pred)
#         else:
#             fn += 1
#     fp += len(preds_masks) - len(matched)
#
# # ===============================
# # Metrics
# # ===============================
# precision = tp / (tp + fp + 1e-6)
# recall = tp / (tp + fn + 1e-6)
# mean_iou = total_iou / (tp + 1e-6)
# avg_time = np.mean(times)
#
# print(f"Images: {len(img_files)}")
# print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, mIoU: {mean_iou:.4f}, Avg time: {avg_time:.2f} ms/img")
