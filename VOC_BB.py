# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
import time
from glob import glob

import cv2
import numpy as np
import onnxruntime as ort

# ===============================
# CONFIG
# ===============================
onnx_path = "runs/segment/yolov8_custom_train5/weights/best.onnx"
img_dir = "D:/new_dataset1/images/test"
label_dir = "D:/new_dataset1/labels/test"
class_names = ["person", "bicycle", "car", "motorcycle", "traffic light"]
conf_thres = 0.25
iou_thres = 0.5
input_size = 640

save_folder = "results"
os.makedirs(save_folder, exist_ok=True)

colors = [
    (0, 255, 0),  # person
    (255, 0, 0),  # bicycle
    (0, 0, 255),  # car
    (255, 255, 0),  # motorcycle
    (0, 255, 255),  # traffic light
]

# ===============================
# ONNX session
# ===============================
session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name


# ===============================
# Helper functions
# ===============================
def preprocess(img, new_shape=(640, 640)):
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

    img_out = img_padded[:, :, ::-1].transpose(2, 0, 1)
    img_out = np.ascontiguousarray(img_out, dtype=np.float32) / 255.0
    return img_out, r, dw, dh


def box_iou(box1, box2):
    xa = max(box1[0], box2[0])
    ya = max(box1[1], box2[1])
    xb = min(box1[2], box2[2])
    yb = min(box1[3], box2[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def load_gt_boxes(label_path, w, h):
    boxes, classes = [], []
    if not os.path.exists(label_path):
        return boxes, classes
    with open(label_path) as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            cls = int(vals[0])
            coords = vals[1:]
            xs = coords[0::2]
            ys = coords[1::2]
            xs = [x * w for x in xs]
            ys = [y * h for y in ys]
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            boxes.append([xmin, ymin, xmax, ymax])
            classes.append(cls)
    return boxes, classes


# ===============================
# Collect predictions & ground truth
# ===============================
preds, gts = [], []
times = []
img_files = glob(os.path.join(img_dir, "*.jpg"))

for img_path in img_files:
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    label_path = os.path.join(label_dir, img_name + ".txt")
    img = cv2.imread(img_path)
    h0, w0 = img.shape[:2]
    inp, r, dw, dh = preprocess(img)
    inp = np.expand_dims(inp, axis=0)

    # Inference
    t1 = time.time()
    outputs = session.run(None, {input_name: inp})
    t2 = time.time()
    times.append((t2 - t1) * 1000)

    dets, proto = outputs[0][0], outputs[1][0]
    dets = dets[dets[:, 4] > conf_thres]

    img_out = img.copy()

    for det in dets:
        x1, y1, x2, y2, conf, cls = det[:6]
        # scale back
        x1 = (x1 - dw) / r
        x2 = (x2 - dw) / r
        y1 = (y1 - dh) / r
        y2 = (y2 - dh) / r
        x1 = max(0, min(x1, w0 - 1))
        x2 = max(0, min(x2, w0 - 1))
        y1 = max(0, min(y1, h0 - 1))
        y2 = max(0, min(y2, h0 - 1))
        preds.append({"image_id": img_name, "class": int(cls), "score": float(conf), "bbox": [x1, y1, x2, y2]})

        # Vẽ bbox
        color = colors[int(cls)]
        cv2.rectangle(img_out, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = f"{class_names[int(cls)]} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_y = int(y1 - th - 4)
        if label_y < 0:
            label_y = int(y2 + 4)
        cv2.rectangle(img_out, (int(x1), label_y), (int(x1 + tw + 4), label_y + th), color, -1)
        cv2.putText(img_out, label, (int(x1 + 2), label_y + th - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Save image
    cv2.imwrite(os.path.join(save_folder, img_name + ".jpg"), img_out)

    # Ground truth
    gt_boxes, gt_classes = load_gt_boxes(label_path, w0, h0)
    for b, c in zip(gt_boxes, gt_classes):
        gts.append({"image_id": img_name, "class": c, "bbox": b})


# ===============================
# Compute mAP VOC style
# ===============================
def compute_map(preds, gts, iou_thr=0.5):
    aps = []
    for cls in set([g["class"] for g in gts]):
        cls_preds = [p for p in preds if p["class"] == cls]
        cls_gts = [g for g in gts if g["class"] == cls]

        cls_preds.sort(key=lambda x: -x["score"])
        npos = len(cls_gts)
        tp = np.zeros(len(cls_preds))
        fp = np.zeros(len(cls_preds))
        matched = {}

        for i, p in enumerate(cls_preds):
            best_iou, best_gt = 0, -1
            for j, g in enumerate(cls_gts):
                if g["image_id"] != p["image_id"]:
                    continue
                if j in matched:
                    continue
                iou = box_iou(p["bbox"], g["bbox"])
                if iou > best_iou:
                    best_iou, best_gt = iou, j
            if best_iou >= iou_thr:
                tp[i] = 1
                matched[best_gt] = True
            else:
                fp[i] = 1

        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        recall = tp / (npos + 1e-6)
        precision = tp / (tp + fp + 1e-6)

        ap = 0
        for t in np.linspace(0, 1, 11):
            p = max(precision[recall >= t]) if np.any(recall >= t) else 0
            ap += p / 11
        aps.append(ap)
        print(f"Class {cls} ({class_names[cls]}): AP={ap:.4f}")
    return np.mean(aps)


mAP = compute_map(preds, gts, iou_thr=iou_thres)
print(f"\nImages: {len(img_files)}")
print(f"mAP@{iou_thres}: {mAP:.4f}, Avg time: {np.mean(times):.2f} ms/img")
print(f"All annotated images saved to folder: {save_folder}")
