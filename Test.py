import os
import cv2
import numpy as np
import onnxruntime as ort
import time
from glob import glob

# ===============================
# CONFIG
# ===============================
onnx_path = "runs/segment/yolov8_custom_train5/weights/best.onnx"
img_dir = "D:/new_dataset1/images/test"
label_dir = "D:/new_dataset1/labels/test"
class_names = ['person', 'bicycle', 'car', 'motorcycle','traffic light']
conf_thres = 0.25
iou_thres = 0.5
input_size = 640
save_folder = "results_segment"
os.makedirs(save_folder, exist_ok=True)

colors = [
    (0, 255, 0),    # person
    (255, 0, 0),    # bicycle
    (0, 0, 255),    # car
    (255, 255, 0),  # motorcycle
    (0, 255, 255)   # traffic light
]

# ===============================
# ONNX session
# ===============================
session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

# ===============================
# HELPER FUNCTIONS
# ===============================
def preprocess(img, new_shape=(640,640)):
    h, w = img.shape[:2]
    r = min(new_shape[0]/h, new_shape[1]/w)
    new_unpad = (int(round(w*r)), int(round(h*r)))
    dw, dh = new_shape[1]-new_unpad[0], new_shape[0]-new_unpad[1]
    dw /= 2
    dh /= 2
    img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh-0.1)), int(round(dh+0.1))
    left, right = int(round(dw-0.1)), int(round(dw+0.1))
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=(114,114,114))
    img_out = img_padded[:,:,::-1].transpose(2,0,1)
    img_out = np.ascontiguousarray(img_out, dtype=np.float32)/255.0
    return img_out, r, dw, dh

def box_iou(box1, box2):
    xa = max(box1[0], box2[0])
    ya = max(box1[1], box2[1])
    xb = min(box1[2], box2[2])
    yb = min(box1[3], box2[3])
    inter = max(0, xb-xa)*max(0, yb-ya)
    area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    area2 = (box2[2]-box2[0])*(box2[3]-box2[1])
    union = area1+area2-inter
    return inter/union if union>0 else 0

def mask_iou(mask1, mask2):
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return inter/union if union>0 else 0

def load_gt_mask(label_path, h, w):
    boxes, masks, classes = [], [], []
    if not os.path.exists(label_path):
        return boxes, masks, classes
    with open(label_path,"r") as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            cls, x, y, bw, bh = vals[:5]
            poly = vals[5:]
            if len(poly)>=6:
                pts = np.array(poly,dtype=np.float32).reshape(-1,2)
                pts[:,0]*=w
                pts[:,1]*=h
                mask = np.zeros((h,w),dtype=np.uint8)
                cv2.fillPoly(mask,[pts.astype(np.int32)],1)
                masks.append(mask.astype(bool))
            # bbox GT
            x1 = (x-bw/2)*w
            y1 = (y-bh/2)*h
            x2 = (x+bw/2)*w
            y2 = (y+bh/2)*h
            boxes.append([x1,y1,x2,y2])
            classes.append(int(cls))
    return boxes, masks, classes

# ===============================
# EVALUATION LOOP
# ===============================
preds, gts = [], []
times = []
tp_mask, fn_mask, total_iou_mask = 0, 0, 0

img_files = glob(os.path.join(img_dir,"*.jpg"))

for img_path in img_files:
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    label_path = os.path.join(label_dir, img_name+".txt")
    img = cv2.imread(img_path)
    h0, w0 = img.shape[:2]
    inp, r, dw, dh = preprocess(img)
    inp = np.expand_dims(inp,axis=0)

    # Inference
    t1 = time.time()
    outputs = session.run(None, {input_name: inp})
    t2 = time.time()
    times.append((t2-t1)*1000)

    dets, proto = outputs[0][0], outputs[1][0]
    dets = dets[dets[:,4]>conf_thres]

    img_out = img.copy()
    preds_masks = []

    # Process detections
    for det in dets:
        x1,y1,x2,y2,conf,cls,*mask_coefs = det
        # scale back to original image
        x1 = (x1-dw)/r; x2 = (x2-dw)/r
        y1 = (y1-dh)/r; y2 = (y2-dh)/r
        x1,x2 = max(0,min(x1,w0-1)), max(0,min(x2,w0-1))
        y1,y2 = max(0,min(y1,h0-1)), max(0,min(y2,h0-1))

        preds.append({"image_id": img_name, "class": int(cls),
                      "score": float(conf), "bbox":[x1,y1,x2,y2]})

        # Mask
        if len(mask_coefs)>0:
            mask = np.dot(mask_coefs, proto.reshape(proto.shape[0],-1))
            mask = mask.reshape(proto.shape[1],proto.shape[2])
            mask = (mask>0.5).astype(np.uint8)
            mask = cv2.resize(mask, (input_size,input_size))
            mask = mask[int(dh):int(dh+h0*r), int(dw):int(dw+w0*r)]
            mask = cv2.resize(mask, (w0,h0))
            preds_masks.append(mask.astype(bool))
            # overlay mask
            color = colors[int(cls)]
            colored_mask = np.zeros_like(img_out,dtype=np.uint8)
            for c in range(3):
                colored_mask[:,:,c] = mask * color[c]
            img_out = cv2.addWeighted(img_out, 1.0, colored_mask, 0.5, 0)

        # Draw bbox
        color = colors[int(cls)]
        cv2.rectangle(img_out,(int(x1),int(y1)),(int(x2),int(y2)),color,2)
        label = f"{class_names[int(cls)]} {conf:.2f}"
        (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,0.6,2)
        ly = int(y1-th-4) if y1-th-4>0 else int(y2+4)
        cv2.rectangle(img_out,(int(x1),ly),(int(x1+tw+4),ly+th),color,-1)
        cv2.putText(img_out,label,(int(x1+2),ly+th-2),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)

    # Save image
    cv2.imwrite(os.path.join(save_folder,img_name+".jpg"),img_out)

    # Load GT
    gt_boxes, gt_masks, gt_classes = load_gt_mask(label_path,h0,w0)
    for b,c in zip(gt_boxes,gt_classes):
        gts.append({"image_id": img_name, "class": c, "bbox":b})

    # Compute mask IoU
    matched = set()
    for gmask in gt_masks:
        best_iou, best_pred = 0, -1
        for i, pmask in enumerate(preds_masks):
            iou = mask_iou(gmask, pmask)
            if iou>best_iou:
                best_iou, best_pred = iou, i
        if best_iou>iou_thres:
            tp_mask += 1
            total_iou_mask += best_iou
            matched.add(best_pred)
        else:
            fn_mask += 1
    fp_mask = len(preds_masks)-len(matched)

# ===============================
# Compute mAP & mask metrics
# ===============================
def compute_map(preds, gts, iou_thr=0.5):
    aps=[]
    for cls in set([g["class"] for g in gts]):
        cls_preds = [p for p in preds if p["class"]==cls]
        cls_gts   = [g for g in gts if g["class"]==cls]
        cls_preds.sort(key=lambda x:-x["score"])
        npos = len(cls_gts)
        tp = np.zeros(len(cls_preds))
        fp = np.zeros(len(cls_preds))
        matched = {}
        for i,p in enumerate(cls_preds):
            best_iou, best_gt = 0,-1
            for j,g in enumerate(cls_gts):
                if g["image_id"]!=p["image_id"]: continue
                if j in matched: continue
                iou = box_iou(p["bbox"],g["bbox"])
                if iou>best_iou:
                    best_iou, best_gt = iou,j
            if best_iou>=iou_thr:
                tp[i]=1
                matched[best_gt]=True
            else:
                fp[i]=1
        fp=np.cumsum(fp)
        tp=np.cumsum(tp)
        recall = tp/(npos+1e-6)
        precision = tp/(tp+fp+1e-6)
        ap=0
        for t in np.linspace(0,1,11):
            p = max(precision[recall>=t]) if np.any(recall>=t) else 0
            ap += p/11
        aps.append(ap)
        print(f"Class {cls} ({class_names[cls]}): AP={ap:.4f}")
    return np.mean(aps)

mAP = compute_map(preds,gts,iou_thr=iou_thres)
precision_mask = tp_mask / (tp_mask+fp_mask+1e-6)
recall_mask = tp_mask / (tp_mask+fn_mask+1e-6)
mean_iou_mask = total_iou_mask / (tp_mask+1e-6)

print(f"\nImages: {len(img_files)}")
print(f"mAP@{iou_thres}: {mAP:.4f}")
print(f"Mask Precision: {precision_mask:.4f}, Recall: {recall_mask:.4f}, mIoU: {mean_iou_mask:.4f}")
print(f"Avg inference time: {np.mean(times):.2f} ms/img")
print(f"Results saved to folder: {save_folder}")
