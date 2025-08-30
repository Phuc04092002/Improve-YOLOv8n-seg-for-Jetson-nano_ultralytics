from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import os
import numpy as np
from skimage import measure

def convert_coco_to_yolo_seg(json_file, output_dir, img_dir):
    coco = COCO(json_file)
    img_ids = coco.getImgIds()
    os.makedirs(output_dir, exist_ok=True)

    # Tạo mapping COCO ID -> YOLO class
    coco_ids = sorted({ann['category_id'] for ann in coco.dataset['annotations']})
    coco2yolo = {cid: idx for idx, cid in enumerate(coco_ids)}
    print(f"✅ Tìm thấy {len(coco_ids)} class COCO. Mapping COCO ID → YOLO class:")
    print(coco2yolo)

    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anns = coco.loadAnns(ann_ids)

        label_file = os.path.join(output_dir, os.path.splitext(img_info['file_name'])[0] + ".txt")
        with open(label_file, "w") as f:
            for ann in anns:
                coco_id = ann['category_id']
                if coco_id not in coco2yolo:
                    continue

                seg_points = None

                # Polygon segmentation
                if 'segmentation' in ann and ann['segmentation']:
                    if isinstance(ann['segmentation'], list):
                        seg = ann['segmentation'][0]
                        seg_points = np.array(seg).reshape(-1, 2)
                    elif isinstance(ann['segmentation'], dict):
                        rle = ann['segmentation']
                        if isinstance(rle['counts'], list):
                            rle = maskUtils.frPyObjects(rle, img_info['height'], img_info['width'])
                        mask = maskUtils.decode(rle)
                        contours = measure.find_contours(mask, 0.5)
                        if len(contours) > 0:
                            contour = max(contours, key=len)
                            seg_points = np.flip(contour, axis=1)  # (y,x) -> (x,y)

                # Fallback dùng bbox nếu không có segmentation
                if seg_points is None:
                    x, y, w, h = ann['bbox']
                    seg_points = np.array([
                        [x, y],
                        [x + w, y],
                        [x + w, y + h],
                        [x, y + h]
                    ])

                # Chuẩn hóa [0,1]
                seg_points[:, 0] /= img_info['width']
                seg_points[:, 1] /= img_info['height']
                seg_str = " ".join([f"{x:.6f} {y:.6f}" for x, y in seg_points])

                # Remap COCO ID -> YOLO class
                label_id = coco2yolo[coco_id]
                f.write(f"{label_id} {seg_str}\n")

# Chuyển train2017
convert_coco_to_yolo_seg(
    "D:/coco/annotations/instances_train2017.json",
    "D:/coco/labels/train2017",
    "D:/coco/images/train2017"
)

# Chuyển val2017
convert_coco_to_yolo_seg(
    "D:/coco/annotations/instances_val2017.json",
    "D:/coco/labels/val2017",
    "D:/coco/images/val2017"
)
