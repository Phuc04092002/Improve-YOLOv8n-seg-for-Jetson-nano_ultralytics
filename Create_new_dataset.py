# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
import random
import shutil
from collections import Counter, defaultdict

# ==== Cấu hình ====
base_dir = "D:/coco"
new_base_dir = "D:/new_dataset1"
num_samples = 6000  # tối thiểu số ảnh
target_classes = [0, 1, 2, 3, 9]
# mapping gốc -> mới
class_remap = {0: 0, 1: 1, 2: 2, 3: 3, 9: 4}


image_train_dir = os.path.join(base_dir, "images", "train2017")
label_train_dir = os.path.join(base_dir, "labels", "train2017")

new_image_train_dir = os.path.join(new_base_dir, "images", "train")
new_label_train_dir = os.path.join(new_base_dir, "labels", "train")
os.makedirs(new_image_train_dir, exist_ok=True)
os.makedirs(new_label_train_dir, exist_ok=True)

# Gom thông tin object từ nhãn
image_to_objects = {}
class_to_images = defaultdict(set)

all_images = [f for f in os.listdir(image_train_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

for img_file in all_images:
    label_file = os.path.splitext(img_file)[0] + ".txt"
    src_label = os.path.join(label_train_dir, label_file)
    if not os.path.exists(src_label) or os.path.getsize(src_label) == 0:
        continue

    obj_counts = Counter()
    with open(src_label, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            class_id = int(parts[0])
            if class_id in target_classes:
                obj_counts[class_id] += 1

    if obj_counts:
        image_to_objects[img_file] = obj_counts
        for cid in obj_counts:
            class_to_images[cid].add(img_file)

# === Cân bằng object ===
selected_images = set()
object_counter = Counter()

# Mục tiêu số object trung bình cho mỗi class
total_objects = sum(sum(cnt.values()) for cnt in image_to_objects.values())
avg_target = total_objects // len(target_classes)

print(f"📊 Tổng object trong COCO (5 class): {total_objects}, target mỗi class ~ {avg_target}")

# Ưu tiên chọn ảnh chứa nhiều object của class đang thiếu
while len(selected_images) < num_samples:
    # Tìm class hiện tại đang thiếu object nhất
    cid = min(target_classes, key=lambda c: object_counter[c])
    candidates = list(class_to_images[cid] - selected_images)
    if not candidates:
        break  # không còn ảnh mới cho class này

    img_file = random.choice(candidates)
    selected_images.add(img_file)
    object_counter.update(image_to_objects[img_file])

# Nếu còn thiếu số ảnh tối thiểu thì bổ sung ngẫu nhiên
if len(selected_images) < num_samples:
    remaining = set(all_images) - selected_images
    extra = random.sample(list(remaining), num_samples - len(selected_images))
    selected_images.update(extra)

# Copy ảnh và nhãn (lọc theo target_classes + remap class_id)
for img_file in selected_images:
    src_img = os.path.join(image_train_dir, img_file)
    dst_img = os.path.join(new_image_train_dir, img_file)
    shutil.copy(src_img, dst_img)

    src_label = os.path.join(label_train_dir, os.path.splitext(img_file)[0] + ".txt")
    dst_label = os.path.join(new_label_train_dir, os.path.splitext(img_file)[0] + ".txt")

    with open(src_label, encoding="utf-8") as f:
        lines = []
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            old_cls = int(parts[0])
            if old_cls in target_classes:
                new_cls = class_remap[old_cls]  # remap sang id mới
                parts[0] = str(new_cls)
                lines.append(" ".join(parts))

    if lines:
        with open(dst_label, "w", encoding="utf-8") as fw:
            fw.write("\n".join(lines))


print(f"✅ Đã tạo dataset với {len(selected_images)} ảnh, object phân bố: {dict(object_counter)}")
