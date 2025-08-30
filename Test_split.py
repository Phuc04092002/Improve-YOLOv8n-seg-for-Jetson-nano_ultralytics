import os
import random
import shutil
from collections import defaultdict

# ==== Cấu hình ====
base_dir = "D:/coco"  # thư mục gốc dataset
new_base_dir = "D:/new_dataset"  # thư mục đích
num_samples = 6000  # số lượng mẫu muốn copy
target_classes = [0, 1, 2, 3, 9]  # class cần lấy

# Thư mục ảnh & nhãn gốc
image_train_dir = os.path.join(base_dir, "images", "train2017")
label_train_dir = os.path.join(base_dir, "labels", "train2017")

# Thư mục ảnh & nhãn đích
new_image_train_dir = os.path.join(new_base_dir, "images", "train")
new_label_train_dir = os.path.join(new_base_dir, "labels", "train")

os.makedirs(new_image_train_dir, exist_ok=True)
os.makedirs(new_label_train_dir, exist_ok=True)

# Gom ảnh theo class
class_to_images = defaultdict(list)
all_images = [f for f in os.listdir(image_train_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

for img_file in all_images:
    label_file = os.path.splitext(img_file)[0] + ".txt"
    src_label = os.path.join(label_train_dir, label_file)

    if os.path.exists(src_label) and os.path.getsize(src_label) > 0:
        with open(src_label, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines()]
            if not lines:
                continue
            found_classes = set()
            for line in lines:
                parts = line.split()
                if len(parts) > 0:
                    class_id = int(parts[0])
                    if class_id in target_classes:
                        found_classes.add(class_id)
            # nếu có ít nhất 1 class hợp lệ thì thêm vào
            for cid in found_classes:
                class_to_images[cid].append(img_file)

# Xác định quota cho từng class
quota = num_samples // len(target_classes)

selected_images = set()
for cid in target_classes:
    imgs = class_to_images[cid]
    if not imgs:
        print(f"⚠️ Class {cid} không có ảnh nào!")
        continue
    chosen = random.sample(imgs, min(quota, len(imgs)))
    selected_images.update(chosen)

# Nếu còn thiếu so với num_samples thì bổ sung ngẫu nhiên từ tất cả valid
all_valid_images = set().union(*class_to_images.values())
if len(selected_images) < num_samples:
    remaining = list(all_valid_images - selected_images)
    extra = random.sample(remaining, min(num_samples - len(selected_images), len(remaining)))
    selected_images.update(extra)

# Copy file
for img_file in selected_images:
    src_img = os.path.join(image_train_dir, img_file)
    dst_img = os.path.join(new_image_train_dir, img_file)
    shutil.copy(src_img, dst_img)

    label_file = os.path.splitext(img_file)[0] + ".txt"
    src_label = os.path.join(label_train_dir, label_file)
    dst_label = os.path.join(new_label_train_dir, label_file)

    # Lọc nhãn
    with open(src_label, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
        lines = [line for line in lines if int(line.split()[0]) in target_classes]

    if lines:  # chỉ ghi nếu còn class hợp lệ
        with open(dst_label, "w", encoding="utf-8") as fw:
            fw.write("\n".join(lines))

print(f"✅ Đã copy {len(selected_images)} mẫu (cân bằng class) sang {new_base_dir}")

