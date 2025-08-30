# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
import random
import shutil

# ==== Cấu hình ====
base_dir = "D:/new_dataset1"
image_dir = os.path.join(base_dir, "images", "data")
label_dir = os.path.join(base_dir, "labels", "data")

# Thư mục đích train/val/test
splits = ["train", "val", "test"]
for split in splits:
    os.makedirs(os.path.join(base_dir, "images", split), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "labels", split), exist_ok=True)

# Lấy danh sách ảnh
all_images = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
all_images.sort()  # để cố định thứ tự

# Shuffle để random
random.seed(42)  # để reproducible
random.shuffle(all_images)

# Chia theo tỉ lệ 8/1/1
n_total = len(all_images)
n_train = int(0.8 * n_total)
n_val = int(0.1 * n_total)
n_test = n_total - n_train - n_val

train_files = all_images[:n_train]
val_files = all_images[n_train : n_train + n_val]
test_files = all_images[n_train + n_val :]


def copy_files(file_list, split):
    for img_file in file_list:
        # copy ảnh
        src_img = os.path.join(image_dir, img_file)
        dst_img = os.path.join(base_dir, "images", split, img_file)
        shutil.copy(src_img, dst_img)

        # copy nhãn
        label_file = os.path.splitext(img_file)[0] + ".txt"
        src_label = os.path.join(label_dir, label_file)
        dst_label = os.path.join(base_dir, "labels", split, label_file)
        if os.path.exists(src_label):  # đề phòng thiếu nhãn
            shutil.copy(src_label, dst_label)


# Copy dữ liệu
copy_files(train_files, "train")
copy_files(val_files, "val")
copy_files(test_files, "test")

print(f"✅ Done! Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
