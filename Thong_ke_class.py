from collections import Counter
import os

label_dir = "D:/new_dataset1/labels/val"
counter = Counter()

for f in os.listdir(label_dir):
    if f.endswith(".txt"):
        with open(os.path.join(label_dir, f), "r") as fh:
            for line in fh:
                if line.strip():
                    cid = int(line.split()[0])
                    counter[cid] += 1

print("Số lượng object theo class_id:")
for cid, cnt in sorted(counter.items()):
    print(f"Class {cid}: {cnt}")
