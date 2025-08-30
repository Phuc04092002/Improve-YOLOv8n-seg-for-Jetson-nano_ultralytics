coco_yaml = '''
path: D:/new_dataset1
train: images/train
val: images/val

nc: 5
names: ['person', 'bicycle', 'car', 'motorcycle','traffic light']
'''

with open('ultralytics/cfg/datasets/coco-seg2.yaml', 'w') as f:
    f.write(coco_yaml)
