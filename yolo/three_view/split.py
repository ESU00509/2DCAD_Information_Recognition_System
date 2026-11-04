import os
import random
import shutil

# 1. 配置你的路径
base_dir = './data' # 当前目录
images_dir = os.path.join(base_dir, 'images')
labels_dir = os.path.join(base_dir, 'labels')
classes_file = os.path.join(base_dir, 'classes.txt')

# 创建 YOLOv8 需要的标准目录结构
for split in ['train', 'val']:
    os.makedirs(os.path.join(images_dir, split), exist_ok=True)
    os.makedirs(os.path.join(labels_dir, split), exist_ok=True)

# 2. 获取所有图片文件名并打乱
all_images = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
random.seed(42) # 固定随机种子，保证每次划分结果一样
random.shuffle(all_images)

# 3. 按 8:2 划分
split_idx = int(len(all_images) * 0.8)
train_images = all_images[:split_idx]
val_images = all_images[split_idx:]

def move_files(file_list, split_type):
    for img_name in file_list:
        # 移动图片
        src_img = os.path.join(images_dir, img_name)
        dst_img = os.path.join(images_dir, split_type, img_name)
        shutil.move(src_img, dst_img)
        
        # 移动对应的 txt 标签文件
        label_name = os.path.splitext(img_name)[0] + '.txt'
        src_label = os.path.join(labels_dir, label_name)
        dst_label = os.path.join(labels_dir, split_type, label_name)
        if os.path.exists(src_label):
            shutil.move(src_label, dst_label)
        else:
            print(f"警告: 找不到 {label_name} 对应的标签文件")

move_files(train_images, 'train')
move_files(val_images, 'val')

print(f"划分完成！训练集: {len(train_images)} 张，验证集: {len(val_images)} 张。")

# 4. 自动读取 classes.txt 并生成 data.yaml
with open(classes_file, 'r', encoding='utf-8') as f:
    classes = [line.strip() for line in f.readlines() if line.strip()]

yaml_content = f"""path: {os.path.abspath(base_dir)}
train: images/train
val: images/val

names:
"""
for i, cls_name in enumerate(classes):
    yaml_content += f"  {i}: {cls_name}\n"

with open(os.path.join(base_dir, 'data.yaml'), 'w', encoding='utf-8') as f:
    f.write(yaml_content)

print("已自动生成 data.yaml 文件！内容如下：")
print(yaml_content)