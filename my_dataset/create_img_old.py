import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import random
import json
import copy

src_img_path = "D:/Project/my_dataset/ShowUI-desktop/images/image0.png"
src_meta_path = "D:/Project/my_dataset/ShowUI-desktop/metadata/metadata.json"
output_dir = "D:/Project/my_dataset/ShowUI-desktop/images"
os.makedirs(output_dir, exist_ok=True)

with open(src_meta_path, "r", encoding="utf-8") as f:
    meta = json.load(f)[0]  # 只取第一个样本

orig_w, orig_h = meta["img_size"]

def apply_affine_to_point(x, y, M):
    pt = np.array([x, y, 1.0])
    x_new, y_new, _ = np.dot(M, pt)
    return x_new, y_new

def random_rotate(img, annos):
    angle = random.uniform(-20, 20)
    w, h = img.size
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    img_np = np.array(img)
    rotated = cv2.warpAffine(img_np, M, (w, h), borderValue=(255,255,255))
    # 标注同步旋转
    new_annos = []
    for ele in annos:
        bbox = ele["bbox"]
        px_min = bbox[0] * w
        py_min = bbox[1] * h
        px_max = bbox[2] * w
        py_max = bbox[3] * h
        # 四个角
        pts = np.array([[px_min, py_min], [px_max, py_min], [px_max, py_max], [px_min, py_max]])
        pts_rot = np.array([apply_affine_to_point(x, y, np.vstack([M, [0,0,1]])) for x, y in pts])
        x_coords, y_coords = pts_rot[:,0], pts_rot[:,1]
        # 新bbox
        new_bbox = [
            max(0, min(x_coords) / w),
            max(0, min(y_coords) / h),
            min(1, max(x_coords) / w),
            min(1, max(y_coords) / h)
        ]
        # point
        px, py = ele["point"][0] * w, ele["point"][1] * h
        px_new, py_new = apply_affine_to_point(px, py, np.vstack([M, [0,0,1]]))
        new_point = [min(1, max(0, px_new / w)), min(1, max(0, py_new / h))]
        ele_new = copy.deepcopy(ele)
        ele_new["bbox"] = new_bbox
        ele_new["point"] = new_point
        new_annos.append(ele_new)
    # 调整成原图大小
    rotated = cv2.resize(rotated, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return Image.fromarray(rotated), new_annos

def add_noise(img, annos):
    img_np = np.array(img)
    noise = np.random.normal(0, 10, img_np.shape).astype(np.uint8)
    img_np = cv2.add(img_np, noise)
    return Image.fromarray(img_np), annos

def random_brightness(img, annos):
    enhancer = ImageEnhance.Brightness(img)
    factor = random.uniform(0.7, 1.3)
    return enhancer.enhance(factor), annos

def random_contrast(img, annos):
    enhancer = ImageEnhance.Contrast(img)
    factor = random.uniform(0.7, 1.3)
    return enhancer.enhance(factor), annos

augmented_meta = []

for i in range(0, 100):
    aug_img = Image.open(src_img_path).convert("RGB")
    annos = copy.deepcopy(meta["element"])
    if random.random() < 0.7:
        aug_img, annos = random_rotate(aug_img, annos)
    if random.random() < 0.5:
        aug_img, annos = add_noise(aug_img, annos)
    if random.random() < 0.5:
        aug_img, annos = random_brightness(aug_img, annos)
    if random.random() < 0.5:
        aug_img, annos = random_contrast(aug_img, annos)
    out_name = f"aug_{i:03d}.png"
    out_path = os.path.join(output_dir, out_name)
    aug_img.save(out_path)
    # 保存标注
    augmented_meta.append({
        "img_url": out_name,
        "img_size": [aug_img.width, aug_img.height],
        "element": annos,
        "element_size": len(annos)
    })

# 保存新的metadata.json
with open(os.path.join(os.path.dirname(src_meta_path), "metadata.json"), "w", encoding="utf-8") as f:
    json.dump(augmented_meta, f, indent=4, ensure_ascii=False)