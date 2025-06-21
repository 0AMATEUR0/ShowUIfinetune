import json
import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import random
import math
from typing import List, Dict, Tuple
import argparse

class DataAugmentor:
    def __init__(self, dataset_dir: str = "."):
        """
        初始化数据增强器
        
        Args:
            dataset_dir: 数据集根目录，包含metadata和images文件夹
        """
        self.dataset_dir = dataset_dir
        self.metadata_path = os.path.join(dataset_dir, "metadata", "metadata.json")
        self.images_dir = os.path.join(dataset_dir, "images")
        self.original_data = []
        self.augmented_data = []
        
    def load_metadata(self):
        """加载原始metadata"""
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            self.original_data = json.load(f)
        print(f"加载了 {len(self.original_data)} 条原始数据")
    
    def save_metadata(self):
        """保存增强后的metadata"""
        all_data = self.original_data + self.augmented_data
        backup_path = self.metadata_path.replace('.json', '_backup.json')
        
        # 备份原始文件
        if os.path.exists(self.metadata_path):
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(self.original_data, f, ensure_ascii=False, indent=2)
            print(f"原始数据已备份到: {backup_path}")
        
        # 保存新数据
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
        print(f"保存了 {len(all_data)} 条数据到 {self.metadata_path}")
    
    def rotate_bbox(self, bbox: List[int], angle: float, img_size: Tuple[int, int]) -> List[int]:
        """
        旋转边界框坐标
        
        Args:
            bbox: [x, y, width, height]
            angle: 旋转角度（度）
            img_size: 图片尺寸 (width, height)
        """
        x, y, w, h = bbox
        img_w, img_h = img_size
        
        # 转换为弧度
        angle_rad = math.radians(angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        # 边界框四个角点
        corners = [
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ]
        
        # 旋转中心
        center_x, center_y = img_w / 2, img_h / 2
        
        # 旋转每个角点
        rotated_corners = []
        for corner_x, corner_y in corners:
            # 平移到原点
            rel_x = corner_x - center_x
            rel_y = corner_y - center_y
            
            # 旋转
            new_x = rel_x * cos_a - rel_y * sin_a
            new_y = rel_x * sin_a + rel_y * cos_a
            
            # 平移回去
            rotated_corners.append([new_x + center_x, new_y + center_y])
        
        # 计算新的边界框
        xs = [corner[0] for corner in rotated_corners]
        ys = [corner[1] for corner in rotated_corners]
        
        new_x = max(0, min(xs))
        new_y = max(0, min(ys))
        new_w = min(img_w - new_x, max(xs) - new_x)
        new_h = min(img_h - new_y, max(ys) - new_y)
        
        return [int(new_x), int(new_y), int(new_w), int(new_h)]
    
    def scale_bbox(self, bbox: List[int], scale_x: float, scale_y: float, img_size: Tuple[int, int]) -> List[int]:
        """
        缩放边界框坐标
        
        Args:
            bbox: [x, y, width, height]
            scale_x: x方向缩放比例
            scale_y: y方向缩放比例
            img_size: 新图片尺寸 (width, height)
        """
        x, y, w, h = bbox
        new_img_w, new_img_h = img_size
        
        new_x = int(x * scale_x)
        new_y = int(y * scale_y)
        new_w = int(w * scale_x)
        new_h = int(h * scale_y)
        
        # 确保边界框在图片范围内
        new_x = max(0, min(new_x, new_img_w - 1))
        new_y = max(0, min(new_y, new_img_h - 1))
        new_w = min(new_w, new_img_w - new_x)
        new_h = min(new_h, new_img_h - new_y)
        
        return [new_x, new_y, new_w, new_h]
    
    def augment_rotation(self, img_path: str, angle: float) -> Tuple[np.ndarray, str]:
        """旋转图片"""
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        
        # 旋转矩阵
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # 旋转图片
        rotated_img = cv2.warpAffine(img, rotation_matrix, (w, h), 
                                   borderMode=cv2.BORDER_CONSTANT, 
                                   borderValue=(255, 255, 255))
        
        # 生成新文件名
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        new_name = f"{base_name}_rot_{int(angle)}.png"
        
        return rotated_img, new_name
    
    def augment_noise(self, img_path: str, noise_level: float = 0.1) -> Tuple[np.ndarray, str]:
        """添加噪声"""
        img = cv2.imread(img_path)
        
        # 生成噪声
        noise = np.random.normal(0, noise_level * 255, img.shape).astype(np.uint8)
        noisy_img = cv2.add(img, noise)
        
        # 生成新文件名
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        new_name = f"{base_name}_noise_{int(noise_level*100)}.png"
        
        return noisy_img, new_name
    
    def augment_brightness(self, img_path: str, factor: float) -> Tuple[np.ndarray, str]:
        """调整亮度"""
        img = Image.open(img_path)
        enhancer = ImageEnhance.Brightness(img)
        bright_img = enhancer.enhance(factor)
        
        # 转换为numpy数组
        bright_img_np = cv2.cvtColor(np.array(bright_img), cv2.COLOR_RGB2BGR)
        
        # 生成新文件名
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        new_name = f"{base_name}_bright_{int(factor*100)}.png"
        
        return bright_img_np, new_name
    
    def augment_contrast(self, img_path: str, factor: float) -> Tuple[np.ndarray, str]:
        """调整对比度"""
        img = Image.open(img_path)
        enhancer = ImageEnhance.Contrast(img)
        contrast_img = enhancer.enhance(factor)
        
        # 转换为numpy数组
        contrast_img_np = cv2.cvtColor(np.array(contrast_img), cv2.COLOR_RGB2BGR)
        
        # 生成新文件名
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        new_name = f"{base_name}_contrast_{int(factor*100)}.png"
        
        return contrast_img_np, new_name
    
    def augment_blur(self, img_path: str, blur_radius: float = 1.0) -> Tuple[np.ndarray, str]:
        """添加模糊"""
        img = Image.open(img_path)
        blurred_img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        # 转换为numpy数组
        blurred_img_np = cv2.cvtColor(np.array(blurred_img), cv2.COLOR_RGB2BGR)
        
        # 生成新文件名
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        new_name = f"{base_name}_blur_{int(blur_radius*10)}.png"
        
        return blurred_img_np, new_name
    
    def augment_scale(self, img_path: str, scale_factor: float) -> Tuple[np.ndarray, str]:
        """缩放图片"""
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        
        scaled_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # 如果缩放后比原图大，裁剪到原尺寸
        if scale_factor > 1.0:
            start_x = (new_w - w) // 2
            start_y = (new_h - h) // 2
            scaled_img = scaled_img[start_y:start_y+h, start_x:start_x+w]
        # 如果缩放后比原图小，填充到原尺寸
        elif scale_factor < 1.0:
            padded_img = np.full((h, w, 3), 255, dtype=np.uint8)
            start_x = (w - new_w) // 2
            start_y = (h - new_h) // 2
            padded_img[start_y:start_y+new_h, start_x:start_x+new_w] = scaled_img
            scaled_img = padded_img
        
        # 生成新文件名
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        new_name = f"{base_name}_scale_{int(scale_factor*100)}.png"
        
        return scaled_img, new_name
    
    def apply_combined_augmentations(self, img_path: str, aug_list: List[Tuple[str, float]]) -> Tuple[np.ndarray, str]:
        """
        应用多种增强的组合
        
        Args:
            img_path: 图片路径
            aug_list: 增强列表，格式为 [(增强类型, 参数), ...]
        
        Returns:
            (增强后的图片, 新文件名)
        """
        img = cv2.imread(img_path)
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        name_parts = [base_name]
        
        # 按顺序应用每种增强
        for aug_type, param in aug_list:
            if aug_type == 'rotation':
                h, w = img.shape[:2]
                center = (w // 2, h // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, param, 1.0)
                img = cv2.warpAffine(img, rotation_matrix, (w, h), 
                                   borderMode=cv2.BORDER_CONSTANT, 
                                   borderValue=(255, 255, 255))
                name_parts.append(f"rot{int(param)}")
                
            elif aug_type == 'noise':
                noise = np.random.normal(0, param * 255, img.shape).astype(np.uint8)
                img = cv2.add(img, noise)
                name_parts.append(f"noise{int(param*100)}")
                
            elif aug_type == 'brightness':
                img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                enhancer = ImageEnhance.Brightness(img_pil)
                img_pil = enhancer.enhance(param)
                img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                name_parts.append(f"bright{int(param*100)}")
                
            elif aug_type == 'contrast':
                img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                enhancer = ImageEnhance.Contrast(img_pil)
                img_pil = enhancer.enhance(param)
                img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                name_parts.append(f"contrast{int(param*100)}")
                
            elif aug_type == 'blur':
                img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=param))
                img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                name_parts.append(f"blur{int(param*10)}")
                
            elif aug_type == 'scale':
                h, w = img.shape[:2]
                new_w = int(w * param)
                new_h = int(h * param)
                scaled_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                
                if param > 1.0:
                    start_x = (new_w - w) // 2
                    start_y = (new_h - h) // 2
                    img = scaled_img[start_y:start_y+h, start_x:start_x+w]
                elif param < 1.0:
                    padded_img = np.full((h, w, 3), 255, dtype=np.uint8)
                    start_x = (w - new_w) // 2
                    start_y = (h - new_h) // 2
                    padded_img[start_y:start_y+new_h, start_x:start_x+new_w] = scaled_img
                    img = padded_img
                name_parts.append(f"scale{int(param*100)}")
                
            elif aug_type == 'flip_h':
                img = cv2.flip(img, 1)  # 水平翻转
                name_parts.append("fliph")
                
            elif aug_type == 'flip_v':
                img = cv2.flip(img, 0)  # 垂直翻转
                name_parts.append("flipv")
                
            elif aug_type == 'hue':
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                hsv[:, :, 0] = (hsv[:, :, 0] + param) % 180
                img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                name_parts.append(f"hue{int(param)}")
                
            elif aug_type == 'saturation':
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[:, :, 1] = hsv[:, :, 1] * param
                hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
                img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
                name_parts.append(f"sat{int(param*100)}")
        
        new_name = "_".join(name_parts) + ".png"
        return img, new_name

    def update_bbox_for_combined_augmentations(self, bbox: List[int], aug_list: List[Tuple[str, float]], img_size: Tuple[int, int]) -> List[int]:
        """
        为组合增强更新bbox坐标
        
        Args:
            bbox: 原始bbox [x, y, width, height]
            aug_list: 增强列表
            img_size: 图片尺寸 (width, height)
        """
        current_bbox = bbox.copy()
        current_size = img_size
        
        for aug_type, param in aug_list:
            if aug_type == 'rotation':
                current_bbox = self.rotate_bbox(current_bbox, param, current_size)
            elif aug_type == 'scale':
                current_bbox = self.scale_bbox(current_bbox, param, param, current_size)
            elif aug_type == 'flip_h':
                # 水平翻转
                x, y, w, h = current_bbox
                img_w, img_h = current_size
                current_bbox = [img_w - x - w, y, w, h]
            elif aug_type == 'flip_v':
                # 垂直翻转
                x, y, w, h = current_bbox
                img_w, img_h = current_size
                current_bbox = [x, img_h - y - h, w, h]
            # 其他增强不影响bbox坐标
        
        return current_bbox

    def generate_random_augmentation_combinations(self, total_count: int, augmentation_ranges: Dict) -> List[List[Tuple[str, float]]]:
        """
        生成随机增强组合
        
        Args:
            total_count: 需要生成的总数量
            augmentation_ranges: 各种增强的参数范围
        
        Returns:
            增强组合列表
        """
        available_augmentations = []
        
        # 单一增强
        for aug_type, config in augmentation_ranges.items():
            if aug_type in ['rotation', 'noise', 'brightness', 'contrast', 'blur', 'scale', 'hue', 'saturation']:
                available_augmentations.append(aug_type)
        
        # 添加翻转（无参数）
        if augmentation_ranges.get('flip', True):
            available_augmentations.extend(['flip_h', 'flip_v'])
        
        combinations = []
        
        for i in range(total_count):
            # 随机选择1-3种增强进行组合
            combo_size = np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])  # 40%单一, 40%双重, 20%三重
            selected_augs = np.random.choice(available_augmentations, size=combo_size, replace=False)
            
            combo = []
            for aug_type in selected_augs:
                if aug_type == 'rotation':
                    param = np.random.uniform(augmentation_ranges['rotation']['min'], 
                                            augmentation_ranges['rotation']['max'])
                    if abs(param) < 3:  # 避免太小的角度
                        param = param + (10 if param >= 0 else -10)
                    combo.append((aug_type, round(param, 1)))
                    
                elif aug_type == 'noise':
                    param = np.random.uniform(augmentation_ranges['noise']['min'], 
                                            augmentation_ranges['noise']['max'])
                    combo.append((aug_type, round(param, 3)))
                    
                elif aug_type == 'brightness':
                    param = np.random.uniform(augmentation_ranges['brightness']['min'], 
                                            augmentation_ranges['brightness']['max'])
                    combo.append((aug_type, round(param, 2)))
                    
                elif aug_type == 'contrast':
                    param = np.random.uniform(augmentation_ranges['contrast']['min'], 
                                            augmentation_ranges['contrast']['max'])
                    combo.append((aug_type, round(param, 2)))
                    
                elif aug_type == 'blur':
                    param = np.random.uniform(augmentation_ranges['blur']['min'], 
                                            augmentation_ranges['blur']['max'])
                    combo.append((aug_type, round(param, 2)))
                    
                elif aug_type == 'scale':
                    param = np.random.uniform(augmentation_ranges['scale']['min'], 
                                            augmentation_ranges['scale']['max'])
                    combo.append((aug_type, round(param, 2)))
                    
                elif aug_type == 'hue':
                    param = np.random.uniform(augmentation_ranges.get('hue', {'min': -20, 'max': 20})['min'], 
                                            augmentation_ranges.get('hue', {'min': -20, 'max': 20})['max'])
                    combo.append((aug_type, round(param, 1)))
                    
                elif aug_type == 'saturation':
                    param = np.random.uniform(augmentation_ranges.get('saturation', {'min': 0.7, 'max': 1.3})['min'], 
                                            augmentation_ranges.get('saturation', {'min': 0.7, 'max': 1.3})['max'])
                    combo.append((aug_type, round(param, 2)))
                    
                elif aug_type in ['flip_h', 'flip_v']:
                    combo.append((aug_type, 1))  # 翻转无参数，用1作为占位符
            
            combinations.append(combo)
        
        return combinations

    def generate_random_params(self, augmentation_config: Dict) -> Dict:
        """
        根据配置生成随机参数
        
        Args:
            augmentation_config: 增强配置字典，现在支持范围定义
            例如: {
                'rotation': {'min': -30, 'max': 30, 'count': 5},
                'noise': {'min': 0.01, 'max': 0.2, 'count': 3},
                'brightness': {'min': 0.5, 'max': 1.5, 'count': 4},
                'contrast': {'min': 0.5, 'max': 1.5, 'count': 3},
                'blur': {'min': 0.1, 'max': 2.0, 'count': 3},
                'scale': {'min': 0.8, 'max': 1.2, 'count': 3}
            }
        """
        random_params = {}
        
        for aug_type, config in augmentation_config.items():
            if isinstance(config, dict) and 'min' in config and 'max' in config:
                # 随机生成参数
                min_val = config['min']
                max_val = config['max']
                count = config.get('count', 3)
                
                if aug_type == 'rotation':
                    # 旋转角度，避免0度（无变化）
                    angles = []
                    while len(angles) < count:
                        angle = np.random.uniform(min_val, max_val)
                        if abs(angle) > 2:  # 避免太小的角度
                            angles.append(round(angle, 1))
                    random_params[aug_type] = angles
                else:
                    # 其他参数
                    values = np.random.uniform(min_val, max_val, count)
                    random_params[aug_type] = [round(v, 3) for v in values]
            else:
                # 兼容旧格式（固定值列表）
                random_params[aug_type] = config
        
        return random_params

    def generate_augmentations_by_count(self, total_augmentations: int, augmentation_ranges: Dict):
        """
        按指定数量生成随机增强（支持组合增强）
        
        Args:
            total_augmentations: 需要生成的增强图片总数
            augmentation_ranges: 增强参数范围配置
        """
        self.load_metadata()
        
        # 按图片分组原始数据
        image_groups = {}
        for item in self.original_data:
            img_name = item['img_url']
            if img_name not in image_groups:
                image_groups[img_name] = []
            image_groups[img_name].append(item)
        
        # 为每张图片生成指定数量的增强
        for img_name, items in image_groups.items():
            img_path = os.path.join(self.images_dir, img_name)
            if not os.path.exists(img_path):
                print(f"警告: 图片 {img_path} 不存在，跳过")
                continue
            
            print(f"正在处理图片: {img_name}")
            print(f"生成 {total_augmentations} 张增强图片...")
            
            # 获取原始图片尺寸
            original_img = cv2.imread(img_path)
            original_h, original_w = original_img.shape[:2]
            original_size = [original_w, original_h]
            
            # 生成随机增强组合
            augmentation_combinations = self.generate_random_augmentation_combinations(
                total_augmentations, augmentation_ranges)
            
            # 打印生成的组合（前5个作为示例）
            print("生成的增强组合示例：")
            for i, combo in enumerate(augmentation_combinations[:5]):
                combo_str = " + ".join([f"{aug_type}({param})" for aug_type, param in combo])
                print(f"  {i+1}: {combo_str}")
            if len(augmentation_combinations) > 5:
                print(f"  ... 还有 {len(augmentation_combinations)-5} 个组合")
            
            # 应用增强并保存
            for i, combo in enumerate(augmentation_combinations):
                try:
                    # 应用组合增强
                    aug_img, new_name = self.apply_combined_augmentations(img_path, combo)
                    
                    # 保存增强图片
                    new_img_path = os.path.join(self.images_dir, new_name)
                    cv2.imwrite(new_img_path, aug_img)
                    
                    # 为每个原始标注创建对应的增强标注
                    for item in items:
                        new_item = item.copy()
                        new_item['img_url'] = new_name
                        
                        # 生成新的ID
                        base_id = item['id']
                        combo_id = "_".join([f"{aug_type}{str(param).replace('.', '_')}" 
                                           for aug_type, param in combo])
                        new_id = f"{base_id}_{combo_id}"
                        new_item['id'] = new_id
                        
                        # 更新边界框
                        original_bbox = item['bbox']
                        new_bbox = self.update_bbox_for_combined_augmentations(
                            original_bbox, combo, original_size)
                        new_item['bbox'] = new_bbox
                        
                        self.augmented_data.append(new_item)
                    
                    if (i + 1) % 10 == 0 or (i + 1) == len(augmentation_combinations):
                        print(f"  已生成: {i + 1}/{len(augmentation_combinations)}")
                        
                except Exception as e:
                    print(f"  生成第 {i+1} 个增强时出错: {e}")
                    continue
        
        print(f"总共生成了 {len(self.augmented_data)} 条增强数据")
        """
        生成数据增强（随机参数）
        
        Args:
            augmentation_config: 增强配置字典
            支持两种格式：
            1. 随机范围格式: {
                'rotation': {'min': -30, 'max': 30, 'count': 5},
                'noise': {'min': 0.01, 'max': 0.2, 'count': 3}
            }
            2. 固定值格式: {
                'rotation': [-15, -10, 10, 15],
                'noise': [0.05, 0.1]
            }
        """
        self.load_metadata()
        
        # 生成随机参数
        print("生成随机增强参数...")
        random_params = self.generate_random_params(augmentation_ranges)
        
        # 打印生成的随机参数
        for aug_type, params in random_params.items():
            print(f"  {aug_type}: {params}")
        print()
        
        # 按图片分组原始数据
        image_groups = {}
        for item in self.original_data:
            img_name = item['img_url']
            if img_name not in image_groups:
                image_groups[img_name] = []
            image_groups[img_name].append(item)
        
        # 为每张图片生成增强
        for img_name, items in image_groups.items():
            img_path = os.path.join(self.images_dir, img_name)
            if not os.path.exists(img_path):
                print(f"警告: 图片 {img_path} 不存在，跳过")
                continue
            
            print(f"正在处理图片: {img_name}")
            
            # 获取原始图片尺寸
            original_img = cv2.imread(img_path)
            original_h, original_w = original_img.shape[:2]
            original_size = [original_w, original_h]
            
            # 应用各种增强
            augmentations = []
            
            # 旋转
            if 'rotation' in random_params:
                for angle in random_params['rotation']:
                    aug_img, new_name = self.augment_rotation(img_path, angle)
                    augmentations.append((aug_img, new_name, 'rotation', angle, original_size))
            
            # 噪声
            if 'noise' in random_params:
                for noise_level in random_params['noise']:
                    aug_img, new_name = self.augment_noise(img_path, noise_level)
                    augmentations.append((aug_img, new_name, 'noise', noise_level, original_size))
            
            # 亮度
            if 'brightness' in random_params:
                for factor in random_params['brightness']:
                    aug_img, new_name = self.augment_brightness(img_path, factor)
                    augmentations.append((aug_img, new_name, 'brightness', factor, original_size))
            
            # 对比度
            if 'contrast' in random_params:
                for factor in random_params['contrast']:
                    aug_img, new_name = self.augment_contrast(img_path, factor)
                    augmentations.append((aug_img, new_name, 'contrast', factor, original_size))
            
            # 模糊
            if 'blur' in random_params:
                for blur_radius in random_params['blur']:
                    aug_img, new_name = self.augment_blur(img_path, blur_radius)
                    augmentations.append((aug_img, new_name, 'blur', blur_radius, original_size))
            
            # 缩放
            if 'scale' in random_params:
                for scale_factor in random_params['scale']:
                    aug_img, new_name = self.augment_scale(img_path, scale_factor)
                    augmentations.append((aug_img, new_name, 'scale', scale_factor, original_size))
            
            # 保存增强图片并更新标注
            for aug_img, new_name, aug_type, param, img_size in augmentations:
                # 保存增强图片
                new_img_path = os.path.join(self.images_dir, new_name)
                cv2.imwrite(new_img_path, aug_img)
                
                # 为每个原始标注创建对应的增强标注
                for item in items:
                    new_item = item.copy()
                    new_item['img_url'] = new_name
                    
                    # 生成新的ID
                    base_id = item['id']
                    new_id = f"{base_id}_{aug_type}_{str(param).replace('.', '_')}"
                    new_item['id'] = new_id
                    
                    # 更新边界框
                    original_bbox = item['bbox']
                    
                    if aug_type == 'rotation':
                        new_bbox = self.rotate_bbox(original_bbox, param, img_size)
                    elif aug_type == 'scale':
                        if param != 1.0:
                            # 对于缩放，需要计算实际的缩放比例
                            scale_x = scale_y = param
                            new_bbox = self.scale_bbox(original_bbox, scale_x, scale_y, img_size)
                        else:
                            new_bbox = original_bbox
                    else:
                        # 其他增强不改变边界框
                        new_bbox = original_bbox
                    
                    new_item['bbox'] = new_bbox
                    self.augmented_data.append(new_item)
                
                print(f"  生成增强图片: {new_name}")
        
        print(f"总共生成了 {len(self.augmented_data)} 条增强数据")

def main():
    parser = argparse.ArgumentParser(description='数据增强脚本')
    parser.add_argument('--dataset_dir', type=str, default='.', 
                       help='数据集目录路径（默认为当前目录）')
    parser.add_argument('--config', type=str, 
                       help='增强配置JSON文件路径')
    parser.add_argument('--count', '-n', type=int, 
                       help='每张图片需要生成的增强图片数量')
    parser.add_argument('--mode', type=str, choices=['count', 'config'], default='count',
                       help='增强模式：count=按数量生成, config=按配置生成')
    
    args = parser.parse_args()
    
    # 默认增强范围配置（用于按数量生成模式）
    default_ranges = {
        'rotation': {'min': -30, 'max': 30},
        'noise': {'min': 0.01, 'max': 0.15},
        'brightness': {'min': 0.6, 'max': 1.4},
        'contrast': {'min': 0.7, 'max': 1.3},
        'blur': {'min': 0.3, 'max': 1.5},
        'scale': {'min': 0.85, 'max': 1.15},
        'hue': {'min': -20, 'max': 20},
        'saturation': {'min': 0.7, 'max': 1.3},
        'flip': True  # 启用翻转
    }
    
    # 默认固定增强配置（用于按配置生成模式）
    default_config = {
        'rotation': {'min': -30, 'max': 30, 'count': 6},
        'noise': {'min': 0.01, 'max': 0.15, 'count': 3},
        'brightness': {'min': 0.6, 'max': 1.4, 'count': 4},
        'contrast': {'min': 0.7, 'max': 1.3, 'count': 3},
        'blur': {'min': 0.3, 'max': 1.5, 'count': 3},
        'scale': {'min': 0.85, 'max': 1.15, 'count': 3}
    }
    
    # 创建数据增强器
    augmentor = DataAugmentor(args.dataset_dir)
    
    try:
        if args.mode == 'count':
            # 按数量生成模式
            if args.count is None:
                # 如果没有指定数量，询问用户
                try:
                    count = int(input("请输入每张图片需要生成的增强图片数量: "))
                except (ValueError, KeyboardInterrupt):
                    print("使用默认数量: 50")
                    count = 50
            else:
                count = args.count
            
            # 加载配置（如果有）
            if args.config and os.path.exists(args.config):
                with open(args.config, 'r', encoding='utf-8') as f:
                    ranges_config = json.load(f)
                print(f"使用配置文件: {args.config}")
            else:
                ranges_config = default_ranges
                print("使用默认增强范围配置")
            
            print(f"将为每张图片生成 {count} 张随机组合增强图片")
            print("增强类型包括: 旋转、噪声、亮度、对比度、模糊、缩放、色调、饱和度、翻转")
            print("支持多种增强组合 (40%单一, 40%双重, 20%三重增强)")
            
            augmentor.generate_augmentations_by_count(count, ranges_config)
            
        else:
            # 按配置生成模式（原有功能）
            if args.config and os.path.exists(args.config):
                with open(args.config, 'r', encoding='utf-8') as f:
                    augmentation_config = json.load(f)
                print(f"使用配置文件: {args.config}")
            else:
                augmentation_config = default_config
                print("使用默认随机增强配置")
                print("每次运行都会生成不同的随机参数！")
            
            augmentor.generate_augmentations(augmentation_config)
        
        augmentor.save_metadata()
        print("数据增强完成！")
        
    except Exception as e:
        print(f"数据增强过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()