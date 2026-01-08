#!/usr/bin/env python3
"""
FASE 1 - AUGMENTASI MINORITY CLASSES (FIXED VERSION)
"""

import os
import cv2
import numpy as np
from pathlib import Path
import albumentations as A
import yaml
from tqdm import tqdm
import shutil

# Konfigurasi
MINORITY_CLASSES = {
    1: {'name': 'Ear-protection', 'target': 500},
    2: {'name': 'Glass', 'target': 500},
    3: {'name': 'Glove', 'target': 300},
    5: {'name': 'Mask', 'target': 300},
}

def load_yolo_annotation(label_path):
    annotations = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    bbox = [float(x) for x in parts[1:5]]
                    annotations.append({'class_id': class_id, 'bbox': bbox})
    return annotations

def save_yolo_annotation(label_path, annotations):
    with open(label_path, 'w') as f:
        for ann in annotations:
            f.write(f"{ann['class_id']} {' '.join(map(str, ann['bbox']))}\n")

def get_augmentation_pipeline(aggressive=True):
    if aggressive:
        return A.Compose([
            A.OneOf([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.2), A.Rotate(limit=30, p=0.5)], p=0.8),
            A.OneOf([A.RandomBrightnessContrast(p=0.8), A.HueSaturationValue(p=0.8)], p=0.9),
            A.OneOf([A.GaussNoise(p=0.5), A.GaussianBlur(p=0.5)], p=0.7),
            A.RandomScale(scale_limit=0.3, p=0.7),
            A.OneOf([A.RandomShadow(p=0.5), A.RandomFog(p=0.3)], p=0.5),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    else:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.7),
            A.GaussNoise(p=0.5),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def augment_image(image, annotations, transform):
    if not annotations:
        return None, None
    
    bboxes = [ann['bbox'] for ann in annotations]
    class_labels = [ann['class_id'] for ann in annotations]
    
    try:
        transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        new_annotations = [
            {'class_id': int(label), 'bbox': list(bbox)}
            for label, bbox in zip(transformed['class_labels'], transformed['bboxes'])
        ]
        return transformed['image'], new_annotations
    except:
        return None, None

def main():
    data_yaml_path = "/mnt/extended-home/frzamzami/Skripsi/Skripsi-Fay/PPE-Dataset-for-Workplace-Safety-1/data.yaml"
    
    print("\n" + "="*80)
    print("FASE 1: MINORITY CLASS AUGMENTATION")
    print("="*80)
    
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    base_path = Path(data_yaml_path).parent
    # FIX: Direct path to train folder
    train_images = base_path / 'train' / 'images'
    train_labels = base_path / 'train' / 'labels'
    
    print(f"\n📁 Dataset paths:")
    print(f"   Images: {train_images} (exists: {train_images.exists()})")
    print(f"   Labels: {train_labels} (exists: {train_labels.exists()})")
    
    output_path = Path('PPE-Dataset-Augmented')
    output_images = output_path / 'train' / 'images'
    output_labels = output_path / 'train' / 'labels'
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)
    
    # Copy original data
    print("\n📁 Copying original training data...")
    for img_file in tqdm(list(train_images.glob('*.jpg')) + list(train_images.glob('*.png'))):
        shutil.copy(img_file, output_images / img_file.name)
        label_file = train_labels / (img_file.stem + '.txt')
        if label_file.exists():
            shutil.copy(label_file, output_labels / label_file.name)
    
    # Augment minority classes
    for class_id, config in MINORITY_CLASSES.items():
        print(f"\n{'='*80}")
        print(f"Augmenting: {config['name']} (Class {class_id})")
        print(f"{'='*80}")
        
        images_with_class = []
        for label_file in train_labels.glob('*.txt'):
            annotations = load_yolo_annotation(label_file)
            if any(ann['class_id'] == class_id for ann in annotations):
                img_file = train_images / (label_file.stem + '.jpg')
                if not img_file.exists():
                    img_file = train_images / (label_file.stem + '.png')
                if img_file.exists():
                    images_with_class.append({
                        'image': img_file,
                        'annotations': annotations
                    })
        
        original_count = len(images_with_class)
        target = config['target']
        print(f"Original: {original_count}, Target: {target}")
        
        if original_count == 0:
            print(f"⚠️  No images with {config['name']}!")
            continue
        
        augmentations_needed = target - original_count
        augmentations_per_image = max(1, augmentations_needed // original_count)
        
        transform = get_augmentation_pipeline(aggressive=original_count < 10)
        
        aug_counter = 0
        with tqdm(total=augmentations_needed, desc=f"Augmenting") as pbar:
            for item in images_with_class:
                img = cv2.imread(str(item['image']))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                for _ in range(augmentations_per_image):
                    aug_img, aug_anns = augment_image(img, item['annotations'], transform)
                    
                    if aug_img is not None and aug_anns:
                        filename = f"{item['image'].stem}_aug{class_id}_{aug_counter}"
                        
                        cv2.imwrite(str(output_images / f"{filename}.jpg"),
                                   cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
                        save_yolo_annotation(output_labels / f"{filename}.txt", aug_anns)
                        
                        aug_counter += 1
                        pbar.update(1)
                        
                        if aug_counter >= augmentations_needed:
                            break
                
                if aug_counter >= augmentations_needed:
                    break
        
        print(f"✅ Generated {aug_counter} augmented images")
    
    # Copy val/test
    print("\n📁 Copying validation and test sets...")
    for split in ['valid', 'test']:
        for img_type in ['images', 'labels']:
            src = base_path / split / img_type
            dst = output_path / split / img_type
            dst.mkdir(parents=True, exist_ok=True)
            for f in src.glob('*.*'):
                shutil.copy(f, dst / f.name)
    
    # Create data.yaml
    new_data_yaml = {
        'train': str(output_path / 'train' / 'images'),
        'val': str(output_path / 'valid' / 'images'),
        'test': str(output_path / 'test' / 'images'),
        'nc': data['nc'],
        'names': data['names'],
    }
    
    with open(output_path / 'data.yaml', 'w') as f:
        yaml.dump(new_data_yaml, f, default_flow_style=False)
    
    print("\n" + "="*80)
    print("✅ AUGMENTATION COMPLETED!")
    print(f"📁 Output: {output_path.absolute()}")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
