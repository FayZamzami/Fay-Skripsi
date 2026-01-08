#!/usr/bin/env python3
"""
OPTION 2: Fix Dataset Split
Re-split augmented dataset agar val/test set juga punya minority classes
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict
import yaml

def collect_images_by_class(labels_dir):
    """Collect images grouped by class"""

    class_images = defaultdict(list)
    labels_path = Path(labels_dir)

    for label_file in labels_path.glob('*.txt'):
        with open(label_file, 'r') as f:
            lines = f.readlines()

            # Get all classes in this image
            classes = set()
            for line in lines:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    classes.add(class_id)

            # Add to each class list
            img_file = label_file.stem + '.jpg'
            for class_id in classes:
                class_images[class_id].append((img_file, label_file.name))

    return class_images

def stratified_split(class_images, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Stratified split: memastikan setiap class ada di train/val/test
    """

    splits = {
        'train': [],
        'val': [],
        'test': []
    }

    print("\n" + "="*80)
    print("STRATIFIED SPLIT")
    print("="*80)
    print(f"Ratio - Train: {train_ratio}, Val: {val_ratio}, Test: {test_ratio}\n")

    # For each class, split proportionally
    for class_id, images in class_images.items():
        # Remove duplicates (same image may have multiple classes)
        unique_images = list(set(images))

        # Shuffle
        random.seed(42)  # For reproducibility
        random.shuffle(unique_images)

        n_total = len(unique_images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        # Split
        train_imgs = unique_images[:n_train]
        val_imgs = unique_images[n_train:n_train+n_val]
        test_imgs = unique_images[n_train+n_val:]

        splits['train'].extend(train_imgs)
        splits['val'].extend(val_imgs)
        splits['test'].extend(test_imgs)

        print(f"Class {class_id}: Total={n_total:4d} | "
              f"Train={len(train_imgs):3d} | "
              f"Val={len(val_imgs):3d} | "
              f"Test={len(test_imgs):3d}")

    # Remove duplicates (images with multiple classes counted multiple times)
    for split_name in splits:
        splits[split_name] = list(set(splits[split_name]))

    return splits

def copy_files(splits, source_images, source_labels, output_dir):
    """Copy files to new split directories"""

    output_path = Path(output_dir)

    print("\n" + "="*80)
    print("COPYING FILES TO NEW SPLIT")
    print("="*80)

    for split_name, file_list in splits.items():
        # Create directories
        img_dir = output_path / split_name / 'images'
        lbl_dir = output_path / split_name / 'labels'
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{split_name.upper()}: {len(file_list)} images")

        copied = 0
        for img_file, lbl_file in file_list:
            # Copy image
            src_img = source_images / img_file
            dst_img = img_dir / img_file

            if src_img.exists():
                shutil.copy2(src_img, dst_img)

                # Copy label
                src_lbl = source_labels / lbl_file
                dst_lbl = lbl_dir / lbl_file
                if src_lbl.exists():
                    shutil.copy2(src_lbl, dst_lbl)
                    copied += 1

        print(f"  ✅ Copied {copied} image-label pairs")

def verify_split(output_dir, class_names):
    """Verify that all classes exist in all splits"""

    output_path = Path(output_dir)

    print("\n" + "="*80)
    print("VERIFICATION: CLASS DISTRIBUTION IN EACH SPLIT")
    print("="*80)

    for split_name in ['train', 'val', 'test']:
        labels_dir = output_path / split_name / 'labels'

        if not labels_dir.exists():
            print(f"\n⚠️  {split_name.upper()}: Directory not found!")
            continue

        # Count classes
        class_counts = defaultdict(int)

        for label_file in labels_dir.glob('*.txt'):
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        class_counts[class_id] += 1

        print(f"\n{split_name.upper()}:")
        print(f"{'Class':<20} {'Count':<10}")
        print("-"*30)

        for class_id, class_name in enumerate(class_names):
            count = class_counts.get(class_id, 0)
            status = "✅" if count > 0 else "❌"
            marker = "🔴" if class_id in [1, 2, 3, 5] else ""  # Minority classes
            print(f"{class_name:<20} {count:<10} {status} {marker}")

def main():
    print("\n" + "="*80)
    print("OPTION 2: FIX DATASET SPLIT")
    print("Re-split augmented dataset dengan stratified sampling")
    print("="*80)

    # Source directories
    source_images = Path("PPE-Dataset-Augmented/train/images")
    source_labels = Path("PPE-Dataset-Augmented/train/labels")

    # Output directory
    output_dir = "PPE-Dataset-Balanced"

    # Load class names
    with open('PPE-Dataset-Augmented/data.yaml', 'r') as f:
        data = yaml.safe_load(f)
    class_names = data['names']

    print(f"\n📁 Source: {source_images.parent}")
    print(f"📁 Output: {output_dir}")
    print(f"📊 Classes: {len(class_names)}")

    # Step 1: Collect images by class
    print("\n" + "-"*80)
    print("STEP 1: Collecting images by class...")
    class_images = collect_images_by_class(source_labels)

    print(f"Found {len(class_images)} classes")
    for class_id, images in class_images.items():
        print(f"  Class {class_id} ({class_names[class_id]}): {len(set(images))} unique images")

    # Step 2: Stratified split
    print("\n" + "-"*80)
    print("STEP 2: Performing stratified split...")
    splits = stratified_split(class_images, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    print(f"\nTotal splits:")
    print(f"  Train: {len(splits['train'])} images")
    print(f"  Val:   {len(splits['val'])} images")
    print(f"  Test:  {len(splits['test'])} images")

    # Step 3: Copy files
    print("\n" + "-"*80)
    print("STEP 3: Copying files to new structure...")
    copy_files(splits, source_images, source_labels, output_dir)

    # Step 4: Create new data.yaml
    print("\n" + "-"*80)
    print("STEP 4: Creating new data.yaml...")

    new_data_yaml = {
        'names': class_names,
        'nc': len(class_names),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images'
    }

    output_yaml = Path(output_dir) / 'data.yaml'
    with open(output_yaml, 'w') as f:
        yaml.dump(new_data_yaml, f, default_flow_style=False, sort_keys=False)

    print(f"✅ Created: {output_yaml}")

    # Step 5: Verify
    print("\n" + "-"*80)
    print("STEP 5: Verifying split...")
    verify_split(output_dir, class_names)

    # Final summary
    print("\n\n" + "="*80)
    print("✅ DATASET SPLIT COMPLETED!")
    print("="*80)
    print(f"\n📁 New balanced dataset: {output_dir}/")
    print(f"📄 New data.yaml: {output_yaml}")
    print("\n🎯 NEXT STEPS:")
    print("1. Re-train model dengan dataset balanced (optional)")
    print("2. Validate model dengan validation set yang proper")
    print("3. Check per-class metrics untuk minority classes")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
