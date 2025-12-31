#!/usr/bin/env python3
"""
Organize the COVID-19 dataset into train/val/test splits
"""

from pathlib import Path
import shutil
import numpy as np

# Set random seed
np.random.seed(42)

# Paths
DATA_DIR = Path('./data')
source_dir = DATA_DIR / 'COVID-19_Radiography_Dataset'
target_dir = DATA_DIR / 'organized'

# Split ratios
split_ratios = (0.7, 0.15, 0.15)

print("Organizing COVID-19 Radiography Dataset...")
print(f"Source: {source_dir}")
print(f"Target: {target_dir}")

# Define source paths
covid_images = source_dir / 'COVID' / 'images'
normal_images = source_dir / 'Normal' / 'images'
pneumonia_images = source_dir / 'Viral Pneumonia' / 'images'

# Create target directories
for split in ['train', 'val', 'test']:
    for class_name in ['COVID', 'Normal', 'Viral_Pneumonia']:
        (target_dir / split / class_name).mkdir(parents=True, exist_ok=True)

# Process each class
class_mapping = {
    'COVID': covid_images,
    'Normal': normal_images,
    'Viral_Pneumonia': pneumonia_images
}

stats = {}
for class_name, class_path in class_mapping.items():
    if not class_path.exists():
        print(f"Warning: {class_path} not found, skipping...")
        continue
    
    # Get all images
    images = list(class_path.glob('*.png'))
    print(f"\nProcessing {class_name}: {len(images)} images")
    np.random.shuffle(images)
    
    # Calculate split indices
    n_train = int(len(images) * split_ratios[0])
    n_val = int(len(images) * split_ratios[1])
    
    # Split data
    train_imgs = images[:n_train]
    val_imgs = images[n_train:n_train + n_val]
    test_imgs = images[n_train + n_val:]
    
    # Copy files
    print(f"  Copying {len(train_imgs)} to train...")
    for img in train_imgs:
        shutil.copy(img, target_dir / 'train' / class_name / img.name)
    
    print(f"  Copying {len(val_imgs)} to val...")
    for img in val_imgs:
        shutil.copy(img, target_dir / 'val' / class_name / img.name)
    
    print(f"  Copying {len(test_imgs)} to test...")
    for img in test_imgs:
        shutil.copy(img, target_dir / 'test' / class_name / img.name)
    
    stats[class_name] = {
        'train': len(train_imgs),
        'val': len(val_imgs),
        'test': len(test_imgs),
        'total': len(images)
    }
    
    print(f"  ✓ {class_name}: {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test")

print("\n" + "="*80)
print("DATASET ORGANIZATION COMPLETE!")
print("="*80)
print("\nSummary:")
for class_name, counts in stats.items():
    print(f"{class_name}: {counts['total']} total images")
    print(f"  Train: {counts['train']}")
    print(f"  Val:   {counts['val']}")
    print(f"  Test:  {counts['test']}")

print(f"\n✓ Organized dataset saved to: {target_dir}")
