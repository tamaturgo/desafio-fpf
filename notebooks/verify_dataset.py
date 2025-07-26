#!/usr/bin/env python3
"""
Verification script to check the merged dataset integrity.
"""

import os
from pathlib import Path
from collections import defaultdict

def verify_dataset(dataset_path):
    """Verify the integrity of the merged dataset."""
    dataset_path = Path(dataset_path)
    
    print("="*50)
    print("DATASET VERIFICATION")
    print("="*50)
    
    # Check class distribution
    class_counts = defaultdict(int)
    total_annotations = 0
    
    for split in ['train', 'valid', 'test']:
        labels_dir = dataset_path / split / 'labels'
        images_dir = dataset_path / split / 'images'
        
        if not labels_dir.exists():
            print(f"Warning: {labels_dir} does not exist")
            continue
            
        label_files = list(labels_dir.glob('*.txt'))
        image_files = list(images_dir.glob('*'))
        
        print(f"\n{split.upper()} SET:")
        print(f"  Images: {len(image_files)}")
        print(f"  Labels: {len(label_files)}")
        
        split_class_counts = defaultdict(int)
        split_annotations = 0
        
        for label_file in label_files:
            with open(label_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        class_id = int(line.split()[0])
                        class_counts[class_id] += 1
                        split_class_counts[class_id] += 1
                        total_annotations += 1
                        split_annotations += 1
        
        print(f"  Annotations: {split_annotations}")
        for class_id in sorted(split_class_counts.keys()):
            class_name = 'box' if class_id == 0 else 'qr_code'
            print(f"    Class {class_id} ({class_name}): {split_class_counts[class_id]}")
    
    print(f"\nOVERALL STATISTICS:")
    print(f"  Total annotations: {total_annotations}")
    for class_id in sorted(class_counts.keys()):
        class_name = 'box' if class_id == 0 else 'qr_code'
        percentage = (class_counts[class_id] / total_annotations) * 100
        print(f"  Class {class_id} ({class_name}): {class_counts[class_id]} ({percentage:.1f}%)")
    
    # Verify data.yaml
    data_yaml = dataset_path / 'data.yaml'
    if data_yaml.exists():
        print(f"\n✓ data.yaml exists at {data_yaml}")
        with open(data_yaml, 'r') as f:
            content = f.read()
            print(f"Configuration preview:\n{content}")
    else:
        print(f"\n✗ data.yaml missing at {data_yaml}")
    
    print("\n" + "="*50)
    print("VERIFICATION COMPLETE")
    print("="*50)

if __name__ == "__main__":
    verify_dataset('/home/tamaturgo/desafio-fpf/notebooks/merged_dataset')
