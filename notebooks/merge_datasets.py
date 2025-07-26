#!/usr/bin/env python3
"""
Script to merge box and qr-code datasets into a single unified dataset.
Box class will remain as 0, QR code class will become 1.
"""

import os
import shutil
import yaml
from pathlib import Path

def create_directory_structure(base_path):
    """Create the directory structure for the merged dataset."""
    directories = [
        'train/images',
        'train/labels', 
        'valid/images',
        'valid/labels',
        'test/images',
        'test/labels'
    ]
    
    for dir_path in directories:
        full_path = base_path / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {full_path}")

def copy_images_and_labels(source_dataset, dest_dataset, split, class_offset=0):
    """Copy images and labels from source to destination, updating class indices."""
    source_images = source_dataset / split / 'images'
    source_labels = source_dataset / split / 'labels'
    dest_images = dest_dataset / split / 'images'
    dest_labels = dest_dataset / split / 'labels'
    
    if not source_images.exists():
        print(f"Warning: {source_images} does not exist, skipping...")
        return 0, 0
    
    copied_images = 0
    copied_labels = 0
    
    # Copy images
    if source_images.exists():
        for image_file in source_images.iterdir():
            if image_file.is_file() and image_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                dest_file = dest_images / image_file.name
                shutil.copy2(image_file, dest_file)
                copied_images += 1
    
    # Copy and update labels
    if source_labels.exists():
        for label_file in source_labels.iterdir():
            if label_file.is_file() and label_file.suffix.lower() == '.txt':
                dest_file = dest_labels / label_file.name
                
                # Read original label file
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                # Update class indices and write to destination
                with open(dest_file, 'w') as f:
                    for line in lines:
                        parts = line.strip().split()
                        if parts:
                            # Update class index
                            old_class = int(parts[0])
                            new_class = old_class + class_offset
                            parts[0] = str(new_class)
                            f.write(' '.join(parts) + '\n')
                
                copied_labels += 1
    
    return copied_images, copied_labels

def create_unified_data_yaml(dest_path):
    """Create the unified data.yaml file."""
    data_config = {
        'train': str(dest_path / 'train' / 'images'),
        'val': str(dest_path / 'valid' / 'images'),
        'test': str(dest_path / 'test' / 'images'),
        'nc': 2,
        'names': ['box', 'qr_code']
    }
    
    yaml_path = dest_path / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    print(f"Created unified data.yaml at: {yaml_path}")
    return yaml_path

def main():
    # Define paths
    notebooks_path = Path('/home/tamaturgo/desafio-fpf/notebooks')
    box_dataset = notebooks_path / 'box'
    qr_dataset = notebooks_path / 'qr-code'
    merged_dataset = notebooks_path / 'merged_dataset'
    
    print("Starting dataset merge process...")
    print(f"Box dataset: {box_dataset}")
    print(f"QR dataset: {qr_dataset}")
    print(f"Merged dataset: {merged_dataset}")
    
    # Create merged dataset directory structure
    create_directory_structure(merged_dataset)
    
    # Copy box dataset (class remains 0)
    print("\nCopying box dataset...")
    for split in ['train', 'valid', 'test']:
        images, labels = copy_images_and_labels(box_dataset, merged_dataset, split, class_offset=0)
        print(f"  {split}: {images} images, {labels} labels")
    
    # Copy QR code dataset (class becomes 1)  
    print("\nCopying QR code dataset...")
    for split in ['train', 'valid', 'test']:
        images, labels = copy_images_and_labels(qr_dataset, merged_dataset, split, class_offset=1)
        print(f"  {split}: {images} images, {labels} labels")
    
    # Create unified data.yaml
    print("\nCreating unified configuration...")
    yaml_path = create_unified_data_yaml(merged_dataset)
    
    # Print summary
    print("\n" + "="*50)
    print("MERGE COMPLETE!")
    print("="*50)
    print(f"Merged dataset location: {merged_dataset}")
    print(f"Configuration file: {yaml_path}")
    print("\nDataset structure:")
    print("- Class 0: box")
    print("- Class 1: qr_code")
    print("\nDirectory structure:")
    for split in ['train', 'valid', 'test']:
        split_path = merged_dataset / split
        if split_path.exists():
            images_count = len(list((split_path / 'images').glob('*'))) if (split_path / 'images').exists() else 0
            labels_count = len(list((split_path / 'labels').glob('*'))) if (split_path / 'labels').exists() else 0
            print(f"  {split}/: {images_count} images, {labels_count} labels")

if __name__ == "__main__":
    main()
