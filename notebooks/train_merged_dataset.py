#!/usr/bin/env python3
"""
Example training script for the merged dataset using YOLOv8.
"""

from ultralytics import YOLO
import yaml
from pathlib import Path

def train_merged_dataset():
    """Train YOLOv8 on the merged dataset."""
    
    # Path to the merged dataset configuration
    data_yaml_path = '/home/tamaturgo/desafio-fpf/notebooks/merged_dataset/data.yaml'
    
    print("="*50)
    print("TRAINING YOLOV8 ON MERGED DATASET")
    print("="*50)
    
    # Load dataset configuration
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    print(f"Dataset configuration:")
    print(f"  Classes: {data_config['nc']}")
    print(f"  Class names: {data_config['names']}")
    print(f"  Train path: {data_config['train']}")
    print(f"  Validation path: {data_config['val']}")
    print(f"  Test path: {data_config['test']}")
    
    # Initialize YOLO model
    model = YOLO('yolov8n.pt')  # Use nano model for faster training
    
    # Training parameters
    train_params = {
        'data': data_yaml_path,
        'epochs': 80,          # Adjust as needed
        'batch': 32,           # Adjust based on GPU memory
        'imgsz': 640,          # Image size
        'patience': 10,        # Early stopping patience
        'save_period': 10,     # Save checkpoint every 10 epochs
        'project': '/home/tamaturgo/desafio-fpf/runs/detect',
        'name': 'merged_model',
        'exist_ok': True
    }
    
    print(f"\nTraining parameters:")
    for key, value in train_params.items():
        print(f"  {key}: {value}")
    
    print(f"\nStarting training...")
    
    # Train the model
    try:
        results = model.train(**train_params)
        print(f"\n✓ Training completed successfully!")
        print(f"Model saved at: {results.save_dir}")
        
        # Validate the model
        print(f"\nRunning validation...")
        metrics = model.val()
        print(f"Validation mAP50: {metrics.box.map50:.4f}")
        print(f"Validation mAP50-95: {metrics.box.map:.4f}")
        
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        return None
    
    return results

if __name__ == "__main__":
    results = train_merged_dataset()
