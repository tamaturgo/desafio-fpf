#!/usr/bin/env python3
"""
Example training script for the merged dataset using YOLOv8.
"""

from ultralytics import YOLO
import yaml
from roboflow import Roboflow
def train_merged_dataset():
    """Train YOLOv8 on the merged dataset."""
    
    data_yaml_path = 'notebooks/datasets/merged_dataset/data.yaml'
    print("="*50)
    print("TRAINING YOLOV8 ON MERGED DATASET")
    print("="*50)
    
    # Load dataset configuration
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    print(f"Dataset configuration:")
    print(f"  Class names: {data_config['names']}")
    print(f"  Train path: {data_config['train']}")
    print(f"  Validation path: {data_config['val']}")
    print(f"  Test path: {data_config['test']}")
    
    # Initialize YOLO model
    model = YOLO('yolov8n.pt')  

    # Training parameters
    train_params = {
        'data': data_yaml_path,
        'epochs': 80,          
        'batch': 8,           
        'imgsz': 640,          
        'patience': 10,        
        'save_period': 10,     
        'project': '/home/tamaturgo/desafio-fpf/runs/detect',
        'name': 'merged_dataset_model',
        'exist_ok': True,
        'device': '0',  
        'workers': 4, 
        'cache': True,  
        'verbose': True,
        'pretrained': True, 
        'optimizer': 'SGD',
        'lr0': 0.01,  
        'momentum': 0.937,
        'weight_decay': 0.0005,  
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
