#!/usr/bin/env python3
"""
Training script for the merged dataset using YOLOv8.
"""

from ultralytics import YOLO
import yaml

def train_merged_dataset():
    """Train YOLOv8 on the merged dataset."""
    
    data_yaml_path = 'notebooks/datasets/merged_dataset/data.yaml'
    
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    model = YOLO('yolov8n.pt')

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
    
    try:
        results = model.train(**train_params)
        metrics = model.val()
        return results
        
    except Exception as e:
        return None

if __name__ == "__main__":
    results = train_merged_dataset()
