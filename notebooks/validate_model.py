#!/usr/bin/env python3
"""
Validation script for the merged dataset using trained YOLOv8 models.
"""

from ultralytics import YOLO
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import numpy as np

def validate_model(model_path, data_yaml_path, save_results=True):
    """Validate a trained YOLO model on the merged dataset."""
    
    print("="*60)
    print("YOLOV8 MODEL VALIDATION")
    print("="*60)
    
    # Load model
    try:
        model = YOLO(model_path)
        print(f"✓ Model loaded: {model_path}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return None
    
    # Load dataset configuration
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    print(f"\nDataset configuration:")
    print(f"  Classes: {data_config['nc']}")
    print(f"  Class names: {data_config['names']}")
    
    # Run validation
    print(f"\nRunning validation...")
    try:
        metrics = model.val(data=data_yaml_path, save_json=save_results, save_hybrid=save_results)
        
        print(f"\n" + "="*40)
        print("VALIDATION RESULTS")
        print("="*40)
        
        # Overall metrics
        print(f"Overall Performance:")
        print(f"  mAP50: {metrics.box.map50:.4f}")
        print(f"  mAP50-95: {metrics.box.map:.4f}")
        print(f"  Precision: {metrics.box.mp:.4f}")
        print(f"  Recall: {metrics.box.mr:.4f}")
        
        # Per-class metrics
        class_names = data_config['names']
        if hasattr(metrics.box, 'ap_class_index') and metrics.box.ap_class_index is not None:
            print(f"\nPer-class Performance:")
            for i, class_idx in enumerate(metrics.box.ap_class_index):
                if i < len(metrics.box.ap50):
                    class_name = class_names[int(class_idx)] if int(class_idx) < len(class_names) else f"Class_{class_idx}"
                    print(f"  {class_name}:")
                    print(f"    mAP50: {metrics.box.ap50[i]:.4f}")
                    print(f"    mAP50-95: {metrics.box.ap[i]:.4f}")
        
        # Save validation report
        if save_results:
            results_dir = Path(f"/home/tamaturgo/desafio-fpf/validation_results")
            results_dir.mkdir(exist_ok=True)
            
            # Save text report
            report_file = results_dir / "validation_report.txt"
            with open(report_file, 'w') as f:
                f.write("YOLO Model Validation Report\n")
                f.write("="*40 + "\n\n")
                f.write(f"Model: {model_path}\n")
                f.write(f"Dataset: {data_yaml_path}\n\n")
                f.write(f"Overall Performance:\n")
                f.write(f"  mAP50: {metrics.box.map50:.4f}\n")
                f.write(f"  mAP50-95: {metrics.box.map:.4f}\n")
                f.write(f"  Precision: {metrics.box.mp:.4f}\n")
                f.write(f"  Recall: {metrics.box.mr:.4f}\n\n")
                
                if hasattr(metrics.box, 'ap_class_index') and metrics.box.ap_class_index is not None:
                    f.write(f"Per-class Performance:\n")
                    for i, class_idx in enumerate(metrics.box.ap_class_index):
                        if i < len(metrics.box.ap50):
                            class_name = class_names[int(class_idx)] if int(class_idx) < len(class_names) else f"Class_{class_idx}"
                            f.write(f"  {class_name}:\n")
                            f.write(f"    mAP50: {metrics.box.ap50[i]:.4f}\n")
                            f.write(f"    mAP50-95: {metrics.box.ap[i]:.4f}\n")
            
            print(f"\n✓ Validation report saved: {report_file}")
        
        return metrics
        
    except Exception as e:
        print(f"✗ Validation failed: {e}")
        return None

def compare_models(model_paths, data_yaml_path):
    """Compare multiple models on the same dataset."""
    
    print("="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    results = {}
    
    for model_path in model_paths:
        print(f"\nValidating {model_path}...")
        metrics = validate_model(model_path, data_yaml_path, save_results=False)
        if metrics:
            results[model_path] = {
                'mAP50': metrics.box.map50,
                'mAP50-95': metrics.box.map,
                'Precision': metrics.box.mp,
                'Recall': metrics.box.mr
            }
    
    if results:
        print(f"\n" + "="*60)
        print("COMPARISON RESULTS")
        print("="*60)
        
        # Create comparison table
        print(f"{'Model':<30} {'mAP50':<8} {'mAP50-95':<10} {'Precision':<10} {'Recall':<8}")
        print("-" * 70)
        
        for model_path, metrics in results.items():
            model_name = Path(model_path).stem
            print(f"{model_name:<30} {metrics['mAP50']:<8.4f} {metrics['mAP50-95']:<10.4f} "
                  f"{metrics['Precision']:<10.4f} {metrics['Recall']:<8.4f}")
        
        # Save comparison
        results_dir = Path("/home/tamaturgo/desafio-fpf/validation_results")
        results_dir.mkdir(exist_ok=True)
        
        comparison_file = results_dir / "model_comparison.txt"
        with open(comparison_file, 'w') as f:
            f.write("Model Comparison Report\n")
            f.write("="*40 + "\n\n")
            f.write(f"{'Model':<30} {'mAP50':<8} {'mAP50-95':<10} {'Precision':<10} {'Recall':<8}\n")
            f.write("-" * 70 + "\n")
            
            for model_path, metrics in results.items():
                model_name = Path(model_path).stem
                f.write(f"{model_name:<30} {metrics['mAP50']:<8.4f} {metrics['mAP50-95']:<10.4f} "
                       f"{metrics['Precision']:<10.4f} {metrics['Recall']:<8.4f}\n")
        
        print(f"\n✓ Comparison saved: {comparison_file}")
    
    return results

def main():
    """Main validation function."""
    
    # Paths
    data_yaml_path = '/home/tamaturgo/desafio-fpf/notebooks/merged_dataset/data.yaml'
    
    # Check for available models
    runs_dir = Path('/home/tamaturgo/desafio-fpf/runs/detect')
    model_paths = []
    
    if runs_dir.exists():
        # Look for trained models
        for run_dir in runs_dir.iterdir():
            if run_dir.is_dir():
                weights_dir = run_dir / 'weights'
                if weights_dir.exists():
                    best_model = weights_dir / 'best.pt'
                    last_model = weights_dir / 'last.pt'
                    if best_model.exists():
                        model_paths.append(str(best_model))
                    elif last_model.exists():
                        model_paths.append(str(last_model))
    
    # Also check for pre-existing models in notebooks directory
    notebooks_dir = Path('/home/tamaturgo/desafio-fpf/notebooks')
    for model_file in notebooks_dir.glob('*.pt'):
        if 'yolov8' in model_file.name:
            model_paths.append(str(model_file))
    
    if not model_paths:
        print("No trained models found!")
        print("Please train a model first using train_merged_dataset.py")
        return
    
    print(f"Found {len(model_paths)} model(s):")
    for i, path in enumerate(model_paths, 1):
        print(f"  {i}. {path}")
    
    # If multiple models, compare them
    if len(model_paths) > 1:
        print(f"\nRunning comparison of all models...")
        compare_models(model_paths, data_yaml_path)
    
    # Validate the best/first model in detail
    print(f"\nRunning detailed validation on: {model_paths[0]}")
    validate_model(model_paths[0], data_yaml_path, save_results=True)

if __name__ == "__main__":
    main()
