#!/usr/bin/env python3
"""
Prediction script for the merged dataset using trained YOLOv8 models.
Makes predictions on test images and saves results with visualizations.
"""

from ultralytics import YOLO
import yaml
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json
from datetime import datetime
import shutil

def load_model_and_config(model_path, data_yaml_path):
    """Load YOLO model and dataset configuration."""
    
    # Load model
    model = YOLO(model_path)
    print(f"✓ Model loaded: {model_path}")
    
    # Load dataset configuration
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    class_names = data_config['names']
    print(f"✓ Classes: {class_names}")
    
    return model, class_names

def predict_on_images(model, image_paths, class_names, output_dir, conf_threshold=0.25):
    """Make predictions on a list of images and save results."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (output_dir / "images_with_predictions").mkdir(exist_ok=True)
    (output_dir / "cropped_detections").mkdir(exist_ok=True)
    (output_dir / "json_results").mkdir(exist_ok=True)
    
    print(f"Saving results to: {output_dir}")
    
    all_results = []
    
    for i, image_path in enumerate(image_paths):
        image_path = Path(image_path)
        print(f"Processing ({i+1}/{len(image_paths)}): {image_path.name}")
        
        try:
            # Make prediction
            results = model(str(image_path), conf=conf_threshold)
            
            # Load original image
            original_image = cv2.imread(str(image_path))
            original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            
            # Prepare result data
            image_results = {
                'image_path': str(image_path),
                'image_name': image_path.name,
                'predictions': [],
                'timestamp': datetime.now().isoformat()
            }
            
            # Process detections
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for j, box in enumerate(boxes):
                        # Get box coordinates
                        xyxy = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        
                        class_name = class_names[cls] if cls < len(class_names) else f"Class_{cls}"
                        
                        # Store prediction data
                        prediction = {
                            'class_id': int(cls),
                            'class_name': class_name,
                            'confidence': float(conf),
                            'bbox': [float(x) for x in xyxy],  # [x1, y1, x2, y2]
                        }
                        image_results['predictions'].append(prediction)
                        
                        # Crop detection
                        x1, y1, x2, y2 = map(int, xyxy)
                        cropped = original_image_rgb[y1:y2, x1:x2]
                        
                        if cropped.size > 0:
                            crop_filename = f"{image_path.stem}_{class_name}_{j}_{conf:.2f}.jpg"
                            crop_path = output_dir / "cropped_detections" / crop_filename
                            
                            # Save cropped detection
                            cropped_pil = Image.fromarray(cropped)
                            cropped_pil.save(crop_path, quality=95)
            
            # Save annotated image
            annotated_image = draw_predictions(original_image_rgb, image_results['predictions'])
            annotated_path = output_dir / "images_with_predictions" / f"pred_{image_path.name}"
            annotated_image.save(annotated_path, quality=95)
            
            # Save JSON results
            json_path = output_dir / "json_results" / f"{image_path.stem}.json"
            with open(json_path, 'w') as f:
                json.dump(image_results, f, indent=2)
            
            all_results.append(image_results)
            
            print(f"  Found {len(image_results['predictions'])} detections")
            
        except Exception as e:
            print(f"  ✗ Error processing {image_path.name}: {e}")
    
    # Save summary
    summary = {
        'total_images': len(image_paths),
        'total_detections': sum(len(r['predictions']) for r in all_results),
        'class_distribution': {},
        'timestamp': datetime.now().isoformat(),
        'model_path': str(model.ckpt_path) if hasattr(model, 'ckpt_path') else 'unknown',
        'confidence_threshold': conf_threshold
    }
    
    # Calculate class distribution
    for result in all_results:
        for pred in result['predictions']:
            class_name = pred['class_name']
            summary['class_distribution'][class_name] = summary['class_distribution'].get(class_name, 0) + 1
    
    summary_path = output_dir / "prediction_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Prediction complete!")
    print(f"  Total images processed: {summary['total_images']}")
    print(f"  Total detections: {summary['total_detections']}")
    print(f"  Class distribution: {summary['class_distribution']}")
    print(f"  Results saved to: {output_dir}")
    
    return all_results

def draw_predictions(image, predictions):
    """Draw bounding boxes and labels on image."""
    
    # Convert to PIL Image
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    
    # Color map for classes
    colors = {
        'box': (255, 0, 0),      # Red
        'qr_code': (0, 255, 0),  # Green
    }
    
    try:
        # Try to load a font
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        # Fallback to default font
        font = ImageFont.load_default()
    
    for pred in predictions:
        x1, y1, x2, y2 = pred['bbox']
        class_name = pred['class_name']
        confidence = pred['confidence']
        
        # Get color for class
        color = colors.get(class_name, (255, 255, 0))  # Default to yellow
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Draw label
        label = f"{class_name}: {confidence:.2f}"
        
        # Get text size for background
        bbox = draw.textbbox((x1, y1), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Draw background for text
        draw.rectangle([x1, y1-text_height-5, x1+text_width+10, y1], fill=color)
        
        # Draw text
        draw.text((x1+5, y1-text_height-2), label, fill=(255, 255, 255), font=font)
    
    return pil_image

def predict_on_test_set(model_path, data_yaml_path, output_dir, max_images=50):
    """Make predictions on the test set."""
    
    print("="*60)
    print("MAKING PREDICTIONS ON TEST SET")
    print("="*60)
    
    # Load model and config
    model, class_names = load_model_and_config(model_path, data_yaml_path)
    
    # Load dataset configuration to get test path
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    test_images_path = Path(data_config['test'])
    
    if not test_images_path.exists():
        print(f"✗ Test images path does not exist: {test_images_path}")
        return
    
    # Get test images
    test_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        test_images.extend(list(test_images_path.glob(ext)))
    
    if not test_images:
        print(f"✗ No test images found in: {test_images_path}")
        return
    
    # Limit number of images
    if len(test_images) > max_images:
        test_images = test_images[:max_images]
        print(f"Processing first {max_images} images out of {len(list(test_images_path.glob('*')))} available")
    
    print(f"Found {len(test_images)} test images")
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir) / f"predictions_{timestamp}"
    
    # Make predictions
    results = predict_on_images(model, test_images, class_names, output_path)
    
    return results

def predict_on_custom_images(model_path, data_yaml_path, images_dir, output_dir):
    """Make predictions on custom images."""
    
    print("="*60)
    print("MAKING PREDICTIONS ON CUSTOM IMAGES")
    print("="*60)
    
    # Load model and config
    model, class_names = load_model_and_config(model_path, data_yaml_path)
    
    images_path = Path(images_dir)
    if not images_path.exists():
        print(f"✗ Images directory does not exist: {images_path}")
        return
    
    # Get custom images
    custom_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        custom_images.extend(list(images_path.glob(ext)))
    
    if not custom_images:
        print(f"✗ No images found in: {images_path}")
        return
    
    print(f"Found {len(custom_images)} custom images")
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir) / f"custom_predictions_{timestamp}"
    
    # Make predictions
    results = predict_on_images(model, custom_images, class_names, output_path)
    
    return results

def main():
    """Main prediction function."""
    
    # Configuration
    data_yaml_path = '/home/tamaturgo/desafio-fpf/notebooks/merged_dataset/data.yaml'
    output_base_dir = '/home/tamaturgo/desafio-fpf/predictions'
    
    # Find available models
    runs_dir = Path('/home/tamaturgo/desafio-fpf/runs/detect')
    model_paths = []
    
    if runs_dir.exists():
        for run_dir in runs_dir.iterdir():
            if run_dir.is_dir():
                weights_dir = run_dir / 'weights'
                if weights_dir.exists():
                    best_model = weights_dir / 'best.pt'
                    if best_model.exists():
                        model_paths.append(str(best_model))
    
    # Also check notebooks directory
    notebooks_dir = Path('/home/tamaturgo/desafio-fpf/notebooks')
    for model_file in notebooks_dir.glob('*.pt'):
        if 'yolov8' in model_file.name and 'trained' in model_file.name:
            model_paths.append(str(model_file))
    
    if not model_paths:
        print("No trained models found!")
        print("Please train a model first using train_merged_dataset.py")
        return
    
    # Use the first available model
    model_path = model_paths[0]
    print(f"Using model: {model_path}")
    
    # Make predictions on test set
    print("\n1. Predicting on test set...")
    test_results = predict_on_test_set(model_path, data_yaml_path, output_base_dir, max_images=30)
    
    # Check for custom images directory
    custom_images_dir = Path('/home/tamaturgo/desafio-fpf/custom_images')
    if custom_images_dir.exists():
        print("\n2. Predicting on custom images...")
        custom_results = predict_on_custom_images(model_path, data_yaml_path, str(custom_images_dir), output_base_dir)
    else:
        print(f"\n2. No custom images directory found at {custom_images_dir}")
        print("Create this directory and add images to test custom predictions.")

if __name__ == "__main__":
    main()
