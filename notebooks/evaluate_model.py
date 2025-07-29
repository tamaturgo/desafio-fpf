#!/usr/bin/env python3
"""
Script para avaliação do modelo YOLOv8 treinado.
"""

from ultralytics import YOLO
import yaml
import os
import sys

def evaluate_model(model_path, data_yaml_path):
    """
    Avalia o modelo YOLOv8 usando o conjunto de validação/teste definido no data.yaml.
    """
    try:
        if not os.path.exists(model_path):
            return None
        if not os.path.exists(data_yaml_path):
            return None

        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)

        dataset_dir = os.path.dirname(os.path.abspath(data_yaml_path))
        
        temp_data_config = data_config.copy()
        temp_data_config['train'] = os.path.join(dataset_dir, data_config['train'])
        temp_data_config['val'] = os.path.join(dataset_dir, data_config['val'])
        if 'test' in data_config:
            temp_data_config['test'] = os.path.join(dataset_dir, data_config['test'])
        
        temp_yaml_path = os.path.join(dataset_dir, 'temp_data.yaml')
        with open(temp_yaml_path, 'w') as f:
            yaml.dump(temp_data_config, f)

        model = YOLO(model_path)
        metrics = model.val(data=temp_yaml_path)
        
        os.remove(temp_yaml_path)
        
        return metrics
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        # Limpa arquivo temporário se existir
        try:
            if 'temp_yaml_path' in locals():
                os.remove(temp_yaml_path)
        except:
            pass
        return None

if __name__ == "__main__":
    # Caminho do modelo treinado e do data.yaml
    model_path = "runs/detect/merged_dataset_model/weights/best.pt"  # ajuste conforme necessário
    data_yaml_path = "notebooks/datasets/merged_dataset/data.yaml"  # ajuste conforme necessário
    evaluate_model(model_path, data_yaml_path)
