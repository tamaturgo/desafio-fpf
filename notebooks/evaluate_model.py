#!/usr/bin/env python3
"""
Script para avaliação do modelo YOLOv8 treinado.
"""

from ultralytics import YOLO
import yaml
import os

def evaluate_model(model_path, data_yaml_path):
    """
    Avalia o modelo YOLOv8 usando o conjunto de validação/teste definido no data.yaml.
    """
    print("="*50)
    print("AVALIAÇÃO DO MODELO YOLOV8")
    print("="*50)

    # Verifica se os arquivos existem
    if not os.path.exists(model_path):
        print(f"Modelo não encontrado: {model_path}")
        return None
    if not os.path.exists(data_yaml_path):
        print(f"Arquivo data.yaml não encontrado: {data_yaml_path}")
        return None

    # Carrega configuração do dataset
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    print(f"Classes: {data_config['names']}")
    print(f"Validação: {data_config['val']}")
    print(f"Teste: {data_config.get('test', 'Não definido')}")

    # Carrega o modelo
    model = YOLO(model_path)

    # Avalia o modelo
    print("\nIniciando avaliação...")
    metrics = model.val(data=data_yaml_path)
    print(f"\nResultados da avaliação:")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precisão: {metrics.box.precision:.4f}")
    print(f"Recall: {metrics.box.recall:.4f}")
    return metrics

if __name__ == "__main__":
    # Caminho do modelo treinado e do data.yaml
    model_path = "../runs/detect/combined_dataset_model/weights/best.pt"  # ajuste conforme necessário
    data_yaml_path = "../notebooks/combined_dataset/data.yaml"  # ajuste conforme necessário
    evaluate_model(model_path, data_yaml_path)
