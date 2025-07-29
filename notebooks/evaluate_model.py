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

        print(f"Modelo encontrado: {model_path}")
        print(f"Data.yaml encontrado: {data_yaml_path}")

        # Carrega configuração do dataset
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        print(f"Classes: {data_config['names']}")
        print(f"Validação: {data_config['val']}")
        print(f"Teste: {data_config.get('test', 'Não definido')}")

        # Ajusta os caminhos para absolutos baseados no diretório do dataset
        dataset_dir = os.path.dirname(os.path.abspath(data_yaml_path))
        
        # Cria um data.yaml temporário com caminhos absolutos
        temp_data_config = data_config.copy()
        temp_data_config['train'] = os.path.join(dataset_dir, data_config['train'])
        temp_data_config['val'] = os.path.join(dataset_dir, data_config['val'])
        if 'test' in data_config:
            temp_data_config['test'] = os.path.join(dataset_dir, data_config['test'])
        
        temp_yaml_path = os.path.join(dataset_dir, 'temp_data.yaml')
        with open(temp_yaml_path, 'w') as f:
            yaml.dump(temp_data_config, f)
        
        print(f"Usando caminhos absolutos:")
        print(f"  Train: {temp_data_config['train']}")
        print(f"  Val: {temp_data_config['val']}")
        if 'test' in temp_data_config:
            print(f"  Test: {temp_data_config['test']}")

        # Carrega o modelo
        print("Carregando modelo...")
        model = YOLO(model_path)
        print("Modelo carregado com sucesso!")

        # Avalia o modelo
        print("\nIniciando avaliação...")
        metrics = model.val(data=temp_yaml_path)
        
        # Remove o arquivo temporário
        os.remove(temp_yaml_path)
        
        print(f"\nResultados da avaliação:")
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        print(f"Precisão: {metrics.box.precision:.4f}")
        print(f"Recall: {metrics.box.recall:.4f}")
        return metrics
    
    except Exception as e:
        print(f"Erro durante a avaliação: {e}")
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
