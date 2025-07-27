#!/usr/bin/env python3

"""
Script para unir os datasets 'dataset_warehouse' e 'merged_dataset', reanotando as labels conforme o novo índice das classes e gerando um novo dataset combinado.
"""


import os
import shutil
import yaml
from pathlib import Path
from glob import glob

def read_classes(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data['names']

def get_yaml_path(dataset_path):
    return os.path.join(dataset_path, 'data.yaml')

def collect_classes(ds1, ds2):
    classes1 = read_classes(get_yaml_path(ds1))
    classes2 = read_classes(get_yaml_path(ds2))
    all_classes = list(dict.fromkeys(classes1 + classes2))
    return all_classes, classes1, classes2

def create_directory_structure(base_path):
    directories = [
        'train/images', 'train/labels',
        'valid/images', 'valid/labels',
        'test/images', 'test/labels'
    ]
    for dir_path in directories:
        full_path = base_path / dir_path
        full_path.mkdir(parents=True, exist_ok=True)

def remap_label(label_path, class_map):
    new_lines = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            old_idx = int(parts[0])
            new_idx = class_map.get(old_idx, old_idx)
            new_line = ' '.join([str(new_idx)] + parts[1:])
            new_lines.append(new_line)
    return new_lines

def copy_images_and_labels(source_dataset, dest_dataset, split, class_map):
    source_images = Path(source_dataset) / split / 'images'
    source_labels = Path(source_dataset) / split / 'labels'
    dest_images = Path(dest_dataset) / split / 'images'
    dest_labels = Path(dest_dataset) / split / 'labels'
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
                new_lbl = remap_label(label_file, class_map)
                with open(dest_file, 'w') as f:
                    f.write('\n'.join(new_lbl) + '\n')
                copied_labels += 1
    return copied_images, copied_labels


def create_unified_data_yaml(dest_path, all_classes):
    data_config = {
        'train': str(dest_path / 'train' / 'images'),
        'val': str(dest_path / 'valid' / 'images'),
        'test': str(dest_path / 'test' / 'images'),
        'names': all_classes
    }
    yaml_path = dest_path / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f, allow_unicode=True)
    print(f"Created unified data.yaml at: {yaml_path}")
    return yaml_path


def main():
    notebooks_path = Path('/home/tamaturgo/desafio-fpf/notebooks')
    ds1 = notebooks_path / 'dataset_warehouse'
    ds2 = notebooks_path / 'merged_dataset'
    out_ds = notebooks_path / 'combined_dataset'

    print("Iniciando merge dos datasets...")
    print(f"Dataset 1: {ds1}")
    print(f"Dataset 2: {ds2}")
    print(f"Dataset combinado: {out_ds}")

    create_directory_structure(out_ds)

    all_classes, classes1, classes2 = collect_classes(ds1, ds2)
    map1 = {i: all_classes.index(name) for i, name in enumerate(classes1)}
    map2 = {i: all_classes.index(name) for i, name in enumerate(classes2)}

    print("\nCopiando dataset 1 (dataset_warehouse)...")
    for split in ['train', 'valid', 'test']:
        images, labels = copy_images_and_labels(ds1, out_ds, split, map1)
        print(f"  {split}: {images} imagens, {labels} labels")

    print("\nCopiando dataset 2 (merged_dataset)...")
    for split in ['train', 'valid', 'test']:
        images, labels = copy_images_and_labels(ds2, out_ds, split, map2)
        print(f"  {split}: {images} imagens, {labels} labels")

    print("\nCriando data.yaml unificado...")
    yaml_path = create_unified_data_yaml(out_ds, all_classes)

    print("\n" + "="*50)
    print("MERGE CONCLUÍDO!")
    print("="*50)
    print(f"Local do dataset combinado: {out_ds}")
    print(f"Arquivo de configuração: {yaml_path}")
    print("\nClasses:")
    for idx, name in enumerate(all_classes):
        print(f"- Classe {idx}: {name}")
    print("\nEstrutura de diretórios:")
    for split in ['train', 'valid', 'test']:
        split_path = out_ds / split
        if split_path.exists():
            images_count = len(list((split_path / 'images').glob('*'))) if (split_path / 'images').exists() else 0
            labels_count = len(list((split_path / 'labels').glob('*'))) if (split_path / 'labels').exists() else 0
            print(f"  {split}/: {images_count} imagens, {labels_count} labels")

if __name__ == "__main__":
    main()
