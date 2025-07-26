"""
Utilitários gerais para o sistema de visão computacional.
"""

import json
import os
from typing import Dict, Any, List
from datetime import datetime
from pathlib import Path


def save_results_to_json(results: Dict, output_path: str, indent: int = 2) -> bool:
    """
    Salva resultados em arquivo JSON.
    
    Args:
        results: Dicionário com os resultados
        output_path: Caminho para salvar o arquivo
        indent: Indentação do JSON
        
    Returns:
        True se salvou com sucesso, False caso contrário
    """
    try:
        # Cria o diretório se não existir
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Converte arrays numpy para listas (se houver)
        serializable_results = make_json_serializable(results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=indent, ensure_ascii=False)
        
        return True
    except Exception as e:
        print(f"Erro ao salvar JSON: {e}")
        return False


def make_json_serializable(obj: Any) -> Any:
    """
    Converte objetos para formato serializável em JSON.
    
    Args:
        obj: Objeto a ser convertido
        
    Returns:
        Objeto serializável
    """
    import numpy as np
    
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    else:
        return obj


def load_results_from_json(file_path: str) -> Dict:
    """
    Carrega resultados de arquivo JSON.
    
    Args:
        file_path: Caminho do arquivo JSON
        
    Returns:
        Dicionário com os resultados carregados
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Erro ao carregar JSON: {e}")
        return {}


def create_output_directory(base_dir: str, timestamp: bool = True) -> str:
    """
    Cria diretório de saída com timestamp opcional.
    
    Args:
        base_dir: Diretório base
        timestamp: Se deve adicionar timestamp ao nome
        
    Returns:
        Caminho do diretório criado
    """
    if timestamp:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(base_dir, f"output_{timestamp_str}")
    else:
        output_dir = base_dir
    
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def validate_image_path(image_path: str) -> bool:
    """
    Valida se o caminho da imagem é válido.
    
    Args:
        image_path: Caminho da imagem
        
    Returns:
        True se válido, False caso contrário
    """
    if not os.path.exists(image_path):
        return False
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    file_extension = Path(image_path).suffix.lower()
    
    return file_extension in valid_extensions


def get_image_files_from_directory(directory: str, recursive: bool = False) -> List[str]:
    """
    Obtém lista de arquivos de imagem de um diretório.
    
    Args:
        directory: Diretório para buscar
        recursive: Se deve buscar recursivamente
        
    Returns:
        Lista de caminhos de imagens
    """
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    if recursive:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if Path(file).suffix.lower() in valid_extensions:
                    image_files.append(os.path.join(root, file))
    else:
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path) and Path(file).suffix.lower() in valid_extensions:
                image_files.append(file_path)
    
    return sorted(image_files)


def format_processing_summary(stats: Dict) -> str:
    """
    Formata um resumo legível das estatísticas de processamento.
    
    Args:
        stats: Dicionário com estatísticas
        
    Returns:
        String formatada com o resumo
    """
    summary = f"""
=== RESUMO DO PROCESSAMENTO ===
Total de imagens: {stats['total_images']}
Processadas com sucesso: {stats['successful_processing']}
Falharam: {stats['failed_processing']}

=== DETECÇÕES ===
Total de objetos detectados: {stats['total_objects_detected']}
Total de QR codes detectados: {stats['total_qr_codes_detected']}
QR crops salvos: {stats['total_qr_crops_saved']}

=== PERFORMANCE ===
Tempo médio de processamento: {stats['average_processing_time_ms']:.1f} ms

=== CLASSES DETECTADAS ===
"""
    
    for class_name, count in stats['classes_summary'].items():
        summary += f"{class_name}: {count} detecções\n"
    
    return summary


def create_directory_structure(base_path: str) -> Dict[str, str]:
    """
    Cria estrutura de diretórios padrão para o projeto.
    
    Args:
        base_path: Caminho base
        
    Returns:
        Dicionário com os caminhos criados
    """
    directories = {
        "qr_crops": os.path.join(base_path, "qr_crops"),
        "outputs": os.path.join(base_path, "outputs"),
        "temp": os.path.join(base_path, "temp"),
        "logs": os.path.join(base_path, "logs")
    }
    
    for name, path in directories.items():
        os.makedirs(path, exist_ok=True)
    
    return directories


class Logger:
    """
    Logger simples para o sistema.
    """
    
    def __init__(self, log_file: str = None):
        """
        Inicializa o logger.
        
        Args:
            log_file: Caminho do arquivo de log (opcional)
        """
        self.log_file = log_file
        
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    def log(self, message: str, level: str = "INFO"):
        """
        Registra uma mensagem de log.
        
        Args:
            message: Mensagem a ser registrada
            level: Nível do log (INFO, WARNING, ERROR)
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {level}: {message}"
        
        print(log_message)
        
        if self.log_file:
            try:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(log_message + "\n")
            except Exception as e:
                print(f"Erro ao escrever no log: {e}")
    
    def info(self, message: str):
        """Registra mensagem de informação."""
        self.log(message, "INFO")
    
    def warning(self, message: str):
        """Registra mensagem de aviso."""
        self.log(message, "WARNING")
    
    def error(self, message: str):
        """Registra mensagem de erro."""
        self.log(message, "ERROR")
