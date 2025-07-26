"""
Configuração simples de logging para Docker.
"""
import logging
import sys

# Configuração básica que aparece no docker logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)

def get_logger(name):
    """Retorna um logger configurado."""
    return logging.getLogger(name)
