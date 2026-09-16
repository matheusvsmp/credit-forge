"""Configuração de logging estruturado (JSON) para rastrear execuções de agentes."""

import logging

from pythonjsonlogger import json as jsonlogger


def configurar_logger(nome: str = "credit_pipeline") -> logging.Logger:
    """Cria (ou reaproveita) um logger que emite cada linha como um objeto JSON."""
    logger = logging.getLogger(nome)

    # Evita adicionar handlers duplicados se a função for chamada mais de uma vez
    # (ex: um script principal e um teste importando o mesmo módulo).
    if logger.handlers:
        return logger

    handler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
