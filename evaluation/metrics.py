"""Cálculo de métricas de qualidade: compara predições do sistema com o gabarito."""

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def calcular_metricas(predicoes: list[str], ground_truth: list[str]) -> dict:
    """Compara decisões do sistema (`predicoes`) com o gabarito (`ground_truth`).

    `average="weighted"`: calcula a métrica separadamente para cada classe
    (Risco Baixo/Médio/Alto) e tira uma média ponderada pelo número de casos
    de cada classe -- assim, uma classe rara não é ignorada nem domina o
    resultado.
    """
    return {
        "accuracy": accuracy_score(ground_truth, predicoes),
        "precision": precision_score(
            ground_truth, predicoes, average="weighted", zero_division=0
        ),
        "recall": recall_score(
            ground_truth, predicoes, average="weighted", zero_division=0
        ),
        "f1_score": f1_score(
            ground_truth, predicoes, average="weighted", zero_division=0
        ),
    }
