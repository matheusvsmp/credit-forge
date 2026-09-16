import json

import pytest


@pytest.fixture(scope="session")
def casos_teste():
    """Os 15 casos do dataset golden, carregados uma vez por sessão de teste."""
    with open("data/test_cases.json", encoding="utf-8") as f:
        return json.load(f)["test_cases"]
