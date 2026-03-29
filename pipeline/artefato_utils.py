"""
artefato_utils.py — Helpers de persistência de artefatos

Reúne operações pequenas e reutilizáveis de escrita de artefatos em disco.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def escrever_json_atomico(
    path: Path,
    payload: Any,
    *,
    indent: int | None = 2,
) -> None:
    """Escreve JSON em UTF-8 de forma atômica.

    Args:
        path: Caminho de saída.
        payload: Estrutura serializável em JSON.
        indent: Indentação opcional do `json.dump`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=indent)
    temp_path.replace(path)


def escrever_csv_atomico(path: Path, df: pd.DataFrame, *, index: bool = False) -> None:
    """Escreve CSV em UTF-8 de forma atômica.

    Args:
        path: Caminho de saída.
        df: DataFrame a ser serializado.
        index: Se `True`, persiste o índice do DataFrame.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(temp_path, index=index)
    temp_path.replace(path)


def escrever_jsonl_atomico(path: Path, registros: list[dict[str, Any]]) -> None:
    """Escreve JSONL em UTF-8 de forma atômica.

    Args:
        path: Caminho de saída.
        registros: Lista de objetos JSON serializáveis, um por linha.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        for registro in registros:
            f.write(json.dumps(registro, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")
    temp_path.replace(path)
