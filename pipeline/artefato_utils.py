"""
artefato_utils.py — Helpers de persistência de artefatos

Reúne operações pequenas e reutilizáveis de escrita de artefatos em disco.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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
