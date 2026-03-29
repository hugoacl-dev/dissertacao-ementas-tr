"""
jsonl_utils.py — Utilitários puros para leitura do formato JSONL do projeto

Centraliza a extração do turno `user` e da ementa a partir do formato
multiturno usado no dataset. Este módulo não depende de paths, logging
ou configuração de fases específicas.
"""
from __future__ import annotations

from typing import Any

MARCADOR_FUNDAMENTACAO = "Gere a ementa para a seguinte fundamentação:\n"


def _extrair_texto_parts(content: dict[str, Any]) -> str:
    """Concatena os textos de todas as partes de um bloco `content`."""
    return "".join(part.get("text", "") for part in content.get("parts", []))


def extrair_fundamentacao_do_texto_user(texto_user: str) -> str:
    """Extrai apenas a fundamentação do texto completo do turno `user`.

    O formato esperado do projeto é:
        {system_prompt}\n\n{MARCADOR_FUNDAMENTACAO}{fundamentacao}

    Se o marcador não for encontrado, retorna o texto original como fallback
    compatível para bases legadas ou formatos inesperados.
    """
    idx = texto_user.find(MARCADOR_FUNDAMENTACAO)
    if idx < 0:
        return texto_user
    return texto_user[idx + len(MARCADOR_FUNDAMENTACAO) :]


def extrair_fundamentacao_e_ementa(obj: dict[str, Any]) -> tuple[str, str]:
    """Extrai fundamentação e ementa de um registro JSONL do projeto."""
    fundamentacao = ""
    ementa = ""

    for content in obj.get("contents", []):
        role = content.get("role", "")
        texto = _extrair_texto_parts(content)
        if role == "user":
            fundamentacao = extrair_fundamentacao_do_texto_user(texto)
        elif role == "model":
            ementa = texto

    return fundamentacao, ementa

