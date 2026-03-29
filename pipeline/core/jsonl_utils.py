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


def extrair_prompt_e_fundamentacao_do_texto_user(
    texto_user: str,
) -> tuple[str | None, str]:
    """Separa prompt canônico e fundamentação do turno `user`.

    Retorna `(prompt, fundamentacao)`. Quando o marcador canônico não está
    presente, assume formato legado e retorna `(None, texto_user)`.
    """
    idx = texto_user.find(MARCADOR_FUNDAMENTACAO)
    if idx < 0:
        return None, texto_user
    prompt = texto_user[:idx].strip()
    fundamentacao = texto_user[idx + len(MARCADOR_FUNDAMENTACAO) :]
    return prompt, fundamentacao


def extrair_fundamentacao_do_texto_user(texto_user: str) -> str:
    """Extrai apenas a fundamentação do texto completo do turno `user`.

    O formato esperado do projeto é:
        {system_prompt}\n\n{MARCADOR_FUNDAMENTACAO}{fundamentacao}

    Se o marcador não for encontrado, retorna o texto original como fallback
    compatível para bases legadas ou formatos inesperados.
    """
    _, fundamentacao = extrair_prompt_e_fundamentacao_do_texto_user(texto_user)
    return fundamentacao


def extrair_prompt_do_registro_jsonl(obj: dict[str, Any]) -> str | None:
    """Extrai o prompt embutido do registro multiturno do projeto."""
    for content in obj.get("contents", []):
        if content.get("role") != "user":
            continue
        texto = _extrair_texto_parts(content)
        prompt, _ = extrair_prompt_e_fundamentacao_do_texto_user(texto)
        return prompt
    return None


def validar_prompt_canonico_do_registro(
    obj: dict[str, Any],
    *,
    prompt_canonico: str,
    contexto: str,
) -> None:
    """Falha se o registro não refletir o prompt canônico congelado.

    Isso evita drift metodológico silencioso quando `system_prompt.txt` muda,
    mas os JSONL não são regenerados.
    """
    prompt_embutido = extrair_prompt_do_registro_jsonl(obj)
    if prompt_embutido is None:
        raise ValueError(
            f"O registro {contexto} não contém o marcador canônico da fundamentação."
        )
    if prompt_embutido.strip() != prompt_canonico.strip():
        raise ValueError(
            f"O prompt embutido no registro {contexto} diverge de `system_prompt.txt`. "
            "Regenere os JSONL da Fase 3 antes de executar as Fases 5–7."
        )


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
