"""
predicoes_utils.py — Helpers compartilhados para predições da Fase 7

Reúne leitura dos casos-base, carregamento de predições existentes,
normalização do texto gerado e persistência incremental das saídas das
condições zero-shot e, futuramente, fine-tuned.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.core.artefato_utils import escrever_jsonl_atomico
from pipeline.core.project_paths import FASE7_CASOS_AVALIACAO_PATH

from .protocolo import validar_registro_caso_avaliacao, validar_registro_predicao


def _ler_jsonl(path: Path) -> list[dict[str, Any]]:
    """Lê um arquivo JSONL e retorna seus objetos em ordem."""
    if not path.exists():
        raise FileNotFoundError(f"Arquivo JSONL não encontrado: {path}")
    registros: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for numero, linha in enumerate(f, start=1):
            conteudo = linha.strip()
            if not conteudo:
                continue
            try:
                registros.append(json.loads(conteudo))
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON inválido em {path}:{numero}: {exc.msg}") from exc
    return registros


def carregar_casos_predicao(path: Path = FASE7_CASOS_AVALIACAO_PATH) -> pd.DataFrame:
    """Carrega os casos-base da Fase 7 já validados."""
    registros = [validar_registro_caso_avaliacao(registro) for registro in _ler_jsonl(path)]
    if not registros:
        raise ValueError(f"Arquivo JSONL vazio: {path}")
    df = pd.DataFrame(registros)
    if df["caso_id"].duplicated().any():
        raise ValueError("Os casos-base da Fase 7 contêm `caso_id` duplicado.")
    return df.sort_values("indice_teste").reset_index(drop=True)


def carregar_predicoes_existentes(path: Path, *, condicao_id: str) -> list[dict[str, Any]]:
    """Carrega e valida predições já persistidas para uma condição."""
    if not path.exists():
        return []
    registros = [
        validar_registro_predicao(registro, condicao_id_esperada=condicao_id)
        for registro in _ler_jsonl(path)
    ]
    caso_ids = [registro["caso_id"] for registro in registros]
    if len(caso_ids) != len(set(caso_ids)):
        raise ValueError(
            f"O arquivo de predições {path} contém `caso_id` duplicado para {condicao_id}."
        )
    return registros


def normalizar_ementa_gerada(texto: str) -> str:
    """Normaliza o texto gerado para persistência do baseline."""
    linhas = [linha.strip() for linha in texto.splitlines() if linha.strip()]
    return " ".join(linhas).strip()


def filtrar_casos_pendentes(
    casos_df: pd.DataFrame,
    predicoes_existentes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Retorna os casos ainda não processados."""
    processados = {registro["caso_id"] for registro in predicoes_existentes}
    pendentes = casos_df[~casos_df["caso_id"].isin(processados)]
    return pendentes.to_dict(orient="records")


def persistir_predicoes(
    path: Path,
    *,
    condicao_id: str,
    registros: list[dict[str, Any]],
) -> None:
    """Valida e escreve o arquivo completo de predições de uma condição."""
    normalizados = [
        validar_registro_predicao(registro, condicao_id_esperada=condicao_id)
        for registro in registros
    ]
    escrever_jsonl_atomico(path, normalizados)
