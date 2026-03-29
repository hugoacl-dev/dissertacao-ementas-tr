"""
fase5_tuning_utils.py — Helpers compartilhados da Fase 5

Concentra a preparação do dataset de tuning, geração de manifests e utilitários
de nomenclatura e persistência usados pelos scripts de fine-tuning.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from artefato_utils import escrever_json_atomico
from jsonl_utils import extrair_fundamentacao_e_ementa
from project_paths import DATASET_TREINO_PATH, SYSTEM_PROMPT_PATH


def _ler_jsonl(path: Path) -> list[dict[str, Any]]:
    """Lê um arquivo JSONL do projeto."""
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
    if not registros:
        raise ValueError(f"Arquivo JSONL vazio: {path}")
    return registros


def carregar_amostras_treino_sft(
    dataset_path: Path = DATASET_TREINO_PATH,
    system_prompt_path: Path = SYSTEM_PROMPT_PATH,
) -> list[dict[str, Any]]:
    """Converte o dataset de treino do projeto em amostras SFT reutilizáveis."""
    system_prompt = system_prompt_path.read_text(encoding="utf-8").strip()
    registros = _ler_jsonl(dataset_path)
    amostras: list[dict[str, Any]] = []

    for indice, registro in enumerate(registros):
        fundamentacao, ementa = extrair_fundamentacao_e_ementa(registro)
        if not fundamentacao.strip() or not ementa.strip():
            raise ValueError(f"Registro de treino inválido no índice {indice}.")
        amostras.append(
            {
                "caso_id": f"treino_{indice:05d}",
                "system_prompt": system_prompt,
                "fundamentacao": fundamentacao.strip(),
                "ementa": ementa.strip(),
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": fundamentacao.strip()},
                    {"role": "assistant", "content": ementa.strip()},
                ],
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": fundamentacao.strip()},
                ],
                "completion": [
                    {"role": "assistant", "content": ementa.strip()},
                ],
            }
        )

    return amostras


def gerar_nome_experimento(prefixo: str) -> str:
    """Gera nome determinístico por timestamp para manifests e jobs."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefixo}_{timestamp}"


def calcular_batch_size_efetivo(
    *,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
) -> int:
    """Calcula batch size efetivo."""
    return per_device_train_batch_size * gradient_accumulation_steps


def construir_uri_gcs(bucket: str, objeto: str) -> str:
    """Constrói URI GCS a partir de bucket e objeto."""
    bucket_limpo = bucket.removeprefix("gs://").strip("/")
    objeto_limpo = objeto.lstrip("/")
    return f"gs://{bucket_limpo}/{objeto_limpo}"


def escrever_manifesto_tuning(path: Path, payload: dict[str, Any]) -> Path:
    """Persiste manifesto do job de tuning."""
    escrever_json_atomico(path, payload, indent=2)
    return path
