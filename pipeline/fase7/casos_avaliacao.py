"""
casos_avaliacao.py — Geração dos casos-base da Fase 7

Converte `data/dataset_teste.jsonl` em `data/fase7/casos_avaliacao.jsonl`,
preservando a ordem do conjunto de teste e extraindo fundamentação e ementa
de referência em formato executável pela etapa de avaliação.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

from artefato_utils import escrever_jsonl_atomico
from jsonl_utils import extrair_fundamentacao_e_ementa
from project_paths import DATASET_TESTE_PATH, FASE7_CASOS_AVALIACAO_PATH

from .protocolo import validar_registro_caso_avaliacao

log = logging.getLogger(__name__)


def _ler_jsonl(path: Path) -> list[dict[str, Any]]:
    """Lê um JSONL do projeto e retorna os objetos em ordem."""
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


def gerar_casos_avaliacao(
    dataset_teste_path: Path = DATASET_TESTE_PATH,
    output_path: Path = FASE7_CASOS_AVALIACAO_PATH,
) -> Path:
    """Gera o arquivo `casos_avaliacao.jsonl` a partir do conjunto de teste."""
    registros = _ler_jsonl(dataset_teste_path)
    casos: list[dict[str, Any]] = []

    for indice_teste, registro in enumerate(registros):
        fundamentacao, ementa_referencia = extrair_fundamentacao_e_ementa(registro)
        caso = validar_registro_caso_avaliacao(
            {
                "caso_id": f"teste_{indice_teste:05d}",
                "indice_teste": indice_teste,
                "fundamentacao": fundamentacao,
                "ementa_referencia": ementa_referencia,
            }
        )
        casos.append(caso)

    escrever_jsonl_atomico(output_path, casos)
    return output_path


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    output_path = gerar_casos_avaliacao()
    log.info("Casos-base da Fase 7 gerados em %s", output_path)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, OSError, ValueError) as exc:
        log.critical("Execução interrompida: %s", exc)
        sys.exit(1)

