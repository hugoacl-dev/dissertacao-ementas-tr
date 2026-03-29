"""
project_paths.py — Caminhos compartilhados do pipeline

Centraliza os caminhos dos artefatos e entradas do projeto para reduzir
duplicação entre os scripts das Fases 1–4 e utilitários.
"""
from __future__ import annotations

from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path("data")
DOCS_DATA_DIR = Path("docs/data")

DUMP_PATH = Path("dump_sistema_judicial.sql")
SQLITE_DB_PATH = DATA_DIR / "banco_sistema_judicial.sqlite"

DADOS_BRUTOS_PATH = DATA_DIR / "dados_brutos.json"
DADOS_LIMPOS_PATH = DATA_DIR / "dados_limpos.json"
DATASET_TREINO_PATH = DATA_DIR / "dataset_treino.jsonl"
DATASET_TESTE_PATH = DATA_DIR / "dataset_teste.jsonl"
ESTATISTICAS_PATH = DATA_DIR / "estatisticas_corpus.json"

INGESTAO_STATS_PATH = DATA_DIR / ".ingestao_stats.json"
ANONIMIZACAO_STATS_PATH = DATA_DIR / ".anonimizacao_stats.json"
PIPELINE_TIMING_PATH = DATA_DIR / ".pipeline_timing.json"

SYSTEM_PROMPT_PATH = PIPELINE_DIR / "system_prompt.txt"

DATASET_PATHS = {
    "treino": DATASET_TREINO_PATH,
    "teste": DATASET_TESTE_PATH,
}
