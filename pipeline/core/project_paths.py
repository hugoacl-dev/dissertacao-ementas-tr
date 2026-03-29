"""
project_paths.py — Caminhos compartilhados do pipeline

Centraliza os caminhos dos artefatos e entradas do projeto para reduzir
duplicação entre os scripts das Fases 1–4 e utilitários.
"""
from __future__ import annotations

from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent.parent
PROMPTS_DIR = PIPELINE_DIR / "prompts"
DATA_DIR = Path("data")
DOCS_DATA_DIR = Path("docs/data")
FASE5_DIR = DATA_DIR / "fase5"
FASE7_DIR = DATA_DIR / "fase7"
FASE7_PREDICOES_DIR = FASE7_DIR / "predicoes"

DUMP_PATH = DATA_DIR / "dump_sistema_judicial.sql"
SQLITE_DB_PATH = DATA_DIR / "banco_sistema_judicial.sqlite"

DADOS_BRUTOS_PATH = DATA_DIR / "dados_brutos.json"
DADOS_LIMPOS_PATH = DATA_DIR / "dados_limpos.json"
DATASET_TREINO_PATH = DATA_DIR / "dataset_treino.jsonl"
DATASET_TESTE_PATH = DATA_DIR / "dataset_teste.jsonl"
ESTATISTICAS_PATH = DATA_DIR / "estatisticas_corpus.json"

INGESTAO_STATS_PATH = DATA_DIR / ".ingestao_stats.json"
ANONIMIZACAO_STATS_PATH = DATA_DIR / ".anonimizacao_stats.json"
PIPELINE_TIMING_PATH = DATA_DIR / ".pipeline_timing.json"

SYSTEM_PROMPT_PATH = PROMPTS_DIR / "system_prompt.txt"
LLM_JUDGE_PROMPT_PATH = PROMPTS_DIR / "llm_judge_prompt.txt"

FASE7_PROTOCOLO_PATH = FASE7_DIR / "protocolo_avaliacao.json"
FASE7_CASOS_AVALIACAO_PATH = FASE7_DIR / "casos_avaliacao.jsonl"
FASE7_METRICAS_AUTOMATICAS_PATH = FASE7_DIR / "metricas_automaticas.csv"
FASE7_AVALIACAO_JUDGE_PATH = FASE7_DIR / "avaliacao_llm_judge.jsonl"
FASE7_AVALIACAO_JUDGE_BRUTA_PATH = FASE7_DIR / "avaliacao_llm_judge_bruta.jsonl"
FASE7_AVALIACAO_JUDGE_MANIFEST_PATH = FASE7_DIR / "avaliacao_llm_judge_manifest.json"
FASE7_AMOSTRA_HUMANA_PATH = FASE7_DIR / "amostra_humana.json"
FASE7_GABARITO_CEGAMENTO_HUMANO_PATH = FASE7_DIR / "gabarito_cegamento_humano.json"
FASE7_AVALIACAO_HUMANA_PATH = FASE7_DIR / "avaliacao_humana.csv"
FASE7_RELATORIO_AVALIACAO_HUMANA_PATH = FASE7_DIR / "relatorio_avaliacao_humana.json"
FASE7_RELATORIO_ESTATISTICO_PATH = FASE7_DIR / "relatorio_estatistico.json"

FASE5_GEMINI_MANIFEST_PATH = FASE5_DIR / "gemini_sft_manifest.json"
FASE5_GEMINI_MODELO_PATH = FASE5_DIR / "modelo_gemini_nome.txt"
FASE5_QWEN_MANIFEST_PATH = FASE5_DIR / "qwen_sft_manifest.json"
FASE5_QWEN_CHECKPOINT_DIR = FASE5_DIR / "modelo_qwen_checkpoint"

DATASET_PATHS = {
    "treino": DATASET_TREINO_PATH,
    "teste": DATASET_TESTE_PATH,
}

FASE7_PREDICAO_PATHS = {
    "gemini_ft": FASE7_PREDICOES_DIR / "gemini_ft.jsonl",
    "gemini_zero_shot": FASE7_PREDICOES_DIR / "gemini_zero_shot.jsonl",
    "qwen_ft": FASE7_PREDICOES_DIR / "qwen_ft.jsonl",
    "qwen_zero_shot": FASE7_PREDICOES_DIR / "qwen_zero_shot.jsonl",
}

FASE7_PREDICAO_MANIFEST_PATHS = {
    condicao_id: path.with_suffix(".manifest.json")
    for condicao_id, path in FASE7_PREDICAO_PATHS.items()
}
