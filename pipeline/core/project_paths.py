"""
project_paths.py — Caminhos compartilhados do pipeline

Centraliza os caminhos dos artefatos e entradas do projeto para reduzir
duplicação entre os scripts das Fases 1–4 e utilitários.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

PIPELINE_DIR = Path(__file__).resolve().parent.parent
PROMPTS_DIR = PIPELINE_DIR / "prompts"
DATA_DIR = Path("data")
DOCS_DATA_DIR = Path("docs/data")
DATA_EXPLORATORIO_DIR = DATA_DIR / "exploratorio"
FASE5_DIR = DATA_DIR / "fase5"
FASE7_DIR = DATA_DIR / "fase7"
FASE7_PREDICOES_DIR = FASE7_DIR / "predicoes"
FASE5_EXPLORATORIA_DIR = DATA_EXPLORATORIO_DIR / "fase5"
FASE7_EXPLORATORIA_DIR = DATA_EXPLORATORIO_DIR / "fase7"
FASE7_PREDICOES_EXPLORATORIA_DIR = FASE7_EXPLORATORIA_DIR / "predicoes"

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

PERFIL_EXECUCAO_EXPLORATORIO = "exploratorio"
PERFIL_EXECUCAO_OFICIAL = "oficial"
PERFIL_EXECUCAO_CLI_PADRAO = PERFIL_EXECUCAO_EXPLORATORIO
PERFIS_EXECUCAO = (
    PERFIL_EXECUCAO_EXPLORATORIO,
    PERFIL_EXECUCAO_OFICIAL,
)


def validar_perfil_execucao(perfil_execucao: str) -> str:
    """Valida o perfil de execução usado para separar testes e runs oficiais."""
    if perfil_execucao not in PERFIS_EXECUCAO:
        raise ValueError(
            f"`perfil_execucao` inválido: {perfil_execucao}. "
            f"Use um dentre {list(PERFIS_EXECUCAO)}."
        )
    return perfil_execucao


def resolver_fase5_dir(perfil_execucao: str) -> Path:
    """Resolve o diretório base da Fase 5 para o perfil informado."""
    perfil_execucao = validar_perfil_execucao(perfil_execucao)
    if perfil_execucao == PERFIL_EXECUCAO_OFICIAL:
        return FASE5_DIR
    return FASE5_EXPLORATORIA_DIR


def resolver_fase7_dir(perfil_execucao: str) -> Path:
    """Resolve o diretório base da Fase 7 para o perfil informado."""
    perfil_execucao = validar_perfil_execucao(perfil_execucao)
    if perfil_execucao == PERFIL_EXECUCAO_OFICIAL:
        return FASE7_DIR
    return FASE7_EXPLORATORIA_DIR


def resolver_artefatos_fase5(perfil_execucao: str) -> dict[str, Path]:
    """Retorna os caminhos canônicos da Fase 5 para um perfil."""
    fase5_dir = resolver_fase5_dir(perfil_execucao)
    return {
        "fase5_dir": fase5_dir,
        "gemini_manifest_path": fase5_dir / "gemini_sft_manifest.json",
        "gemini_modelo_path": fase5_dir / "modelo_gemini_nome.txt",
        "qwen_manifest_path": fase5_dir / "qwen_sft_manifest.json",
        "qwen_checkpoint_dir": fase5_dir / "modelo_qwen_checkpoint",
    }


def resolver_predicoes_fase7(perfil_execucao: str) -> dict[str, Path]:
    """Retorna os caminhos das predições da Fase 7 para um perfil."""
    fase7_dir = resolver_fase7_dir(perfil_execucao)
    predicoes_dir = fase7_dir / "predicoes"
    return {
        "gemini_ft": predicoes_dir / "gemini_ft.jsonl",
        "gemini_zero_shot": predicoes_dir / "gemini_zero_shot.jsonl",
        "qwen_ft": predicoes_dir / "qwen_ft.jsonl",
        "qwen_zero_shot": predicoes_dir / "qwen_zero_shot.jsonl",
    }


def resolver_manifestos_predicoes_fase7(perfil_execucao: str) -> dict[str, Path]:
    """Retorna os manifests das predições da Fase 7 para um perfil."""
    return {
        condicao_id: path.with_suffix(".manifest.json")
        for condicao_id, path in resolver_predicoes_fase7(perfil_execucao).items()
    }


def resolver_prefixo_gcs_fase5(perfil_execucao: str) -> str:
    """Retorna o prefixo GCS padrão para a Fase 5 conforme o perfil."""
    perfil_execucao = validar_perfil_execucao(perfil_execucao)
    if perfil_execucao == PERFIL_EXECUCAO_OFICIAL:
        return "dissertacao-ementas-tr/fase5"
    return "testes/fase5"


def resolver_artefatos_fase7(perfil_execucao: str) -> dict[str, Any]:
    """Retorna os caminhos canônicos da Fase 7 para um perfil."""
    fase7_dir = resolver_fase7_dir(perfil_execucao)
    predicao_paths = resolver_predicoes_fase7(perfil_execucao)
    predicao_manifest_paths = resolver_manifestos_predicoes_fase7(perfil_execucao)
    return {
        "fase7_dir": fase7_dir,
        "protocolo_path": fase7_dir / "protocolo_avaliacao.json",
        "casos_avaliacao_path": fase7_dir / "casos_avaliacao.jsonl",
        "predicao_paths": predicao_paths,
        "predicao_manifest_paths": predicao_manifest_paths,
        "metricas_automaticas_path": fase7_dir / "metricas_automaticas.csv",
        "avaliacao_judge_path": fase7_dir / "avaliacao_llm_judge.jsonl",
        "avaliacao_judge_bruta_path": fase7_dir / "avaliacao_llm_judge_bruta.jsonl",
        "avaliacao_judge_manifest_path": fase7_dir / "avaliacao_llm_judge_manifest.json",
        "amostra_humana_path": fase7_dir / "amostra_humana.json",
        "gabarito_cegamento_humano_path": fase7_dir / "gabarito_cegamento_humano.json",
        "avaliacao_humana_path": fase7_dir / "avaliacao_humana.csv",
        "relatorio_avaliacao_humana_path": fase7_dir / "relatorio_avaliacao_humana.json",
        "relatorio_estatistico_path": fase7_dir / "relatorio_estatistico.json",
    }
