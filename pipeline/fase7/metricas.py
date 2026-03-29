"""
metricas.py — Consolidação de métricas da Fase 7

Lê os casos do conjunto de avaliação, as predições das quatro condições e as
avaliações do LLM-as-a-Judge, produzindo a tabela consolidada
`data/fase7/metricas_automaticas.csv`.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from artefato_utils import escrever_csv_atomico
from project_paths import (
    FASE7_AVALIACAO_JUDGE_PATH,
    FASE7_CASOS_AVALIACAO_PATH,
    FASE7_METRICAS_AUTOMATICAS_PATH,
    FASE7_PREDICAO_PATHS,
)

from .protocolo import (
    CONDICOES_EXPERIMENTAIS,
    DIMENSOES_JUIZ,
    calcular_score_global_llm_judge,
    validar_registro_caso_avaliacao,
    validar_registro_predicao,
    validar_resposta_llm_judge,
)

log = logging.getLogger(__name__)

COLUNAS_CASOS = ("caso_id", "indice_teste", "fundamentacao", "ementa_referencia")
COLUNAS_PREDICAO = ("caso_id", "condicao_id", "ementa_gerada")
COLUNAS_AVALIACAO_JUDGE = ("caso_id", "condicao_id", "avaliacao")


def _ler_jsonl(path: Path) -> list[dict[str, Any]]:
    """Lê um arquivo JSONL, ignorando linhas em branco."""
    if not path.exists():
        raise FileNotFoundError(f"Arquivo JSONL não encontrado: {path}")
    linhas: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for numero, linha in enumerate(f, start=1):
            conteudo = linha.strip()
            if not conteudo:
                continue
            try:
                linhas.append(json.loads(conteudo))
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON inválido em {path}:{numero}: {exc.msg}") from exc
    if not linhas:
        raise ValueError(f"Arquivo JSONL vazio: {path}")
    return linhas


def carregar_casos_avaliacao(path: Path = FASE7_CASOS_AVALIACAO_PATH) -> pd.DataFrame:
    """Carrega e valida os casos-base da Fase 7."""
    registros = _ler_jsonl(path)
    registros_normalizados = [validar_registro_caso_avaliacao(registro) for registro in registros]
    df = pd.DataFrame(registros_normalizados)
    faltantes = [coluna for coluna in COLUNAS_CASOS if coluna not in df.columns]
    if faltantes:
        raise ValueError(f"Casos de avaliação sem colunas obrigatórias: {faltantes}")
    tabela = df.loc[:, COLUNAS_CASOS].copy()
    if tabela["caso_id"].duplicated().any():
        raise ValueError("Casos de avaliação contêm `caso_id` duplicado.")
    return tabela.sort_values("caso_id").reset_index(drop=True)


def carregar_predicoes_condicao(path: Path, *, condicao_id: str) -> pd.DataFrame:
    """Carrega e valida as predições de uma condição experimental."""
    registros = _ler_jsonl(path)
    registros_normalizados = [
        validar_registro_predicao(registro, condicao_id_esperada=condicao_id)
        for registro in registros
    ]
    df = pd.DataFrame(registros_normalizados)
    faltantes = [coluna for coluna in COLUNAS_PREDICAO if coluna not in df.columns]
    if faltantes:
        raise ValueError(f"Predições da condição '{condicao_id}' sem colunas obrigatórias: {faltantes}")
    tabela = df.loc[:, COLUNAS_PREDICAO].copy()
    if tabela["caso_id"].duplicated().any():
        raise ValueError(f"Predições da condição '{condicao_id}' contêm `caso_id` duplicado.")
    return tabela.sort_values("caso_id").reset_index(drop=True)


def carregar_todas_predicoes(
    predicao_paths: dict[str, Path] = FASE7_PREDICAO_PATHS,
) -> pd.DataFrame:
    """Carrega todas as predições previstas no protocolo da Fase 7."""
    tabelas = [
        carregar_predicoes_condicao(path, condicao_id=condicao_id)
        for condicao_id, path in predicao_paths.items()
    ]
    return pd.concat(tabelas, ignore_index=True)


def carregar_avaliacoes_judge(path: Path = FASE7_AVALIACAO_JUDGE_PATH) -> pd.DataFrame:
    """Carrega e valida as saídas do LLM-as-a-Judge."""
    registros = _ler_jsonl(path)
    df = pd.DataFrame(registros)
    faltantes = [coluna for coluna in COLUNAS_AVALIACAO_JUDGE if coluna not in df.columns]
    if faltantes:
        raise ValueError(f"Avaliações do juiz sem colunas obrigatórias: {faltantes}")

    tabela = df.loc[:, COLUNAS_AVALIACAO_JUDGE].copy()
    if tabela.isnull().any().any():
        raise ValueError("Avaliações do juiz contêm valores nulos em colunas obrigatórias.")

    tabela["caso_id"] = tabela["caso_id"].astype(str)
    tabela["condicao_id"] = tabela["condicao_id"].astype(str)
    if tabela.duplicated(subset=["caso_id", "condicao_id"]).any():
        raise ValueError("Avaliações do juiz contêm duplicidade por caso e condição.")

    condicoes_validas = {item["id"] for item in CONDICOES_EXPERIMENTAIS}
    invalidas = sorted(set(tabela["condicao_id"]) - condicoes_validas)
    if invalidas:
        raise ValueError(f"Avaliações do juiz com condição inválida: {invalidas}")

    tabela["avaliacao"] = tabela["avaliacao"].apply(validar_resposta_llm_judge)
    return tabela.sort_values(["condicao_id", "caso_id"]).reset_index(drop=True)


def consolidar_casos_e_predicoes(
    casos_df: pd.DataFrame,
    predicoes_df: pd.DataFrame,
) -> pd.DataFrame:
    """Consolida casos-base com predições das quatro condições."""
    caso_ids = set(casos_df["caso_id"])
    resultado: list[pd.DataFrame] = []

    for condicao in CONDICOES_EXPERIMENTAIS:
        condicao_id = condicao["id"]
        tabela_condicao = predicoes_df[predicoes_df["condicao_id"] == condicao_id].copy()
        ids_condicao = set(tabela_condicao["caso_id"])
        faltantes = sorted(caso_ids - ids_condicao)
        extras = sorted(ids_condicao - caso_ids)
        if faltantes:
            raise ValueError(
                f"A condição '{condicao_id}' não contém todos os casos esperados. "
                f"Exemplos faltantes: {faltantes[:5]}"
            )
        if extras:
            raise ValueError(
                f"A condição '{condicao_id}' contém casos inexistentes na base. "
                f"Exemplos extras: {extras[:5]}"
            )

        merged = casos_df.merge(tabela_condicao, on="caso_id", how="inner")
        resultado.append(merged)

    return pd.concat(resultado, ignore_index=True)


def _calcular_metricas_rouge_por_par(
    ementa_referencia: str,
    ementa_gerada: str,
) -> dict[str, float]:
    """Calcula ROUGE-1/2/L F1 para um par referência-candidata."""
    try:
        from rouge_score import rouge_scorer
    except ImportError as exc:
        raise ImportError(
            "Dependência ausente: instale `rouge-score` no ambiente das fases avançadas."
        ) from exc

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    scores = scorer.score(ementa_referencia, ementa_gerada)
    return {
        "rouge_1_f1": float(scores["rouge1"].fmeasure),
        "rouge_2_f1": float(scores["rouge2"].fmeasure),
        "rouge_l_f1": float(scores["rougeL"].fmeasure),
    }


def _calcular_bertscore_f1_lote(
    ementas_geradas: list[str],
    ementas_referencia: list[str],
    *,
    model_type: str = "xlm-roberta-large",
    batch_size: int = 32,
) -> list[float]:
    """Calcula BERTScore F1 em lote."""
    try:
        from bert_score import score as bert_score
    except ImportError as exc:
        raise ImportError(
            "Dependência ausente: instale `bert-score` e `transformers` no ambiente das fases avançadas."
        ) from exc

    _, _, f1 = bert_score(
        ementas_geradas,
        ementas_referencia,
        model_type=model_type,
        lang="pt",
        rescale_with_baseline=True,
        batch_size=batch_size,
        verbose=False,
    )
    return [float(valor) for valor in f1.tolist()]


def gerar_tabela_metricas_fase7(
    casos_df: pd.DataFrame,
    predicoes_df: pd.DataFrame,
    avaliacoes_judge_df: pd.DataFrame,
) -> pd.DataFrame:
    """Gera a tabela consolidada de métricas da Fase 7."""
    consolidado = consolidar_casos_e_predicoes(casos_df, predicoes_df)

    ids_esperados = set(zip(consolidado["caso_id"], consolidado["condicao_id"]))
    ids_judge = set(zip(avaliacoes_judge_df["caso_id"], avaliacoes_judge_df["condicao_id"]))
    faltantes = sorted(ids_esperados - ids_judge)
    extras = sorted(ids_judge - ids_esperados)
    if faltantes:
        raise ValueError(
            "Avaliações do LLM-as-a-Judge não cobrem todos os pares caso-condição. "
            f"Exemplos faltantes: {faltantes[:5]}"
        )
    if extras:
        raise ValueError(
            "Avaliações do LLM-as-a-Judge contêm pares caso-condição inexistentes. "
            f"Exemplos extras: {extras[:5]}"
        )

    rows: list[dict[str, Any]] = []
    bertscores = _calcular_bertscore_f1_lote(
        consolidado["ementa_gerada"].tolist(),
        consolidado["ementa_referencia"].tolist(),
    )

    avaliacao_map = {
        (row["caso_id"], row["condicao_id"]): row["avaliacao"]
        for _, row in avaliacoes_judge_df.iterrows()
    }

    for idx, row in consolidado.iterrows():
        met_rouge = _calcular_metricas_rouge_por_par(
            row["ementa_referencia"],
            row["ementa_gerada"],
        )
        for metrica, score in met_rouge.items():
            rows.append(
                {
                    "caso_id": row["caso_id"],
                    "condicao_id": row["condicao_id"],
                    "metrica": metrica,
                    "score": score,
                }
            )
        rows.append(
            {
                "caso_id": row["caso_id"],
                "condicao_id": row["condicao_id"],
                "metrica": "bertscore_f1",
                "score": bertscores[idx],
            }
        )

        avaliacao = avaliacao_map[(row["caso_id"], row["condicao_id"])]
        rows.append(
            {
                "caso_id": row["caso_id"],
                "condicao_id": row["condicao_id"],
                "metrica": "judge_score_global",
                "score": calcular_score_global_llm_judge(avaliacao),
            }
        )
        for dimensao in DIMENSOES_JUIZ:
            rows.append(
                {
                    "caso_id": row["caso_id"],
                    "condicao_id": row["condicao_id"],
                    "metrica": f"judge_{dimensao}",
                    "score": float(avaliacao[dimensao]["score"]),
                }
            )

    tabela = pd.DataFrame(rows)
    return tabela.sort_values(["caso_id", "condicao_id", "metrica"]).reset_index(drop=True)


def escrever_metricas_fase7(
    *,
    casos_path: Path = FASE7_CASOS_AVALIACAO_PATH,
    predicao_paths: dict[str, Path] = FASE7_PREDICAO_PATHS,
    avaliacao_judge_path: Path = FASE7_AVALIACAO_JUDGE_PATH,
    output_path: Path = FASE7_METRICAS_AUTOMATICAS_PATH,
) -> Path:
    """Gera e persiste a tabela consolidada de métricas da Fase 7."""
    casos_df = carregar_casos_avaliacao(casos_path)
    predicoes_df = carregar_todas_predicoes(predicao_paths)
    avaliacoes_judge_df = carregar_avaliacoes_judge(avaliacao_judge_path)
    tabela = gerar_tabela_metricas_fase7(casos_df, predicoes_df, avaliacoes_judge_df)
    escrever_csv_atomico(output_path, tabela, index=False)
    return output_path


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    output_path = escrever_metricas_fase7()
    log.info("Tabela de métricas da Fase 7 gerada em %s", output_path)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, OSError, ValueError, pd.errors.ParserError, ImportError) as exc:
        log.critical("Execução interrompida: %s", exc)
        sys.exit(1)

