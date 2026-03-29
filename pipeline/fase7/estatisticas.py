"""
estatisticas.py — Inferência estatística pareada da Fase 7

Consome a tabela consolidada de métricas por caso e condição experimental,
aplica bootstrap pareado, teste de permutação pareado e ajustes de
multiplicidade, gerando um relatório JSON reproduzível.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipeline.core.artefato_utils import escrever_json_atomico
from pipeline.core.project_paths import (
    FASE7_METRICAS_AUTOMATICAS_PATH,
    FASE7_PROTOCOLO_PATH,
    FASE7_RELATORIO_ESTATISTICO_PATH,
)

from .protocolo import CONDICOES_EXPERIMENTAIS, gerar_manifesto_fase7

log = logging.getLogger(__name__)

SEMENTE_INFERENCIA = 20260329
ALPHA_PADRAO = 0.05
COLUNAS_METRICAS_FASE7 = ("caso_id", "condicao_id", "metrica", "score")


def bootstrap_pareado(
    deltas: np.ndarray,
    *,
    iteracoes: int,
    seed: int = SEMENTE_INFERENCIA,
) -> tuple[float, float]:
    """Estima intervalo de confiança de 95% via bootstrap pareado."""
    if deltas.ndim != 1 or len(deltas) == 0:
        raise ValueError("Os deltas do bootstrap devem formar vetor unidimensional não vazio.")
    if iteracoes <= 0:
        raise ValueError("O número de iterações do bootstrap deve ser positivo.")

    rng = np.random.default_rng(seed)
    n = len(deltas)
    medias = np.empty(iteracoes, dtype=float)
    for i in range(iteracoes):
        indices = rng.integers(0, n, size=n)
        medias[i] = float(np.mean(deltas[indices]))
    return float(np.percentile(medias, 2.5)), float(np.percentile(medias, 97.5))


def calcular_pvalue_permutacao_pareada(
    scores_ft: np.ndarray,
    scores_zs: np.ndarray,
    *,
    iteracoes: int,
    seed: int = SEMENTE_INFERENCIA,
) -> float:
    """Calcula p-value bicaudal por permutação pareada via inversão de sinal."""
    if scores_ft.shape != scores_zs.shape:
        raise ValueError("As séries comparadas na permutação devem ter o mesmo formato.")
    if scores_ft.ndim != 1 or len(scores_ft) == 0:
        raise ValueError("As séries comparadas na permutação devem ser vetores não vazios.")
    if iteracoes <= 0:
        raise ValueError("O número de iterações da permutação deve ser positivo.")

    deltas = scores_ft - scores_zs
    observado = float(np.mean(deltas))
    rng = np.random.default_rng(seed)
    contagem = 0
    for _ in range(iteracoes):
        sinais = rng.choice(np.array([-1.0, 1.0]), size=len(deltas))
        permutado = float(np.mean(deltas * sinais))
        if abs(permutado) >= abs(observado):
            contagem += 1
    return float((contagem + 1) / (iteracoes + 1))


def ajustar_pvalues_holm(pvalues: list[float]) -> list[float]:
    """Aplica ajuste Holm-Bonferroni."""
    n = len(pvalues)
    if n == 0:
        return []

    ordem = sorted(range(n), key=pvalues.__getitem__)
    ajustados = [0.0] * n
    acumulado = 0.0
    for posicao, indice in enumerate(ordem):
        fator = n - posicao
        valor = min(1.0, pvalues[indice] * fator)
        acumulado = max(acumulado, valor)
        ajustados[indice] = acumulado
    return ajustados


def ajustar_pvalues_bh(pvalues: list[float]) -> list[float]:
    """Aplica ajuste Benjamini-Hochberg (FDR)."""
    n = len(pvalues)
    if n == 0:
        return []

    ordem = sorted(range(n), key=pvalues.__getitem__)
    ajustados = [0.0] * n
    minimo = 1.0
    for posicao_reversa, indice in enumerate(reversed(ordem), start=1):
        rank = n - posicao_reversa + 1
        valor = pvalues[indice] * n / rank
        minimo = min(minimo, valor)
        ajustados[indice] = min(1.0, minimo)
    return ajustados


def carregar_manifesto_fase7(path: Path = FASE7_PROTOCOLO_PATH) -> dict[str, Any]:
    """Carrega o manifesto da Fase 7 do disco ou o gera em memória."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return gerar_manifesto_fase7()


def carregar_metricas_fase7(path: Path = FASE7_METRICAS_AUTOMATICAS_PATH) -> pd.DataFrame:
    """Carrega e valida a tabela consolidada de métricas da Fase 7."""
    if not path.exists():
        raise FileNotFoundError(
            f"Tabela de métricas não encontrada em {path}. "
            "Gere primeiro o artefato consolidado da Fase 7."
        )

    df = pd.read_csv(path)
    return validar_tabela_metricas_fase7(df)


def validar_tabela_metricas_fase7(df: pd.DataFrame) -> pd.DataFrame:
    """Valida o schema mínimo da tabela de métricas da Fase 7."""
    faltantes = [coluna for coluna in COLUNAS_METRICAS_FASE7 if coluna not in df.columns]
    if faltantes:
        raise ValueError(f"Colunas obrigatórias ausentes em metricas_automaticas.csv: {faltantes}")

    if df.empty:
        raise ValueError("A tabela de métricas da Fase 7 está vazia.")

    tabela = df.loc[:, COLUNAS_METRICAS_FASE7].copy()
    if tabela.isnull().any().any():
        raise ValueError("A tabela de métricas contém valores nulos em colunas obrigatórias.")

    tabela["condicao_id"] = tabela["condicao_id"].astype(str)
    tabela["metrica"] = tabela["metrica"].astype(str)
    tabela["caso_id"] = tabela["caso_id"].astype(str)
    tabela["score"] = pd.to_numeric(tabela["score"], errors="raise")

    if not np.isfinite(tabela["score"]).all():
        raise ValueError("A tabela de métricas contém scores não finitos.")

    condicoes_validas = {item["id"] for item in CONDICOES_EXPERIMENTAIS}
    condicoes_invalidas = sorted(set(tabela["condicao_id"]) - condicoes_validas)
    if condicoes_invalidas:
        raise ValueError(f"Condições experimentais inválidas: {condicoes_invalidas}")

    duplicados = tabela.duplicated(subset=["caso_id", "condicao_id", "metrica"])
    if duplicados.any():
        raise ValueError(
            "A tabela de métricas contém linhas duplicadas para a mesma combinação "
            "caso-condição-métrica."
        )

    return tabela


def mapear_condicoes_por_familia() -> dict[str, dict[str, str]]:
    """Mapeia família de modelo para ids das condições FT e zero-shot."""
    familias: dict[str, dict[str, str]] = {}
    for item in CONDICOES_EXPERIMENTAIS:
        familias.setdefault(item["familia"], {})[item["condicao"]] = item["id"]
    return familias


def construir_pares_metricos(
    df: pd.DataFrame,
    *,
    familia: str,
    metrica: str,
) -> pd.DataFrame:
    """Constrói a tabela pareada por caso para uma família e métrica."""
    mapeamento = mapear_condicoes_por_familia()
    ids = mapeamento[familia]
    ft_id = ids["fine_tuned"]
    zs_id = ids["zero_shot"]

    subset = df[(df["condicao_id"].isin([ft_id, zs_id])) & (df["metrica"] == metrica)].copy()
    tabela = subset.pivot(index="caso_id", columns="condicao_id", values="score")
    faltantes_ft = tabela[ft_id].isna() if ft_id in tabela else pd.Series(dtype=bool)
    faltantes_zs = tabela[zs_id].isna() if zs_id in tabela else pd.Series(dtype=bool)
    if ft_id not in tabela.columns or zs_id not in tabela.columns:
        raise ValueError(
            f"A métrica '{metrica}' da família '{familia}' não possui ambas as condições "
            "fine_tuned e zero_shot."
        )
    if faltantes_ft.any() or faltantes_zs.any():
        raise ValueError(
            f"A métrica '{metrica}' da família '{familia}' não preserva pareamento "
            "completo por caso entre FT e zero-shot."
        )
    return tabela.sort_index()


def comparar_condicoes_pareadas(
    df: pd.DataFrame,
    *,
    familia: str,
    metrica: str,
    iteracoes_bootstrap: int,
    iteracoes_permutacao: int,
    seed: int = SEMENTE_INFERENCIA,
) -> dict[str, Any]:
    """Executa a comparação pareada FT vs zero-shot para uma métrica."""
    tabela = construir_pares_metricos(df, familia=familia, metrica=metrica)
    ft_id = mapear_condicoes_por_familia()[familia]["fine_tuned"]
    zs_id = mapear_condicoes_por_familia()[familia]["zero_shot"]

    ft_scores = tabela[ft_id].to_numpy(dtype=float)
    zs_scores = tabela[zs_id].to_numpy(dtype=float)
    deltas = ft_scores - zs_scores

    ic_inferior, ic_superior = bootstrap_pareado(
        deltas,
        iteracoes=iteracoes_bootstrap,
        seed=seed,
    )
    pvalue = calcular_pvalue_permutacao_pareada(
        ft_scores,
        zs_scores,
        iteracoes=iteracoes_permutacao,
        seed=seed,
    )

    return {
        "familia": familia,
        "metrica": metrica,
        "n_casos": int(len(tabela)),
        "media_fine_tuned": float(np.mean(ft_scores)),
        "media_zero_shot": float(np.mean(zs_scores)),
        "delta_medio": float(np.mean(deltas)),
        "ic95_delta": [ic_inferior, ic_superior],
        "p_value_bruto": pvalue,
    }


def aplicar_ajustes_multiplicidade(
    comparacoes: list[dict[str, Any]],
    *,
    alpha: float = ALPHA_PADRAO,
) -> list[dict[str, Any]]:
    """Aplica ajustes primários e secundários por família."""
    resultado: list[dict[str, Any]] = []
    familias = sorted({item["familia"] for item in comparacoes})
    for familia in familias:
        bloco = [item.copy() for item in comparacoes if item["familia"] == familia]
        primarias = [item for item in bloco if item["escopo"] == "primario"]
        secundarias = [item for item in bloco if item["escopo"] == "secundario"]

        pvals_primarios = [item["p_value_bruto"] for item in primarias]
        pvals_secundarios = [item["p_value_bruto"] for item in secundarias]

        ajustados_primarios = ajustar_pvalues_holm(pvals_primarios)
        ajustados_secundarios = ajustar_pvalues_bh(pvals_secundarios)

        for item, ajustado in zip(primarias, ajustados_primarios):
            item["metodo_ajuste"] = "holm_bonferroni"
            item["p_value_ajustado"] = ajustado
            item["significativo_ajustado"] = ajustado < alpha
            resultado.append(item)
        for item, ajustado in zip(secundarias, ajustados_secundarios):
            item["metodo_ajuste"] = "benjamini_hochberg"
            item["p_value_ajustado"] = ajustado
            item["significativo_ajustado"] = ajustado < alpha
            resultado.append(item)
    return sorted(resultado, key=lambda item: (item["familia"], item["metrica"]))


def _resumir_consistencia_entre_familias(
    comparacoes: list[dict[str, Any]],
    *,
    metricas_primarias: list[str],
) -> dict[str, Any]:
    """Resume, de forma exploratória, a consistência entre Gemini e Qwen."""
    resumo: dict[str, Any] = {}
    for metrica in metricas_primarias:
        itens = [
            item
            for item in comparacoes
            if item["escopo"] == "primario" and item["metrica"] == metrica
        ]
        if not itens:
            continue
        sinais_nao_nulos = {
            1 if item["delta_medio"] > 0 else -1
            for item in itens
            if item["delta_medio"] != 0
        }
        resumo[metrica] = {
            "direcao_consistente": len(sinais_nao_nulos) <= 1,
            "familias": {
                item["familia"]: {
                    "delta_medio": item["delta_medio"],
                    "p_value_ajustado": item["p_value_ajustado"],
                    "significativo_ajustado": item["significativo_ajustado"],
                }
                for item in itens
            },
        }
    return resumo


def gerar_relatorio_estatistico(
    df: pd.DataFrame,
    manifesto: dict[str, Any],
    *,
    metricas_path: Path = FASE7_METRICAS_AUTOMATICAS_PATH,
    manifesto_path: Path = FASE7_PROTOCOLO_PATH,
) -> dict[str, Any]:
    """Gera o relatório estatístico consolidado da Fase 7."""
    tabela = validar_tabela_metricas_fase7(df)
    familias = mapear_condicoes_por_familia()
    metricas_disponiveis = sorted(set(tabela["metrica"]))
    metricas_primarias = list(manifesto["co_desfechos_primarios"])

    comparacoes: list[dict[str, Any]] = []
    for familia in sorted(familias):
        faltantes_primarias = [
            metrica
            for metrica in metricas_primarias
            if tabela[
                (tabela["condicao_id"].isin(familias[familia].values()))
                & (tabela["metrica"] == metrica)
            ].empty
        ]
        if faltantes_primarias:
            raise ValueError(
                f"A família '{familia}' não contém todas as métricas primárias: "
                f"{faltantes_primarias}"
            )

        metricas_familia = sorted(
            {
                metrica
                for metrica in metricas_disponiveis
                if not tabela[
                    (tabela["condicao_id"].isin(familias[familia].values()))
                    & (tabela["metrica"] == metrica)
                ].empty
            }
        )
        for metrica in metricas_familia:
            comparacao = comparar_condicoes_pareadas(
                tabela,
                familia=familia,
                metrica=metrica,
                iteracoes_bootstrap=manifesto["inferencia"]["bootstrap_pareado_iteracoes"],
                iteracoes_permutacao=manifesto["inferencia"]["permutacao_pareada_iteracoes"],
            )
            comparacao["escopo"] = (
                "primario" if metrica in metricas_primarias else "secundario"
            )
            comparacoes.append(comparacao)

    comparacoes_ajustadas = aplicar_ajustes_multiplicidade(comparacoes)

    resumo_familias: dict[str, Any] = {}
    for familia in sorted(familias):
        primarias = [
            item
            for item in comparacoes_ajustadas
            if item["familia"] == familia and item["escopo"] == "primario"
        ]
        sucesso = all(
            item["delta_medio"] > 0 and item["significativo_ajustado"] for item in primarias
        )
        resumo_familias[familia] = {
            "sucesso_confirmatorio": sucesso,
            "metricas_primarias": [item["metrica"] for item in primarias],
        }

    consistencia_entre_familias = _resumir_consistencia_entre_familias(
        comparacoes_ajustadas,
        metricas_primarias=metricas_primarias,
    )

    return {
        "versao_protocolo": manifesto["versao_protocolo"],
        "semente_inferencia": SEMENTE_INFERENCIA,
        "alpha": ALPHA_PADRAO,
        "arquivo_metricas": str(metricas_path),
        "arquivo_manifesto": str(manifesto_path),
        "comparacoes": comparacoes_ajustadas,
        "resumo_familias": resumo_familias,
        "consistencia_entre_familias": consistencia_entre_familias,
    }


def escrever_relatorio_estatistico(
    metricas_path: Path = FASE7_METRICAS_AUTOMATICAS_PATH,
    output_path: Path = FASE7_RELATORIO_ESTATISTICO_PATH,
) -> Path:
    """Carrega métricas, gera o relatório e persiste o artefato final."""
    manifesto = carregar_manifesto_fase7()
    tabela = carregar_metricas_fase7(metricas_path)
    relatorio = gerar_relatorio_estatistico(
        tabela,
        manifesto,
        metricas_path=metricas_path,
        manifesto_path=FASE7_PROTOCOLO_PATH,
    )
    escrever_json_atomico(output_path, relatorio, indent=2)
    return output_path


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    output_path = escrever_relatorio_estatistico()
    log.info("Relatório estatístico da Fase 7 gerado em %s", output_path)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, OSError, ValueError, pd.errors.ParserError) as exc:
        log.critical("Execução interrompida: %s", exc)
        sys.exit(1)
