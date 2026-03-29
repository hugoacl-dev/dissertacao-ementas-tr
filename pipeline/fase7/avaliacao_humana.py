"""
avaliacao_humana.py — Amostragem cega e análise da avaliação humana da Fase 7

Gera a amostra estratificada de 40 casos, monta o instrumento cego para os
avaliadores e, após o preenchimento do CSV, consolida médias e weighted
Cohen's kappa quadrático por critério.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipeline.core.artefato_utils import escrever_csv_atomico, escrever_json_atomico
from pipeline.core.project_paths import (
    FASE7_AMOSTRA_HUMANA_PATH,
    FASE7_AVALIACAO_HUMANA_PATH,
    FASE7_CASOS_AVALIACAO_PATH,
    FASE7_GABARITO_CEGAMENTO_HUMANO_PATH,
    PERFIL_EXECUCAO_CLI_PADRAO,
    PERFIS_EXECUCAO,
    FASE7_PREDICAO_PATHS,
    FASE7_RELATORIO_AVALIACAO_HUMANA_PATH,
    resolver_artefatos_fase7,
    resolver_predicoes_fase7,
)

from .metricas import carregar_casos_avaliacao, carregar_todas_predicoes, consolidar_casos_e_predicoes
from .protocolo import (
    AVALIADORES_AVALIACAO_HUMANA,
    CASOS_AVALIACAO_HUMANA,
    CASOS_POR_ESTRATO_AVALIACAO_HUMANA,
    CRITERIOS_AVALIACAO_HUMANA,
    SEED_AMOSTRAGEM_AVALIACAO_HUMANA,
    SEED_CEGAMENTO_AVALIACAO_HUMANA,
    VERSAO_PROTOCOLO_FASE7,
)

log = logging.getLogger(__name__)

ROTULOS_CEGOS = ("A", "B", "C", "D")
COLUNAS_AVALIACAO_HUMANA = (
    "caso_id",
    "item_id",
    "rotulo_cego",
    "avaliador_id",
    "criterio",
    "nota",
    "observacoes",
)


def _contar_palavras(texto: str) -> int:
    return len(texto.split())


def atribuir_estratos_quartis(casos_df: pd.DataFrame) -> pd.DataFrame:
    """Estratifica por quartis do comprimento da fundamentação."""
    if len(casos_df) < 4:
        raise ValueError("A estratificação em quartis exige ao menos 4 casos.")

    tabela = casos_df.copy()
    tabela["comprimento_fundamentacao_palavras"] = tabela["fundamentacao"].map(_contar_palavras)
    tabela = tabela.sort_values(
        ["comprimento_fundamentacao_palavras", "caso_id"],
    ).reset_index(drop=True)
    tabela["estrato_quartil"] = (tabela.index * 4 // len(tabela)) + 1
    tabela.loc[tabela["estrato_quartil"] > 4, "estrato_quartil"] = 4
    return tabela


def selecionar_casos_amostra(
    casos_df: pd.DataFrame,
    *,
    seed: int = SEED_AMOSTRAGEM_AVALIACAO_HUMANA,
    casos_por_estrato: int = CASOS_POR_ESTRATO_AVALIACAO_HUMANA,
) -> pd.DataFrame:
    """Seleciona casos sem reposição, balanceados por estrato."""
    if casos_por_estrato * 4 != CASOS_AVALIACAO_HUMANA:
        raise ValueError(
            f"O protocolo exige {CASOS_AVALIACAO_HUMANA} casos no total "
            f"({CASOS_POR_ESTRATO_AVALIACAO_HUMANA} por estrato)."
        )
    estratificados = atribuir_estratos_quartis(casos_df)
    rng = np.random.default_rng(seed)
    selecionados: list[pd.DataFrame] = []

    for estrato in range(1, 5):
        bloco = estratificados[estratificados["estrato_quartil"] == estrato].copy()
        if len(bloco) < casos_por_estrato:
            raise ValueError(
                f"Estrato {estrato} sem casos suficientes para amostrar {casos_por_estrato} exemplos."
            )
        indices_escolhidos = rng.choice(bloco.index.to_numpy(), size=casos_por_estrato, replace=False)
        selecionados.append(bloco.loc[indices_escolhidos].copy())

    amostra = pd.concat(selecionados, ignore_index=True)
    return amostra.sort_values(["estrato_quartil", "indice_teste", "caso_id"]).reset_index(drop=True)


def montar_amostra_humana(
    *,
    casos_path: Path = FASE7_CASOS_AVALIACAO_PATH,
    predicao_paths: dict[str, Path] = FASE7_PREDICAO_PATHS,
    seed_amostragem: int = SEED_AMOSTRAGEM_AVALIACAO_HUMANA,
    seed_cegamento: int = SEED_CEGAMENTO_AVALIACAO_HUMANA,
    casos_por_estrato: int = CASOS_POR_ESTRATO_AVALIACAO_HUMANA,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Monta a amostra cega e o gabarito restrito da avaliação humana."""
    casos_df = carregar_casos_avaliacao(casos_path)
    predicoes_df = carregar_todas_predicoes(predicao_paths)
    consolidado = consolidar_casos_e_predicoes(casos_df, predicoes_df)
    amostra_df = selecionar_casos_amostra(
        casos_df,
        seed=seed_amostragem,
        casos_por_estrato=casos_por_estrato,
    )

    rng_cegamento = np.random.default_rng(seed_cegamento)
    casos_payload: list[dict[str, Any]] = []
    gabarito_cegamento: list[dict[str, Any]] = []

    for _, caso in amostra_df.iterrows():
        observacoes = consolidado[consolidado["caso_id"] == caso["caso_id"]].copy()
        if len(observacoes) != len(ROTULOS_CEGOS):
            raise ValueError(
                f"O caso {caso['caso_id']} não possui as quatro condições experimentais necessárias."
            )
        ordem = rng_cegamento.permutation(len(ROTULOS_CEGOS))
        itens_avaliacao: list[dict[str, Any]] = []
        for posicao, indice_observacao in enumerate(ordem):
            rotulo = ROTULOS_CEGOS[posicao]
            observacao = observacoes.iloc[int(indice_observacao)]
            item_id = f"{caso['caso_id']}_{rotulo}"
            itens_avaliacao.append(
                {
                    "item_id": item_id,
                    "rotulo_cego": rotulo,
                    "ementa_gerada": observacao["ementa_gerada"],
                }
            )
            gabarito_cegamento.append(
                {
                    "item_id": item_id,
                    "caso_id": caso["caso_id"],
                    "rotulo_cego": rotulo,
                    "condicao_id_real": observacao["condicao_id"],
                }
            )

        casos_payload.append(
            {
                "caso_id": caso["caso_id"],
                "indice_teste": int(caso["indice_teste"]),
                "estrato_quartil": int(caso["estrato_quartil"]),
                "comprimento_fundamentacao_palavras": int(caso["comprimento_fundamentacao_palavras"]),
                "fundamentacao": caso["fundamentacao"],
                "itens_avaliacao": itens_avaliacao,
            }
        )

    amostra_cega = {
        "versao_protocolo": VERSAO_PROTOCOLO_FASE7,
        "seed_amostragem": seed_amostragem,
        "seed_cegamento": seed_cegamento,
        "casos_por_estrato": casos_por_estrato,
        "total_casos": len(casos_payload),
        "total_itens": len(gabarito_cegamento),
        "casos": casos_payload,
    }
    gabarito = {
        "versao_protocolo": VERSAO_PROTOCOLO_FASE7,
        "seed_cegamento": seed_cegamento,
        "total_itens": len(gabarito_cegamento),
        "itens": gabarito_cegamento,
    }
    return amostra_cega, gabarito


def montar_template_avaliacao_humana(amostra: dict[str, Any]) -> pd.DataFrame:
    """Cria o CSV vazio a ser preenchido pelos dois avaliadores."""
    linhas: list[dict[str, Any]] = []
    for caso in amostra["casos"]:
        for item in caso["itens_avaliacao"]:
            for avaliador_idx in range(1, AVALIADORES_AVALIACAO_HUMANA + 1):
                for criterio in CRITERIOS_AVALIACAO_HUMANA:
                    linhas.append(
                        {
                            "caso_id": caso["caso_id"],
                            "item_id": item["item_id"],
                            "rotulo_cego": item["rotulo_cego"],
                            "avaliador_id": f"avaliador_{avaliador_idx}",
                            "criterio": criterio,
                            "nota": pd.NA,
                            "observacoes": "",
                        }
                    )
    return pd.DataFrame(linhas, columns=COLUNAS_AVALIACAO_HUMANA)


def preparar_avaliacao_humana(
    *,
    casos_path: Path = FASE7_CASOS_AVALIACAO_PATH,
    predicao_paths: dict[str, Path] = FASE7_PREDICAO_PATHS,
    amostra_path: Path = FASE7_AMOSTRA_HUMANA_PATH,
    gabarito_path: Path = FASE7_GABARITO_CEGAMENTO_HUMANO_PATH,
    template_path: Path = FASE7_AVALIACAO_HUMANA_PATH,
    seed_amostragem: int = SEED_AMOSTRAGEM_AVALIACAO_HUMANA,
    seed_cegamento: int = SEED_CEGAMENTO_AVALIACAO_HUMANA,
    casos_por_estrato: int = CASOS_POR_ESTRATO_AVALIACAO_HUMANA,
) -> tuple[Path, Path]:
    """Gera a amostra cega e o template CSV da avaliação humana."""
    amostra, gabarito = montar_amostra_humana(
        casos_path=casos_path,
        predicao_paths=predicao_paths,
        seed_amostragem=seed_amostragem,
        seed_cegamento=seed_cegamento,
        casos_por_estrato=casos_por_estrato,
    )
    template = montar_template_avaliacao_humana(amostra)
    escrever_json_atomico(amostra_path, amostra, indent=2)
    escrever_json_atomico(gabarito_path, gabarito, indent=2)
    escrever_csv_atomico(template_path, template, index=False)
    return amostra_path, template_path


def carregar_amostra_humana(path: Path = FASE7_AMOSTRA_HUMANA_PATH) -> dict[str, Any]:
    """Carrega e valida minimamente a amostra humana já preparada."""
    if not path.exists():
        raise FileNotFoundError(f"Amostra humana não encontrada: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "casos" not in payload:
        raise ValueError("A amostra humana está sem o bloco obrigatório `casos`.")
    if "gabarito_cegamento" in payload or "itens" in payload:
        raise ValueError(
            "A amostra humana não pode conter o gabarito de cegamento no mesmo artefato."
        )
    if len(payload["casos"]) != CASOS_AVALIACAO_HUMANA:
        raise ValueError(
            f"A amostra humana deve conter {CASOS_AVALIACAO_HUMANA} casos."
        )
    return payload


def carregar_gabarito_cegamento_humano(
    path: Path = FASE7_GABARITO_CEGAMENTO_HUMANO_PATH,
) -> pd.DataFrame:
    """Carrega o gabarito restrito de correspondência entre rótulos e condições."""
    if not path.exists():
        raise FileNotFoundError(f"Gabarito de cegamento não encontrado: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    itens = payload.get("itens")
    if not isinstance(itens, list) or not itens:
        raise ValueError("O gabarito de cegamento está vazio ou malformado.")
    tabela = pd.DataFrame(itens)
    faltantes = [coluna for coluna in ("item_id", "caso_id", "rotulo_cego", "condicao_id_real") if coluna not in tabela.columns]
    if faltantes:
        raise ValueError(f"O gabarito de cegamento está sem colunas obrigatórias: {faltantes}")
    if tabela["item_id"].duplicated().any():
        raise ValueError("O gabarito de cegamento contém `item_id` duplicado.")
    return tabela.loc[:, ["item_id", "caso_id", "rotulo_cego", "condicao_id_real"]].copy()


def _linhas_esperadas_avaliacao_humana(amostra: dict[str, Any]) -> set[tuple[str, str, str, str, str]]:
    """Expande a grade esperada caso-item-avaliador-critério."""
    esperadas: set[tuple[str, str, str, str, str]] = set()
    for caso in amostra["casos"]:
        for item in caso["itens_avaliacao"]:
            for avaliador_idx in range(1, AVALIADORES_AVALIACAO_HUMANA + 1):
                for criterio in CRITERIOS_AVALIACAO_HUMANA:
                    esperadas.add(
                        (
                            str(caso["caso_id"]),
                            str(item["item_id"]),
                            str(item["rotulo_cego"]),
                            f"avaliador_{avaliador_idx}",
                            criterio,
                        )
                    )
    return esperadas


def carregar_avaliacao_humana(
    path: Path = FASE7_AVALIACAO_HUMANA_PATH,
    *,
    amostra: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Carrega e valida a planilha preenchida pelos avaliadores."""
    if not path.exists():
        raise FileNotFoundError(f"Avaliação humana não encontrada: {path}")
    df = pd.read_csv(path)
    faltantes = [coluna for coluna in COLUNAS_AVALIACAO_HUMANA if coluna not in df.columns]
    if faltantes:
        raise ValueError(f"A avaliação humana está sem colunas obrigatórias: {faltantes}")
    if df.empty:
        raise ValueError("A avaliação humana está vazia.")

    tabela = df.loc[:, COLUNAS_AVALIACAO_HUMANA].copy()
    tabela["caso_id"] = tabela["caso_id"].astype(str)
    tabela["item_id"] = tabela["item_id"].astype(str)
    tabela["rotulo_cego"] = tabela["rotulo_cego"].astype(str)
    tabela["avaliador_id"] = tabela["avaliador_id"].astype(str)
    tabela["criterio"] = tabela["criterio"].astype(str)
    tabela["observacoes"] = tabela["observacoes"].fillna("").astype(str)
    tabela["nota"] = pd.to_numeric(tabela["nota"], errors="raise").astype(int)

    if set(tabela["criterio"]) - set(CRITERIOS_AVALIACAO_HUMANA):
        raise ValueError("A avaliação humana contém critérios inválidos.")
    if set(tabela["nota"]) - {1, 2, 3, 4, 5}:
        raise ValueError("A avaliação humana contém notas fora da escala Likert 1–5.")
    if tabela.duplicated(subset=["item_id", "avaliador_id", "criterio"]).any():
        raise ValueError("A avaliação humana contém linhas duplicadas por item-avaliador-critério.")
    if len(set(tabela["avaliador_id"])) != AVALIADORES_AVALIACAO_HUMANA:
        raise ValueError(
            f"A avaliação humana deve conter exatamente {AVALIADORES_AVALIACAO_HUMANA} avaliadores."
        )
    if amostra is not None:
        linhas_esperadas = _linhas_esperadas_avaliacao_humana(amostra)
        linhas_recebidas = {
            (
                str(row["caso_id"]),
                str(row["item_id"]),
                str(row["rotulo_cego"]),
                str(row["avaliador_id"]),
                str(row["criterio"]),
            )
            for _, row in tabela.iterrows()
        }
        faltantes = sorted(linhas_esperadas - linhas_recebidas)
        extras = sorted(linhas_recebidas - linhas_esperadas)
        if faltantes:
            raise ValueError(
                "A avaliação humana está incompleta. "
                f"Exemplos faltantes: {faltantes[:5]}"
            )
        if extras:
            raise ValueError(
                "A avaliação humana contém linhas fora da grade experimental esperada. "
                f"Exemplos extras: {extras[:5]}"
            )
    return tabela


def cohen_kappa_ponderado_quadratico(
    notas_a: list[int],
    notas_b: list[int],
) -> float:
    """Calcula weighted Cohen's kappa quadrático para escala 1–5."""
    if len(notas_a) != len(notas_b) or not notas_a:
        raise ValueError("As listas de notas devem ter o mesmo tamanho e não podem ser vazias.")

    categorias = np.array([1, 2, 3, 4, 5], dtype=int)
    matriz_observada = np.zeros((len(categorias), len(categorias)), dtype=float)
    for nota_a, nota_b in zip(notas_a, notas_b):
        matriz_observada[nota_a - 1, nota_b - 1] += 1.0
    matriz_observada /= matriz_observada.sum()

    marginais_a = matriz_observada.sum(axis=1)
    marginais_b = matriz_observada.sum(axis=0)
    matriz_esperada = np.outer(marginais_a, marginais_b)

    diferencas = np.subtract.outer(categorias - 1, categorias - 1)
    pesos = (diferencas ** 2) / ((len(categorias) - 1) ** 2)

    discordancia_observada = float(np.sum(pesos * matriz_observada))
    discordancia_esperada = float(np.sum(pesos * matriz_esperada))
    if discordancia_esperada == 0:
        return 1.0 if discordancia_observada == 0 else 0.0
    return float(1.0 - (discordancia_observada / discordancia_esperada))


def gerar_relatorio_avaliacao_humana(
    amostra: dict[str, Any],
    gabarito_df: pd.DataFrame,
    avaliacoes_df: pd.DataFrame,
) -> dict[str, Any]:
    """Consolida concordância e médias descritivas da avaliação humana."""
    tabela = avaliacoes_df.merge(
        gabarito_df,
        on=["item_id", "caso_id", "rotulo_cego"],
        how="left",
        validate="many_to_one",
    )
    if tabela["condicao_id_real"].isnull().any():
        raise ValueError("A avaliação humana contém itens não presentes no gabarito cego.")

    avaliadores = sorted(tabela["avaliador_id"].unique())
    kappas: dict[str, float] = {}
    for criterio in CRITERIOS_AVALIACAO_HUMANA:
        bloco = tabela[tabela["criterio"] == criterio].copy()
        pivot = bloco.pivot(
            index="item_id",
            columns="avaliador_id",
            values="nota",
        )
        if set(pivot.columns) != set(avaliadores):
            raise ValueError(
                f"O critério '{criterio}' não possui notas completas para ambos os avaliadores."
            )
        kappas[criterio] = cohen_kappa_ponderado_quadratico(
            pivot[avaliadores[0]].astype(int).tolist(),
            pivot[avaliadores[1]].astype(int).tolist(),
        )

    medias_por_condicao = (
        tabela.groupby("condicao_id_real", as_index=False)["nota"]
        .mean()
        .rename(columns={"condicao_id_real": "condicao_id", "nota": "media_nota"})
        .sort_values("condicao_id")
        .to_dict(orient="records")
    )
    medias_por_condicao_criterio = (
        tabela.groupby(["condicao_id_real", "criterio"], as_index=False)["nota"]
        .mean()
        .rename(columns={"condicao_id_real": "condicao_id", "nota": "media_nota"})
        .sort_values(["condicao_id", "criterio"])
        .to_dict(orient="records")
    )

    return {
        "versao_protocolo": amostra["versao_protocolo"],
        "n_casos": len(amostra["casos"]),
        "n_itens": int(len(gabarito_df)),
        "n_registros_avaliacao": int(len(tabela)),
        "avaliadores": avaliadores,
        "kappa_quadratico_por_criterio": kappas,
        "medias_por_condicao": medias_por_condicao,
        "medias_por_condicao_criterio": medias_por_condicao_criterio,
    }


def escrever_relatorio_avaliacao_humana(
    *,
    amostra_path: Path = FASE7_AMOSTRA_HUMANA_PATH,
    gabarito_path: Path = FASE7_GABARITO_CEGAMENTO_HUMANO_PATH,
    avaliacao_path: Path = FASE7_AVALIACAO_HUMANA_PATH,
    output_path: Path = FASE7_RELATORIO_AVALIACAO_HUMANA_PATH,
) -> Path:
    """Lê a planilha preenchida e gera o relatório consolidado."""
    amostra = carregar_amostra_humana(amostra_path)
    gabarito_df = carregar_gabarito_cegamento_humano(gabarito_path)
    avaliacoes_df = carregar_avaliacao_humana(avaliacao_path, amostra=amostra)
    relatorio = gerar_relatorio_avaliacao_humana(amostra, gabarito_df, avaliacoes_df)
    escrever_json_atomico(output_path, relatorio, indent=2)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infraestrutura da avaliação humana da Fase 7.")
    parser.add_argument(
        "--perfil-execucao",
        choices=PERFIS_EXECUCAO,
        default=PERFIL_EXECUCAO_CLI_PADRAO,
    )
    parser.add_argument("--modo", choices=["preparar", "analisar"], default="preparar")
    parser.add_argument("--casos-path", type=Path, default=None)
    parser.add_argument("--amostra-path", type=Path, default=None)
    parser.add_argument("--gabarito-path", type=Path, default=None)
    parser.add_argument("--avaliacao-path", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--seed-amostragem", type=int, default=SEED_AMOSTRAGEM_AVALIACAO_HUMANA)
    parser.add_argument("--seed-cegamento", type=int, default=SEED_CEGAMENTO_AVALIACAO_HUMANA)
    parser.add_argument("--casos-por-estrato", type=int, default=CASOS_POR_ESTRATO_AVALIACAO_HUMANA)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = _parse_args()
    artefatos = resolver_artefatos_fase7(args.perfil_execucao)
    if args.modo == "preparar":
        amostra_path, template_path = preparar_avaliacao_humana(
            casos_path=args.casos_path or artefatos["casos_avaliacao_path"],
            predicao_paths=resolver_predicoes_fase7(args.perfil_execucao),
            amostra_path=args.amostra_path or artefatos["amostra_humana_path"],
            gabarito_path=args.gabarito_path or artefatos["gabarito_cegamento_humano_path"],
            template_path=args.avaliacao_path or artefatos["avaliacao_humana_path"],
            seed_amostragem=args.seed_amostragem,
            seed_cegamento=args.seed_cegamento,
            casos_por_estrato=args.casos_por_estrato,
        )
        log.info("Amostra humana persistida em %s", amostra_path)
        log.info("Template de avaliação humana persistido em %s", template_path)
        return

    output_path = escrever_relatorio_avaliacao_humana(
        amostra_path=args.amostra_path or artefatos["amostra_humana_path"],
        gabarito_path=args.gabarito_path or artefatos["gabarito_cegamento_humano_path"],
        avaliacao_path=args.avaliacao_path or artefatos["avaliacao_humana_path"],
        output_path=args.output_path or artefatos["relatorio_avaliacao_humana_path"],
    )
    log.info("Relatório da avaliação humana persistido em %s", output_path)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, OSError, ValueError, pd.errors.ParserError) as exc:
        log.critical("Execução interrompida: %s", exc)
        sys.exit(1)
