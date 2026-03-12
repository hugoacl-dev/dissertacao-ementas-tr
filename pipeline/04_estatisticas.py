"""
04_estatisticas.py — Fase 4: Estatísticas Descritivas do Corpus

Produz uma análise quantitativa completa do dataset, fornecendo à banca
uma caracterização do corpus de trabalho.

Métricas calculadas:
  - Funil de attrition (brutos → limpos → finais)
  - Distribuição de comprimento (em palavras) das fundamentações e ementas
  - Razão de compressão (fundamentação / ementa)
  - Novel n-grams (grau de abstratividade do corpus)
  - Período temporal coberto pelos julgamentos
  - Totais de palavras para fine-tuning

Unidade de medida: palavras (split por espaço). Justificativa: é o padrão
em pesquisa de sumarização (ROUGE opera sobre palavras) e evita dependência
de chamadas à API para contagem de tokens de subword.

Entradas : data/dados_brutos.json, data/dados_limpos.json,
           data/dataset_treino.jsonl, data/dataset_teste.jsonl
Saídas   : data/estatisticas_corpus.json
Executar a partir da raiz do projeto: python3 pipeline/04_estatisticas.py
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import math
import statistics as stats

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

log = logging.getLogger(__name__)

BRUTOS_PATH = Path("data/dados_brutos.json")
LIMPOS_PATH = Path("data/dados_limpos.json")
TREINO_PATH = Path("data/dataset_treino.jsonl")
TESTE_PATH = Path("data/dataset_teste.jsonl")
OUTPUT_PATH = Path("data/estatisticas_corpus.json")

# Número total de registros lidos do dump (pré-filtro de nulos).
# Obtido do log da Fase 1, não pode ser recalculado a partir dos JSONs.
TOTAL_DUMP = 32_478


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------


def _contar_palavras(texto: str) -> int:
    """Conta palavras via split por espaço (proxy aceito em pesquisa de sumarização)."""
    return len(texto.split())


def _carregar_json(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _carregar_jsonl(path: Path) -> list[dict]:
    registros = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                registros.append(json.loads(line))
    return registros


def _extrair_texto_do_jsonl(obj: dict) -> tuple[str, str]:
    """Extrai fundamentação e ementa de um registro JSONL do Gemini.

    O turno 'user' contém a instrução de sistema + fundamentação.
    O turno 'model' contém a ementa.
    """
    fundamentacao = ""
    ementa = ""
    for content in obj.get("contents", []):
        role = content.get("role", "")
        text = content["parts"][0]["text"]
        if role == "user":
            # Remove o prefixo fixo da instrução de sistema
            marcador = "Gere a ementa para a seguinte fundamentação:\n"
            idx = text.find(marcador)
            if idx >= 0:
                fundamentacao = text[idx + len(marcador):]
            else:
                fundamentacao = text
        elif role == "model":
            ementa = text
    return fundamentacao, ementa


# ---------------------------------------------------------------------------
# Cálculos estatísticos (stdlib only, sem numpy)
# ---------------------------------------------------------------------------


def _percentil(valores_ordenados: list[int], p: float) -> float:
    """Calcula o percentil p (0–100) via interpolação linear."""
    n = len(valores_ordenados)
    k = (p / 100) * (n - 1)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(valores_ordenados[int(k)])
    return valores_ordenados[f] + (k - f) * (valores_ordenados[c] - valores_ordenados[f])


def _distribuicao(valores: list[int]) -> dict[str, Any]:
    """Calcula estatísticas descritivas sobre uma lista de inteiros."""
    ordenados = sorted(valores)
    n = len(ordenados)
    media = sum(ordenados) / n
    mediana = stats.median(ordenados)
    desvio = stats.pstdev(ordenados)  # population std (não amostral)

    return {
        "contagem": n,
        "media": round(media, 1),
        "mediana": round(mediana, 1),
        "desvio_padrao": round(desvio, 1),
        "min": ordenados[0],
        "max": ordenados[-1],
        "p5": round(_percentil(ordenados, 5), 1),
        "p25": round(_percentil(ordenados, 25), 1),
        "p75": round(_percentil(ordenados, 75), 1),
        "p95": round(_percentil(ordenados, 95), 1),
        "total_palavras": sum(ordenados),
    }


def _ngrams(palavras: list[str], n: int) -> set[tuple[str, ...]]:
    """Gera o conjunto de n-grams a partir de uma lista de palavras."""
    return {tuple(palavras[i:i + n]) for i in range(len(palavras) - n + 1)}


def calcular_novel_ngrams(
    pares: list[tuple[str, str]],
) -> dict[str, dict[str, float]]:
    """Calcula a taxa de novel n-grams (uni/bi/tri) das ementas vs. fundamentações.

    Novel n-gram = n-gram presente na ementa que NÃO aparece na fundamentação.
    Taxa alta → corpus genuinamente abstrativo.
    Referência: See et al. (2017), 'Get To The Point'.

    Args:
        pares: Lista de tuplas (fundamentação, ementa) em texto limpo.

    Returns:
        Dicionário com taxas médias de novelty para uni/bi/trigramas.
    """
    novelty: dict[str, list[float]] = {"unigrams": [], "bigrams": [], "trigrams": []}

    for fund, ementa in pares:
        palavras_fund = fund.lower().split()
        palavras_ementa = ementa.lower().split()

        if len(palavras_ementa) < 2:
            continue

        for n, nome in [(1, "unigrams"), (2, "bigrams"), (3, "trigrams")]:
            if len(palavras_ementa) < n:
                continue
            ng_ementa = _ngrams(palavras_ementa, n)
            ng_fund = _ngrams(palavras_fund, n)
            novel = ng_ementa - ng_fund
            taxa = len(novel) / len(ng_ementa) * 100 if ng_ementa else 0.0
            novelty[nome].append(taxa)

    resultado = {}
    for nome, valores in novelty.items():
        if valores:
            resultado[nome] = {
                "media": round(sum(valores) / len(valores), 1),
                "mediana": round(stats.median(valores), 1),
                "desvio_padrao": round(stats.pstdev(valores), 1),
            }
    return resultado


def calcular_funil(brutos: list, limpos: list, treino: list, teste: list) -> dict:
    """Calcula o funil de attrition do pipeline."""
    return {
        "dump_postgresql": TOTAL_DUMP,
        "apos_filtro_nulos_fase1": len(brutos),
        "apos_limpeza_fase2": len(limpos),
        "dataset_final_fase3": len(treino) + len(teste),
        "treino": len(treino),
        "teste": len(teste),
        "perda_fase1": TOTAL_DUMP - len(brutos),
        "perda_fase2": len(brutos) - len(limpos),
        "perda_total": TOTAL_DUMP - (len(treino) + len(teste)),
        "taxa_retencao_global": round((len(treino) + len(teste)) / TOTAL_DUMP * 100, 1),
    }


def calcular_periodo_temporal(brutos: list[dict]) -> dict:
    """Extrai o período temporal do campo data_cadastro (se disponível nos dados brutos).

    Nota: dados_brutos.json não contém data_cadastro (foi excluído na exportação
    da Fase 1 por design). Se necessário, ler diretamente do SQLite.
    """
    # Tenta usar dados_brutos.json primeiro
    datas = [r.get("data_cadastro", "") for r in brutos if r.get("data_cadastro")]

    if not datas:
        # Fallback: ler do SQLite
        import sqlite3
        db_path = Path("data/banco_tr_one.sqlite")
        if db_path.exists():
            conn = sqlite3.connect(db_path)
            cursor = conn.execute(
                "SELECT data_cadastro FROM turmarecursal_processo "
                "WHERE data_cadastro IS NOT NULL ORDER BY data_cadastro"
            )
            datas = [row[0] for row in cursor.fetchall() if row[0]]
            conn.close()

    if not datas:
        return {"erro": "data_cadastro não disponível"}

    # Extrair apenas a parte de data (YYYY-MM-DD) e o ano
    datas_ordenadas = sorted(datas)
    anos = {}
    for d in datas_ordenadas:
        ano = d[:4]
        if ano.isdigit():
            anos[ano] = anos.get(ano, 0) + 1

    return {
        "data_mais_antiga": datas_ordenadas[0][:10] if datas_ordenadas else "—",
        "data_mais_recente": datas_ordenadas[-1][:10] if datas_ordenadas else "—",
        "distribuicao_por_ano": dict(sorted(anos.items())),
    }


# ---------------------------------------------------------------------------
# Relatório
# ---------------------------------------------------------------------------


def _imprimir_distribuicao(nome: str, dist: dict) -> None:
    """Imprime uma distribuição formatada no log."""
    log.info(
        "  %s: média=%.1f | mediana=%.1f | std=%.1f | min=%d | max=%d",
        nome, dist["media"], dist["mediana"], dist["desvio_padrao"], dist["min"], dist["max"],
    )
    log.info(
        "    Percentis: P5=%.1f | P25=%.1f | P75=%.1f | P95=%.1f",
        dist["p5"], dist["p25"], dist["p75"], dist["p95"],
    )
    log.info("    Total de palavras: %s", f"{dist['total_palavras']:,}")


def gerar_relatorio(
    brutos_path: Path = BRUTOS_PATH,
    limpos_path: Path = LIMPOS_PATH,
    treino_path: Path = TREINO_PATH,
    teste_path: Path = TESTE_PATH,
    output_path: Path = OUTPUT_PATH,
) -> dict:
    """Pipeline completo da Fase 4: calcula todas as estatísticas e grava JSON."""

    log.info("=== Fase 4: Estatísticas Descritivas do Corpus ===")

    # --- Carregar dados ---
    log.info("Carregando dados...")
    brutos = _carregar_json(brutos_path)
    limpos = _carregar_json(limpos_path)
    treino = _carregar_jsonl(treino_path)
    teste = _carregar_jsonl(teste_path)

    # --- Funil ---
    funil = calcular_funil(brutos, limpos, treino, teste)
    log.info("Funil de Attrition:")
    log.info("  Dump PostgreSQL:        %d", funil["dump_postgresql"])
    log.info("  Após filtro nulos (F1): %d (-%d)", funil["apos_filtro_nulos_fase1"], funil["perda_fase1"])
    log.info("  Após limpeza (F2):      %d (-%d)", funil["apos_limpeza_fase2"], funil["perda_fase2"])
    log.info("  Dataset final (F3):     %d", funil["dataset_final_fase3"])
    log.info("    Treino: %d | Teste: %d", funil["treino"], funil["teste"])
    log.info("  Taxa de retenção global: %.1f%%", funil["taxa_retencao_global"])

    # --- Extrair textos do JSONL final ---
    log.info("Extraindo textos do dataset final...")
    all_data = treino + teste
    fundamentos_palavras: list[int] = []
    ementas_palavras: list[int] = []
    razoes: list[float] = []
    pares_texto: list[tuple[str, str]] = []

    for obj in all_data:
        fund, ementa = _extrair_texto_do_jsonl(obj)
        n_fund = _contar_palavras(fund)
        n_ementa = _contar_palavras(ementa)
        fundamentos_palavras.append(n_fund)
        ementas_palavras.append(n_ementa)
        pares_texto.append((fund, ementa))
        if n_ementa > 0:
            razoes.append(n_fund / n_ementa)

    # --- Distribuições ---
    dist_fund = _distribuicao(fundamentos_palavras)
    dist_ementa = _distribuicao(ementas_palavras)

    log.info("Distribuição de Comprimento (palavras):")
    _imprimir_distribuicao("Fundamentação", dist_fund)
    _imprimir_distribuicao("Ementa", dist_ementa)

    # --- Razão de compressão ---
    razao_stats = {
        "media": round(sum(razoes) / len(razoes), 2),
        "mediana": round(stats.median(razoes), 2),
        "desvio_padrao": round(stats.pstdev(razoes), 2),
        "min": round(min(razoes), 2),
        "max": round(max(razoes), 2),
    }
    log.info(
        "Razão de Compressão (fund./ementa): média=%.2f | mediana=%.2f | std=%.2f",
        razao_stats["media"], razao_stats["mediana"], razao_stats["desvio_padrao"],
    )

    # --- Novel n-grams (abstratividade) ---
    log.info("Calculando novel n-grams (abstratividade)...")
    novelty = calcular_novel_ngrams(pares_texto)
    for nome, vals in novelty.items():
        log.info(
            "  Novel %s: média=%.1f%% | mediana=%.1f%% | std=%.1f%%",
            nome, vals["media"], vals["mediana"], vals["desvio_padrao"],
        )

    # --- Período temporal ---
    log.info("Analisando período temporal...")
    periodo = calcular_periodo_temporal(brutos)
    if "erro" not in periodo:
        log.info(
            "  Período: %s a %s",
            periodo["data_mais_antiga"], periodo["data_mais_recente"],
        )
        log.info("  Distribuição por ano: %s", periodo.get("distribuicao_por_ano", {}))
    else:
        log.warning("  %s", periodo["erro"])

    # --- Montar resultado final ---
    resultado = {
        "unidade_de_medida": "palavras (split por espaço)",
        "funil": funil,
        "fundamentacao": dist_fund,
        "ementa": dist_ementa,
        "razao_compressao": razao_stats,
        "novel_ngrams": novelty,
        "periodo_temporal": periodo,
    }

    # --- Gravar JSON ---
    log.info("Gravando resultados em %s ...", output_path)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)

    log.info("=== Fase 4 finalizada com sucesso. ===")
    return resultado


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    gerar_relatorio()


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, OSError) as exc:
        log.critical("Execução interrompida: %s", exc)
        sys.exit(1)
