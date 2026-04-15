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
Executar a partir da raiz do projeto: python3 -m pipeline.fase1_4.fase04_estatisticas
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
from pipeline.core.artefato_utils import escrever_json_atomico
from pipeline.core.data_cadastro_utils import validar_e_converter_data_cadastro
from pipeline.core.jsonl_utils import extrair_fundamentacao_e_ementa
from pipeline.core.project_paths import (
    ANONIMIZACAO_STATS_PATH as PII_STATS_PATH,
    DADOS_BRUTOS_PATH as BRUTOS_PATH,
    DADOS_LIMPOS_PATH as LIMPOS_PATH,
    DATASET_TESTE_PATH as TESTE_PATH,
    DATASET_TREINO_PATH as TREINO_PATH,
    DOCS_DATA_DIR as DOCS_DATA_PATH,
    DUMP_PATH,
    ESTATISTICAS_PATH as OUTPUT_PATH,
    INGESTAO_STATS_PATH,
    PIPELINE_TIMING_PATH as TIMING_PATH,
    SQLITE_DB_PATH,
    SYSTEM_PROMPT_PATH,
)

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

log = logging.getLogger(__name__)

# Número total de registros lidos do dump (pré-filtro de nulos).
# Fallback para execuções legadas em que a Fase 1 ainda não persiste stats.
TOTAL_DUMP_FALLBACK = 32_478

# Percentis padrão para distribuições de comprimento
_PERCENTIS = [0.05, 0.25, 0.75, 0.95]

_PREFIXO_INPUT_CONTAMINADO = "VOTO-EMENTA"
_EMENTA_TRUNCADA_EXATA = "AMPARO ASSISTENCIAL. SENTENÇA DE IMPROCEDÊNCIA. RECORRE A PARTE-AUTORA"
_HEADER_INSTITUCIONAL_PREFIXO = "JUSTIÇA FEDERAL DA 5ª REGIÃO"
_QWEN_MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
_QWEN_MAX_INPUT_TOKENS = 8192
_GEMINI_MODEL_ID = "gemini-2.5-flash"


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def _carregar_json(path: Path) -> pd.DataFrame:
    """Lê um arquivo JSON (array de objetos) em um DataFrame."""
    return pd.read_json(path, orient="records", dtype=False)


def _carregar_jsonl(path: Path) -> list[dict]:
    """Lê um arquivo JSONL em lista de dicionários (formato Gemini)."""
    registros = []
    with path.open("r", encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if linha:
                registros.append(json.loads(linha))
    return registros


def _carregar_stats_ingestao(path: Path) -> dict[str, int]:
    """Lê as estatísticas estruturadas da Fase 1, se disponíveis."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    campos_obrigatorios = ("total_lidos", "descartados_nulos", "exportados")
    faltantes = [campo for campo in campos_obrigatorios if campo not in payload]
    if faltantes:
        raise ValueError(
            f"{path} não contém os campos obrigatórios: {', '.join(faltantes)}"
        )

    stats = {campo: int(payload[campo]) for campo in campos_obrigatorios}
    if stats["total_lidos"] <= 0:
        raise ValueError("total_lidos deve ser maior que zero.")
    if stats["descartados_nulos"] < 0 or stats["exportados"] < 0:
        raise ValueError("descartados_nulos e exportados não podem ser negativos.")
    if stats["exportados"] + stats["descartados_nulos"] != stats["total_lidos"]:
        raise ValueError(
            "As estatísticas de ingestão são inconsistentes: "
            "exportados + descartados_nulos deve ser igual a total_lidos."
        )
    return stats


def _file_size_mb(path: Path) -> float | None:
    """Retorna o tamanho do arquivo em MB, quando disponível."""
    try:
        return round(os.path.getsize(path) / (1024 * 1024), 1)
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Distribuições estatísticas (pandas/numpy)
# ---------------------------------------------------------------------------

# Nota sobre ddof=0 (desvio padrão populacional):
# O corpus de 32.312 pares constitui a população completa de votos-ementa
# da TR/JFPB no período coberto (2024-04 a 2026-02), não uma amostra
# aleatória de uma população maior. Usa-se ddof=0 (denominador N) para
# desvio padrão e coeficiente de variação.


def _distribuicao(serie: pd.Series) -> dict[str, Any]:
    """Calcula estatísticas descritivas de uma Series numérica.

    Usa pandas/numpy nativamente: mean, median, std (populacional),
    quantis e total. Substitui a implementação manual anterior com
    math.floor e statistics.pstdev.

    Args:
        serie: Series com contagens de palavras (inteiros).

    Returns:
        Dicionário compatível com o schema do dashboard.
    """
    if serie.empty:
        return {
            "contagem": 0, "media": 0.0, "mediana": 0.0, "desvio_padrao": 0.0,
            "assimetria": 0.0, "curtose": 0.0, "coeficiente_variacao_pct": None,
            "min": 0, "max": 0, "p5": 0.0, "p25": 0.0, "p75": 0.0, "p95": 0.0,
            "total_palavras": 0,
        }

    quantis = serie.quantile(_PERCENTIS)  # pandas interpola linearmente por padrão

    std = float(serie.std(ddof=0))
    media = float(serie.mean())
    raw_skew = serie.skew()
    raw_kurt = serie.kurtosis()

    return {
        "contagem":      int(len(serie)),
        "media":         round(media, 1),
        "mediana":       round(float(serie.median()), 1),
        "desvio_padrao": round(std, 1),                       # ddof=0 → população completa
        "assimetria":    round(float(0.0 if pd.isna(raw_skew) else raw_skew), 2),
        "curtose":       round(float(0.0 if pd.isna(raw_kurt) else raw_kurt), 2),
        "coeficiente_variacao_pct": round(std / media * 100, 1) if media != 0 else None,
        "min":           int(serie.min()),
        "max":           int(serie.max()),
        "p5":            round(float(quantis[0.05]), 1),
        "p25":           round(float(quantis[0.25]), 1),
        "p75":           round(float(quantis[0.75]), 1),
        "p95":           round(float(quantis[0.95]), 1),
        "total_palavras": int(serie.sum()),
    }


def _histograma(serie: pd.Series, bin_size: int, limite: int = 15) -> dict[str, int]:
    """Gera histograma de frequências com bins de tamanho fixo.

    Substitui a implementação manual anterior com dict e operações de
    arredondamento. pd.cut() faz o binning vetorizado.

    Args:
        serie: Series com contagens de palavras.
        bin_size: Largura de cada bin (em palavras).
        limite: Número máximo de bins a retornar (os mais frequentes primeiros).

    Returns:
        Dicionário {label_bin: contagem} ordenado pelo valor mínimo do bin.
    """
    if serie.empty:
        return {}

    maximo = int(serie.max())
    # Gera fronteiras de bins cobrindo toda a faixa de dados
    fronteiras = range(0, maximo + bin_size + 1, bin_size)
    labels = [f"{b}-{b + bin_size - 1}" for b in fronteiras[:-1]]

    bins = pd.cut(serie, bins=list(fronteiras), labels=labels, right=False)
    contagens = (
        bins.value_counts()
        .reindex(labels, fill_value=0)  # preserva ordem dos bins
        .head(limite)
    )

    return {str(k): int(v) for k, v in contagens.items() if v > 0}


def _resumo_iqr(serie: pd.Series, *, precisao: int) -> dict[str, float | int]:
    """Resume sinais distribucionais via IQR, sem presumir exclusão automática."""
    if serie.empty:
        zero = round(0.0, precisao)
        return {
            "q1": zero,
            "q3": zero,
            "iqr": zero,
            "limite_iqr_superior": zero,
            "acima_limite_iqr": 0,
            "limite_iqr_superior_severo": zero,
            "acima_limite_iqr_severo": 0,
        }

    q1 = float(serie.quantile(0.25))
    q3 = float(serie.quantile(0.75))
    iqr = q3 - q1
    limite_iqr = q3 + 1.5 * iqr
    limite_iqr_severo = q3 + 3.0 * iqr
    return {
        "q1": round(q1, precisao),
        "q3": round(q3, precisao),
        "iqr": round(iqr, precisao),
        "limite_iqr_superior": round(limite_iqr, precisao),
        "acima_limite_iqr": int((serie > limite_iqr).sum()),
        "limite_iqr_superior_severo": round(limite_iqr_severo, precisao),
        "acima_limite_iqr_severo": int((serie > limite_iqr_severo).sum()),
    }


def _classificar_anomalias_ementa(ementa: str) -> list[str]:
    """Identifica rótulos claramente corrompidos por regras estruturais conservadoras."""
    texto = (ementa or "").strip()
    if not texto:
        return []

    texto_upper = texto.upper()
    anomalias: list[str] = []
    if texto_upper == _EMENTA_TRUNCADA_EXATA:
        anomalias.append("trunc_8w_exata")
    if texto_upper.startswith(_HEADER_INSTITUCIONAL_PREFIXO):
        anomalias.append("header_ementa")
    if texto_upper.startswith("DESPACHO"):
        anomalias.append("despacho_ementa")
    if re.match(r"^\d+[\).\-]", texto):
        anomalias.append("ementa_start_digit")
    if len(texto.split()) > 300:
        anomalias.append("ementa_longa_anomala")
    return anomalias


def _resumir_anomalias_estruturais(df: pd.DataFrame) -> dict[str, dict[str, int | str]]:
    """Conta pares com rótulo corrompido e inputs contaminados por voto-ementa."""
    rotulo_corrompido = df["ementa"].apply(lambda texto: bool(_classificar_anomalias_ementa(texto)))
    input_contaminado = df["fundamentacao"].str.strip().str.upper().str.startswith(_PREFIXO_INPUT_CONTAMINADO)

    return {
        "rotulos_corrompidos": {
            "total": int(rotulo_corrompido.sum()),
            "treino": int((rotulo_corrompido & df["split"].eq("treino")).sum()),
            "teste": int((rotulo_corrompido & df["split"].eq("teste")).sum()),
            "acao": "quarentena_exclusao",
        },
        "inputs_contaminados": {
            "total": int(input_contaminado.sum()),
            "treino": int((input_contaminado & df["split"].eq("treino")).sum()),
            "teste": int((input_contaminado & df["split"].eq("teste")).sum()),
            "acao": "sanitizacao_prefixo_voto_ementa",
        },
    }


def _compatibilidade_qwen(df: pd.DataFrame) -> dict[str, str | int | None]:
    """Conta fundamentações que excedem o limite de contexto explicitado para o Qwen."""
    base = {
        "id": "qwen",
        "model_id": _QWEN_MODEL_ID,
        "limite_tokens_input": _QWEN_MAX_INPUT_TOKENS,
        "acima_limite": None,
        "treino": None,
        "teste": None,
        "acao": "estratificar_ou_excluir_do_benchmark_qwen",
    }

    try:
        from transformers import AutoTokenizer
    except ImportError:
        log.warning("Outliers/Qwen: `transformers` indisponível; compatibilidade ficará sem contagem.")
        return {**base, "status": "dependencia_ausente_transformers"}

    try:
        tokenizer = AutoTokenizer.from_pretrained(_QWEN_MODEL_ID)
    except Exception as exc:  # noqa: BLE001 - falha operacional de cache/rede
        log.warning("Outliers/Qwen: tokenizer indisponível: %s", exc)
        return {**base, "status": "tokenizer_indisponivel"}

    acima_limite = {"treino": 0, "teste": 0}
    for split, fundamentacao in df[["split", "fundamentacao"]].itertuples(index=False):
        n_tokens = len(tokenizer.encode(fundamentacao, add_special_tokens=False))
        if n_tokens > _QWEN_MAX_INPUT_TOKENS:
            acima_limite[split] += 1

    total = acima_limite["treino"] + acima_limite["teste"]
    return {
        **base,
        "acima_limite": total,
        "treino": acima_limite["treino"],
        "teste": acima_limite["teste"],
    }


def calcular_outliers(df: pd.DataFrame) -> dict[str, Any]:
    """Consolida sinais distribucionais, anomalias estruturais e compatibilidade experimental."""
    anomalias_estruturais = _resumir_anomalias_estruturais(df)
    compat_qwen = _compatibilidade_qwen(df)

    return {
        "status": "proposta_metodologica",
        "enquadramento": {
            "unidade_distribucional": "palavras",
            "unidade_compatibilidade_modelo": "tokens",
            "mensagem_principal": (
                "IQR é triagem descritiva; tratamento de outliers depende de integridade "
                "estrutural e compatibilidade experimental."
            ),
        },
        "sinais_distribucionais": {
            "fundamentacao": _resumo_iqr(df["n_fund"], precisao=1),
            "ementa": _resumo_iqr(df["n_ementa"], precisao=1),
            "razao_compressao": _resumo_iqr(df["razao"].dropna(), precisao=2),
        },
        "anomalias_estruturais": anomalias_estruturais,
        "compatibilidade_modelos": [
            compat_qwen,
            {
                "id": "gemini",
                "model_id": _GEMINI_MODEL_ID,
                "limite_tokens_input": None,
                "status": "sem_limite_local_codificado",
            },
        ],
        "principios": [
            "IQR não implica exclusão automática.",
            "Curto não implica erro.",
            "Cauda longa válida deve ser preservada e, quando necessário, estratificada.",
            "Erro estrutural e incompatibilidade experimental são problemas distintos.",
        ],
    }


# ---------------------------------------------------------------------------
# Novel n-grams (abstratividade) — mantida em stdlib (sets são ideais aqui)
# ---------------------------------------------------------------------------


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

    Nota: mantida em stdlib pura — operações de conjunto sobre listas de
    n-grams variáveis não se vetorizam naturalmente em pandas.

    Args:
        pares: Lista de tuplas (fundamentação, ementa) em texto limpo.

    Returns:
        Dicionário com taxas médias de novelty para uni/bi/trigramas.
    """
    novelty: dict[str, list[float]] = {"unigrams": [], "bigrams": [], "trigrams": []}

    for fund, ementa in pares:
        palavras_fund  = fund.lower().split()
        palavras_ementa = ementa.lower().split()

        if len(palavras_ementa) < 2:
            continue

        for n, nome in [(1, "unigrams"), (2, "bigrams"), (3, "trigrams")]:
            if len(palavras_ementa) < n:
                continue
            ng_ementa = _ngrams(palavras_ementa, n)
            ng_fund   = _ngrams(palavras_fund, n)
            novel = ng_ementa - ng_fund
            taxa  = len(novel) / len(ng_ementa) * 100 if ng_ementa else 0.0
            novelty[nome].append(taxa)

    resultado = {}
    for nome, valores in novelty.items():
        if valores:
            s = pd.Series(valores)
            resultado[nome] = {
                "media":         round(float(s.mean()), 1),
                "mediana":       round(float(s.median()), 1),
                "desvio_padrao": round(float(s.std(ddof=0)), 1),
            }
    return resultado


# ---------------------------------------------------------------------------
# Funil de attrition
# ---------------------------------------------------------------------------


def calcular_funil(
    total_dump: int,
    n_brutos: int,
    n_limpos: int,
    n_treino: int,
    n_teste: int,
) -> dict:
    """Calcula o funil de attrition do pipeline."""
    total_final = n_treino + n_teste
    return {
        "dump_postgresql":          total_dump,
        "apos_filtro_nulos_fase1":  n_brutos,
        "apos_limpeza_fase2":       n_limpos,
        "dataset_final_fase3":      total_final,
        "treino":                   n_treino,
        "teste":                    n_teste,
        "perda_fase1":              total_dump - n_brutos,
        "perda_fase2":              n_brutos - n_limpos,
        "perda_total":              total_dump - total_final,
        "taxa_retencao_global":     round(total_final / total_dump * 100, 1) if total_dump else 0.0,
    }


# ---------------------------------------------------------------------------
# Período temporal (pandas datetime)
# ---------------------------------------------------------------------------


def calcular_periodo_temporal(df_brutos: pd.DataFrame) -> dict:
    """Extrai o período temporal do campo data_cadastro usando pandas.

    Usa parse estrito de datetime para garantir consistência com o split
    cronológico realizado na Fase 3.
    """
    if "data_cadastro" in df_brutos.columns and df_brutos["data_cadastro"].notna().any():
        datas_brutas = df_brutos["data_cadastro"]
    else:
        # Fallback: ler do SQLite
        import sqlite3
        if not SQLITE_DB_PATH.exists():
            raise ValueError("Fase 4 / período temporal: `data_cadastro` não disponível.")
        conn = sqlite3.connect(SQLITE_DB_PATH)
        rows = conn.execute(
            "SELECT data_cadastro FROM turmarecursal_processo "
            "WHERE data_cadastro IS NOT NULL"
        ).fetchall()
        conn.close()
        if not rows:
            raise ValueError("Fase 4 / período temporal: `data_cadastro` não disponível.")
        datas_brutas = pd.Series([r[0] for r in rows], dtype=str, name="data_cadastro")

    datas = validar_e_converter_data_cadastro(
        datas_brutas,
        contexto="Fase 4 / período temporal",
    )

    anos = datas.dt.year.astype(str)
    distribuicao = (
        anos.value_counts()
        .sort_index()
        .to_dict()
    )

    datas_ordenadas = datas.sort_values(kind="stable")
    return {
        "data_mais_antiga":      datas_ordenadas.iloc[0].strftime("%Y-%m-%d"),
        "data_mais_recente":     datas_ordenadas.iloc[-1].strftime("%Y-%m-%d"),
        "distribuicao_por_ano":  {str(k): int(v) for k, v in distribuicao.items()},
    }


# ---------------------------------------------------------------------------
# Word Cloud (pandas str.split + explode + value_counts)
# ---------------------------------------------------------------------------

_STOP_WORDS_PT = {
    "a", "à", "ao", "aos", "as", "às", "até", "com", "como", "da", "das",
    "de", "del", "dem", "des", "do", "dos", "e", "é", "ela", "elas",
    "ele", "eles", "em", "entre", "era", "essa", "essas", "esse", "esses",
    "esta", "estar", "estas", "este", "estes", "eu", "foi", "for",
    "foram", "há", "isso", "isto", "já", "lhe", "lhes", "lo", "mas",
    "me", "mesmo", "meu", "minha", "muito", "na", "nas", "não", "nem",
    "no", "nos", "nós", "num", "numa", "o", "os", "ou", "para",
    "pela", "pelas", "pelo", "pelos", "por", "qual", "quando",
    "que", "quem", "são", "se", "sem", "ser", "seu", "seus", "sua",
    "suas", "só", "também", "te", "tem", "ter", "tu", "tua", "um",
    "uma", "uns", "umas", "você", "vos",
    # Tokens de anonimização
    "[nome_pessoa]", "[nome_ocultado]", "[cpf]", "[cnpj]", "[npu]",
    "[email]", "[telefone]", "[conta-digito]", "[endereço_completo]", "[data]",
    "nome_pessoa", "nome_ocultado", "cpf", "cnpj", "npu",
    "email", "telefone", "conta-digito", "endereço_completo", "data",
    # Conectivos e termos gramaticais
    "nº", "n°", "art", "art.", "inc", "inc.", "cf", "§", "ii", "iii",
    "iv", "vi", "vii", "viii", "ix", "sob",
}


def calcular_wordcloud(ementas: pd.Series, top_n: int = 100) -> list[dict]:
    """Gera os dados da word cloud a partir das ementas.

    Usa pandas str.split().explode() para tokenização vetorizada e
    value_counts() para frequência, substituindo o loop manual com Counter.

    Args:
        ementas: Series com os textos das ementas.
        top_n: Número de termos mais frequentes a retornar.

    Returns:
        Lista de dicionários [{text, weight}] para wordcloud2.js.
    """
    contagens = (
        ementas
        .str.lower()
        .str.split()
        .explode()
        .str.strip(".,;:!?()[]{}\"'/—–-")
        .loc[lambda s: s.str.len() >= 3]
        .loc[lambda s: ~s.isin(_STOP_WORDS_PT)]
        .value_counts()
        .head(top_n)
    )
    return [{"text": word, "weight": int(count)} for word, count in contagens.items()]


# ---------------------------------------------------------------------------
# Distribuição temática (matérias)
# ---------------------------------------------------------------------------

_MATERIAS_REGRAS: list[tuple[str, list[str]]] = [
    ("Previdenciário",        ["PREVIDENCI", "REVIDENCIÁRIO", "APOSENTADORIA", "PENSÃO POR MORTE",
                               "AUXÍLIO", "SALÁRIO-MATERNIDADE", "REVISÃO DA RMI",
                               "BENEFÍCIO POR INCAPACIDADE"]),
    ("Assistencial",          ["ASSISTENCIAL", "AMPARO", "BPC", "BENEFÍCIO ASSISTENCIAL",
                               "ASSISTÊNCIA SOCIAL", "BENEFÍCIO DE PRESTAÇÃO CONTINUADA"]),
    ("Seguridade Social",     ["SEGURIDADE"]),
    ("Processual",            ["PROCESSUAL", "AGRAVO", "ADEQUAÇÃO", "RECURSO ORDINÁRIO",
                               "RECURSO INOMINADO", "QUESTÃO DE ORDEM", "RETORNO DOS AUTOS"]),
    ("Administrativo",        ["ADMINISTRATIV", "SERVIDOR PÚBLICO", "SAÚDE"]),
    ("FGTS",                  ["FGTS"]),
    ("Embargos",              ["EMBARGO"]),
    ("Civil",                 ["CIVIL"]),
    ("Tributário",            ["TRIBUTÁRI", "FISCAL"]),
    ("Criminal",              ["PENAL", "CRIMINAL"]),
    ("Constitucional",        ["CONSTITUCIONAL"]),
    ("Financeiro/Habitacional",["FINANCIAMENTO HABITACIONAL", "SISTEMA FINANCEIRO"]),
]


def _classificar_materia(prefixo: str) -> str:
    """Classifica uma ementa pelo prefixo (primeira sentença) em área do direito."""
    for materia, keywords in _MATERIAS_REGRAS:
        if any(k in prefixo for k in keywords):
            return materia
    return "Outros"


def calcular_distribuicao_materias(ementas: pd.Series, top_n: int = 10) -> list[dict]:
    """Distribui as ementas por área do direito usando apply() vetorizado.

    Substitui o loop manual com Counter por uma cadeia pandas:
    extração do prefixo → classificação → value_counts().

    Args:
        ementas: Series com os textos das ementas.
        top_n: Número máximo de categorias a exibir antes de agrupar em "Outros".

    Returns:
        Lista de dicionários [{materia, contagem}] ordenada decrescentemente.
    """
    prefixos = ementas.str.split(".").str[0].str.strip().str.upper()
    materias = prefixos.apply(_classificar_materia)

    contagens = materias.value_counts()
    top = contagens.head(top_n)
    outros_residual = contagens.iloc[top_n:].sum()

    if outros_residual > 0:
        # Soma com "Outros" se já existir entre os top
        outros_total = int(top.get("Outros", 0)) + int(outros_residual)
        top = top.drop("Outros", errors="ignore")
        top["Outros"] = outros_total
        top = top.sort_values(ascending=False)

    return [{"materia": str(k), "contagem": int(v)} for k, v in top.items()]


# ---------------------------------------------------------------------------
# Relatório principal
# ---------------------------------------------------------------------------


def _imprimir_distribuicao(nome: str, dist: dict) -> None:
    """Imprime uma distribuição formatada no log."""
    log.info(
        "  %s: média=%.1f | mediana=%.1f | std=%.1f | min=%d | max=%d",
        nome, dist["media"], dist["mediana"], dist["desvio_padrao"],
        dist["min"], dist["max"],
    )
    log.info(
        "    Percentis: P5=%.1f | P25=%.1f | P75=%.1f | P95=%.1f",
        dist["p5"], dist["p25"], dist["p75"], dist["p95"],
    )
    log.info(
        "    Assimetria=%.2f | Curtose=%.2f | CV=%.1f%%",
        dist["assimetria"], dist["curtose"],
        dist.get("coeficiente_variacao_pct") or 0.0,
    )
    log.info("    Total de palavras: %s", f"{dist['total_palavras']:,}")


def gerar_relatorio(
    brutos_path: Path = BRUTOS_PATH,
    limpos_path: Path = LIMPOS_PATH,
    treino_path: Path = TREINO_PATH,
    teste_path:  Path = TESTE_PATH,
    output_path: Path = OUTPUT_PATH,
) -> dict:
    """Pipeline completo da Fase 4: calcula todas as estatísticas e grava JSON."""

    log.info("=== Fase 4: Estatísticas Descritivas do Corpus ===")

    # --- Carregar dados ---
    log.info("Carregando dados...")
    df_brutos = _carregar_json(brutos_path)
    df_limpos = _carregar_json(limpos_path)
    treino_raw = _carregar_jsonl(treino_path)
    teste_raw = _carregar_jsonl(teste_path)
    try:
        ingestao_stats = _carregar_stats_ingestao(INGESTAO_STATS_PATH)
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        log.warning("Não foi possível ler %s: %s", INGESTAO_STATS_PATH, exc)
        ingestao_stats = {}
    total_dump = ingestao_stats.get("total_lidos", TOTAL_DUMP_FALLBACK)

    # --- Extrair textos do JSONL final para um DataFrame ---
    log.info("Extraindo textos do dataset final...")
    pares: list[dict[str, str]] = []
    for split, registros in (("treino", treino_raw), ("teste", teste_raw)):
        for obj in registros:
            fundamentacao, ementa = extrair_fundamentacao_e_ementa(obj)
            pares.append(
                {
                    "split": split,
                    "fundamentacao": fundamentacao,
                    "ementa": ementa,
                }
            )
    df = pd.DataFrame(pares)

    # Contagem de palavras vetorizada
    df["n_fund"]  = df["fundamentacao"].str.split().str.len()
    df["n_ementa"] = df["ementa"].str.split().str.len()
    df["razao"]   = np.where(df["n_ementa"] > 0, df["n_fund"] / df["n_ementa"], np.nan)

    # --- Funil ---
    funil = calcular_funil(total_dump, len(df_brutos), len(df_limpos), len(treino_raw), len(teste_raw))
    log.info("Funil de Attrition:")
    log.info("  Dump PostgreSQL:        %d", funil["dump_postgresql"])
    log.info("  Após filtro nulos (F1): %d (-%d)", funil["apos_filtro_nulos_fase1"], funil["perda_fase1"])
    log.info("  Após limpeza (F2):      %d (-%d)", funil["apos_limpeza_fase2"], funil["perda_fase2"])
    log.info("  Dataset final (F3):     %d", funil["dataset_final_fase3"])
    log.info("    Treino: %d | Teste: %d", funil["treino"], funil["teste"])
    log.info("  Taxa de retenção global: %.1f%%", funil["taxa_retencao_global"])

    # --- Distribuições ---
    dist_fund   = _distribuicao(df["n_fund"])
    dist_ementa = _distribuicao(df["n_ementa"])

    log.info("Distribuição de Comprimento (palavras):")
    _imprimir_distribuicao("Fundamentação", dist_fund)
    _imprimir_distribuicao("Ementa", dist_ementa)

    # --- Razão de compressão ---
    razao_serie = df["razao"].dropna()
    if not razao_serie.empty:
        razao_stats = {
            "media":         round(float(razao_serie.mean()), 2),
            "mediana":       round(float(razao_serie.median()), 2),
            "desvio_padrao": round(float(razao_serie.std(ddof=0)), 2),
            "min":           round(float(razao_serie.min()), 2),
            "max":           round(float(razao_serie.max()), 2),
        }
    else:
        razao_stats = {"media": 0.0, "mediana": 0.0, "desvio_padrao": 0.0, "min": 0.0, "max": 0.0}
    log.info(
        "Razão de Compressão (fund./ementa): média=%.2f | mediana=%.2f | std=%.2f",
        razao_stats["media"], razao_stats["mediana"], razao_stats["desvio_padrao"],
    )

    # --- Novel n-grams (abstratividade) ---
    log.info("Calculando novel n-grams (abstratividade)...")
    pares_texto = list(zip(df["fundamentacao"], df["ementa"]))
    novelty = calcular_novel_ngrams(pares_texto)
    for nome, vals in novelty.items():
        log.info(
            "  Novel %s: média=%.1f%% | mediana=%.1f%% | std=%.1f%%",
            nome, vals["media"], vals["mediana"], vals["desvio_padrao"],
        )

    # --- Período temporal ---
    log.info("Analisando período temporal...")
    periodo = calcular_periodo_temporal(df_brutos)
    log.info("  Período: %s a %s", periodo["data_mais_antiga"], periodo["data_mais_recente"])
    log.info("  Distribuição por ano: %s", periodo["distribuicao_por_ano"])

    # --- Timing do pipeline ---
    timing: dict = {}
    if TIMING_PATH.exists():
        try:
            with TIMING_PATH.open("r") as f:
                timing = json.load(f)
            log.info("Timing do pipeline carregado: %s", timing)
        except (json.JSONDecodeError, OSError):
            log.warning("Não foi possível ler timing do pipeline.")

    # --- Vocabulário ---
    log.info("Calculando vocabulário...")
    vocab_fund   = set(df["fundamentacao"].str.lower().str.split().explode())
    vocab_ementa = set(df["ementa"].str.lower().str.split().explode())
    vocab_total  = vocab_fund | vocab_ementa

    vocabulario = {
        "fundamentacao": len(vocab_fund),
        "ementa":        len(vocab_ementa),
        "total_unico":   len(vocab_total),
        "sobreposicao":  len(vocab_fund & vocab_ementa),
    }
    log.info(
        "Vocabulário: fund=%d | ementa=%d | total=%d | sobreposição=%d",
        vocabulario["fundamentacao"], vocabulario["ementa"],
        vocabulario["total_unico"], vocabulario["sobreposicao"],
    )

    # --- Histogramas ---
    hist_fund   = _histograma(df["n_fund"], bin_size=200)
    hist_ementa = _histograma(df["n_ementa"], bin_size=10)

    # --- Scatter (amostragem reprodutível) ---
    log.info("Gerando dados de scatter (compressão)...")
    scatter_sample_size = min(2_000, len(df))
    df_scatter = df[["n_fund", "n_ementa"]].sample(
        n=scatter_sample_size, random_state=42
    ).sort_values("n_fund")
    scatter_data = [
        {"x": int(row.n_fund), "y": int(row.n_ementa)}
        for row in df_scatter.itertuples(index=False)
    ]
    # Correlação de Spearman (rank + Pearson nos ranks, sem scipy)
    spearman_rho = float(df["n_fund"].rank().corr(df["n_ementa"].rank()))
    log.info("  Scatter: %d pontos amostrados de %d | Spearman ρ = %.4f",
             len(scatter_data), len(df), spearman_rho)

    # --- Word Cloud ---
    log.info("Gerando dados de word cloud...")
    wordcloud_data = calcular_wordcloud(df["ementa"])
    log.info("  Word cloud: top-%d termos.", len(wordcloud_data))

    # --- Distribuição temática ---
    log.info("Calculando distribuição temática (matérias)...")
    distribuicao_materias = calcular_distribuicao_materias(df["ementa"])
    log.info("  Matérias: %s", {d["materia"]: d["contagem"] for d in distribuicao_materias[:5]})

    # --- Outliers (metodologia) ---
    log.info("Consolidando sinais de outliers...")
    outliers = calcular_outliers(df)
    log.info(
        "  Outliers/IQR: fund=%d | ementa=%d | razão=%d",
        outliers["sinais_distribucionais"]["fundamentacao"]["acima_limite_iqr"],
        outliers["sinais_distribucionais"]["ementa"]["acima_limite_iqr"],
        outliers["sinais_distribucionais"]["razao_compressao"]["acima_limite_iqr"],
    )
    log.info(
        "  Anomalias estruturais: rótulos=%d | inputs contaminados=%d",
        outliers["anomalias_estruturais"]["rotulos_corrompidos"]["total"],
        outliers["anomalias_estruturais"]["inputs_contaminados"]["total"],
    )
    compat_qwen = outliers["compatibilidade_modelos"][0]
    if compat_qwen.get("acima_limite") is not None:
        log.info(
            "  Compatibilidade Qwen: %d fundamentações acima de %d tokens",
            compat_qwen["acima_limite"],
            compat_qwen["limite_tokens_input"],
        )
    else:
        log.info("  Compatibilidade Qwen: %s", compat_qwen.get("status", "indisponível"))

    # --- PII Stats ---
    pii_contagem: dict = {}
    if PII_STATS_PATH.exists():
        try:
            with PII_STATS_PATH.open("r") as f:
                pii_contagem = json.load(f)
            log.info("  PII stats carregadas: %s", pii_contagem)
        except (json.JSONDecodeError, OSError):
            log.warning("Não foi possível ler PII stats.")

    # --- Narrativas ---
    taxa_ret_f1 = round(len(df_brutos) / total_dump * 100, 1) if total_dump else 0
    perda_f1    = total_dump - len(df_brutos)
    narrativa_f1 = (
        f"O sistema judicial armazena votos e ementas em PostgreSQL. "
        f"O dump original contém {total_dump:,} registros. "
        f"Após descartar {perda_f1:,} registros sem voto (fundamentação) ou ementa preenchida, "
        f"{len(df_brutos):,} pares válidos foram exportados, uma retenção de {taxa_ret_f1}%, "
        f"indicando alta completude da base de dados."
    ).replace(",", ".")

    taxa_ret_f2 = round(len(df_limpos) / len(df_brutos) * 100, 1) if len(df_brutos) else 0
    perda_f2    = len(df_brutos) - len(df_limpos)
    narrativa_f2 = (
        f"Textos jurídicos contêm ruído processual (identificadores PJe, carimbos de "
        f"publicação DJe, assinaturas) que polui o treinamento de modelos de linguagem. "
        f"Após aplicar as regras de limpeza via expressões regulares e os filtros de qualidade "
        f"da Fase 2, {perda_f2:,} registros foram descartados. A retenção de {taxa_ret_f2}% "
        f"confirma que o ruído era pontual e que a maior parte do corpus permaneceu utilizável."
    ).replace(",", ".")

    narrativa_f3 = (
        f"Para conformidade com a LGPD, dados pessoais (CPF, CNPJ, nomes, endereços) foram "
        f"substituídos por tokens genéricos como [NOME_PESSOA] e [CPF]. O corpus anonimizado "
        f"foi dividido em treino ({len(treino_raw):,} registros, 90%) e teste ({len(teste_raw):,} registros, "
        f"10%) por critério cronológico (data_cadastro), evitando temporal leakage."
    ).replace(",", ".")

    narrativa_f4 = (
        "Esta fase analisa quantitativamente o corpus para caracterizar "
        "a complexidade da tarefa de sumarização e validar a qualidade dos dados."
    )

    # --- Dicas didáticas ---
    perda_total_pct = round((1 - funil.get("taxa_retencao_global", 100) / 100) * 100, 1)
    mediana_fund    = dist_fund.get("mediana", 0)
    mediana_ementa  = dist_ementa.get("mediana", 1)
    novel_tri       = novelty.get("trigrams", {}).get("media", 0)
    hist_keys       = list(hist_fund.keys())

    dicas = {
        "funil": (
            f"O funil mostra quantos registros sobreviveram a cada etapa do pipeline. "
            f"A perda total foi de apenas {perda_total_pct}%, indicando que a base original "
            f"é consistente e quase não há dados inutilizáveis."
        ),
        "distribuicao": (
            f"A fundamentação tem mediana de {mediana_fund} palavras enquanto a ementa "
            f"tem apenas {mediana_ementa}, uma razão de ~{int(mediana_fund // mediana_ementa) if mediana_ementa else '?'}:1. "
            f"Essa assimetria extrema define a complexidade da tarefa de sumarização."
        ),
        "novel_ngrams": (
            f"Novel n-grams medem quanto do texto da ementa NÃO aparece na fundamentação. "
            f"Valores altos ({novel_tri}% de trigrams novos) confirmam que os juízes "
            f"reformulam significativamente o texto. É sumarização abstrativa, não extrativa."
        ),
        "histograma": (
            f"A maioria das fundamentações concentra-se entre {hist_keys[0] if hist_keys else '?'} e "
            f"{hist_keys[2] if len(hist_keys) > 2 else '?'} palavras, mas a cauda longa revela casos de "
            f"alta complexidade com textos muito extensos."
        ),
        "temporal": (
            f"Quantidade de processos registrados no sistema a cada ano, "
            f"abrangendo o período de {periodo.get('data_mais_antiga', '—')[:4]} a "
            f"{periodo.get('data_mais_recente', '—')[:4]}."
        ),
    }

    # --- Metadados por fase ---
    regras_higienizacao = [
        "Remoção de tags HTML (artefatos de renderização)",
        "Remoção de metadados processuais (Processo nº, NPU)",
        "Remoção de identificadores PJe (id. 48772689, etc.)",
        "Remoção de carimbos DJe completos",
        "Remoção de carimbos DJe simples e datas DJe",
        "Remoção de 'DIVULG' isolado (resíduo DJe)",
        "Remoção de Ação Civil Pública + DPU com nº (cabeçalho formatado)",
        "Substituição de datas prefixadas por cidade por token [DATA]",
        "Remoção de Súmula de Julgamento (rodapé processual)",
        "Remoção de honoríficos de juízes no final (assinatura)",
        "Remoção de blocos em CAPSLOCK no final (assinatura)",
        "Descarte de registros vazios após a limpeza",
        "Descarte de registros com fundamentação idêntica à ementa",
        "Filtro: fundamentação com menos de 10 palavras",
        "Filtro: ementa com menos de 5 palavras",
    ]

    fase1 = {
        "nome": "Ingestão",
        "descricao": "Extração dos votos (fundamentações) e ementas do banco PostgreSQL do sistema judicial.",
        "narrativa": narrativa_f1, "status": "concluida",
        "script": "python3 -m pipeline.fase1_4.fase01_ingestao",
        "duracao_segundos": timing.get("fase1_ingestao"),
        "registros_dump": total_dump,
        "registros_exportados": len(df_brutos),
        "perda": perda_f1, "taxa_retencao": taxa_ret_f1,
        "fonte": "Sistema Judicial / PostgreSQL",
        "artefatos": [
            {"nome": "dump.sql", "tamanho_mb": _file_size_mb(DUMP_PATH), "tipo": "entrada", "conteudo": "Dump binário PostgreSQL (custom format)"},
            {"nome": "dados_brutos.json", "tamanho_mb": _file_size_mb(brutos_path), "tipo": "saida", "conteudo": "{id, fundamentação, ementa}"},
        ],
    }

    fase2 = {
        "nome": "Higienização",
        "descricao": "Limpeza do corpus para remoção de ruídos processuais via expressões regulares.",
        "narrativa": narrativa_f2, "status": "concluida",
        "script": "python3 -m pipeline.fase1_4.fase02_higienizacao",
        "duracao_segundos": timing.get("fase2_higienizacao"),
        "registros_entrada": len(df_brutos), "registros_saida": len(df_limpos),
        "perda": perda_f2, "taxa_retencao": taxa_ret_f2,
        "regras_aplicadas": regras_higienizacao,
        "artefatos": [
            {"nome": "dados_brutos.json", "tamanho_mb": _file_size_mb(brutos_path), "tipo": "entrada", "conteudo": "{id, fundamentação, ementa}"},
            {"nome": "dados_limpos.json", "tamanho_mb": _file_size_mb(limpos_path), "tipo": "saida", "conteudo": "{id, fundamentação, ementa} limpos"},
        ],
    }

    fase3 = {
        "nome": "Anonimização (LGPD)",
        "descricao": "Substituição de dados pessoais por tokens genéricos e formatação para fine-tuning.",
        "narrativa": narrativa_f3, "status": "concluida",
        "script": "python3 -m pipeline.fase1_4.fase03_anonimizacao",
        "duracao_segundos": timing.get("fase3_anonimizacao"),
        "registros_entrada": len(df_limpos),
        "registros_saida": len(treino_raw) + len(teste_raw),
        "treino": len(treino_raw), "teste": len(teste_raw),
        "split_ratio": "90/10", "split_criterio": "cronológico (data_cadastro)",
        "categorias_pii": [
            "CPF", "CNPJ", "NPU", "CONTA-DIGITO",
            "EMAIL", "TELEFONE", "NOME_OCULTADO", "NOME_PESSOA", "ENDEREÇO_COMPLETO",
        ],
        "pii_contagem": pii_contagem,
        "system_prompt": SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip(),
        "artefatos": [
            {"nome": "dados_limpos.json", "tamanho_mb": _file_size_mb(limpos_path), "tipo": "entrada", "conteudo": "{id, fundamentação, ementa} limpos"},
            {"nome": "dataset_treino.jsonl", "tamanho_mb": _file_size_mb(treino_path), "tipo": "saida", "conteudo": "{contents: [{role, parts}]} anonimizado · 90%"},
            {"nome": "dataset_teste.jsonl", "tamanho_mb": _file_size_mb(teste_path), "tipo": "saida", "conteudo": "{contents: [{role, parts}]} anonimizado · 10%"},
        ],
    }

    fase4 = {
        "nome": "Estatísticas Descritivas",
        "descricao": "Análise quantitativa do corpus: distribuições, compressão e grau de abstratividade.",
        "narrativa": narrativa_f4, "status": "concluida",
        "script": "python3 -m pipeline.fase1_4.fase04_estatisticas",
        "duracao_segundos": timing.get("fase4_estatisticas"),
        "funil": funil,
        "fundamentacao": dist_fund,
        "ementa": dist_ementa,
        "razao_compressao": razao_stats,
        "novel_ngrams": novelty,
        "periodo_temporal": periodo,
        "vocabulario": vocabulario,
        "histograma_fundamentacao": hist_fund,
        "histograma_ementa": hist_ementa,
        "scatter_compressao": {
            "pontos": scatter_data,
            "n_amostra": len(scatter_data),
            "n_total": len(df),
            "spearman_rho": round(spearman_rho, 4),
        },
        "wordcloud": wordcloud_data,
        "distribuicao_materias": distribuicao_materias,
        "outliers": outliers,
        "dicas": dicas,
        "artefatos": [
            {"nome": "dataset_treino.jsonl", "tamanho_mb": _file_size_mb(treino_path), "tipo": "entrada", "conteudo": "{contents: [{role, parts}]} anonimizado"},
            {"nome": "estatisticas_corpus.json", "tamanho_mb": None, "tipo": "saida", "conteudo": "Métricas, distribuições, funil"},
        ],
    }

    resultado = {
        "meta": {
            "gerado_em": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "titulo_pesquisa": "Geração Abstrativa de Ementas Judiciais",
            "unidade_de_medida": "palavras (split por espaço)",
            "pipeline_total_segundos": timing.get("pipeline_total"),
        },
        "fases": {
            "fase1_ingestao":    fase1,
            "fase2_higienizacao": fase2,
            "fase3_anonimizacao": fase3,
            "fase4_estatisticas": fase4,
            "fase5_finetuning":  {"nome": "Fine-Tuning", "status": "pendente", "scripts": ["python3 -m pipeline.fase5.finetuning_gemini", "python3 -m pipeline.fase5.finetuning_qwen"]},
            "fase6_baseline":    {"nome": "Baseline Zero-Shot", "status": "pendente", "scripts": ["python3 -m pipeline.fase6.baseline_gemini", "python3 -m pipeline.fase6.baseline_qwen"]},
            "fase7_avaliacao":   {"nome": "Avaliação", "status": "pendente", "script": "Apresentacao_Dissertacao_Colab.ipynb"},
        },
    }

    # --- Gravar JSON ---
    log.info("Gravando resultados em %s ...", output_path)
    escrever_json_atomico(output_path, resultado)

    # Cópia para GitHub Pages
    DOCS_DATA_PATH.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output_path, DOCS_DATA_PATH / output_path.name)

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
    except (FileNotFoundError, OSError, ValueError) as exc:
        log.critical("Execução interrompida: %s", exc)
        sys.exit(1)
