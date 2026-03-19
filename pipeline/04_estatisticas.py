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
    if not valores:
        return {
            "contagem": 0, "media": 0.0, "mediana": 0.0, "desvio_padrao": 0.0,
            "min": 0, "max": 0, "p25": 0.0, "p75": 0.0, "p95": 0.0,
        }
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
        db_path = Path("data/banco_sistema_judicial.sqlite")
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
    if razoes:
        razao_stats = {
            "media": round(sum(razoes) / len(razoes), 2),
            "mediana": round(stats.median(razoes), 2),
            "desvio_padrao": round(stats.pstdev(razoes), 2),
            "min": round(min(razoes), 2),
            "max": round(max(razoes), 2),
        }
    else:
        razao_stats = {"media": 0.0, "mediana": 0.0, "desvio_padrao": 0.0, "min": 0.0, "max": 0.0}
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

    # --- Metadados por fase (para o dashboard) ---
    import os
    from datetime import datetime
    from collections import Counter

    def _file_size_mb(path: Path) -> float | None:
        try:
            return round(os.path.getsize(path) / (1024 * 1024), 1)
        except OSError:
            return None

    # --- Ler timing do pipeline (gerado por run_all.sh) ---
    timing_path = Path("data/.pipeline_timing.json")
    timing: dict = {}
    if timing_path.exists():
        try:
            with timing_path.open("r") as f:
                timing = json.load(f)
            log.info("Timing do pipeline carregado: %s", timing)
        except (json.JSONDecodeError, OSError):
            log.warning("Não foi possível ler timing do pipeline.")

    # --- Vocabulário e histograma ---
    log.info("Calculando vocabulário e histogramas...")
    vocab_fund: set[str] = set()
    vocab_ementa: set[str] = set()
    for fund, ementa in pares_texto:
        vocab_fund.update(fund.lower().split())
        vocab_ementa.update(ementa.lower().split())
    vocab_total = vocab_fund | vocab_ementa

    vocabulario = {
        "fundamentacao": len(vocab_fund),
        "ementa": len(vocab_ementa),
        "total_unico": len(vocab_total),
        "sobreposicao": len(vocab_fund & vocab_ementa),
    }
    log.info(
        "Vocabulário: fund=%d | ementa=%d | total=%d | sobreposição=%d",
        vocabulario["fundamentacao"], vocabulario["ementa"],
        vocabulario["total_unico"], vocabulario["sobreposicao"],
    )

    # Histograma de comprimento (bins de 100 palavras para fundamentação, 10 para ementa)
    def _histograma(valores: list[int], bin_size: int) -> dict:
        bins: dict[str, int] = {}
        for v in valores:
            bucket = (v // bin_size) * bin_size
            label = f"{bucket}-{bucket + bin_size - 1}"
            bins[label] = bins.get(label, 0) + 1
        # Ordenar por bucket numérico e limitar aos primeiros 15 buckets
        sorted_bins = dict(sorted(bins.items(), key=lambda x: int(x[0].split("-")[0]))[:15])
        return sorted_bins

    hist_fund = _histograma(fundamentos_palavras, 200)
    hist_ementa = _histograma(ementas_palavras, 10)

    # --- Scatter data (amostragem para performance no navegador) ---
    log.info("Gerando dados de scatter (compressão)...")
    scatter_full = [
        {"x": fundamentos_palavras[i], "y": ementas_palavras[i]}
        for i in range(len(fundamentos_palavras))
    ]
    # Amostrar ~2000 pontos uniformemente
    import random as _rng
    _rng.seed(42)  # reprodutível (apenas para amostragem visual, não afeta dados)
    scatter_sample_size = min(2000, len(scatter_full))
    scatter_data = sorted(
        _rng.sample(scatter_full, scatter_sample_size),
        key=lambda p: p["x"],
    )
    log.info("  Scatter: %d pontos amostrados de %d.", len(scatter_data), len(scatter_full))

    # --- Word Cloud (top-100 termos das ementas, sem stop words) ---
    log.info("Gerando dados de word cloud...")
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
        # Tokens de anonimização (com e sem colchetes — strip() remove [])
        "[nome_pessoa]", "[nome_ocultado]", "[cpf]", "[cnpj]", "[npu]",
        "[email]", "[telefone]", "[conta-digito]", "[endereço_completo]", "[data]",
        "nome_pessoa", "nome_ocultado", "cpf", "cnpj", "npu",
        "email", "telefone", "conta-digito", "endereço_completo", "data",
        # Preposições/artigos compostos e conectivos
        "nº", "n°", "art", "art.", "inc", "inc.", "cf", "§", "ii", "iii",
        "iv", "vi", "vii", "viii", "ix", "sob",
    }
    from collections import Counter as _Counter
    word_counts: dict[str, int] = _Counter()
    for _, ementa in pares_texto:
        palavras = ementa.lower().split()
        for p in palavras:
            # Remove pontuação grudada
            p_clean = p.strip(".,;:!?()[]{}\"'/—–-")
            if len(p_clean) >= 3 and p_clean not in _STOP_WORDS_PT:
                word_counts[p_clean] += 1
    wordcloud_data = [
        {"text": word, "weight": count}
        for word, count in word_counts.most_common(100)
    ]
    log.info("  Word cloud: top-%d termos.", len(wordcloud_data))

    # --- PII Stats (lidas do JSON auxiliar da Fase 3) ---
    pii_stats_path = Path("data/.anonimizacao_stats.json")
    pii_contagem: dict = {}
    if pii_stats_path.exists():
        try:
            with pii_stats_path.open("r") as f:
                pii_contagem = json.load(f)
            log.info("  PII stats carregadas: %s", pii_contagem)
        except (json.JSONDecodeError, OSError):
            log.warning("Não foi possível ler PII stats.")

    # --- Artefatos completos por fase ---
    db_path = Path("data/banco_sistema_judicial.sqlite")
    dump_path = Path("dump_sistema_judicial.sql")

    # --- Narrativas contextuais (didáticas) ---
    taxa_ret_f1 = round(len(brutos) / TOTAL_DUMP * 100, 1)
    perda_f1 = TOTAL_DUMP - len(brutos)
    narrativa_f1 = (
        f"O sistema judicial armazena votos e ementas em PostgreSQL. "
        f"O dump original contém {TOTAL_DUMP:,} registros. "
        f"Após descartar {perda_f1:,} registros sem voto (fundamentação) ou ementa preenchida, "
        f"{len(brutos):,} pares válidos foram exportados, uma retenção de {taxa_ret_f1}%, "
        f"indicando alta completude da base de dados."
    ).replace(",", ".")

    taxa_ret_f2 = round(len(limpos) / len(brutos) * 100, 1) if brutos else 0
    perda_f2 = len(brutos) - len(limpos)
    narrativa_f2 = (
        f"Textos jurídicos contêm ruído processual (identificadores PJe, carimbos de "
        f"publicação DJe, assinaturas) que polui o treinamento de modelos de linguagem. "
        f"Após aplicar 6 regras de limpeza via expressões regulares, {perda_f2:,} registros "
        f"foram descartados por fundamentação ou ementa muito curta. A retenção de {taxa_ret_f2}% "
        f"confirma que o ruído era pontual. A maioria do corpus é utilizável."
    ).replace(",", ".")

    narrativa_f3 = (
        f"Para conformidade com a LGPD, dados pessoais (CPF, CNPJ, nomes, endereços) foram "
        f"substituídos por tokens genéricos como [NOME_PESSOA] e [CPF]. O corpus anonimizado "
        f"foi dividido em treino ({len(treino):,} registros, 90%) e teste ({len(teste):,} registros, "
        f"10%) por critério cronológico (data_cadastro), evitando temporal leakage."
    ).replace(",", ".")

    razao_media = razao_stats.get("media", 0)
    novel_uni = novelty.get("unigrams", {}).get("media", 0)
    novel_tri = novelty.get("trigrams", {}).get("media", 0)
    narrativa_f4 = (
        "Esta fase analisa quantitativamente o corpus para caracterizar "
        "a complexidade da tarefa de sumarização e validar a qualidade dos dados."
    )

    fase1 = {
        "nome": "Ingestão",
        "descricao": "Extração dos votos (fundamentações) e ementas do banco PostgreSQL do sistema judicial.",
        "narrativa": narrativa_f1,
        "status": "concluida",
        "script": "pipeline/01_ingestao.py",
        "duracao_segundos": timing.get("fase1_ingestao"),
        "registros_dump": TOTAL_DUMP,
        "registros_exportados": len(brutos),
        "perda": perda_f1,
        "taxa_retencao": taxa_ret_f1,
        "fonte": "Sistema Judicial / PostgreSQL",
        "artefatos": [
            {"nome": "dump.sql", "tamanho_mb": _file_size_mb(dump_path), "tipo": "entrada", "conteudo": "Dump binário PostgreSQL (custom format)"},
            {"nome": "dados_brutos.json", "tamanho_mb": _file_size_mb(brutos_path), "tipo": "saida", "conteudo": "{id, fundamentação, ementa}"},
        ],
    }

    regras_higienizacao = [
        "Remoção de tags HTML (artefatos de renderização)",
        "Remoção de metadados processuais (Processo nº, NPU)",
        "Remoção de identificadores PJe (id. 48772689, etc.)",
        "Remoção de carimbos DJe completos e simples",
        "Remoção de 'DIVULG' isolado (resíduo DJe)",
        "Remoção de Ação Civil Pública + DPU com nº (cabeçalho formatado)",
        "Substituição de datas prefixadas por cidade por token [DATA]",
        "Remoção de Súmula de Julgamento (rodapé processual)",
        "Remoção de honoríficos de juízes no final (assinatura)",
        "Remoção de blocos em CAPSLOCK no final (assinatura)",
        "Filtro: fundamentação < 50 caracteres → descarte",
        "Filtro: ementa < 20 caracteres → descarte",
    ]

    fase2 = {
        "nome": "Higienização",
        "descricao": "Limpeza do corpus para remoção de ruídos processuais via expressões regulares.",
        "narrativa": narrativa_f2,
        "status": "concluida",
        "script": "pipeline/02_higienizacao.py",
        "duracao_segundos": timing.get("fase2_higienizacao"),
        "registros_entrada": len(brutos),
        "registros_saida": len(limpos),
        "perda": perda_f2,
        "taxa_retencao": taxa_ret_f2,
        "regras_aplicadas": regras_higienizacao,
        "artefatos": [
            {"nome": "dados_brutos.json", "tamanho_mb": _file_size_mb(brutos_path), "tipo": "entrada", "conteudo": "{id, fundamentação, ementa}"},
            {"nome": "dados_limpos.json", "tamanho_mb": _file_size_mb(limpos_path), "tipo": "saida", "conteudo": "{id, fundamentação, ementa} limpos"},
        ],
    }

    fase3 = {
        "nome": "Anonimização (LGPD)",
        "descricao": "Substituição de dados pessoais por tokens genéricos e formatação para fine-tuning.",
        "narrativa": narrativa_f3,
        "status": "concluida",
        "script": "pipeline/03_anonimizacao.py",
        "duracao_segundos": timing.get("fase3_anonimizacao"),
        "registros_entrada": len(limpos),
        "registros_saida": len(treino) + len(teste),
        "treino": len(treino),
        "teste": len(teste),
        "split_ratio": "90/10",
        "split_criterio": "cronológico (data_cadastro)",
        "categorias_pii": [
            "CPF", "CNPJ", "NPU", "CONTA-DIGITO",
            "EMAIL", "TELEFONE",
            "NOME_OCULTADO", "NOME_PESSOA",
            "ENDEREÇO_COMPLETO",
        ],
        "pii_contagem": pii_contagem,
        "artefatos": [
            {"nome": "dados_limpos.json", "tamanho_mb": _file_size_mb(limpos_path), "tipo": "entrada", "conteudo": "{id, fundamentação, ementa} limpos"},
            {"nome": "dataset_treino.jsonl", "tamanho_mb": _file_size_mb(treino_path), "tipo": "saida", "conteudo": "{contents: [{role, parts}]} anonimizado · 90%"},
            {"nome": "dataset_teste.jsonl", "tamanho_mb": _file_size_mb(teste_path), "tipo": "saida", "conteudo": "{contents: [{role, parts}]} anonimizado · 10%"},
        ],
    }

    # Dicas didáticas para cada quadro da Fase 4
    perda_total_pct = round((1 - funil.get("taxa_retencao_global", 100) / 100) * 100, 1)
    mediana_fund = dist_fund.get("mediana", 0)
    mediana_ementa = dist_ementa.get("mediana", 0)
    dicas = {
        "funil": (
            f"O funil mostra quantos registros sobreviveram a cada etapa do pipeline. "
            f"A perda total foi de apenas {perda_total_pct}%, indicando que a base original "
            f"é consistente e quase não há dados inutilizáveis."
        ),
        "distribuicao": (
            f"A fundamentação tem mediana de {mediana_fund} palavras enquanto a ementa "
            f"tem apenas {mediana_ementa}, uma razão de ~{mediana_fund // mediana_ementa}:1. "
            f"Essa assimetria extrema define a complexidade da tarefa de sumarização."
        ),
        "novel_ngrams": (
            f"Novel n-grams medem quanto do texto da ementa NÃO aparece na fundamentação. "
            f"Valores altos ({novel_tri}% de trigrams novos) confirmam que os juízes "
            f"reformulam significativamente o texto. É sumarização abstrativa, não extrativa."
        ),
        "histograma": (
            f"A maioria das fundamentações concentra-se entre {list(hist_fund.keys())[0]} e "
            f"{list(hist_fund.keys())[2]} palavras, mas a cauda longa revela casos de "
            f"alta complexidade com textos muito extensos."
        ),
        "temporal": (
            f"Quantidade de processos registrados no sistema a cada ano, "
            f"abrangendo o período de {periodo.get('data_mais_antiga', '—')[:4]} a "
            f"{periodo.get('data_mais_recente', '—')[:4]}."
        ),
    }

    fase4 = {
        "nome": "Estatísticas Descritivas",
        "descricao": "Análise quantitativa do corpus: distribuições, compressão e grau de abstratividade.",
        "narrativa": narrativa_f4,
        "status": "concluida",
        "script": "pipeline/04_estatisticas.py",
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
        "scatter_compressao": scatter_data,
        "wordcloud": wordcloud_data,
        "dicas": dicas,
        "artefatos": [
            {"nome": "dataset_treino.jsonl", "tamanho_mb": _file_size_mb(treino_path), "tipo": "entrada", "conteudo": "{contents: [{role, parts}]} anonimizado"},
            {"nome": "estatisticas_corpus.json", "tamanho_mb": None, "tipo": "saida", "conteudo": "Métricas, distribuições, funil"},
        ],
    }

    # --- Montar resultado final ---
    resultado = {
        "meta": {
            "gerado_em": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "titulo_pesquisa": "Geração Abstrativa de Ementas Judiciais",
            "autor": "AUTOR_ANONIMIZADO",
            "unidade_de_medida": "palavras (split por espaço)",
            "pipeline_total_segundos": timing.get("pipeline_total"),
        },
        "fases": {
            "fase1_ingestao": fase1,
            "fase2_higienizacao": fase2,
            "fase3_anonimizacao": fase3,
            "fase4_estatisticas": fase4,
            "fase5_finetuning": {"nome": "Fine-Tuning", "status": "pendente", "scripts": ["pipeline/05_finetuning_gemini.py", "pipeline/05_finetuning_qwen.py"]},
            "fase6_baseline": {"nome": "Baseline Zero-Shot", "status": "pendente", "scripts": ["pipeline/06_baseline_gemini.py", "pipeline/06_baseline_qwen.py"]},
            "fase7_avaliacao": {"nome": "Avaliação", "status": "pendente", "script": "Apresentacao_Dissertacao_Colab.ipynb"},
        },
    }

    # --- Gravar JSON ---
    log.info("Gravando resultados em %s ...", output_path)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)

    # Cópia para GitHub Pages (docs/)
    docs_data = Path("docs/data")
    docs_data.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(output_path, docs_data / output_path.name)

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
