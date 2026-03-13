"""
02_higienizacao.py — Fase 2: Saneamento e Higienização Avançada

Aplica expressões regulares pré-compiladas sobre os textos brutos gerados
pela Fase 1, removendo **ruído estrutural** (HTML, IDs do PJe,
carimbos DJe, assinaturas de juízes). Datas e conteúdo de mérito são preservados.

Entradas : data/dados_brutos.json
Saídas   : data/dados_limpos.json
Executar a partir da raiz do projeto: python3 pipeline/02_higienizacao.py
"""
from __future__ import annotations

import html
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

log = logging.getLogger(__name__)

INPUT_PATH = Path("data/dados_brutos.json")
OUTPUT_PATH = Path("data/dados_limpos.json")

# Comprimento mínimo aceitável (em caracteres) após limpeza.
MIN_FUNDAMENTACAO_LEN: int = 50
MIN_EMENTA_LEN: int = 20


# ---------------------------------------------------------------------------
# Padrões de limpeza pré-compilados
# Compilar uma única vez no import evita recompilação a cada chamada.
# ---------------------------------------------------------------------------

_MESES = (
    r"(?:janeiro|fevereiro|março|abril|maio|junho|"
    r"julho|agosto|setembro|outubro|novembro|dezembro)"
)
_CIDADES_DATA = (
    r"(?:joão pessoa|campina grande|guarabira|patos|monteiro|sousa)"
)

_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    # Formato: (nome, pattern, substituição)
    # Datas são substituídas por [DATA] (não removidas) para preservar
    # informação semântica de mérito — prazos, óbitos, DIB de benefícios.

    # 1. Tags HTML — <p style="...">, <strong>, <ol>, <li> …
    ("html_tags", re.compile(r"<[^>]+>", re.IGNORECASE), " "),

    # 2. Metadados processuais — "Processo nº 0500148-54.2016.4.05.8200"
    (
        "metadados_processuais",
        re.compile(
            r"(?:processo|protocolo|autos)[\s\w]*n[º°o]?[\s:]*[\d\.\-]+(?:/\d+)?",
            re.IGNORECASE,
        ),
        " ",
    ),

    # 3. IDs do PJe — todos os formatos: "id. 48772689", "(id 48772 689)", "id.. 42000664", "Id. . 25146010"
    ("ids_pje", re.compile(r"\(?id[\s.]*[\s:]*\d{5,}\)?", re.IGNORECASE), " "),

    # 4. Carimbo DJe + linha inteira — "PROCESSO ELETRÔNICO DJe-s/n DIVULG 23-05-2024"
    (
        "dје_carimbo_longo",
        re.compile(
            r"(?:Data de Julgamento.*?|PROCESSO ELETRÔNICO.*?DJe[\w\-\s/]*(?:\d{4})?)",
            re.IGNORECASE,
        ),
        " ",
    ),

    # 5a. DJe simples — "DJe 26.09.2012", "DJe-s/n"
    ("dje_simples", re.compile(r"\bDJe[\-\s]*[s/n]*\b", re.IGNORECASE), " "),
    (
        "dje_data",
        re.compile(r"DJe\s+\d{1,2}[./]\d{1,2}[./]\d{2,4}", re.IGNORECASE),
        " ",
    ),

    # 5b. Carimbo "DIVULG" isolado (remanescentem do DJe sem prefixo completo)
    ("divulg_isolado", re.compile(r"\bDIVULG\b", re.IGNORECASE), " "),

    # 6a. Ação Civil Pública + DPU (formato completo)
    (
        "acp_dpu",
        re.compile(
            r"Ação Civil Pública[\s\-]+ACP\s*n[º°o]?\s*[\d\.\-]+[^.]*DPU",
            re.IGNORECASE,
        ),
        " ",
    ),


    # 7. Datas com cidade prefixada — "João Pessoa, 12 de março de 2021"
    # [C1] Substituição por token semântico [DATA] em vez de espaço vazio.
    (
        "data_cidade",
        re.compile(
            rf"{_CIDADES_DATA}[\s,]+\d{{1,2}}[\sde]+{_MESES}[\sde]+\d{{4}}",
            re.IGNORECASE,
        ),
        " [DATA] ",
    ),


    # 12. "Súmula de Julgamento: …" até o final da string
    (
        "sumula_julgamento",
        re.compile(r"Súmula de Julgamento:?.*$", re.IGNORECASE),
        " ",
    ),

    # 13. Honoríficos de juízes até o final da string
    (
        "honorificos_juiz",
        re.compile(
            r"(?:Juiz Federal|Juíza Federal|Relator|Relatora|Excelentíssimo|Desembargador)[\w\s]*?$",
            re.IGNORECASE,
        ),
        " ",
    ),

    # 14. Blocos isolados CAPSLOCK no final (assinaturas de juízes)
    (
        "capslock_assinatura",
        re.compile(r"\.?[ \n]*[A-ZÀ-Ÿ\s]{10,}$"),
        " ",
    ),
]

# Normalização de espaçamento (aplicada sempre ao final)
_RE_WHITESPACE = re.compile(r"[\r\n\t]+")
_RE_MULTI_SPACE = re.compile(r" {2,}")


# ---------------------------------------------------------------------------
# Lógica de limpeza
# ---------------------------------------------------------------------------


@dataclass
class CleaningStats:
    """Contadores de auditoria do processo de limpeza."""

    total_entrada: int = 0
    descartados_vazios: int = 0
    descartados_curtos: int = 0
    exportados: int = 0

    @property
    def taxa_retencao(self) -> float:
        if self.total_entrada == 0:
            return 0.0
        return self.exportados / self.total_entrada * 100


def _aplicar_patterns(texto: str) -> str:
    """Aplica todos os padrões de limpeza sequencialmente com sua substituição específica."""
    for _nome, pattern, repl in _PATTERNS:
        texto = pattern.sub(repl, texto)
    return texto


def _normalizar_espacos(texto: str) -> str:
    """Colapsa quebras de linha, tabs e espaços múltiplos em um único espaço."""
    texto = _RE_WHITESPACE.sub(" ", texto)
    texto = _RE_MULTI_SPACE.sub(" ", texto)
    return texto.strip()


def limpar_texto(texto: str | None) -> str:
    """Pipeline de limpeza de um único texto jurídico.

    Etapas:
        1. Decode de HTML entities (&nbsp;, &quot;, etc.)
        2. Remoção de tags HTML.
        3. Remoção de ruídos processuais via padrões pré-compilados.
        4. Normalização de espaços.

    Args:
        texto: String bruta extraída do dump; pode ser None.

    Returns:
        String higienizada (pode ser vazia se o texto era apenas ruído).
    """
    if not texto:
        return ""

    texto = html.unescape(texto)
    texto = _aplicar_patterns(texto)
    return _normalizar_espacos(texto)


# ---------------------------------------------------------------------------
# I/O e pipeline
# ---------------------------------------------------------------------------


def _iterar_registros(path: Path) -> Iterator[dict]:
    """Lê o JSON de entrada linha a linha de forma lazy (evita carregar tudo)."""
    with path.open("r", encoding="utf-8") as f:
        registros = json.load(f)
    yield from registros


def _limpar_registro(
    item: dict,
    stats: CleaningStats,
) -> dict | None:
    """Aplica limpeza a um único registro e atualiza os contadores.

    Returns:
        Dicionário limpo ou None se o registro deve ser descartado.
    """
    stats.total_entrada += 1

    fundamentacao_raw = item.get("fundamentacao", "")
    ementa_raw = item.get("ementa", "")

    if not fundamentacao_raw or not ementa_raw:
        stats.descartados_vazios += 1
        return None

    fundamentacao = limpar_texto(fundamentacao_raw)
    ementa = limpar_texto(ementa_raw)

    if not fundamentacao or not ementa:
        stats.descartados_vazios += 1
        return None

    if len(fundamentacao) < MIN_FUNDAMENTACAO_LEN or len(ementa) < MIN_EMENTA_LEN:
        stats.descartados_curtos += 1
        return None

    stats.exportados += 1
    return {"id": item.get("id"), "fundamentacao": fundamentacao, "ementa": ementa}


def processar(input_path: Path = INPUT_PATH, output_path: Path = OUTPUT_PATH) -> CleaningStats:
    """Pipeline da Fase 2: limpeza de todos os registros brutos.

    Args:
        input_path: Caminho para `dados_brutos.json`.
        output_path: Caminho de saída para `dados_limpos.json`.

    Returns:
        Estatísticas de execução.
    """
    if not input_path.exists():
        raise FileNotFoundError(
            f"Arquivo de entrada não encontrado: {input_path}\n"
            "Execute a Fase 1 primeiro (01_ingestao.py)."
        )

    log.info("=== Fase 2: Saneamento e Higienização (Regex) ===")
    log.info("Lendo registros de %s ...", input_path)

    stats = CleaningStats()
    dados_limpos: list[dict] = []

    for item in _iterar_registros(input_path):
        registro_limpo = _limpar_registro(item, stats)
        if registro_limpo:
            dados_limpos.append(registro_limpo)

        if stats.total_entrada % 5_000 == 0:
            log.info(
                "%d registros processados | Exportados até agora: %d",
                stats.total_entrada,
                stats.exportados,
            )

    log.info(
        "Saneamento concluído. Entrada: %d | Exportados: %d (%.1f%%) | "
        "Descartados (vazios): %d | Descartados (muito curtos): %d",
        stats.total_entrada,
        stats.exportados,
        stats.taxa_retencao,
        stats.descartados_vazios,
        stats.descartados_curtos,
    )

    log.info("Gravando %d registros limpos em %s ...", len(dados_limpos), output_path)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(dados_limpos, f, ensure_ascii=False, separators=(",", ":"))

    log.info("=== Fase 2 finalizada com sucesso. ===")
    return stats


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    processar()


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, OSError) as exc:
        log.critical("Execução interrompida: %s", exc)
        sys.exit(1)
