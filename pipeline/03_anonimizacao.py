"""
03_anonimizacao.py — Fase 3: Anonimização (LGPD) e Formatação JSONL

Substitui dados pessoais (LGPD) por tokens neutros,
formata os pares {fundamentacao, ementa} no padrão multiturno conversacional
exigido pela API de fine-tuning do Gemini, e realiza a divisão treino/teste.

Entradas : data/dados_limpos.json
Saídas   : data/dataset_treino.jsonl, data/dataset_teste.jsonl
Executar a partir da raiz do projeto: python3 pipeline/03_anonimizacao.py

Formato JSONL (compatível com Gemini Supervised Fine-Tuning):
  {"contents": [{"role": "user", "parts": [{"text": "..."}]},
                {"role": "model", "parts": [{"text": "..."}]}]}

Nota: a API de tuning NÃO suporta role "system" no array `contents`.
A instrução de sistema é embutida no turno `user`.
"""
from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from random import Random

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

log = logging.getLogger(__name__)

INPUT_PATH = Path("data/dados_limpos.json")
TRAIN_PATH = Path("data/dataset_treino.jsonl")
TEST_PATH = Path("data/dataset_teste.jsonl")

TEST_SIZE: float = 0.10  # 10% para avaliação da banca; 90% para fine-tuning
RANDOM_SEED: int = 42    # Seed fixa para divisão reproduzível Treino/Teste

# [C14] Comprimento mínimo pós-anonimização — registros cujo conteúdo era
# majoritariamente dados pessoais ficam apenas com tokens como [NOME_PESSOA] [LOCAL_OCULTADO].
MIN_FUND_ANON_LEN: int = 50
MIN_EMENTA_ANON_LEN: int = 20

# Instrução de sistema embutida no prompt do usuário (ver nota no cabeçalho)
_INSTRUCAO_SISTEMA = (
    "Você é um assistente jurídico experiente que auxilia juízes a escreverem "
    "Ementas Judiciais, que são resumos curtos, estruturados e objetivos do que "
    "foi decidido numa fundamentação (voto). Ao ser fornecida a fundamentação de um "
    "Recurso, você deve responder única e exclusivamente com o texto da Ementa correspondente."
)


# ---------------------------------------------------------------------------
# Padrões de anonimização pré-compilados
# ---------------------------------------------------------------------------
# Nota sobre escopo LGPD: a Lei 13.709/2018 protege dados pessoais de
# PESSOA NATURAL (Art. 1º, Art. 5º-I). Razões sociais (PJ), municípios
# e CEPs são informação pública — não são dados pessoais e não são anonimizados.
# ---------------------------------------------------------------------------

_RE_CPF = re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b")
_RE_CNPJ = re.compile(r"\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b")

# NPU (Número Único do Processo) — formato CNJ: NNNNNNN-DD.AAAA.J.TR.OOOO
# Embora públicos, permitem localizar processos e identificar partes.
_RE_NPU = re.compile(r"\b\d{7}-\d{2}\.\d{4}\.\d{1,2}\.\d{2}\.\d{4}\b")

# Conta bancária — exige âncora de contexto para evitar falsos positivos
# com números de benefício INSS (formato XXXXXXX-X).
_RE_CONTA = re.compile(
    r"(?:conta\s*(?:corrente|poupança|bancária)?|ag[êe]ncia|ag\.?|c/c|c\.c\.)"
    r"\s*(?:n[º°o]?\s*)?\d{4,5}-\d\b",
    re.IGNORECASE,
)

# E-mail — dado pessoal inequívoco (Art. 5º, I LGPD)
_RE_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")

# Telefone — exige separador (hífen, espaço ou parênteses) para evitar
# falsos positivos com números de benefício INSS (11 dígitos sem separador).
_RE_TELEFONE = re.compile(
    r"(?:\(\d{2}\)\s*\d{4,5}-?\d{4}|\d{2}\s+\d{4,5}-\d{4})\b"
)

# Honorífico + nome — separados em dois grupos por natureza jurídica.
#
# Grupo 1: PARTES PRIVADAS → anonimizar (dados pessoais LGPD)
#   autor/réu, advogado, Sr./Sra., Dr./Dra. (geralmente advogados), perito
#
# Grupo 2: AGENTES PÚBLICOS → preservar (Art. 93, IX CF — publicidade)
#   relator, desembargador, juiz, Ministro/Min. — nomes públicos em citações
#   de precedentes são referências bibliográficas, não dados pessoais.

_HONORIFICOS_PRIVADOS = (
    r"(?:Sr\.|Sra\.|Dr\.|Dra\.|advogado|advogada|autor|autora|"
    r"réu|ré|perito|perita)"
)
_RE_NOME_HONORIFICO = re.compile(
    rf"(?i)\b({_HONORIFICOS_PRIVADOS})\s+([A-ZÀ-Ÿ][a-zà-ÿ]+\s*){{1,4}}[A-ZÀ-Ÿ][a-zà-ÿ]+\b"
)

# Nome próprio isolado: 3+ palavras com inicial maiúscula consecutivas
_RE_NOME_PROPRIO = re.compile(
    r"\b([A-ZÀ-Ÿ][a-zà-ÿ]+\s+){2,5}[A-ZÀ-Ÿ][a-zà-ÿ]+\b"
)

# Termos jurídicos que NÃO devem ser substituídos por [NOME_PESSOA].
# Usa PREFIX MATCHING — "Tribunal Regional Federal da 5ª Região" dá match
# porque começa com "Tribunal Regional Federal" que está na lista.
_PREFIXOS_JURIDICOS = (
    # Tribunais e órgãos
    "Superior Tribunal",
    "Supremo Tribunal",
    "Tribunal Regional",
    "Tribunal de Justiça",
    "Turma Nacional",
    "Turma Recursal",
    "Juizado Especial",
    "Juizados Especiais",
    "Conselho Nacional",
    "Instituto Nacional",
    "Ministério Público",
    "Defensoria Pública",
    "Advocacia Geral",
    "Procuradoria Geral",
    "Caixa Econômica",
    "Banco Central",
    "Banco do Brasil",
    # Diplomas legais e institutos
    "Código de Processo",
    "Código Civil",
    "Código Penal",
    "Consolidação das Leis",
    "Constituição Federal",
    "Lei de Benefícios",
    "Lei Orgânica",
    "Regime Geral",
    "Fundo de Garantia",
    "Benefício de Prestação",
    # Expressões processuais capitalizadas
    "Recurso Cível",
    "Recurso Especial",
    "Recurso Extraordinário",
    "Embargos de Declaração",
    "Mandado de Segurança",
    "Ação Civil Pública",
    "Ação Penal",
    "Projeto de Lei",
    "Medida Provisória",
)


def _substituir_nome_proprio(match: re.Match) -> str:
    """Callback para re.sub: substitui nomes próprios preservando termos jurídicos.

    Usa prefix matching — se o texto capturado COMEÇA com um prefixo
    jurídico conhecido, é preservado. Isso cobre variações como
    "Tribunal Regional Federal da 5ª Região" sem exigir match exato.
    """
    texto_match = match.group(0)
    for prefixo in _PREFIXOS_JURIDICOS:
        if texto_match.startswith(prefixo):
            return texto_match
    return "[NOME_PESSOA]"


# Logradouros com âncora numérica — Rua X, nº 10 / Av. Y, CEP 58000
_RE_LOGRADOURO = re.compile(
    r"\b(?:rua|r\.|avenida|av\.|praça|pça\.|travessa|tv\.|rodovia|br-\d+|sítio|fazenda)"
    r"[^.,]{1,60}?"
    r"(?:n[º°o]?\s*\d+|cep\s*\d|bloco\s*\d|lote\s*\d)",
    re.IGNORECASE,
)



# ---------------------------------------------------------------------------
# Lógica de anonimização
# ---------------------------------------------------------------------------


@dataclass
class AnonimizationStats:
    """Contadores de tokens de dados pessoais substituídos ao longo de toda a base."""

    cpfs: int = 0
    cnpjs: int = 0
    npus: int = 0
    contas: int = 0
    emails: int = 0
    telefones: int = 0
    nomes_honorificos: int = 0
    nomes_proprios: int = 0
    logradouros: int = 0
    descartados_pos_anon: int = 0  # registros destruídos pela anonimização

    @property
    def total(self) -> int:
        return (
            self.cpfs + self.cnpjs + self.npus + self.contas
            + self.emails + self.telefones
            + self.nomes_honorificos + self.nomes_proprios
            + self.logradouros
        )


def _count(pattern: re.Pattern[str], texto: str) -> int:
    return len(pattern.findall(texto))


def anonimizar_texto(texto: str | None, stats: AnonimizationStats | None = None) -> str:
    """Substitui dados pessoais no texto por tokens genéricos, em conformidade com a LGPD.

    Camadas de proteção (em ordem de aplicação):
        1. CPF formatado → [CPF]
        2. CNPJ formatado → [CNPJ]
        3. Conta bancária (com âncora) → [CONTA-DIGITO]
        4. E-mail → [EMAIL]
        5. Telefone → [TELEFONE]
        6. Nomes após honorífico → [NOME_OCULTADO]
        7. Nomes próprios isolados (3+ palavras) → [NOME_PESSOA]
        8. Logradouros com âncora numérica → [ENDEREÇO_COMPLETO]

    Nota LGPD: cidades, CEPs e razões sociais (PJ) NÃO são anonimizados
    pois não constituem dados pessoais de pessoa natural (Art. 5º, I).

    Args:
        texto: String a ser anonimizada; pode ser None.
        stats: Objeto de contagem; se fornecido, é atualizado in-place.

    Returns:
        String com dados pessoais substituídos por tokens neutros.
    """
    if not texto:
        return ""

    # Contagem ANTES das substituições (exceto nomes próprios — ver abaixo).
    if stats:
        stats.cpfs += _count(_RE_CPF, texto)
        stats.cnpjs += _count(_RE_CNPJ, texto)
        stats.npus += _count(_RE_NPU, texto)
        stats.contas += _count(_RE_CONTA, texto)
        stats.emails += _count(_RE_EMAIL, texto)
        stats.telefones += _count(_RE_TELEFONE, texto)
        stats.nomes_honorificos += _count(_RE_NOME_HONORIFICO, texto)
        stats.logradouros += _count(_RE_LOGRADOURO, texto)

    texto = _RE_CPF.sub("[CPF]", texto)
    texto = _RE_CNPJ.sub("[CNPJ]", texto)
    texto = _RE_NPU.sub("[NPU]", texto)
    texto = _RE_CONTA.sub("[CONTA-DIGITO]", texto)
    texto = _RE_EMAIL.sub("[EMAIL]", texto)
    texto = _RE_TELEFONE.sub("[TELEFONE]", texto)
    texto = _RE_NOME_HONORIFICO.sub(r"\1 [NOME_OCULTADO]", texto)

    # Nomes próprios com exclusão de termos jurídicos via callback.
    # Contagem pós-sub para refletir apenas substituições efetivas.
    nomes_antes = texto.count("[NOME_PESSOA]")
    texto = _RE_NOME_PROPRIO.sub(_substituir_nome_proprio, texto)
    if stats:
        stats.nomes_proprios += texto.count("[NOME_PESSOA]") - nomes_antes

    texto = _RE_LOGRADOURO.sub("[ENDEREÇO_COMPLETO]", texto)

    return texto


# ---------------------------------------------------------------------------
# Formatação JSONL para Gemini Fine-Tuning
# ---------------------------------------------------------------------------


def formatar_exemplo_gemini(fundamentacao: str, ementa: str) -> dict:
    """Formata um par {fundamentacao, ementa} no formato multiturno do Gemini.

    O formato usa apenas roles "user" e "model" — a instrução de sistema é
    embutida no turno "user" pois a API de fine-tuning não suporta role
    "system" no array `contents`.

    Returns:
        Dicionário pronto para serialização JSONL.
    """
    return {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            f"{_INSTRUCAO_SISTEMA}\n\n"
                            f"Gere a ementa para a seguinte fundamentação:\n{fundamentacao}"
                        )
                    }
                ],
            },
            {
                "role": "model",
                "parts": [{"text": ementa}],
            },
        ]
    }


# ---------------------------------------------------------------------------
# Pipeline de geração dos datasets
# ---------------------------------------------------------------------------


def _carregar_registros(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _anonimizar_registros(
    registros: list[dict],
) -> tuple[list[dict], AnonimizationStats]:
    """Anonimiza todos os registros e retorna os exemplos formatados + stats."""
    stats = AnonimizationStats()
    exemplos: list[dict] = []

    for i, item in enumerate(registros, start=1):
        fund = anonimizar_texto(item["fundamentacao"], stats)
        ementa = anonimizar_texto(item["ementa"], stats)

        # [C14] Descarta registros destruídos pela anonimização.
        if len(fund) < MIN_FUND_ANON_LEN or len(ementa) < MIN_EMENTA_ANON_LEN:
            stats.descartados_pos_anon += 1
            continue

        exemplos.append(formatar_exemplo_gemini(fund, ementa))

        if i % 5_000 == 0:
            log.info("%d/%d registros anonimizados...", i, len(registros))

    return exemplos, stats


def _dividir_e_gravar(
    exemplos: list[dict],
    train_path: Path,
    test_path: Path,
    test_size: float,
    seed: int,
) -> tuple[int, int]:
    """Embaralha, divide e grava os datasets de treino e teste.

    Usa `random.Random(seed)` para não poluir o estado global do módulo `random`.

    Returns:
        Tupla (qtd_treino, qtd_teste).
    """
    rng = Random(seed)
    rng.shuffle(exemplos)

    qtd_teste = int(len(exemplos) * test_size)
    dataset_teste = exemplos[:qtd_teste]
    dataset_treino = exemplos[qtd_teste:]

    log.info(
        "Divisão (seed=%d): %d para TREINO (%.0f%%) | %d para TESTE (%.0f%%)",
        seed,
        len(dataset_treino),
        (1 - test_size) * 100,
        len(dataset_teste),
        test_size * 100,
    )

    for path, dataset in [(train_path, dataset_treino), (test_path, dataset_teste)]:
        log.info("Gravando %s (%d exemplos)...", path, len(dataset))
        with path.open("w", encoding="utf-8") as f:
            for ex in dataset:
                f.write(json.dumps(ex, ensure_ascii=False, separators=(",", ":")) + "\n")

    return len(dataset_treino), len(dataset_teste)


def gerar_datasets(
    input_path: Path = INPUT_PATH,
    train_path: Path = TRAIN_PATH,
    test_path: Path = TEST_PATH,
    test_size: float = TEST_SIZE,
    seed: int = RANDOM_SEED,
) -> AnonimizationStats:
    """Pipeline completo da Fase 3: anonimização LGPD + formatação JSONL + split.

    Returns:
        Estatísticas de tokens de dados pessoais substituídos.
    """
    if not input_path.exists():
        raise FileNotFoundError(
            f"Arquivo de entrada não encontrado: {input_path}\n"
            "Execute a Fase 2 primeiro (02_higienizacao.py)."
        )

    log.info("=== Fase 3: Anonimização (LGPD) e Formatação JSONL ===")
    log.info("Lendo %s ...", input_path)
    registros = _carregar_registros(input_path)
    log.info("%d registros carregados.", len(registros))

    log.info("Aplicando anonimização LGPD...")
    exemplos, stats = _anonimizar_registros(registros)

    log.info(
        "Anonimização concluída. Total de tokens de dados pessoais substituídos: %d "
        "(CPFs: %d | CNPJs: %d | NPUs: %d | Emails: %d | Telefones: %d | "
        "Nomes hon.: %d | Nomes próprios: %d | Logradouros: %d)",
        stats.total,
        stats.cpfs,
        stats.cnpjs,
        stats.npus,
        stats.emails,
        stats.telefones,
        stats.nomes_honorificos,
        stats.nomes_proprios,
        stats.logradouros,
    )

    if stats.descartados_pos_anon > 0:
        log.warning(
            "[C14] %d registros descartados por ficarem abaixo do comprimento "
            "mínimo após anonimização (fund < %d ou ementa < %d chars).",
            stats.descartados_pos_anon,
            MIN_FUND_ANON_LEN,
            MIN_EMENTA_ANON_LEN,
        )

    _dividir_e_gravar(exemplos, train_path, test_path, test_size, seed)
    log.info("=== Fase 3 finalizada com sucesso. ===")
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
    gerar_datasets()


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, OSError) as exc:
        log.critical("Execução interrompida: %s", exc)
        sys.exit(1)
