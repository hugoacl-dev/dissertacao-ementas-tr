"""
03_anonimizacao.py — Fase 3: Anonimização (LGPD) e Formatação JSONL

Substitui dados pessoais (LGPD) por tokens neutros,
formata os pares {fundamentacao, ementa} no padrão multiturno conversacional
exigido pela API de fine-tuning do Gemini e do Qwen, e realiza a divisão
treino/teste por critério cronológico.

Entradas : data/dados_limpos.json
           (deve conter o campo 'data_cadastro' exportado pela Fase 1)
Saídas   : data/dataset_treino.jsonl, data/dataset_teste.jsonl
Executar a partir da raiz do projeto: python3 pipeline/03_anonimizacao.py

Divisão treino/teste: CRONOLÓGICA por 'data_cadastro'.
As 90% decisões mais antigas vão para treino; as 10% mais recentes para teste.
Isso evita temporal leakage — o modelo não treina em dados futuros relativos
ao conjunto de avaliação — seguindo o protocolo do SLDS (Rolshoven et al.,
2025). NÃO é utilizado shuffle aleatório.

Formato JSONL (compatível com Gemini Supervised Fine-Tuning e Unsloth/Qwen):
  {"contents": [{"role": "user", "parts": [{"text": "..."}]},
                {"role": "model", "parts": [{"text": "..."}]}]}

Nota: a API de tuning do Gemini NÃO suporta role "system" no array `contents`.
A instrução de sistema é embutida no turno `user`.
"""
from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from data_cadastro_utils import validar_e_converter_data_cadastro
from jsonl_utils import MARCADOR_FUNDAMENTACAO


# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

log = logging.getLogger(__name__)

INPUT_PATH = Path("data/dados_limpos.json")
TRAIN_PATH = Path("data/dataset_treino.jsonl")
TEST_PATH = Path("data/dataset_teste.jsonl")

TEST_SIZE: float = 0.10  # 10% para avaliação da banca; 90% para fine-tuning

# Divisão CRONOLÓGICA — não utiliza shuffle aleatório.
# Os 90% de decisões mais antigas (por data_cadastro) vão para treino;
# os 10% mais recentes vão para teste. Isso evita temporal leakage.
# Referência: SLDS (Rolshoven et al., 2025).

# [C14] Comprimento mínimo pós-anonimização — registros cujo conteúdo era
# majoritariamente dados pessoais ficam apenas com tokens como [NOME_PESSOA] [LOCAL_OCULTADO].
MIN_FUND_ANON_LEN: int = 50
MIN_EMENTA_ANON_LEN: int = 20

# Instrução de sistema carregada de arquivo externo para facilitar edição
# e reutilização nas Fases 5, 6 e 7 (ver nota no cabeçalho sobre role "system").
_SYSTEM_PROMPT_PATH = Path(__file__).parent / "system_prompt.txt"
_INSTRUCAO_SISTEMA = _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()


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
    "juizado especial",
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


def _contar_ocorrencias(pattern: re.Pattern[str], texto: str) -> int:
    """Conta quantas ocorrências de um padrão existem no texto."""
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
        7. Back-reference: fragmentos de 2+ palavras dos nomes já capturados → [NOME_PESSOA]
        8. Nomes próprios isolados (3+ palavras) → [NOME_PESSOA]
        9. Logradouros com âncora numérica → [ENDEREÇO_COMPLETO]

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
        stats.cpfs += _contar_ocorrencias(_RE_CPF, texto)
        stats.cnpjs += _contar_ocorrencias(_RE_CNPJ, texto)
        stats.npus += _contar_ocorrencias(_RE_NPU, texto)
        stats.contas += _contar_ocorrencias(_RE_CONTA, texto)
        stats.emails += _contar_ocorrencias(_RE_EMAIL, texto)
        stats.telefones += _contar_ocorrencias(_RE_TELEFONE, texto)
        stats.nomes_honorificos += _contar_ocorrencias(_RE_NOME_HONORIFICO, texto)
        stats.logradouros += _contar_ocorrencias(_RE_LOGRADOURO, texto)

    texto = _RE_CPF.sub("[CPF]", texto)
    texto = _RE_CNPJ.sub("[CNPJ]", texto)
    texto = _RE_NPU.sub("[NPU]", texto)
    texto = _RE_CONTA.sub("[CONTA-DIGITO]", texto)
    texto = _RE_EMAIL.sub("[EMAIL]", texto)
    texto = _RE_TELEFONE.sub("[TELEFONE]", texto)

    # --- Etapa 6: Nomes com honorífico (captura + coleta para back-ref) ---
    # Antes de substituir, coletar os nomes completos para segunda passagem.
    nomes_coletados: set[str] = set()
    for m in _RE_NOME_HONORIFICO.finditer(texto):
        # O grupo 0 é "título Nome Sobrenome"; removemos o título (grupo 1).
        nome_completo = m.group(0)
        titulo = m.group(1)
        # Extrair só a parte do nome (sem o título)
        nome_sem_titulo = nome_completo[len(titulo):].strip()
        if nome_sem_titulo:
            nomes_coletados.add(nome_sem_titulo)

    texto = _RE_NOME_HONORIFICO.sub(r"\1 [NOME_OCULTADO]", texto)

    # --- Etapa 7: Back-reference — fragmentos de 2+ palavras dos nomes ---
    # Gera substrings de 2+ palavras consecutivas de cada nome coletado.
    # Ex.: "ALFA BETA GAMA" → {"ALFA BETA", "Silva Santos"}
    fragmentos: set[str] = set()
    for nome in nomes_coletados:
        partes = nome.split()
        if len(partes) >= 2:
            # Gerar todas as janelas de 2+ palavras consecutivas
            for tamanho in range(2, len(partes) + 1):
                for inicio in range(len(partes) - tamanho + 1):
                    fragmento = " ".join(partes[inicio:inicio + tamanho])
                    fragmentos.add(fragmento)

    # Substituir fragmentos (mais longos primeiro para evitar substituição parcial)
    nomes_backref_antes = texto.count("[NOME_PESSOA]")
    for fragmento in sorted(fragmentos, key=len, reverse=True):
        texto = texto.replace(fragmento, "[NOME_PESSOA]")
    if stats:
        stats.nomes_proprios += texto.count("[NOME_PESSOA]") - nomes_backref_antes

    # --- Etapa 8: Nomes próprios isolados (3+ palavras, regex original) ---
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
                            f"{MARCADOR_FUNDAMENTACAO}{fundamentacao}"
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


def _carregar_registros(path: Path) -> pd.DataFrame:
    """Carrega `dados_limpos.json` em um DataFrame pandas."""
    return pd.read_json(path, orient="records", dtype=False)


def _anonimizar_registros(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, AnonimizationStats]:
    """Anonimiza todos os registros via apply() e retorna o DataFrame + stats."""
    stats = AnonimizationStats()

    def _processar_linha(row: pd.Series) -> pd.Series:
        fund = anonimizar_texto(row["fundamentacao"], stats)
        ementa = anonimizar_texto(row["ementa"], stats)

        # [C14] Descarta registros destruidos pela anonimizacao.
        if len(fund) < MIN_FUND_ANON_LEN or len(ementa) < MIN_EMENTA_ANON_LEN:
            stats.descartados_pos_anon += 1
            return pd.Series({"exemplo": None})

        exemplo = formatar_exemplo_gemini(fund, ementa)
        exemplo["data_cadastro"] = row.get("data_cadastro", "")
        return pd.Series({"exemplo": exemplo})

    # apply() e filtro de nulos pós-anon
    resultado = df[["fundamentacao", "ementa", "data_cadastro"]].apply(
        lambda row: _processar_linha(row), axis=1
    )

    # Reconstruir DataFrame de exemplos descartando os linhas marcadas como None
    exemplos_list: list[dict] = []
    for i, ex in enumerate(resultado["exemplo"], start=1):
        if ex is not None:
            exemplos_list.append(ex)
        if i % 5_000 == 0:
            log.info("%d/%d registros anonimizados...", i, len(df))

    return pd.DataFrame({"exemplo": exemplos_list}), stats


def _dividir_e_gravar(
    df_exemplos: pd.DataFrame,
    train_path: Path,
    test_path: Path,
    test_size: float,
) -> tuple[int, int]:
    """Divide cronologicamente e grava os datasets de treino e teste.

    Usa `data_cadastro` validada como datetime para ordenar e fazer o split
    cronológico. Datas inválidas abortam a execução para preservar a
    integridade metodológica do experimento.

    A divisão é feita por 'data_cadastro' (campo preservado desde a Fase 1):
    as decisões mais antigas vão para treino e as mais recentes para teste.
    Isso evita temporal leakage — garantia de que o modelo não treina em
    dados futuros relativos ao conjunto de avaliação.

    Referência: SLDS (Rolshoven et al., 2025).

    Returns:
        Tupla (qtd_treino, qtd_teste).
    """
    datas_brutas = pd.Series(
        [ex.get("data_cadastro") for ex in df_exemplos["exemplo"]],
        name="data_cadastro",
    )
    datas = validar_e_converter_data_cadastro(
        datas_brutas,
        contexto="Fase 3 / split cronológico",
    )
    df_com_data = df_exemplos.copy()
    df_com_data["_data_cadastro"] = datas

    df_ordenado = df_com_data.sort_values("_data_cadastro", kind="stable").reset_index(drop=True)

    qtd_total = len(df_ordenado)
    qtd_teste = int(qtd_total * test_size)

    df_treino = df_ordenado.iloc[:-qtd_teste] if qtd_teste > 0 else df_ordenado
    df_teste  = df_ordenado.iloc[-qtd_teste:] if qtd_teste > 0 else df_ordenado.iloc[0:0]

    log.info(
        "Divisão CRONOLÓGICA: %d para TREINO (%.0f%%) | %d para TESTE (%.0f%%)",
        len(df_treino), (1 - test_size) * 100,
        len(df_teste),  test_size * 100,
    )

    # Validação cronológica
    treino_ultima = df_treino["_data_cadastro"].iloc[-1].isoformat() if len(df_treino) else "?"
    teste_primeira = df_teste["_data_cadastro"].iloc[0].isoformat()  if len(df_teste)  else "?"
    log.info(
        "Validação cronológica: treino até %s | teste a partir de %s",
        treino_ultima, teste_primeira,
    )

    for path, df_split in [(train_path, df_treino), (test_path, df_teste)]:
        log.info("Gravando %s (%d exemplos)...", path, len(df_split))
        with path.open("w", encoding="utf-8") as f:
            for ex in df_split["exemplo"]:
                # Remove data_cadastro antes de gravar — não é campo do JSONL de treino.
                ex_sem_data = {k: v for k, v in ex.items() if k != "data_cadastro"}
                f.write(json.dumps(ex_sem_data, ensure_ascii=False, separators=(",", ":")) + "\n")

    return len(df_treino), len(df_teste)


def gerar_datasets(
    input_path: Path = INPUT_PATH,
    train_path: Path = TRAIN_PATH,
    test_path: Path = TEST_PATH,
    test_size: float = TEST_SIZE,
) -> AnonimizationStats:
    """Pipeline completo da Fase 3: anonimização LGPD + formatação JSONL + split cronológico.

    A divisão treino/teste é feita por critério CRONOLÓGICO (campo 'data_cadastro'),
    não por shuffle aleatório. As 90% decisões mais antigas vão para treino;
    as 10% mais recentes vão para teste. Isso evita temporal leakage.

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
    df = _carregar_registros(input_path)
    log.info("%d registros carregados.", len(df))

    log.info("Aplicando anonimização LGPD...")
    df_exemplos, stats = _anonimizar_registros(df)

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

    _dividir_e_gravar(df_exemplos, train_path, test_path, test_size)

    # Persistir contagens PII para consumo pelo dashboard (04_estatisticas.py)
    pii_stats_path = Path("data/.anonimizacao_stats.json")
    pii_payload = {
        "CPF": stats.cpfs,
        "CNPJ": stats.cnpjs,
        "NPU": stats.npus,
        "CONTA_DIGITO": stats.contas,
        "EMAIL": stats.emails,
        "TELEFONE": stats.telefones,
        "NOME_OCULTADO": stats.nomes_honorificos,
        "NOME_PESSOA": stats.nomes_proprios,
        "ENDERECO_COMPLETO": stats.logradouros,
        "total": stats.total,
        "descartados_pos_anon": stats.descartados_pos_anon,
    }
    with pii_stats_path.open("w", encoding="utf-8") as f:
        json.dump(pii_payload, f, ensure_ascii=False, indent=2)
    log.info("Stats PII salvas em %s", pii_stats_path)

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
    except (FileNotFoundError, OSError, ValueError) as exc:
        log.critical("Execução interrompida: %s", exc)
        sys.exit(1)
