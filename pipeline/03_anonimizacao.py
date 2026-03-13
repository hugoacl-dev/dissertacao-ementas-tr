"""
03_anonimizacao.py — Fase 3: Anonimização (LGPD) e Formatação JSONL

Substitui Informações Pessoalmente Identificáveis (PII) por tokens neutros,
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

# Municípios da Paraíba com maior frequência no corpus (~40 dos 223 totais)
# Suficiente para cobertura de ~95% das ocorrências geográficas no corpus.
_CIDADES_PB = re.compile(
    r"\b(?:em\s+|no\s+|na\s+|de\s+)?"
    r"(?:João Pessoa|Campina Grande|Santa Rita|Patos|Bayeux|Sousa|"
    r"Cajazeiras|Guarabira|Cabedelo|Sapé|Mamanguape|Queimadas|São Bento|"
    r"Monteiro|Esperança|Pombal|Catolé do Rocha|Alagoa Grande|Pedras de Fogo|"
    r"Lagoa Seca|Santa Luzia|São João do Rio do Peixe|Itaporanga|Rio Tinto|"
    r"Princesa Isabel|Areia|Mari|Jacaraú|Bananal|Conde|São Miguel de Taipu|"
    r"Boa Vista|Boqueirão|Coremas|Cuité|Itabaiana|Lucena|Picuí|Pitimbu|"
    r"Solânea|Taperoá|Umbuzeiro|Recife|Natal|Maceió|Fortaleza|"
    r"Paraíba|PB|zona rural|sítio corredor)\b",
    re.IGNORECASE,
)

_RE_CPF = re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b")
_RE_CNPJ = re.compile(r"\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b")
_RE_CEP = re.compile(r"\b\d{5}-\d{3}\b")
_RE_CONTA = re.compile(r"\b\d{4,5}-\d{1}\b")

# Honorífico + nome — ex: "Dr. PESSOA TESTE ALFA", "autora PESSOA TESTE BETA"
_HONORIFICOS = (
    r"(?:Sr\.|Sra\.|Dr\.|Dra\.|advogado|advogada|autor|autora|"
    r"réu|ré|juiz|juíza|relator|relatora|desembargador|desembargadora)"
)
_RE_NOME_HONORÍFICO = re.compile(
    rf"(?i)\b({_HONORIFICOS})\s+([A-ZÀ-Ÿ][a-zà-ÿ]+\s*){{1,4}}[A-ZÀ-Ÿ][a-zà-ÿ]+\b"
)

# Nome próprio isolado: 3+ palavras com inicial maiúscula consecutivas
_RE_NOME_PROPRIO = re.compile(
    r"\b([A-ZÀ-Ÿ][a-zà-ÿ]+\s+){2,5}[A-ZÀ-Ÿ][a-zà-ÿ]+\b"
)

# Logradouros com âncora numérica — Rua X, nº 10 / Av. Y, CEP 58000
_RE_LOGRADOURO = re.compile(
    r"\b(?:rua|r\.|avenida|av\.|praça|pça\.|travessa|tv\.|rodovia|br-\d+|sítio|fazenda)"
    r"[^.,]{1,60}?"
    r"(?:n[º°o]?\s*\d+|cep\s*\d|bloco\s*\d|lote\s*\d)",
    re.IGNORECASE,
)

# Empresa: sufixo jurídico inequívoco (Ltda, S/A, EPP) antecedido por 2–40 chars
# Mais restrito do que antes para evitar falsos positivos com "ME" e "SA".
_RE_EMPRESA_LTDA = re.compile(
    r"\b[\w\s\-&À-ÿ]{2,40}\s+(?:Ltda\.?|LTDA\.?|S/A|S\.A\.?|EPP|Empreendimentos|Comércio|Serviços)(?:\b|\s)",
    re.IGNORECASE,
)
# "ME" e "SA" são sufixos curtos demais — só substitui quando precedidos de
# nome que começa com maiúscula (sinal de razão social).
_RE_EMPRESA_ME = re.compile(
    r"\b[A-ZÀ-Ÿ][\w\s\-&À-ÿ]{2,38}\s+(?:ME|SA)\b"
)

# Empresa com prefixo já anonimizado — ex: "[NOME_PESSOA] LTDA" / "[LOCAL_OCULTADO] Ltda."
# Ocorre quando o nome foi substituído por token mas o sufixo jurídico sobreviveu.
_RE_EMPRESA_POS_TOKEN = re.compile(
    r"(?:\[(?:NOME_PESSOA|NOME_OCULTADO|LOCAL_OCULTADO|EMPRESA)\])\s+"
    r"(?:Ltda\.?|LTDA\.?|EPP|S/A|S\.A\.?|ME|SA)\b",
    re.IGNORECASE,
)

# Passe final: any surviving Ltda/LTDA/EPP is unambiguously a company suffix.
# Applied last so it only catches what slipped through all previous patterns.
_RE_EMPRESA_SUFIXO_RESIDUAL = re.compile(
    r"\b(?:Ltda\.?|LTDA\.?|EPP)\b"
)



# ---------------------------------------------------------------------------
# Lógica de anonimização
# ---------------------------------------------------------------------------


@dataclass
class AnonimizationStats:
    """Contadores de tokens PII substituídos ao longo de toda a base."""

    cpfs: int = 0
    cnpjs: int = 0
    ceps: int = 0
    contas: int = 0
    nomes_honorificos: int = 0
    nomes_proprios: int = 0
    locais: int = 0
    logradouros: int = 0
    empresas: int = 0

    @property
    def total(self) -> int:
        return (
            self.cpfs + self.cnpjs + self.ceps + self.contas
            + self.nomes_honorificos + self.nomes_proprios
            + self.locais + self.logradouros + self.empresas
        )


def _count(pattern: re.Pattern[str], texto: str) -> int:
    return len(pattern.findall(texto))


def anonimizar_texto(texto: str | None, stats: AnonimizationStats | None = None) -> str:
    """Substitui PII no texto por tokens genéricos, em conformidade com a LGPD.

    Camadas de proteção (em ordem de aplicação):
        1. CPF formatado → [CPF]
        2. CNPJ formatado → [CNPJ]
        3. CEP numérico → [CEP]
        4. Conta bancária com dígito → [CONTA-DIGITO]
        5. Nomes após honorífico → [NOME_OCULTADO]
        6. Nomes próprios isolados (3+ palavras capitalizadas) → [NOME_PESSOA]
        7. Municípios da Paraíba (corpus representativo) → [LOCAL_OCULTADO]
        8. Logradouros com âncora numérica → [ENDEREÇO_COMPLETO]
        9. Razões sociais com sufixo jurídico → [EMPRESA]

    Args:
        texto: String a ser anonimizada; pode ser None.
        stats: Objeto de contagem; se fornecido, é atualizado in-place.

    Returns:
        String com PII substituída por tokens neutros.
    """
    if not texto:
        return ""

    if stats:
        stats.cpfs += _count(_RE_CPF, texto)
        stats.cnpjs += _count(_RE_CNPJ, texto)
        stats.ceps += _count(_RE_CEP, texto)
        stats.contas += _count(_RE_CONTA, texto)
        stats.nomes_honorificos += _count(_RE_NOME_HONORÍFICO, texto)
        stats.nomes_proprios += _count(_RE_NOME_PROPRIO, texto)
        stats.locais += _count(_CIDADES_PB, texto)
        stats.logradouros += _count(_RE_LOGRADOURO, texto)
        stats.empresas += _count(_RE_EMPRESA_LTDA, texto) + _count(_RE_EMPRESA_ME, texto)

    texto = _RE_CPF.sub("[CPF]", texto)
    texto = _RE_CNPJ.sub("[CNPJ]", texto)
    texto = _RE_CEP.sub("[CEP]", texto)
    texto = _RE_CONTA.sub("[CONTA-DIGITO]", texto)
    texto = _RE_NOME_HONORÍFICO.sub(r"\1 [NOME_OCULTADO]", texto)
    texto = _RE_NOME_PROPRIO.sub("[NOME_PESSOA]", texto)
    texto = _CIDADES_PB.sub(" [LOCAL_OCULTADO] ", texto)
    texto = _RE_LOGRADOURO.sub("[ENDEREÇO_COMPLETO]", texto)
    texto = _RE_EMPRESA_LTDA.sub("[EMPRESA] ", texto)
    texto = _RE_EMPRESA_ME.sub("[EMPRESA]", texto)
    # Captura sufixos que sobreviveram após anonimização do nome da empresa
    texto = _RE_EMPRESA_POS_TOKEN.sub("[EMPRESA]", texto)
    # Passe final: captura qualquer Ltda/LTDA/EPP remanescente (inequivocamente sufixo de empresa)
    texto = _RE_EMPRESA_SUFIXO_RESIDUAL.sub("[EMPRESA]", texto)

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
        Estatísticas de tokens PII substituídos.
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
        "Anonimização concluída. Total de tokens PII substituídos: %d "
        "(CPFs: %d | CNPJs: %d | Nomes: %d | Locais: %d | Empresas: %d)",
        stats.total,
        stats.cpfs,
        stats.cnpjs,
        stats.nomes_honorificos + stats.nomes_proprios,
        stats.locais,
        stats.empresas + stats.logradouros,
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
