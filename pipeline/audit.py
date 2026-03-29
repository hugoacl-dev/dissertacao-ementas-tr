"""
audit.py — Validação pós-Fase 3: Auditoria de Vazamentos de Dados Pessoais

Varre os arquivos JSONL gerados pela Fase 3 em busca de resíduos de dados pessoais
que tenham escapado da limpeza e anonimização.

Executar a partir da raiz do projeto: python3 pipeline/audit.py

Lógica de extração de texto:
  - Turno "user": extrai apenas o conteúdo da fundamentação por meio do
    helper compartilhado do projeto, removendo a instrução embutida.
  - Turno "model": extrai o texto da ementa diretamente.

Isso evita falsos positivos causados pela instrução de sistema embutida.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from jsonl_utils import extrair_fundamentacao_e_ementa

# ---------------------------------------------------------------------------
# Padrões de auditoria
# ---------------------------------------------------------------------------

PATTERNS: dict[str, re.Pattern[str]] = {
    "1. CPF": re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b"),
    "2. CNPJ": re.compile(r"\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b"),
    "3. NPU (Número do Processo)": re.compile(
        r"\b\d{7}-\d{2}\.\d{4}\.\d{1,2}\.\d{2}\.\d{4}\b"
    ),
    "4. ID PJe": re.compile(r"(?i)\bid[\s.]*[\s:]*\d{5,}"),
    "5. E-mail": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "6. Telefone": re.compile(r"(?:\(\d{2}\)\s*\d{4,5}-?\d{4}|\d{2}\s+\d{4,5}-\d{4})\b"),
    # DJe apenas quando seguido de data (carimbo real)
    "7. DJe com data residual": re.compile(
        r"(?i)\bDJe\s+\d{1,2}[./]\d{1,2}[./]\d{2,4}"
    ),
}

# ---------------------------------------------------------------------------
# Extração dos campos de dado (excluindo instrução embutida)
# ---------------------------------------------------------------------------


def _extrair_textos_de_dado(obj: dict) -> str:
    """Extrai apenas os campos de conteúdo real (fundamentação e ementa).

    Usa o mesmo contrato do JSONL consumido pela Fase 4 e pelos utilitários
    do projeto, removendo a instrução embutida do turno "user".
    """
    fundamentacao, ementa = extrair_fundamentacao_e_ementa(obj)
    return " ".join(parte for parte in (fundamentacao, ementa) if parte)


# ---------------------------------------------------------------------------
# Auditoria
# ---------------------------------------------------------------------------


def audit(file_path: str | Path) -> bool:
    """Audita um arquivo JSONL em busca de vazamentos de dados pessoais.

    Args:
        file_path: Caminho para o arquivo JSONL a auditar.

    Returns:
        True se a base passou em todos os checks; False se houver vazamentos.
    """
    path = Path(file_path)
    print(f"\n{'=' * 50}")
    print(f"  Auditando: {path.name}")
    print(f"{'=' * 50}")

    counts: defaultdict[str, int] = defaultdict(int)
    examples: defaultdict[str, list[str]] = defaultdict(list)
    total = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
                text = _extrair_textos_de_dado(obj)
                for cat, regex in PATTERNS.items():
                    matches = regex.findall(text)
                    if matches:
                        counts[cat] += len(matches)
                        if len(examples[cat]) < 3:
                            m = matches[0]
                            examples[cat].append(m if isinstance(m, str) else m[0])
            except (json.JSONDecodeError, KeyError):
                pass

    print(f"Total de registros analisados: {total}\n")
    all_clean = True
    for cat in PATTERNS:
        if counts[cat] > 0:
            all_clean = False
            print(f"[!] VAZAMENTO - {cat}: {counts[cat]} ocorrências")
            print(f"    Amostras: {examples[cat]}")
        else:
            print(f"[OK] {cat}")

    if all_clean:
        print("\n✓ Base 100% limpa — nenhum vazamento de dados pessoais detectado.")
    else:
        print(f"\n✗ Vazamentos detectados em {sum(1 for c in counts.values() if c > 0)} categorias.")

    return all_clean


def main() -> None:
    targets = [Path("data/dataset_treino.jsonl"), Path("data/dataset_teste.jsonl")]
    all_passed = True
    for target in targets:
        if not target.exists():
            print(f"[ERRO] Arquivo não encontrado: {target}")
            all_passed = False
            continue
        passed = audit(target)
        all_passed = all_passed and passed

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
