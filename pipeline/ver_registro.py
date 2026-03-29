"""
ver_registro.py — Inspeciona um registro do dataset JSONL

Uso (a partir da raiz do projeto):
    python3 pipeline/ver_registro.py [índice] [arquivo]

Exemplos:
    python3 pipeline/ver_registro.py 0               # 1º registro do dataset de teste
    python3 pipeline/ver_registro.py 42              # registro 42 do dataset de teste
    python3 pipeline/ver_registro.py 0 treino        # 1º registro do dataset de treino
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from jsonl_utils import extrair_fundamentacao_e_ementa

# Argumentos: índice (padrão 0) e dataset (padrão "teste")
idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
dataset = sys.argv[2] if len(sys.argv) > 2 else "teste"

paths = {
    "teste": Path("data/dataset_teste.jsonl"),
    "treino": Path("data/dataset_treino.jsonl"),
}

path = paths.get(dataset)
if path is None or not path.exists():
    print(f"[ERRO] Dataset '{dataset}' não encontrado. Use: teste | treino")
    sys.exit(1)

lines = path.read_text(encoding="utf-8").splitlines()

if idx < 0 or idx >= len(lines):
    print(f"[ERRO] Índice {idx} fora do intervalo (0–{len(lines)-1})")
    sys.exit(1)

obj = json.loads(lines[idx])
fundamentacao, ementa = extrair_fundamentacao_e_ementa(obj)

sep = "─" * 60
print(f"\n{sep}")
print(f"  Registro {idx} / {len(lines)-1}  —  {path.name}")
print(sep)
print("\n📄 FUNDAMENTAÇÃO:")
print(fundamentacao)
print(f"\n📋 EMENTA:")
print(ementa)
print(sep)
