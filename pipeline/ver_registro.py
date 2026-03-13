"""
ver_registro.py — Inspeciona um registro do dataset JSONL

Uso (a partir da raiz do projeto):
    python3 pipeline/ver_registro.py [índice] [arquivo]

Exemplos:
    python3 pipeline/ver_registro.py 0               # 1º registro do dataset de teste
    python3 pipeline/ver_registro.py 42              # registro 42 do dataset de teste
    python3 pipeline/ver_registro.py 0 treino        # 1º registro do dataset de treino
"""
import json
import sys
from pathlib import Path

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
# O turno "user" contém: instrução de sistema + "\n\n" + fundamentação
user_text = obj["contents"][0]["parts"][0]["text"]
# Remove o prefixo da instrução de sistema (tudo antes do \n\n duplo final)
fundamentacao = user_text.split("\n\n", 1)[-1]
ementa = obj["contents"][1]["parts"][0]["text"]

sep = "─" * 60
print(f"\n{sep}")
print(f"  Registro {idx} / {len(lines)-1}  —  {path.name}")
print(sep)
print("\n📄 FUNDAMENTAÇÃO:")
print(fundamentacao)
print(f"\n📋 EMENTA:")
print(ementa)
print(sep)
