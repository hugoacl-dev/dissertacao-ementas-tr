#!/usr/bin/env bash
# run_all.sh — Executa as Fases 1-3 do pipeline sequencialmente
# Uso: bash pipeline/run_all.sh  (a partir da raiz do projeto)
set -euo pipefail

echo "=== Pipeline de Dados — Dissertação TR-ONE ==="
echo ""

echo "[1/5] Fase 1: Ingestão (pg_restore → SQLite + JSON)"
python3 pipeline/01_ingestao.py
echo ""

echo "[2/5] Fase 2: Higienização (Regex)"
python3 pipeline/02_higienizacao.py
echo ""

echo "[3/5] Fase 3: Anonimização (LGPD) + JSONL"
python3 pipeline/03_anonimizacao.py
echo ""

echo "[4/5] Auditoria LGPD"
python3 pipeline/audit.py
echo ""

echo "[5/5] Fase 4: Estatísticas Descritivas do Corpus"
python3 pipeline/04_estatisticas.py
echo ""

echo "=== Pipeline concluído. ==="
