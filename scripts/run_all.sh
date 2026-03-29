#!/usr/bin/env bash
# run_all.sh — Executa as Fases 1-4 do pipeline sequencialmente
# Uso: bash scripts/run_all.sh  (a partir da raiz do projeto)
set -euo pipefail

# Ensure Homebrew binaries (including pg_restore) are in PATH
# Needed because keg-only formulae (e.g. postgresql@16) may not be symlinked
export PATH="/opt/homebrew/bin:/opt/homebrew/opt/postgresql@16/bin:$PATH"

TIMING_FILE="data/.pipeline_timing.json"
mkdir -p data

echo "=== Pipeline de Dados — Dissertação ==="
echo ""

t_pipeline=$(date +%s)

echo "[1/5] Fase 1: Ingestão (pg_restore → SQLite + JSON)"
t0=$(date +%s)
python3 -m pipeline.fases1_4.fase01_ingestao
t_fase1=$(( $(date +%s) - t0 ))
echo ""

echo "[2/5] Fase 2: Higienização (Regex)"
t0=$(date +%s)
python3 -m pipeline.fases1_4.fase02_higienizacao
t_fase2=$(( $(date +%s) - t0 ))
echo ""

echo "[3/5] Fase 3: Anonimização (LGPD) + JSONL"
t0=$(date +%s)
python3 -m pipeline.fases1_4.fase03_anonimizacao
t_fase3=$(( $(date +%s) - t0 ))
echo ""

echo "[4/5] Auditoria LGPD"
t0=$(date +%s)
python3 -m pipeline.ferramentas.auditoria
t_audit=$(( $(date +%s) - t0 ))
echo ""

echo "[5/5] Fase 4: Estatísticas Descritivas do Corpus"
t0=$(date +%s)
python3 -m pipeline.fases1_4.fase04_estatisticas
t_fase4=$(( $(date +%s) - t0 ))
echo ""

t_total=$(( $(date +%s) - t_pipeline ))

cat > "$TIMING_FILE" <<EOF
{
  "fase1_ingestao": $t_fase1,
  "fase2_higienizacao": $t_fase2,
  "fase3_anonimizacao": $t_fase3,
  "auditoria_lgpd": $t_audit,
  "fase4_estatisticas": $t_fase4,
  "pipeline_total": $t_total
}
EOF

echo "Timing salvo em $TIMING_FILE"
echo "=== Pipeline concluído em ${t_total}s. ==="
