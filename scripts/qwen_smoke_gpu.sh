#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAIN_LIMIT="${TRAIN_LIMIT:-16}"
CASES_LIMIT="${CASES_LIMIT:-4}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
SMOKE_ID="${SMOKE_ID:-qwen_smoke_$(date +%Y%m%d_%H%M%S)}"
HF_HOME_DEFAULT="${HF_HOME_DEFAULT:-/workspace/.cache/huggingface}"

TRAIN_DATASET_PATH="data/exploratorio/fase5/${SMOKE_ID}_dataset_treino.jsonl"
CHECKPOINT_DIR="data/exploratorio/fase5/${SMOKE_ID}_modelo_qwen_checkpoint"
QWEN_MANIFEST_PATH="data/exploratorio/fase5/${SMOKE_ID}_qwen_sft_manifest.json"
CASES_PATH="data/exploratorio/fase7/${SMOKE_ID}_casos_avaliacao.jsonl"
QWEN_ZS_OUTPUT_PATH="data/exploratorio/fase7/predicoes/${SMOKE_ID}_qwen_zero_shot.jsonl"
QWEN_ZS_MANIFEST_PATH="data/exploratorio/fase7/predicoes/${SMOKE_ID}_qwen_zero_shot.manifest.json"
QWEN_FT_OUTPUT_PATH="data/exploratorio/fase7/predicoes/${SMOKE_ID}_qwen_ft.jsonl"
QWEN_FT_MANIFEST_PATH="data/exploratorio/fase7/predicoes/${SMOKE_ID}_qwen_ft.manifest.json"

if [[ -d "/workspace" ]]; then
  export HF_HOME="${HF_HOME:-$HF_HOME_DEFAULT}"
else
  export HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/huggingface}"
fi
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

if [[ ! -f "data/dataset_treino.jsonl" ]]; then
  echo "Arquivo ausente: data/dataset_treino.jsonl"
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU NVIDIA não detectada. Este smoke test deve ser executado em ambiente com GPU."
  exit 1
fi

echo "== GPU detectada =="
nvidia-smi

echo "== Verificando dependências Python =="
"$PYTHON_BIN" - <<'PY'
import importlib
mods = ["torch", "transformers", "accelerate", "datasets", "trl", "peft", "unsloth"]
faltantes = [m for m in mods if importlib.util.find_spec(m) is None]
if faltantes:
    raise SystemExit(f"Dependências ausentes para o smoke test do Qwen: {faltantes}")
print("Dependências OK")
PY

mkdir -p data/exploratorio/fase5 data/exploratorio/fase7 data/exploratorio/fase7/predicoes
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE"

echo "== Cache Hugging Face =="
echo "HF_HOME=$HF_HOME"
echo "HF_HUB_CACHE=$HF_HUB_CACHE"
echo "TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
echo "HF_HUB_DISABLE_XET=$HF_HUB_DISABLE_XET"

echo "== Gerando subset de treino (${TRAIN_LIMIT} linhas) =="
head -n "$TRAIN_LIMIT" data/dataset_treino.jsonl > "$TRAIN_DATASET_PATH"

echo "== Gerando casos-base exploratórios =="
"$PYTHON_BIN" -m pipeline.fase7.casos_avaliacao --perfil-execucao exploratorio

echo "== Gerando subset de casos (${CASES_LIMIT} linhas) =="
head -n "$CASES_LIMIT" data/exploratorio/fase7/casos_avaliacao.jsonl > "$CASES_PATH"

echo "== Fine-tuning LoRA mínimo do Qwen =="
"$PYTHON_BIN" -m pipeline.fase5.finetuning_qwen \
  --perfil-execucao exploratorio \
  --dataset-path "$TRAIN_DATASET_PATH" \
  --output-dir "$CHECKPOINT_DIR" \
  --manifest-path "$QWEN_MANIFEST_PATH" \
  --max-seq-length "$MAX_SEQ_LENGTH" \
  --per-device-train-batch-size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
  --num-train-epochs "$NUM_TRAIN_EPOCHS" \
  --warmup-steps 1 \
  --logging-steps 1 \
  --save-total-limit 1

echo "== Inferência zero-shot do Qwen =="
"$PYTHON_BIN" -m pipeline.fase6.baseline_qwen \
  --perfil-execucao exploratorio \
  --casos-path "$CASES_PATH" \
  --output-path "$QWEN_ZS_OUTPUT_PATH" \
  --manifest-path "$QWEN_ZS_MANIFEST_PATH" \
  --model-id Qwen/Qwen2.5-14B-Instruct \
  --limit "$CASES_LIMIT"

echo "== Inferência fine-tuned do Qwen =="
"$PYTHON_BIN" -m pipeline.fase6.baseline_qwen \
  --perfil-execucao exploratorio \
  --casos-path "$CASES_PATH" \
  --output-path "$QWEN_FT_OUTPUT_PATH" \
  --manifest-path "$QWEN_FT_MANIFEST_PATH" \
  --condicao-id qwen_ft \
  --model-id "$CHECKPOINT_DIR" \
  --limit "$CASES_LIMIT"

cat <<EOF
Smoke test do Qwen concluído.

Artefatos principais:
- $QWEN_MANIFEST_PATH
- $CHECKPOINT_DIR
- $QWEN_ZS_OUTPUT_PATH
- $QWEN_ZS_MANIFEST_PATH
- $QWEN_FT_OUTPUT_PATH
- $QWEN_FT_MANIFEST_PATH
EOF
