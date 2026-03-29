#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
TORCH_VERSION="${TORCH_VERSION:-2.9.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.24.1}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.9.1}"
TORCH_CHANNEL="${TORCH_CHANNEL:-cu130}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
UNSLOTH_EXTRA="${UNSLOTH_EXTRA:-cu130-torch291}"
UNSLOTH_ZOO_REF="${UNSLOTH_ZOO_REF:-git+https://github.com/unslothai/unsloth-zoo.git}"
UNSLOTH_REF="${UNSLOTH_REF:-git+https://github.com/unslothai/unsloth.git}"
RUN_SMOKE="${RUN_SMOKE:-0}"
SMOKE_SCRIPT="${SMOKE_SCRIPT:-scripts/qwen_smoke_gpu.sh}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU NVIDIA não detectada. Este bootstrap foi feito para RunPod/ambiente com GPU."
  exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
  echo "== Criando virtualenv em $VENV_DIR =="
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

echo "== GPU detectada =="
nvidia-smi

echo "== Atualizando pip/setuptools =="
python -m pip install --upgrade pip wheel setuptools

echo "== Instalando requirements base =="
pip install -r requirements.txt
pip install -r requirements_fases_avancadas.txt

echo "== Verificando torch atual =="
CURRENT_TORCH="$(python - <<'PY'
import importlib.util
if importlib.util.find_spec("torch") is None:
    print("ausente")
else:
    import torch
    print(torch.__version__)
PY
)"
echo "torch atual: $CURRENT_TORCH"

EXPECTED_TORCH="${TORCH_VERSION}+${TORCH_CHANNEL}"
if [[ "$CURRENT_TORCH" != "$EXPECTED_TORCH" ]]; then
  echo "== Alinhando torch para o combo validado com Unsloth ($EXPECTED_TORCH) =="
  pip uninstall -y torch torchvision torchaudio || true
  pip install \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}" \
    --index-url "$TORCH_INDEX_URL"
else
  echo "torch já alinhado com o combo validado."
fi

echo "== Instalando/atualizando Unsloth =="
pip install --no-deps "$UNSLOTH_ZOO_REF"
pip install "unsloth[${UNSLOTH_EXTRA}] @ ${UNSLOTH_REF}" --no-build-isolation

if [[ -d "/workspace" ]]; then
  export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
else
  export HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/huggingface}"
fi
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE"

ENV_FILE="${ENV_FILE:-$REPO_ROOT/.runpod_qwen_env.sh}"
cat > "$ENV_FILE" <<EOF
export HF_HOME="$HF_HOME"
export HF_HUB_CACHE="$HF_HUB_CACHE"
export TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE"
export HF_HUB_DISABLE_XET="$HF_HUB_DISABLE_XET"
EOF

echo "== Verificação final do ambiente =="
python - <<'PY'
import importlib
mods = ["torch", "transformers", "accelerate", "datasets", "trl", "peft", "unsloth"]
faltantes = [m for m in mods if importlib.util.find_spec(m) is None]
if faltantes:
    raise SystemExit(f"Dependências ausentes após bootstrap: {faltantes}")

import torch
import unsloth

print("torch =", torch.__version__)
print("cuda =", torch.version.cuda)
print("gpu =", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("unsloth ok")
PY

cat <<EOF

Bootstrap do RunPod/Qwen concluído.

Artefatos locais:
- virtualenv: $VENV_DIR
- env de cache: $ENV_FILE

Variáveis exportadas nesta sessão:
- HF_HOME=$HF_HOME
- HF_HUB_CACHE=$HF_HUB_CACHE
- TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE
- HF_HUB_DISABLE_XET=$HF_HUB_DISABLE_XET

Próximos passos:
1. source "$ENV_FILE"
2. garantir que data/dataset_treino.jsonl e data/dataset_teste.jsonl estejam disponíveis no pod
3. rodar o smoke:
   bash "$SMOKE_SCRIPT"
EOF

if [[ "$RUN_SMOKE" == "1" ]]; then
  echo "== RUN_SMOKE=1: executando $SMOKE_SCRIPT =="
  bash "$SMOKE_SCRIPT"
fi
