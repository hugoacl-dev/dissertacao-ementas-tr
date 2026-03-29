# RUNBOOK_RUNPOD_QWEN.md

Runbook local de reativação do ambiente GPU do Qwen no RunPod.

Objetivo:
- permitir retomada rápida após `Stop Pod` ou `Terminate Pod`
- evitar redescoberta de problemas já conhecidos
- registrar o caminho mínimo validado em smoke real

Data de referência: `2026-03-29`

## 1. Quando consultar este arquivo

Leia este runbook se a tarefa envolver:
- smoke test do Qwen em GPU
- retomada do `finetuning_qwen.py`
- retomada do `baseline_qwen.py`
- recriação do ambiente RunPod depois de encerrar o pod anterior

## 2. O que já foi validado

Smoke exploratório real já validado em:
- RunPod
- `H100 80GB`
- Ubuntu do template `Runpod Pytorch 2.4.0`

Fluxos já executados com sucesso:
- `prepare-only` do Qwen
- `qwen_zero_shot` com caso sintético mínimo
- LoRA mínimo do Qwen com 1 amostra
- `qwen_ft` com checkpoint local gerado no próprio pod

## 3. O que se perde ao terminar o pod

Ao usar `Terminate Pod`, perde-se:
- `.venv` do pod
- cache do Hugging Face em `/workspace/.cache/huggingface`
- download do modelo base
- checkpoint LoRA sintético do smoke
- qualquer dataset sintético criado só no pod

Não se perde:
- código já commitado no repositório
- correções de compatibilidade do `trl`
- `scripts/qwen_smoke_gpu.sh`
- notas operacionais do `README.md`
- este runbook local

## 4. Configuração recomendada do pod

Configuração validada:
- template: `Runpod Pytorch 2.4.0`
- GPU: `H100 SXM x1`
- pricing: `On-Demand`
- `SSH terminal access`: habilitado
- `Container disk`: `20 GB`
- `Volume disk`: `60 GB`

Observação:
- `Container disk` pode estourar se o cache do Hugging Face ficar em `/root/.cache`
- o cache útil deve ficar em `/workspace/.cache/huggingface`

## 5. Bootstrap mínimo do ambiente

### 5.1. Clonar o repositório

```bash
cd /workspace
git clone https://github.com/hugoacl-dev/dissertacao-ementas-tr.git
cd dissertacao-ementas-tr
git checkout main
git pull --ff-only origin main
```

### 5.2. Criar venv e instalar dependências base

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
pip install -r requirements_fases_avancadas.txt
```

### 5.3. Ajustar `torch` para o combo validado com Unsloth

O ambiente do template pode instalar `torch` mais novo que o combo documentado do Unsloth.

Combo validado em smoke:
- `torch 2.9.1+cu130`
- `CUDA 13.0`

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu130
```

### 5.4. Instalar Unsloth

```bash
pip install --no-deps git+https://github.com/unslothai/unsloth-zoo.git
pip install "unsloth[cu130-torch291] @ git+https://github.com/unslothai/unsloth.git" --no-build-isolation
```

### 5.5. Verificação mínima

```bash
python - <<'PY'
import torch, unsloth
print("torch =", torch.__version__)
print("cuda =", torch.version.cuda)
print("gpu =", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("unsloth ok")
PY
```

Saída esperada:
- `torch = 2.9.1+cu130`
- `cuda = 13.0`
- GPU NVIDIA visível

## 6. Variáveis de ambiente obrigatórias para evitar falhas

Antes de carregar modelo ou rodar smoke:

```bash
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_CACHE=/workspace/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers
export HF_HUB_DISABLE_XET=1
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE"
```

Motivos:
- evitar uso de `/root/.cache`, que esgota o `container disk`
- contornar falhas observadas com o backend Xet

## 7. O que o smoke script faz e o que ele NÃO faz

O script versionado é:
- `scripts/qwen_smoke_gpu.sh`

Ele faz:
- verifica GPU e dependências Python
- cria subsets exploratórios
- roda LoRA mínimo do Qwen
- roda `qwen_zero_shot`
- roda `qwen_ft`

Ele NÃO faz:
- instalar `torch`
- instalar `unsloth`
- copiar `data/dataset_treino.jsonl` para o pod

## 8. Execução do smoke

Pré-requisito:
- `data/dataset_treino.jsonl` disponível no pod

Execução padrão:

```bash
bash scripts/qwen_smoke_gpu.sh
```

Execução reduzida:

```bash
TRAIN_LIMIT=16 CASES_LIMIT=4 MAX_SEQ_LENGTH=2048 bash scripts/qwen_smoke_gpu.sh
```

## 9. Dados reais vs. dados sintéticos

Smoke de infraestrutura:
- pode usar dataset sintético mínimo
- serve só para validar ambiente, compatibilidade e fluxo

Rodada metodologicamente relevante:
- usar apenas `data/dataset_treino.jsonl`
- usar apenas `data/dataset_teste.jsonl`
- executar somente após congelamento metodológico

## 10. Falhas já conhecidas

### Cache no lugar errado

Sintoma:
- falta de espaço mesmo com `/workspace` livre

Causa:
- downloads indo para `/root/.cache`

Correção:
- exportar `HF_HOME`, `HF_HUB_CACHE` e `TRANSFORMERS_CACHE` para `/workspace/.cache/huggingface`

### Xet / reconstrução de shards

Sintoma:
- falhas intermitentes no download pelo `huggingface_hub`

Correção:
- `export HF_HUB_DISABLE_XET=1`

### Drift de API do `trl`

Sintoma:
- erro com `max_seq_length` ou `tokenizer`

Situação atual:
- já corrigido em `pipeline/fase5/finetuning_qwen.py`

Compatibilidade implementada:
- `max_length` / `max_seq_length`
- `processing_class` / `tokenizer`

## 11. Decisão operacional ao fim da sessão

Se for retomar em 24–48h:
- preferir `Stop Pod`

Se não houver retomada iminente:
- preferir `Terminate Pod`

Razão:
- o custo relevante é o compute da GPU; o `volume disk` parado ainda custa, mas muito menos

## 12. Checklist de retomada para o próximo agente

1. Ler `.claude/AGENTS.md`
2. Ler `.claude/CONTEXTO_CONTINUIDADE.md`
3. Ler este arquivo
4. Confirmar se a tarefa é smoke exploratório ou rodada oficial
5. Se for rodar no RunPod:
   - subir pod 80 GB
   - reinstalar ambiente
   - configurar caches
   - só então executar `scripts/qwen_smoke_gpu.sh` ou os CLIs do Qwen
