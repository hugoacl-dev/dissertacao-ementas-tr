from __future__ import annotations

import importlib
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = REPO_ROOT / "pipeline"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


MAPA_MODULOS_PIPELINE = {
    "01_ingestao.py": "pipeline.fase1_4.fase01_ingestao",
    "02_higienizacao.py": "pipeline.fase1_4.fase02_higienizacao",
    "03_anonimizacao.py": "pipeline.fase1_4.fase03_anonimizacao",
    "04_estatisticas.py": "pipeline.fase1_4.fase04_estatisticas",
    "05_finetuning_gemini.py": "pipeline.fase5.finetuning_gemini",
    "05_finetuning_qwen.py": "pipeline.fase5.finetuning_qwen",
    "06_baseline_gemini.py": "pipeline.fase6.baseline_gemini",
    "06_baseline_qwen.py": "pipeline.fase6.baseline_qwen",
    "audit.py": "pipeline.ferramentas.auditoria",
    "ver_registro.py": "pipeline.ferramentas.ver_registro",
}


def carregar_modulo_pipeline(nome_arquivo: str):
    """Carrega um módulo do pipeline pelo nome legado do arquivo."""
    nome_modulo = MAPA_MODULOS_PIPELINE.get(nome_arquivo)
    if nome_modulo is None:
        raise RuntimeError(f"Módulo não mapeado para o arquivo: {nome_arquivo}")
    return importlib.import_module(nome_modulo)
