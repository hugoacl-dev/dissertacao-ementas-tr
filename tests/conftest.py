from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = REPO_ROOT / "pipeline"

if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))


def carregar_modulo_pipeline(nome_arquivo: str):
    """Carrega um módulo do diretório `pipeline/` pelo nome do arquivo."""
    caminho = PIPELINE_DIR / nome_arquivo
    nome_modulo = caminho.stem.replace("-", "_")
    spec = importlib.util.spec_from_file_location(nome_modulo, caminho)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Não foi possível carregar o módulo: {caminho}")
    modulo = importlib.util.module_from_spec(spec)
    sys.modules[nome_modulo] = modulo
    spec.loader.exec_module(modulo)
    return modulo
