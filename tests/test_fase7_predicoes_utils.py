from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pipeline.fase7.predicoes_utils import (
    carregar_casos_predicao,
    carregar_predicoes_existentes,
    filtrar_casos_pendentes,
    normalizar_ementa_gerada,
    persistir_predicoes,
)


def _escrever_jsonl(path: Path, registros: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for registro in registros:
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")


def test_normalizar_ementa_gerada_colapsa_linhas_em_branco() -> None:
    texto = "  AREA.\n\n TEMA. \n FUNDAMENTO. \n RESULTADO.  "
    assert normalizar_ementa_gerada(texto) == "AREA. TEMA. FUNDAMENTO. RESULTADO."


def test_carregar_casos_predicao_respeita_schema() -> None:
    path = Path("/tmp/casos_predicao_schema.jsonl")
    _escrever_jsonl(
        path,
        [
            {
                "caso_id": "teste_00000",
                "indice_teste": 0,
                "fundamentacao": "fund",
                "ementa_referencia": "ementa",
            }
        ],
    )

    df = carregar_casos_predicao(path)
    assert list(df.columns) == ["caso_id", "indice_teste", "fundamentacao", "ementa_referencia"]


def test_persistir_e_carregar_predicoes_existentes(tmp_path) -> None:
    path = tmp_path / "predicoes.jsonl"
    registros = [
        {"caso_id": "teste_00000", "condicao_id": "gemini_zero_shot", "ementa_gerada": "ementa 1"},
        {"caso_id": "teste_00001", "condicao_id": "gemini_zero_shot", "ementa_gerada": "ementa 2"},
    ]

    persistir_predicoes(path, condicao_id="gemini_zero_shot", registros=registros)
    carregadas = carregar_predicoes_existentes(path, condicao_id="gemini_zero_shot")

    assert carregadas == registros


def test_filtrar_casos_pendentes_remove_processados() -> None:
    casos_df = pd.DataFrame(
        [
            {"caso_id": "teste_00000", "indice_teste": 0, "fundamentacao": "f1", "ementa_referencia": "e1"},
            {"caso_id": "teste_00001", "indice_teste": 1, "fundamentacao": "f2", "ementa_referencia": "e2"},
        ]
    )
    existentes = [{"caso_id": "teste_00000", "condicao_id": "qwen_zero_shot", "ementa_gerada": "ok"}]

    pendentes = filtrar_casos_pendentes(casos_df, existentes)

    assert [item["caso_id"] for item in pendentes] == ["teste_00001"]


def test_carregar_predicoes_existentes_rejeita_condicao_errada(tmp_path) -> None:
    path = tmp_path / "predicoes_invalidas.jsonl"
    _escrever_jsonl(
        path,
        [{"caso_id": "teste_00000", "condicao_id": "qwen_zero_shot", "ementa_gerada": "ementa"}],
    )

    with pytest.raises(ValueError, match="divergente"):
        carregar_predicoes_existentes(path, condicao_id="gemini_zero_shot")
