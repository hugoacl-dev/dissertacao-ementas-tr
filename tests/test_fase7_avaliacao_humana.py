from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pipeline.fase7.avaliacao_humana import (
    carregar_amostra_humana,
    carregar_gabarito_cegamento_humano,
    escrever_relatorio_avaliacao_humana,
    preparar_avaliacao_humana,
)


def _escrever_jsonl(path: Path, registros: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for registro in registros:
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")


def test_preparar_e_analisar_avaliacao_humana(tmp_path) -> None:
    casos_path = tmp_path / "casos.jsonl"
    amostra_path = tmp_path / "amostra_humana.json"
    gabarito_path = tmp_path / "gabarito_cegamento_humano.json"
    avaliacao_path = tmp_path / "avaliacao_humana.csv"
    relatorio_path = tmp_path / "relatorio_avaliacao_humana.json"
    predicao_paths = {
        nome: tmp_path / f"{nome}.jsonl"
        for nome in ["gemini_ft", "gemini_zero_shot", "qwen_ft", "qwen_zero_shot"]
    }

    _escrever_jsonl(
        casos_path,
        [
            {
                "caso_id": f"c{i:02d}",
                "indice_teste": i,
                "fundamentacao": ("fundamentacao " * (30 + i)).strip(),
                "ementa_referencia": f"referencia {i}",
            }
            for i in range(40)
        ],
    )
    for condicao_id, path in predicao_paths.items():
        _escrever_jsonl(
            path,
            [
                {
                    "caso_id": f"c{i:02d}",
                    "condicao_id": condicao_id,
                    "ementa_gerada": f"{condicao_id} caso {i}",
                }
                for i in range(40)
            ],
        )

    out_amostra, out_template = preparar_avaliacao_humana(
        casos_path=casos_path,
        predicao_paths=predicao_paths,
        amostra_path=amostra_path,
        gabarito_path=gabarito_path,
        template_path=avaliacao_path,
    )

    assert out_amostra == amostra_path
    assert out_template == avaliacao_path
    amostra = carregar_amostra_humana(amostra_path)
    gabarito = carregar_gabarito_cegamento_humano(gabarito_path)
    assert amostra["total_casos"] == 40
    assert "gabarito_cegamento" not in amostra
    assert len(gabarito) == 160

    template_df = pd.read_csv(avaliacao_path)
    assert len(template_df) == 40 * 4 * 2 * 4
    template_df["nota"] = 4
    template_df.to_csv(avaliacao_path, index=False)

    output = escrever_relatorio_avaliacao_humana(
        amostra_path=amostra_path,
        gabarito_path=gabarito_path,
        avaliacao_path=avaliacao_path,
        output_path=relatorio_path,
    )

    assert output == relatorio_path
    relatorio = json.loads(relatorio_path.read_text(encoding="utf-8"))
    assert set(relatorio["kappa_quadratico_por_criterio"]) == {
        "adequacao",
        "completude",
        "concisao",
        "fluencia",
    }
    assert all(valor == 1.0 for valor in relatorio["kappa_quadratico_por_criterio"].values())
    assert all(item["media_nota"] == 4.0 for item in relatorio["medias_por_condicao"])


def test_relatorio_humano_rejeita_grade_incompleta(tmp_path) -> None:
    casos_path = tmp_path / "casos.jsonl"
    amostra_path = tmp_path / "amostra_humana.json"
    gabarito_path = tmp_path / "gabarito_cegamento_humano.json"
    avaliacao_path = tmp_path / "avaliacao_humana.csv"
    relatorio_path = tmp_path / "relatorio_avaliacao_humana.json"
    predicao_paths = {
        nome: tmp_path / f"{nome}.jsonl"
        for nome in ["gemini_ft", "gemini_zero_shot", "qwen_ft", "qwen_zero_shot"]
    }

    _escrever_jsonl(
        casos_path,
        [
            {
                "caso_id": f"c{i:02d}",
                "indice_teste": i,
                "fundamentacao": ("fundamentacao " * (30 + i)).strip(),
                "ementa_referencia": f"referencia {i}",
            }
            for i in range(40)
        ],
    )
    for condicao_id, path in predicao_paths.items():
        _escrever_jsonl(
            path,
            [
                {
                    "caso_id": f"c{i:02d}",
                    "condicao_id": condicao_id,
                    "ementa_gerada": f"{condicao_id} caso {i}",
                }
                for i in range(40)
            ],
        )

    preparar_avaliacao_humana(
        casos_path=casos_path,
        predicao_paths=predicao_paths,
        amostra_path=amostra_path,
        gabarito_path=gabarito_path,
        template_path=avaliacao_path,
    )

    template_df = pd.read_csv(avaliacao_path)
    template_df["nota"] = 4
    template_df = template_df.iloc[1:].copy()
    template_df.to_csv(avaliacao_path, index=False)

    with pytest.raises(ValueError, match="incompleta"):
        escrever_relatorio_avaliacao_humana(
            amostra_path=amostra_path,
            gabarito_path=gabarito_path,
            avaliacao_path=avaliacao_path,
            output_path=relatorio_path,
        )
