from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from artefato_utils import escrever_csv_atomico
from fase7_metricas import (
    carregar_avaliacoes_judge,
    carregar_casos_avaliacao,
    carregar_predicoes_condicao,
    escrever_metricas_fase7,
    gerar_tabela_metricas_fase7,
)


def _escrever_jsonl(path: Path, registros: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for registro in registros:
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")


def test_escrever_csv_atomico_sobrescreve_arquivo(tmp_path) -> None:
    path = tmp_path / "fase7" / "metricas.csv"
    escrever_csv_atomico(path, pd.DataFrame([{"a": 1}, {"a": 2}]), index=False)
    escrever_csv_atomico(path, pd.DataFrame([{"a": 3}]), index=False)

    df = pd.read_csv(path)
    assert df.to_dict(orient="records") == [{"a": 3}]


def test_carregadores_basicos_da_fase7(tmp_path) -> None:
    casos_path = tmp_path / "casos.jsonl"
    predicao_path = tmp_path / "gemini_ft.jsonl"
    judge_path = tmp_path / "judge.jsonl"

    _escrever_jsonl(
        casos_path,
        [{"caso_id": "c1", "indice_teste": 0, "fundamentacao": "fund", "ementa_referencia": "ref"}],
    )
    _escrever_jsonl(
        predicao_path,
        [{"caso_id": "c1", "condicao_id": "gemini_ft", "ementa_gerada": "pred"}],
    )
    _escrever_jsonl(
        judge_path,
        [
            {
                "caso_id": "c1",
                "condicao_id": "gemini_ft",
                "avaliacao": {
                    "pertinencia_tematica": {"score": 4, "justificativa": "ok"},
                    "completude_dispositiva": {"score": 4, "justificativa": "ok"},
                    "fidelidade_factual": {"score": 5, "justificativa": "ok"},
                    "concisao": {"score": 3, "justificativa": "ok"},
                    "adequacao_terminologica": {"score": 4, "justificativa": "ok"},
                },
            }
        ],
    )

    assert carregar_casos_avaliacao(casos_path)["caso_id"].tolist() == ["c1"]
    assert carregar_predicoes_condicao(predicao_path, condicao_id="gemini_ft")["condicao_id"].tolist() == ["gemini_ft"]
    assert carregar_avaliacoes_judge(judge_path)["condicao_id"].tolist() == ["gemini_ft"]


def test_gerar_tabela_metricas_fase7_com_metricas_mockadas(monkeypatch, tmp_path) -> None:
    casos_df = pd.DataFrame(
        [
            {"caso_id": "c1", "indice_teste": 0, "fundamentacao": "fund 1", "ementa_referencia": "ref 1"},
            {"caso_id": "c2", "indice_teste": 1, "fundamentacao": "fund 2", "ementa_referencia": "ref 2"},
        ]
    )
    predicoes_df = pd.DataFrame(
        [
            {
                "caso_id": caso_id,
                "condicao_id": condicao_id,
                "ementa_gerada": f"{condicao_id}-{caso_id}",
            }
            for caso_id in ["c1", "c2"]
            for condicao_id in ["gemini_ft", "gemini_zero_shot", "qwen_ft", "qwen_zero_shot"]
        ]
    )
    avaliacoes_df = pd.DataFrame(
        [
            {
                "caso_id": caso_id,
                "condicao_id": condicao_id,
                "avaliacao": {
                    "pertinencia_tematica": {"score": 4, "justificativa": "ok"},
                    "completude_dispositiva": {"score": 4, "justificativa": "ok"},
                    "fidelidade_factual": {"score": 5, "justificativa": "ok"},
                    "concisao": {"score": 3, "justificativa": "ok"},
                    "adequacao_terminologica": {"score": 4, "justificativa": "ok"},
                },
            }
            for caso_id in ["c1", "c2"]
            for condicao_id in ["gemini_ft", "gemini_zero_shot", "qwen_ft", "qwen_zero_shot"]
        ]
    )

    monkeypatch.setattr(
        "fase7_metricas._calcular_metricas_rouge_por_par",
        lambda referencia, candidata: {
            "rouge_1_f1": 0.1,
            "rouge_2_f1": 0.05,
            "rouge_l_f1": 0.08,
        },
    )
    monkeypatch.setattr(
        "fase7_metricas._calcular_bertscore_f1_lote",
        lambda geradas, referencias, **kwargs: [0.9] * len(geradas),
    )

    tabela = gerar_tabela_metricas_fase7(casos_df, predicoes_df, avaliacoes_df)

    assert set(tabela.columns) == {"caso_id", "condicao_id", "metrica", "score"}
    assert len(tabela) == 80
    assert "judge_score_global" in set(tabela["metrica"])
    assert "judge_fidelidade_factual" in set(tabela["metrica"])


def test_escrever_metricas_fase7_com_artefatos_sinteticos(monkeypatch, tmp_path) -> None:
    casos_path = tmp_path / "casos.jsonl"
    judge_path = tmp_path / "judge.jsonl"
    output_path = tmp_path / "metricas.csv"
    predicao_paths = {
        nome: tmp_path / f"{nome}.jsonl"
        for nome in ["gemini_ft", "gemini_zero_shot", "qwen_ft", "qwen_zero_shot"]
    }

    _escrever_jsonl(
        casos_path,
        [
            {"caso_id": "c1", "indice_teste": 0, "fundamentacao": "fund 1", "ementa_referencia": "ref 1"},
            {"caso_id": "c2", "indice_teste": 1, "fundamentacao": "fund 2", "ementa_referencia": "ref 2"},
        ],
    )
    for nome, path in predicao_paths.items():
        _escrever_jsonl(
            path,
            [
                {"caso_id": "c1", "condicao_id": nome, "ementa_gerada": f"{nome}-1"},
                {"caso_id": "c2", "condicao_id": nome, "ementa_gerada": f"{nome}-2"},
            ],
        )
    _escrever_jsonl(
        judge_path,
        [
            {
                "caso_id": caso_id,
                "condicao_id": condicao_id,
                "avaliacao": {
                    "pertinencia_tematica": {"score": 4, "justificativa": "ok"},
                    "completude_dispositiva": {"score": 4, "justificativa": "ok"},
                    "fidelidade_factual": {"score": 5, "justificativa": "ok"},
                    "concisao": {"score": 3, "justificativa": "ok"},
                    "adequacao_terminologica": {"score": 4, "justificativa": "ok"},
                },
            }
            for caso_id in ["c1", "c2"]
            for condicao_id in predicao_paths
        ],
    )

    monkeypatch.setattr(
        "fase7_metricas._calcular_metricas_rouge_por_par",
        lambda referencia, candidata: {
            "rouge_1_f1": 0.1,
            "rouge_2_f1": 0.05,
            "rouge_l_f1": 0.08,
        },
    )
    monkeypatch.setattr(
        "fase7_metricas._calcular_bertscore_f1_lote",
        lambda geradas, referencias, **kwargs: [0.9] * len(geradas),
    )

    path = escrever_metricas_fase7(
        casos_path=casos_path,
        predicao_paths=predicao_paths,
        avaliacao_judge_path=judge_path,
        output_path=output_path,
    )

    assert path == output_path
    df = pd.read_csv(output_path)
    assert len(df) == 80
    assert {"bertscore_f1", "judge_score_global", "rouge_l_f1"} <= set(df["metrica"])
