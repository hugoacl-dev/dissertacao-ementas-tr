from __future__ import annotations

import json
from pathlib import Path

from pipeline.fase7.casos_avaliacao import gerar_casos_avaliacao
from pipeline.core.project_paths import SYSTEM_PROMPT_PATH


def _escrever_dataset_teste(path: Path, registros: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for registro in registros:
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")


def test_gerar_casos_avaliacao_a_partir_do_dataset_teste(tmp_path) -> None:
    dataset_path = tmp_path / "dataset_teste.jsonl"
    output_path = tmp_path / "casos_avaliacao.jsonl"
    prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()

    _escrever_dataset_teste(
        dataset_path,
        [
            {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": (
                                    f"{prompt}\n\n"
                                    "Gere a ementa para a seguinte fundamentação:\n"
                                    "fundamentação A"
                                )
                            }
                        ],
                    },
                    {"role": "model", "parts": [{"text": "ementa A"}]},
                ]
            },
            {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": (
                                    f"{prompt}\n\n"
                                    "Gere a ementa para a seguinte fundamentação:\n"
                                    "fundamentação B"
                                )
                            }
                        ],
                    },
                    {"role": "model", "parts": [{"text": "ementa B"}]},
                ]
            },
        ],
    )

    path = gerar_casos_avaliacao(dataset_teste_path=dataset_path, output_path=output_path)

    linhas = output_path.read_text(encoding="utf-8").splitlines()
    assert path == output_path
    assert len(linhas) == 2
    primeiro = json.loads(linhas[0])
    segundo = json.loads(linhas[1])
    assert primeiro == {
        "caso_id": "teste_00000",
        "indice_teste": 0,
        "fundamentacao": "fundamentação A",
        "ementa_referencia": "ementa A",
    }
    assert segundo["caso_id"] == "teste_00001"
    assert segundo["indice_teste"] == 1
