from __future__ import annotations

import json
from pathlib import Path

from fase5_tuning_utils import (
    calcular_batch_size_efetivo,
    carregar_amostras_treino_sft,
    construir_uri_gcs,
    gerar_nome_experimento,
)


def _escrever_jsonl(path: Path, registros: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for registro in registros:
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")


def test_carregar_amostras_treino_sft_reconstroi_messages_prompt_completion(tmp_path) -> None:
    dataset_path = tmp_path / "dataset_treino.jsonl"
    system_prompt_path = tmp_path / "system_prompt.txt"
    system_prompt_path.write_text("prompt canônico", encoding="utf-8")
    _escrever_jsonl(
        dataset_path,
        [
            {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": (
                                    "prompt canônico\n\n"
                                    "Gere a ementa para a seguinte fundamentação:\n"
                                    "fundamentação 1"
                                )
                            }
                        ],
                    },
                    {"role": "model", "parts": [{"text": "ementa 1"}]},
                ]
            }
        ],
    )

    amostras = carregar_amostras_treino_sft(dataset_path, system_prompt_path)

    assert len(amostras) == 1
    assert amostras[0]["caso_id"] == "treino_00000"
    assert amostras[0]["messages"][0]["role"] == "system"
    assert amostras[0]["messages"][2]["content"] == "ementa 1"
    assert amostras[0]["completion"][0]["role"] == "assistant"


def test_helpers_basicos_da_fase5() -> None:
    assert calcular_batch_size_efetivo(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
    ) == 32
    assert construir_uri_gcs("gs://meu-bucket", "pasta/arquivo.jsonl") == "gs://meu-bucket/pasta/arquivo.jsonl"
    assert gerar_nome_experimento("gemini_sft").startswith("gemini_sft_")
