from __future__ import annotations

from pathlib import Path

from conftest import carregar_modulo_pipeline


gemini_sft = carregar_modulo_pipeline("05_finetuning_gemini.py")
qwen_sft = carregar_modulo_pipeline("05_finetuning_qwen.py")


def test_gemini_prepare_only_escreve_manifesto(tmp_path) -> None:
    dataset_path = tmp_path / "dataset_treino.jsonl"
    dataset_path.write_text('{"contents":[]}\n', encoding="utf-8")
    manifest_path = tmp_path / "gemini_manifest.json"

    output = gemini_sft.executar_finetuning_gemini(
        project_id="projeto-teste",
        dataset_path=dataset_path,
        train_dataset_gcs_uri="gs://bucket/train.jsonl",
        prepare_only=True,
        manifest_path=manifest_path,
    )

    assert output == manifest_path
    assert manifest_path.exists()


def test_qwen_prepare_only_escreve_manifesto(tmp_path, monkeypatch) -> None:
    manifest_path = tmp_path / "qwen_manifest.json"
    monkeypatch.setattr(
        qwen_sft,
        "preparar_dataset_qwen",
        lambda dataset_path: [{"messages": [], "prompt": [], "completion": [], "ementa": "x", "fundamentacao": "y", "system_prompt": "z", "caso_id": "treino_00000"}],
    )

    output = qwen_sft.executar_finetuning_qwen(
        dataset_path=Path("data/dataset_treino.jsonl"),
        prepare_only=True,
        manifest_path=manifest_path,
    )

    assert output == manifest_path
    assert manifest_path.exists()
