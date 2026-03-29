from __future__ import annotations

import json
from pathlib import Path

from pipeline.fase7.avaliacao_judge import executar_avaliacao_judge


def _escrever_jsonl(path: Path, registros: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for registro in registros:
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")


def test_executar_avaliacao_judge_com_retomada_incremental(monkeypatch, tmp_path) -> None:
    casos_path = tmp_path / "casos.jsonl"
    output_path = tmp_path / "avaliacao_judge.jsonl"
    raw_output_path = tmp_path / "avaliacao_judge_bruta.jsonl"
    manifest_path = tmp_path / "avaliacao_judge_manifest.json"
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
    _escrever_jsonl(
        predicao_paths["gemini_ft"],
        [
            {"caso_id": "c1", "condicao_id": "gemini_ft", "ementa_gerada": "pred 1"},
            {"caso_id": "c2", "condicao_id": "gemini_ft", "ementa_gerada": "pred 2"},
        ],
    )

    monkeypatch.setattr(
        "pipeline.fase7.avaliacao_judge.avaliar_observacao_com_judge",
        lambda **kwargs: (
            {
                "pertinencia_tematica": {"score": 4, "justificativa": "ok"},
                "completude_dispositiva": {"score": 4, "justificativa": "ok"},
                "fidelidade_factual": {"score": 5, "justificativa": "ok"},
                "concisao": {"score": 3, "justificativa": "ok"},
                "adequacao_terminologica": {"score": 4, "justificativa": "ok"},
            },
            {
                "model_id_api": "deepseek-chat",
                "mensagem_usuario": "mensagem de teste",
                "resposta_bruta": {"choices": []},
            },
        ),
    )

    path = executar_avaliacao_judge(
        casos_path=casos_path,
        predicao_paths=predicao_paths,
        output_path=output_path,
        raw_output_path=raw_output_path,
        manifest_path=manifest_path,
        flush_every=1,
    )

    assert path == output_path
    linhas = output_path.read_text(encoding="utf-8").splitlines()
    assert len(linhas) == 2
    primeiro = json.loads(linhas[0])
    assert primeiro["condicao_id"] == "gemini_ft"
    linhas_brutas = raw_output_path.read_text(encoding="utf-8").splitlines()
    assert len(linhas_brutas) == 2
    primeiro_bruto = json.loads(linhas_brutas[0])
    assert primeiro_bruto["model_id_api"] == "deepseek-chat"

    manifesto = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifesto["status"] == "completed"
    assert sorted(manifesto["condicoes_ausentes"]) == ["gemini_zero_shot", "qwen_ft", "qwen_zero_shot"]
