from __future__ import annotations

import json
from pathlib import Path

from pipeline.ferramentas.status_fases57 import (
    agregar_status_execucao_oficial,
    agregar_status_validacao,
    carregar_manifesto_com_perfil,
    checkpoint_qwen_existe,
    ler_json_se_existir,
    status_oficial_por_manifesto,
)


def test_ler_json_se_existir_retorna_none_para_arquivo_ausente(tmp_path: Path) -> None:
    assert ler_json_se_existir(tmp_path / "ausente.json") is None


def test_carregar_manifesto_com_perfil_ignora_manifesto_legado_sem_perfil(tmp_path: Path) -> None:
    manifesto_path = tmp_path / "manifesto.json"
    manifesto_path.write_text(
        json.dumps({"status": "completed", "condicao_id": "qwen_zero_shot"}),
        encoding="utf-8",
    )

    assert carregar_manifesto_com_perfil(
        manifesto_path,
        perfil_execucao="oficial",
    ) is None


def test_carregar_manifesto_com_perfil_retorna_payload_valido(tmp_path: Path) -> None:
    manifesto = {
        "perfil_execucao": "exploratorio",
        "status": "completed",
        "condicao_id": "gemini_zero_shot",
    }
    manifesto_path = tmp_path / "manifesto.json"
    manifesto_path.write_text(json.dumps(manifesto), encoding="utf-8")

    assert carregar_manifesto_com_perfil(
        manifesto_path,
        perfil_execucao="exploratorio",
    ) == manifesto


def test_agregar_status_validacao_e_execucao_oficial() -> None:
    assert agregar_status_validacao(["pendente", "pendente"]) == "pendente"
    assert agregar_status_validacao(["validada", "validada"]) == "validada"
    assert agregar_status_validacao(["validada", "pendente"]) == "parcial"

    assert agregar_status_execucao_oficial(["pendente", "pendente"]) == "pendente"
    assert agregar_status_execucao_oficial(["concluida", "concluida"]) == "concluida"
    assert agregar_status_execucao_oficial(["concluida", "pendente"]) == "em_andamento"


def test_status_oficial_por_manifesto_exige_output_quando_informado(tmp_path: Path) -> None:
    manifesto = {
        "perfil_execucao": "oficial",
        "status": "completed",
        "condicao_id": "gemini_ft",
    }
    manifesto_path = tmp_path / "manifesto.json"
    manifesto_path.write_text(json.dumps(manifesto), encoding="utf-8")

    status, payload = status_oficial_por_manifesto(
        manifesto_path,
        output_path=tmp_path / "modelo.txt",
    )

    assert status == "em_andamento"
    assert payload == manifesto


def test_checkpoint_qwen_existe_so_quando_ha_conteudo(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()

    assert checkpoint_qwen_existe(checkpoint_dir) is False

    (checkpoint_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

    assert checkpoint_qwen_existe(checkpoint_dir) is True
