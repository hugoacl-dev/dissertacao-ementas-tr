from __future__ import annotations

import json

import pandas as pd
import pytest

from pipeline.core.artefato_utils import escrever_json_atomico
from pipeline.core.jsonl_utils import (
    MARCADOR_FUNDAMENTACAO,
    extrair_prompt_do_registro_jsonl,
    extrair_prompt_e_fundamentacao_do_texto_user,
    extrair_fundamentacao_do_texto_user,
    extrair_fundamentacao_e_ementa,
    validar_prompt_canonico_do_registro,
)
from pipeline.core.data_cadastro_utils import validar_e_converter_data_cadastro
from pipeline.core.project_paths import (
    DATASET_PATHS,
    DATASET_TESTE_PATH,
    DATASET_TREINO_PATH,
    PERFIL_EXECUCAO_EXPLORATORIO,
    PERFIL_EXECUCAO_OFICIAL,
    SYSTEM_PROMPT_PATH,
    resolver_artefatos_fase5,
    resolver_artefatos_fase7,
    resolver_manifestos_predicoes_fase7,
    resolver_predicoes_fase7,
    resolver_prefixo_gcs_fase5,
    validar_perfil_execucao,
)


def test_extrai_fundamentacao_e_ementa_do_formato_jsonl() -> None:
    obj = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            "prompt do sistema\n\n"
                            f"{MARCADOR_FUNDAMENTACAO}"
                            "fundamentação real"
                        )
                    }
                ],
            },
            {"role": "model", "parts": [{"text": "ementa final"}]},
        ]
    }

    fundamentacao, ementa = extrair_fundamentacao_e_ementa(obj)

    assert fundamentacao == "fundamentação real"
    assert ementa == "ementa final"


def test_fallback_da_extracao_quando_marcador_nao_existe() -> None:
    texto_user = "texto legado sem marcador"
    assert extrair_fundamentacao_do_texto_user(texto_user) == texto_user


def test_extrai_prompt_e_fundamentacao_do_texto_user() -> None:
    texto_user = "prompt canônico\n\n" + MARCADOR_FUNDAMENTACAO + "fundamentação real"

    prompt, fundamentacao = extrair_prompt_e_fundamentacao_do_texto_user(texto_user)

    assert prompt == "prompt canônico"
    assert fundamentacao == "fundamentação real"


def test_valida_prompt_canonico_do_registro() -> None:
    obj = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": "prompt canônico\n\n" + MARCADOR_FUNDAMENTACAO + "fund"}],
            },
            {"role": "model", "parts": [{"text": "ementa"}]},
        ]
    }

    assert extrair_prompt_do_registro_jsonl(obj) == "prompt canônico"
    validar_prompt_canonico_do_registro(
        obj,
        prompt_canonico="prompt canônico",
        contexto="sintético",
    )


def test_prompt_divergente_aborta_fases_avancadas() -> None:
    obj = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": "prompt antigo\n\n" + MARCADOR_FUNDAMENTACAO + "fund"}],
            },
            {"role": "model", "parts": [{"text": "ementa"}]},
        ]
    }

    with pytest.raises(ValueError, match="diverge de `system_prompt.txt`"):
        validar_prompt_canonico_do_registro(
            obj,
            prompt_canonico="prompt novo",
            contexto="sintético",
        )


def test_valida_e_converte_data_cadastro_iso8601() -> None:
    serie = pd.Series(
        [
            "2025-01-01 12:34:56.123456-03",
            "2025-02-01 01:02:03.000000-03",
        ]
    )

    convertido = validar_e_converter_data_cadastro(serie, contexto="teste")

    assert str(convertido.dtype).startswith("datetime64")
    assert convertido.name == "data_cadastro"


def test_data_cadastro_vazia_aborta_execucao() -> None:
    serie = pd.Series(["2025-01-01 12:00:00-03", ""])

    with pytest.raises(ValueError, match="nula ou vazia"):
        validar_e_converter_data_cadastro(serie, contexto="teste")


def test_data_cadastro_invalida_aborta_execucao() -> None:
    serie = pd.Series(["2025-01-01 12:00:00-03", "31/01/2025"])

    with pytest.raises(ValueError, match="inválida"):
        validar_e_converter_data_cadastro(serie, contexto="teste")


def test_project_paths_mantem_mapeamento_dos_datasets() -> None:
    assert DATASET_PATHS["treino"] == DATASET_TREINO_PATH
    assert DATASET_PATHS["teste"] == DATASET_TESTE_PATH
    assert SYSTEM_PROMPT_PATH.name == "system_prompt.txt"
    assert SYSTEM_PROMPT_PATH.exists()


def test_project_paths_separam_artefatos_exploratorios_e_oficiais() -> None:
    assert validar_perfil_execucao(PERFIL_EXECUCAO_EXPLORATORIO) == PERFIL_EXECUCAO_EXPLORATORIO
    assert validar_perfil_execucao(PERFIL_EXECUCAO_OFICIAL) == PERFIL_EXECUCAO_OFICIAL

    artefatos_fase5_expl = resolver_artefatos_fase5(PERFIL_EXECUCAO_EXPLORATORIO)
    artefatos_fase5_of = resolver_artefatos_fase5(PERFIL_EXECUCAO_OFICIAL)
    artefatos_fase7_expl = resolver_artefatos_fase7(PERFIL_EXECUCAO_EXPLORATORIO)
    predicoes_expl = resolver_predicoes_fase7(PERFIL_EXECUCAO_EXPLORATORIO)
    predicoes_of = resolver_predicoes_fase7(PERFIL_EXECUCAO_OFICIAL)
    manifestos_expl = resolver_manifestos_predicoes_fase7(PERFIL_EXECUCAO_EXPLORATORIO)

    assert "exploratorio/fase5" in str(artefatos_fase5_expl["gemini_manifest_path"])
    assert "exploratorio/fase7/predicoes" in str(predicoes_expl["gemini_zero_shot"])
    assert "exploratorio/fase7" in str(artefatos_fase7_expl["protocolo_path"])
    assert artefatos_fase5_of["gemini_manifest_path"].name == "gemini_sft_manifest.json"
    assert predicoes_of["gemini_zero_shot"].name == "gemini_zero_shot.jsonl"
    assert manifestos_expl["gemini_zero_shot"].name == "gemini_zero_shot.manifest.json"
    assert resolver_prefixo_gcs_fase5(PERFIL_EXECUCAO_EXPLORATORIO) == "testes/fase5"
    assert resolver_prefixo_gcs_fase5(PERFIL_EXECUCAO_OFICIAL) == "dissertacao-ementas-tr/fase5"


def test_project_paths_rejeitam_perfil_execucao_invalido() -> None:
    with pytest.raises(ValueError, match="perfil_execucao"):
        validar_perfil_execucao("rascunho")


def test_escrever_json_atomico_cria_pasta_e_sobrescreve_arquivo(tmp_path) -> None:
    path = tmp_path / "subdir" / "stats.json"

    escrever_json_atomico(path, {"fase": 1, "ok": True})
    assert json.loads(path.read_text(encoding="utf-8")) == {"fase": 1, "ok": True}

    escrever_json_atomico(path, {"fase": 2, "ok": False})
    assert json.loads(path.read_text(encoding="utf-8")) == {"fase": 2, "ok": False}
