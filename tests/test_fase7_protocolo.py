from __future__ import annotations

import json

import pytest

from pipeline.fase7.protocolo import (
    CASOS_AVALIACAO_HUMANA,
    CASOS_POR_ESTRATO_AVALIACAO_HUMANA,
    CONDICOES_EXPERIMENTAIS,
    CRITERIOS_AVALIACAO_HUMANA,
    DIMENSOES_JUIZ,
    MODELO_JUIZ_API_OBSERVACAO,
    MODELO_JUIZ_API_PADRAO,
    SEED_AMOSTRAGEM_AVALIACAO_HUMANA,
    SEED_CEGAMENTO_AVALIACAO_HUMANA,
    VERSAO_PROTOCOLO_FASE7,
    calcular_score_global_llm_judge,
    gerar_manifesto_fase7,
    ler_prompt_llm_judge,
    schema_registro_avaliacao_judge,
    schema_registro_caso_avaliacao,
    schema_registro_predicao,
    schema_resposta_llm_judge,
    validar_registro_avaliacao_judge,
    validar_registro_caso_avaliacao,
    validar_registro_predicao,
    validar_resposta_llm_judge,
)
from pipeline.core.project_paths import (
    FASE7_AVALIACAO_JUDGE_BRUTA_PATH,
    FASE7_AVALIACAO_JUDGE_MANIFEST_PATH,
    FASE7_GABARITO_CEGAMENTO_HUMANO_PATH,
    FASE7_PREDICAO_MANIFEST_PATHS,
    FASE7_PREDICAO_PATHS,
    FASE7_PROTOCOLO_PATH,
    LLM_JUDGE_PROMPT_PATH,
    PERFIL_EXECUCAO_EXPLORATORIO,
    resolver_artefatos_fase7,
)


def test_prompt_judge_versionado_existe_e_define_regras_essenciais() -> None:
    prompt = ler_prompt_llm_judge()

    assert LLM_JUDGE_PROMPT_PATH.exists()
    assert "JSON válido" in prompt
    assert "não inclua score_global" in prompt
    for dimensao in DIMENSOES_JUIZ:
        assert dimensao in prompt


def test_schema_resposta_judge_explica_dimensoes_sem_score_global() -> None:
    schema = schema_resposta_llm_judge()

    assert schema["required"] == list(DIMENSOES_JUIZ)
    assert "score_global" not in schema["properties"]


def test_schemas_de_caso_e_predicao_exigem_campos_esperados() -> None:
    schema_caso = schema_registro_caso_avaliacao()
    schema_predicao = schema_registro_predicao()
    schema_avaliacao_judge = schema_registro_avaliacao_judge()

    assert schema_caso["required"] == ["caso_id", "indice_teste", "fundamentacao", "ementa_referencia"]
    assert schema_predicao["required"] == ["caso_id", "condicao_id", "ementa_gerada"]
    assert schema_avaliacao_judge["required"] == ["caso_id", "condicao_id", "avaliacao"]


def test_validar_resposta_judge_aceita_payload_valido() -> None:
    payload = {
        dimensao: {"score": 4, "justificativa": "Adequado ao caso."}
        for dimensao in DIMENSOES_JUIZ
    }

    validado = validar_resposta_llm_judge(payload)

    assert validado == payload
    assert calcular_score_global_llm_judge(validado) == 4.0


def test_validar_registro_caso_avaliacao_aceita_payload_valido() -> None:
    payload = {
        "caso_id": "teste_00000",
        "indice_teste": 0,
        "fundamentacao": "Fundamentação válida.",
        "ementa_referencia": "Ementa válida.",
    }

    assert validar_registro_caso_avaliacao(payload) == payload


def test_validar_registro_predicao_aceita_payload_valido() -> None:
    payload = {
        "caso_id": "teste_00000",
        "condicao_id": "gemini_ft",
        "ementa_gerada": "Ementa gerada.",
    }

    assert validar_registro_predicao(payload, condicao_id_esperada="gemini_ft") == payload


def test_validar_registro_avaliacao_judge_aceita_payload_valido() -> None:
    payload = {
        "caso_id": "teste_00000",
        "condicao_id": "gemini_ft",
        "avaliacao": {
            dimensao: {"score": 4, "justificativa": "Adequado ao caso."}
            for dimensao in DIMENSOES_JUIZ
        },
    }

    assert validar_registro_avaliacao_judge(payload, condicao_id_esperada="gemini_ft") == payload


def test_validar_resposta_judge_rejeita_score_fora_da_faixa() -> None:
    payload = {
        dimensao: {"score": 4, "justificativa": "Adequado ao caso."}
        for dimensao in DIMENSOES_JUIZ
    }
    payload["fidelidade_factual"]["score"] = 6

    with pytest.raises(ValueError, match="entre 1 e 5"):
        validar_resposta_llm_judge(payload)


def test_validar_resposta_judge_rejeita_dimensao_inesperada() -> None:
    payload = {
        dimensao: {"score": 4, "justificativa": "Adequado ao caso."}
        for dimensao in DIMENSOES_JUIZ
    }
    payload["score_global"] = 4

    with pytest.raises(ValueError, match="inesperadas"):
        validar_resposta_llm_judge(payload)


def test_validar_registro_predicao_rejeita_condicao_incompativel() -> None:
    payload = {
        "caso_id": "teste_00000",
        "condicao_id": "qwen_ft",
        "ementa_gerada": "Ementa gerada.",
    }

    with pytest.raises(ValueError, match="divergente"):
        validar_registro_predicao(payload, condicao_id_esperada="gemini_ft")


def test_manifesto_fase7_tem_contrato_estavel() -> None:
    manifesto = gerar_manifesto_fase7()

    assert manifesto["versao_protocolo"] == VERSAO_PROTOCOLO_FASE7
    assert manifesto["perfil_execucao"] == "oficial"
    assert manifesto["llm_judge"]["prompt_path"] == str(LLM_JUDGE_PROMPT_PATH)
    assert manifesto["llm_judge"]["modelo_api_padrao"] == MODELO_JUIZ_API_PADRAO
    assert manifesto["llm_judge"]["observacao_modelo_api"] == MODELO_JUIZ_API_OBSERVACAO
    assert manifesto["artefatos"]["manifesto"] == str(FASE7_PROTOCOLO_PATH)
    assert manifesto["schema_caso_avaliacao"]["required"] == ["caso_id", "indice_teste", "fundamentacao", "ementa_referencia"]
    assert manifesto["schema_predicao"]["required"] == ["caso_id", "condicao_id", "ementa_gerada"]
    assert manifesto["avaliacao_humana"]["criterios"] == list(CRITERIOS_AVALIACAO_HUMANA)
    assert manifesto["avaliacao_humana"]["casos_amostrados"] == CASOS_AVALIACAO_HUMANA
    assert manifesto["avaliacao_humana"]["casos_por_estrato"] == CASOS_POR_ESTRATO_AVALIACAO_HUMANA
    assert manifesto["avaliacao_humana"]["seed_amostragem"] == SEED_AMOSTRAGEM_AVALIACAO_HUMANA
    assert manifesto["avaliacao_humana"]["seed_cegamento"] == SEED_CEGAMENTO_AVALIACAO_HUMANA
    assert manifesto["avaliacao_humana"]["pacote_cego_separado_do_gabarito"] is True
    assert sorted(manifesto["artefatos"]["predicoes"]) == sorted(FASE7_PREDICAO_PATHS)
    assert sorted(manifesto["artefatos"]["manifestos_predicoes"]) == sorted(FASE7_PREDICAO_MANIFEST_PATHS)
    assert manifesto["artefatos"]["avaliacao_llm_judge_bruta"] == str(FASE7_AVALIACAO_JUDGE_BRUTA_PATH)
    assert manifesto["artefatos"]["avaliacao_llm_judge_manifesto"] == str(FASE7_AVALIACAO_JUDGE_MANIFEST_PATH)
    assert manifesto["artefatos"]["gabarito_cegamento_humano"] == str(FASE7_GABARITO_CEGAMENTO_HUMANO_PATH)
    assert [item["id"] for item in manifesto["condicoes_experimentais"]] == [
        item["id"] for item in CONDICOES_EXPERIMENTAIS
    ]
    json.dumps(manifesto, ensure_ascii=False)


def test_manifesto_fase7_pode_ser_gerado_para_perfil_exploratorio() -> None:
    manifesto = gerar_manifesto_fase7(perfil_execucao=PERFIL_EXECUCAO_EXPLORATORIO)
    artefatos = resolver_artefatos_fase7(PERFIL_EXECUCAO_EXPLORATORIO)

    assert manifesto["perfil_execucao"] == PERFIL_EXECUCAO_EXPLORATORIO
    assert manifesto["artefatos"]["manifesto"] == str(artefatos["protocolo_path"])
    assert manifesto["artefatos"]["casos_avaliacao"] == str(artefatos["casos_avaliacao_path"])
    assert manifesto["artefatos"]["avaliacao_llm_judge"] == str(artefatos["avaliacao_judge_path"])
    assert manifesto["artefatos"]["relatorio_estatistico"] == str(artefatos["relatorio_estatistico_path"])
