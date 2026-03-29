from __future__ import annotations

import json

from conftest import carregar_modulo_pipeline


anonimizacao = carregar_modulo_pipeline("03_anonimizacao.py")
auditoria = carregar_modulo_pipeline("audit.py")


def test_anonimiza_nome_privado_em_caixa_alta_no_contexto_movido_por() -> None:
    texto = (
        "Pedido movido por PESSOA TESTE UM, representado por PESSOA TESTE DOIS, "
        "em face da UNIÃO."
    )

    anonimizado = anonimizacao.anonimizar_texto(texto)

    assert "PESSOA TESTE UM" not in anonimizado
    assert "PESSOA TESTE DOIS" not in anonimizado
    assert anonimizado.count("[NOME_OCULTADO]") == 2


def test_anonimiza_nome_privado_em_label_autor() -> None:
    texto = (
        "PROCESSO: [NPU] AUTOR: PESSOA TESTE TRES "
        "RÉU: Instituto Nacional do Seguro Social - INSS"
    )

    anonimizado = anonimizacao.anonimizar_texto(texto)

    assert "PESSOA TESTE TRES" not in anonimizado
    assert "AUTOR: [NOME_OCULTADO]" in anonimizado
    assert "Instituto Nacional do Seguro Social" in anonimizado


def test_preserva_nome_de_agente_publico_em_precedente() -> None:
    texto = (
        "Reafirmação da jurisprudência do Supremo Tribunal Federal. "
        "(RE 635729 RG, Relator Min. Ministro do Supremo Tribunal Federal, julgado em 30/06/2011)"
    )

    anonimizado = anonimizacao.anonimizar_texto(texto)

    assert "Ministro do Supremo Tribunal Federal" in anonimizado
    assert "[NOME_OCULTADO]" not in anonimizado


def test_auditoria_detecta_nome_privado_residual_em_contexto() -> None:
    texto = "Pedido movido por PESSOA TESTE UM em face da UNIÃO."
    encontrados = auditoria._detectar_nomes_privados_residuais(texto)

    assert encontrados == ["PESSOA TESTE UM"]


def test_auditoria_jsonl_falha_com_nome_privado_residual(tmp_path) -> None:
    registro = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            "prompt\n\n"
                            "Gere a ementa para a seguinte fundamentação:\n"
                            "Pedido movido por PESSOA TESTE UM em face da UNIÃO."
                        )
                    }
                ],
            },
            {"role": "model", "parts": [{"text": "ementa"}]},
        ]
    }
    path = tmp_path / "dataset.jsonl"
    path.write_text(json.dumps(registro, ensure_ascii=False) + "\n", encoding="utf-8")

    assert auditoria.audit(path) is False
