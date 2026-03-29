from __future__ import annotations

import pandas as pd
import pytest

from jsonl_utils import (
    MARCADOR_FUNDAMENTACAO,
    extrair_fundamentacao_do_texto_user,
    extrair_fundamentacao_e_ementa,
)
from data_cadastro_utils import validar_e_converter_data_cadastro


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
