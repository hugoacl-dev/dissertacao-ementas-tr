from __future__ import annotations

import pandas as pd

from conftest import carregar_modulo_pipeline


estatisticas = carregar_modulo_pipeline("04_estatisticas.py")


def test_classificar_anomalias_ementa_reconhece_rotulos_corrompidos() -> None:
    assert estatisticas._classificar_anomalias_ementa(
        "AMPARO ASSISTENCIAL. SENTENÇA DE IMPROCEDÊNCIA. RECORRE A PARTE-AUTORA"
    ) == ["trunc_8w_exata"]
    assert estatisticas._classificar_anomalias_ementa(
        "JUSTIÇA FEDERAL DA 5ª REGIÃO"
    ) == ["header_ementa"]
    assert estatisticas._classificar_anomalias_ementa(
        "DESPACHO. Intime-se."
    ) == ["despacho_ementa"]
    assert estatisticas._classificar_anomalias_ementa(
        "1. A sentença foi improcedente e o restante do texto segue em formato narrativo."
    ) == ["ementa_start_digit"]


def test_resumo_iqr_preserva_contagens_e_limites() -> None:
    serie = pd.Series([10, 11, 11, 12, 13, 14, 15, 100], dtype=float)
    resumo = estatisticas._resumo_iqr(serie, precisao=2)

    assert resumo["q1"] == 11.0
    assert resumo["q3"] == 14.25
    assert resumo["iqr"] == 3.25
    assert resumo["limite_iqr_superior"] == 19.12
    assert resumo["acima_limite_iqr"] == 1
    assert resumo["limite_iqr_superior_severo"] == 24.0
    assert resumo["acima_limite_iqr_severo"] == 1


def test_resumir_anomalias_estruturais_separa_rotulo_e_input_contaminado() -> None:
    df = pd.DataFrame(
        [
            {
                "split": "treino",
                "fundamentacao": "VOTO-EMENTA texto do voto",
                "ementa": "AMPARO ASSISTENCIAL. SENTENÇA DE IMPROCEDÊNCIA. RECORRE A PARTE-AUTORA",
            },
            {
                "split": "teste",
                "fundamentacao": "Fundamentação válida.",
                "ementa": "PREVIDENCIÁRIO. Ementa regular.",
            },
        ]
    )

    resumo = estatisticas._resumir_anomalias_estruturais(df)

    assert resumo["rotulos_corrompidos"]["total"] == 1
    assert resumo["rotulos_corrompidos"]["treino"] == 1
    assert resumo["rotulos_corrompidos"]["teste"] == 0
    assert resumo["inputs_contaminados"]["total"] == 1
    assert resumo["inputs_contaminados"]["treino"] == 1
    assert resumo["inputs_contaminados"]["teste"] == 0
