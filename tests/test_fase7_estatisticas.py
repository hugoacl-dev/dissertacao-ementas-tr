from __future__ import annotations

import json

import pandas as pd
import pytest

from pipeline.fase7.estatisticas import (
    ajustar_pvalues_bh,
    ajustar_pvalues_holm,
    bootstrap_pareado,
    calcular_pvalue_permutacao_pareada,
    gerar_relatorio_estatistico,
    validar_tabela_metricas_fase7,
)
from pipeline.fase7.protocolo import gerar_manifesto_fase7


def _metricas_sinteticas() -> pd.DataFrame:
    linhas: list[dict[str, object]] = []
    for caso_idx in range(1, 11):
        caso_id = f"caso-{caso_idx}"
        for condicao_id, familia, base_bert, base_judge, rouge in [
            ("gemini_ft", "gemini", 0.92, 4.8, 0.81),
            ("gemini_zero_shot", "gemini", 0.61, 3.2, 0.54),
            ("qwen_ft", "qwen", 0.88, 4.4, 0.77),
            ("qwen_zero_shot", "qwen", 0.58, 2.9, 0.49),
        ]:
            for metrica, score in [
                ("bertscore_f1", base_bert + caso_idx * 0.001),
                ("judge_score_global", base_judge + caso_idx * 0.01),
                ("rouge_l_f1", rouge + caso_idx * 0.002),
            ]:
                linhas.append(
                    {
                        "caso_id": caso_id,
                        "condicao_id": condicao_id,
                        "metrica": metrica,
                        "score": score,
                        "familia": familia,
                    }
                )
    return pd.DataFrame(linhas)


def test_bootstrap_pareado_retorna_intervalo_ordenado() -> None:
    ic_inferior, ic_superior = bootstrap_pareado(
        deltas=pd.Series([0.1, 0.2, 0.3, 0.4]).to_numpy(),
        iteracoes=500,
        seed=123,
    )

    assert ic_inferior <= ic_superior
    assert ic_superior > 0


def test_permutacao_pareada_detecta_diferenca_clara() -> None:
    ft = pd.Series([0.9, 0.91, 0.92, 0.93, 0.94]).to_numpy()
    zs = pd.Series([0.4, 0.41, 0.42, 0.43, 0.44]).to_numpy()

    ft = pd.Series([0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97]).to_numpy()
    zs = pd.Series([0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47]).to_numpy()

    pvalue = calcular_pvalue_permutacao_pareada(ft, zs, iteracoes=4000, seed=123)

    assert pvalue < 0.05


def test_ajustes_holm_e_bh_preservam_faixa_unitaria() -> None:
    pvalues = [0.01, 0.03, 0.04, 0.20]

    ajustado_holm = ajustar_pvalues_holm(pvalues)
    ajustado_bh = ajustar_pvalues_bh(pvalues)

    assert all(0 <= valor <= 1 for valor in ajustado_holm)
    assert all(0 <= valor <= 1 for valor in ajustado_bh)


def test_validar_tabela_metricas_rejeita_duplicidade() -> None:
    df = pd.DataFrame(
        [
            {"caso_id": "1", "condicao_id": "gemini_ft", "metrica": "bertscore_f1", "score": 0.9},
            {"caso_id": "1", "condicao_id": "gemini_ft", "metrica": "bertscore_f1", "score": 0.9},
        ]
    )

    with pytest.raises(ValueError, match="duplicadas"):
        validar_tabela_metricas_fase7(df)


def test_gerar_relatorio_estatistico_sintetico() -> None:
    manifesto = gerar_manifesto_fase7()
    df = _metricas_sinteticas()

    relatorio = gerar_relatorio_estatistico(df, manifesto)

    assert relatorio["versao_protocolo"] == manifesto["versao_protocolo"]
    assert relatorio["resumo_familias"]["gemini"]["sucesso_confirmatorio"] is True
    assert relatorio["resumo_familias"]["qwen"]["sucesso_confirmatorio"] is True
    assert len(relatorio["comparacoes"]) == 6
    json.dumps(relatorio, ensure_ascii=False)
