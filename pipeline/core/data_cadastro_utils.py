"""
data_cadastro_utils.py — Validação e parse estrito de `data_cadastro`

Centraliza a validação do campo `data_cadastro`, que sustenta o split
cronológico do experimento e a análise temporal do corpus.
"""
from __future__ import annotations

import pandas as pd


def _formatar_amostras_invalidas(
    serie_original: pd.Series,
    mascara: pd.Series,
    *,
    limite: int = 5,
) -> str:
    """Formata poucas amostras problemáticas para mensagens de erro."""
    amostras: list[str] = []
    for idx, valor in serie_original[mascara].head(limite).items():
        texto = "<vazio>" if pd.isna(valor) or str(valor).strip() == "" else str(valor)
        amostras.append(f"linha {idx}: {texto!r}")
    return "; ".join(amostras)


def validar_e_converter_data_cadastro(
    datas: pd.Series,
    *,
    contexto: str,
) -> pd.Series:
    """Valida e converte `data_cadastro` para datetime.

    A pesquisa depende de split cronológico confiável. Portanto, valores
    nulos, vazios ou não parseáveis devem abortar a execução em vez de
    serem tolerados silenciosamente.
    """
    datas_str = datas.astype("string").str.strip()

    mascara_vazios = datas_str.isna() | datas_str.eq("")
    if mascara_vazios.any():
        raise ValueError(
            f"{contexto}: {int(mascara_vazios.sum())} registros com "
            f"`data_cadastro` nula ou vazia. Amostras: "
            f"{_formatar_amostras_invalidas(datas, mascara_vazios)}"
        )

    datas_dt = pd.to_datetime(datas_str, format="ISO8601", errors="coerce")
    mascara_invalidos = datas_dt.isna()
    if mascara_invalidos.any():
        raise ValueError(
            f"{contexto}: {int(mascara_invalidos.sum())} registros com "
            f"`data_cadastro` inválida. Esperado timestamp ISO8601 parseável. "
            f"Amostras: {_formatar_amostras_invalidas(datas_str, mascara_invalidos)}"
        )

    return datas_dt.rename("data_cadastro")

