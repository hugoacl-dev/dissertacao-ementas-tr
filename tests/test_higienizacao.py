from __future__ import annotations

from conftest import carregar_modulo_pipeline


higienizacao = carregar_modulo_pipeline("02_higienizacao.py")


def test_remove_sumula_terminal_sem_apagar_resultado() -> None:
    texto = (
        "PREVIDENCIÁRIO. RECURSO PROVIDO. "
        "Súmula do julgamento: por unanimidade, deu-se provimento."
    )

    limpo = higienizacao.limpar_texto(texto)

    assert "Súmula do julgamento" not in limpo
    assert "RECURSO PROVIDO" in limpo


def test_remove_assinatura_terminal_do_magistrado() -> None:
    texto = "Fundamentação válida. NOME MAGISTRADO GENERICO Juiz Federal Relator"

    limpo = higienizacao.limpar_texto(texto)

    assert "NOME MAGISTRADO GENERICO" not in limpo
    assert "Juiz Federal Relator" not in limpo
    assert limpo == "Fundamentação válida"


def test_preserva_formula_dispositiva_em_capslock() -> None:
    texto = "PREVIDENCIÁRIO. BENEFÍCIO ASSISTENCIAL. RECURSO IMPROVIDO."

    limpo = higienizacao.limpar_texto(texto)

    assert "RECURSO IMPROVIDO" in limpo
