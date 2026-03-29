from __future__ import annotations

from types import SimpleNamespace

from conftest import carregar_modulo_pipeline


baseline_gemini = carregar_modulo_pipeline("06_baseline_gemini.py")
baseline_qwen = carregar_modulo_pipeline("06_baseline_qwen.py")


def test_extrair_texto_resposta_gemini_prioriza_atributo_text() -> None:
    resposta = SimpleNamespace(text="  ementa final  ")
    assert baseline_gemini._extrair_texto_resposta_gemini(resposta) == "  ementa final  "


def test_gerar_ementa_qwen_funciona_com_modelo_e_tokenizer_fake() -> None:
    class FakeTensor:
        def __init__(self, rows: list[list[int]]):
            self.rows = rows

        def to(self, _device):
            return self

        @property
        def shape(self):
            return (len(self.rows), len(self.rows[0]))

        def __getitem__(self, item):
            if isinstance(item, tuple):
                row_sel, col_sel = item
                rows = self.rows[row_sel] if isinstance(row_sel, slice) else [self.rows[row_sel]]
                if isinstance(col_sel, slice):
                    return FakeTensor([row[col_sel] for row in rows])
                raise TypeError("Seleção não suportada no fake tensor.")
            return self.rows[item]

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 1

        def apply_chat_template(self, mensagens, **kwargs):
            assert mensagens[0]["role"] == "system"
            assert mensagens[1]["role"] == "user"
            assert kwargs["add_generation_prompt"] is True
            return {"input_ids": FakeTensor([[10, 11, 12]])}

        def batch_decode(self, ids_gerados, skip_special_tokens=True):
            assert skip_special_tokens is True
            assert ids_gerados.rows == [[21, 22]]
            return ["AREA. TEMA. FUNDAMENTO. RESULTADO."]

    class FakeModel:
        device = "cpu"

        def generate(self, **kwargs):
            assert "input_ids" in kwargs
            return FakeTensor([[10, 11, 12, 21, 22]])

    class FakeNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    fake_torch = SimpleNamespace(no_grad=lambda: FakeNoGrad())
    baseline_qwen.sys.modules["torch"] = fake_torch

    texto = baseline_qwen.gerar_ementa_qwen(
        FakeModel(),
        FakeTokenizer(),
        system_prompt="prompt",
        fundamentacao="fundamentação",
        max_new_tokens=64,
        temperature=0.0,
        top_p=1.0,
    )

    assert texto == "AREA. TEMA. FUNDAMENTO. RESULTADO."
