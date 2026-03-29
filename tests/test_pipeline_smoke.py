from __future__ import annotations

import json
from pathlib import Path

from conftest import carregar_modulo_pipeline
from jsonl_utils import extrair_fundamentacao_e_ementa


higienizacao = carregar_modulo_pipeline("02_higienizacao.py")
anonimizacao = carregar_modulo_pipeline("03_anonimizacao.py")
auditoria = carregar_modulo_pipeline("audit.py")
estatisticas = carregar_modulo_pipeline("04_estatisticas.py")


def _gravar_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_smoke_pipeline_fases_2_a_4_com_fixture_sintetica(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    data_dir = tmp_path / "data"
    docs_data_dir = tmp_path / "docs" / "data"
    data_dir.mkdir(parents=True)
    docs_data_dir.mkdir(parents=True)

    registros_brutos = [
        {
            "id": 1,
            "fundamentacao": (
                "<p>Pedido movido por PESSOA TESTE UM em face da UNIÃO.</p> "
                "A sentença analisou detalhadamente a prova documental e concluiu "
                "pela inexistência de incapacidade laboral permanente. "
                "Súmula do julgamento: por unanimidade, negou-se provimento."
            ),
            "ementa": (
                "PREVIDENCIÁRIO. BENEFÍCIO POR INCAPACIDADE. "
                "AUSÊNCIA DE INCAPACIDADE LABORAL. RECURSO DESPROVIDO."
            ),
            "data_cadastro": "2025-01-10 10:00:00-03",
        },
        {
            "id": 2,
            "fundamentacao": (
                "PROCESSO: 0000001 AUTOR: PESSOA TESTE TRES "
                "RÉU: Instituto Nacional do Seguro Social - INSS. "
                "O laudo pericial foi claro e suficiente, afastando o impedimento "
                "de longo prazo e confirmando a improcedência."
            ),
            "ementa": (
                "ASSISTENCIAL. BENEFÍCIO ASSISTENCIAL. "
                "AUSÊNCIA DE IMPEDIMENTO DE LONGO PRAZO. SENTENÇA MANTIDA."
            ),
            "data_cadastro": "2025-01-11 10:00:00-03",
        },
        {
            "id": 3,
            "fundamentacao": (
                "A renda do grupo familiar do(a) titular, Sr(a). PESSOA TESTE QUATRO, "
                "afasta a situação de miserabilidade extrema. "
                "O estudo social descreveu moradia adequada e despesas ordinárias compatíveis."
            ),
            "ementa": (
                "SEGURIDADE SOCIAL. BENEFÍCIO ASSISTENCIAL AO IDOSO. "
                "VULNERABILIDADE SOCIOECONÔMICA NÃO COMPROVADA."
            ),
            "data_cadastro": "2025-01-12 10:00:00-03",
        },
        {
            "id": 4,
            "fundamentacao": (
                "A parte autora apresentou documentos rurais contemporâneos, "
                "mas o conjunto probatório permaneceu frágil e insuficiente "
                "para demonstrar a qualidade de segurado especial durante a carência."
            ),
            "ementa": (
                "PREVIDENCIÁRIO. APOSENTADORIA RURAL POR IDADE. "
                "QUALIDADE DE SEGURADO ESPECIAL NÃO COMPROVADA."
            ),
            "data_cadastro": "2025-01-13 10:00:00-03",
        },
    ]

    _gravar_json(data_dir / "dados_brutos.json", registros_brutos)
    _gravar_json(
        data_dir / ".ingestao_stats.json",
        {"total_lidos": 4, "descartados_nulos": 0, "exportados": 4},
    )
    _gravar_json(
        data_dir / ".pipeline_timing.json",
        {
            "fase1_ingestao": 1,
            "fase2_higienizacao": 1,
            "fase3_anonimizacao": 1,
            "fase4_estatisticas": 1,
            "pipeline_total": 4,
        },
    )

    stats_f2 = higienizacao.processar(
        input_path=data_dir / "dados_brutos.json",
        output_path=data_dir / "dados_limpos.json",
    )
    stats_f3 = anonimizacao.gerar_datasets(
        input_path=data_dir / "dados_limpos.json",
        train_path=data_dir / "dataset_treino.jsonl",
        test_path=data_dir / "dataset_teste.jsonl",
        test_size=0.25,
    )
    relatorio = estatisticas.gerar_relatorio(
        brutos_path=data_dir / "dados_brutos.json",
        limpos_path=data_dir / "dados_limpos.json",
        treino_path=data_dir / "dataset_treino.jsonl",
        teste_path=data_dir / "dataset_teste.jsonl",
        output_path=data_dir / "estatisticas_corpus.json",
    )

    assert stats_f2.exportados == 4
    assert stats_f3.total > 0
    assert auditoria.audit(data_dir / "dataset_treino.jsonl") is True
    assert auditoria.audit(data_dir / "dataset_teste.jsonl") is True

    treino = (data_dir / "dataset_treino.jsonl").read_text(encoding="utf-8").strip().splitlines()
    teste = (data_dir / "dataset_teste.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(treino) == 3
    assert len(teste) == 1

    textos = []
    for linha in treino + teste:
        fund, ementa = extrair_fundamentacao_e_ementa(json.loads(linha))
        textos.append((fund, ementa))

    corpus_final = "\n".join(f"{fund}\n{ementa}" for fund, ementa in textos)
    assert "PESSOA TESTE UM" not in corpus_final
    assert "PESSOA TESTE TRES" not in corpus_final
    assert "PESSOA TESTE QUATRO" not in corpus_final
    assert "[NOME_OCULTADO]" in corpus_final
    assert "Súmula do julgamento" not in corpus_final

    estatisticas_json = json.loads((data_dir / "estatisticas_corpus.json").read_text(encoding="utf-8"))
    docs_json = json.loads((docs_data_dir / "estatisticas_corpus.json").read_text(encoding="utf-8"))

    assert estatisticas_json["fases"]["fase4_estatisticas"]["funil"]["dataset_final_fase3"] == 4
    assert estatisticas_json["fases"]["fase3_anonimizacao"]["treino"] == 3
    assert estatisticas_json["fases"]["fase3_anonimizacao"]["teste"] == 1
    assert estatisticas_json["fases"]["fase3_anonimizacao"]["pii_contagem"]["NOME_OCULTADO"] >= 3
    assert docs_json["fases"]["fase4_estatisticas"]["funil"]["dataset_final_fase3"] == 4
    assert relatorio["fases"]["fase4_estatisticas"]["funil"]["dataset_final_fase3"] == 4
