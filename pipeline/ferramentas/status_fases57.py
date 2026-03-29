"""
status_fases57.py — Gera um snapshot estático de prontidão das Fases 5–7

Objetivo:
- expor, no dashboard estático em `docs/`, o estado atual de implementação,
  validação exploratória e execução oficial das Fases 5–7;
- deixar explícito que smoke tests e artefatos exploratórios não são
  resultados finais da dissertação;
- preparar o front-end para a futura execução oficial sem trocar o schema
  do JSON consumido pelo dashboard.

Uso:
    python3 -m pipeline.ferramentas.status_fases57
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pipeline.core.artefato_utils import escrever_json_atomico
from pipeline.core.project_paths import (
    DOCS_DATA_DIR,
    PERFIL_EXECUCAO_EXPLORATORIO,
    PERFIL_EXECUCAO_OFICIAL,
    resolver_artefatos_fase5,
    resolver_artefatos_fase7,
)

STATUS_FASES_5_7_PATH = DOCS_DATA_DIR / "fases_5_7_status.json"


def ler_json_se_existir(path: Path) -> dict[str, Any] | None:
    """Lê um JSON se o arquivo existir e for válido; senão, retorna `None`."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def contar_linhas(path: Path) -> int | None:
    """Conta linhas em um arquivo texto; retorna `None` se ele não existir."""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def checkpoint_qwen_existe(path: Path) -> bool:
    """Verifica se o diretório de checkpoint do Qwen contém arquivos."""
    return path.exists() and path.is_dir() and any(path.iterdir())


def carregar_manifesto_com_perfil(
    path: Path,
    *,
    perfil_execucao: str,
    status_validos: tuple[str, ...] = ("completed", "prepared"),
) -> dict[str, Any] | None:
    """
    Carrega um manifesto apenas se ele corresponder ao perfil informado.

    Artefatos antigos sem `perfil_execucao` explícito são ignorados de forma
    conservadora para não sugerir falsamente uma rodada oficial já iniciada.
    """
    payload = ler_json_se_existir(path)
    if not payload:
        return None
    if payload.get("perfil_execucao") != perfil_execucao:
        return None
    if payload.get("status") not in status_validos:
        return None
    return payload


def agregar_status_validacao(statuses: list[str]) -> str:
    """Agrega status de validação exploratória por componente."""
    if not statuses or all(status == "pendente" for status in statuses):
        return "pendente"
    if all(status == "validada" for status in statuses):
        return "validada"
    return "parcial"


def agregar_status_execucao_oficial(statuses: list[str]) -> str:
    """Agrega status da execução oficial por componente."""
    if not statuses or all(status == "pendente" for status in statuses):
        return "pendente"
    if all(status == "concluida" for status in statuses):
        return "concluida"
    return "em_andamento"


def status_oficial_por_manifesto(
    manifest_path: Path,
    *,
    output_path: Path | None = None,
    status_validos: tuple[str, ...] = ("completed", "prepared"),
) -> tuple[str, dict[str, Any] | None]:
    """Classifica o estado oficial a partir de um manifesto compatível."""
    manifesto = carregar_manifesto_com_perfil(
        manifest_path,
        perfil_execucao=PERFIL_EXECUCAO_OFICIAL,
        status_validos=status_validos,
    )
    if manifesto is None:
        return "pendente", None

    if output_path is not None and not output_path.exists():
        return "em_andamento", manifesto

    return "concluida", manifesto


def carregar_manifesto_juiz_exploratorio() -> tuple[dict[str, Any] | None, Path | None]:
    """
    Retorna o melhor manifesto exploratório do juiz automático.

    Prioridade:
    1. smoke real com múltiplas observações (`*real_limit10*.manifest.json`)
    2. smoke real simples (`*real*.manifest.json`)
    3. smoke de checagem (`*chatcheck*.manifest.json`)
    4. manifesto canônico exploratório
    """
    artefatos = resolver_artefatos_fase7(PERFIL_EXECUCAO_EXPLORATORIO)
    fase7_dir: Path = artefatos["fase7_dir"]
    candidatos = sorted(
        fase7_dir.glob("avaliacao_llm_judge*.manifest.json"),
        key=lambda path: path.name,
    )
    ordenacao = (
        "avaliacao_llm_judge_smoke40_real_limit10.manifest.json",
        "avaliacao_llm_judge_smoke40_real.manifest.json",
        "avaliacao_llm_judge_smoke40_chatcheck.manifest.json",
        "avaliacao_llm_judge_manifest.json",
    )
    for nome in ordenacao:
        path = fase7_dir / nome
        if path in candidatos:
            payload = carregar_manifesto_com_perfil(
                path,
                perfil_execucao=PERFIL_EXECUCAO_EXPLORATORIO,
                status_validos=("completed",),
            )
            if payload is not None:
                return payload, path
    return None, None


def gerar_status_fases_5_7() -> dict[str, Any]:
    """Gera o payload consumido pelo dashboard estático das Fases 5–7."""
    fase5_expl = resolver_artefatos_fase5(PERFIL_EXECUCAO_EXPLORATORIO)
    fase5_ofc = resolver_artefatos_fase5(PERFIL_EXECUCAO_OFICIAL)
    fase7_expl = resolver_artefatos_fase7(PERFIL_EXECUCAO_EXPLORATORIO)
    fase7_ofc = resolver_artefatos_fase7(PERFIL_EXECUCAO_OFICIAL)

    gemini_expl_manifest = carregar_manifesto_com_perfil(
        fase5_expl["gemini_manifest_path"],
        perfil_execucao=PERFIL_EXECUCAO_EXPLORATORIO,
        status_validos=("prepared", "completed"),
    )
    qwen_expl_manifest = carregar_manifesto_com_perfil(
        fase5_expl["qwen_manifest_path"],
        perfil_execucao=PERFIL_EXECUCAO_EXPLORATORIO,
        status_validos=("prepared", "completed"),
    )

    gemini_ofc_status, gemini_ofc_manifest = status_oficial_por_manifesto(
        fase5_ofc["gemini_manifest_path"],
        output_path=fase5_ofc["gemini_modelo_path"],
    )
    if gemini_ofc_status == "concluida" and not fase5_ofc["gemini_modelo_path"].exists():
        gemini_ofc_status = "em_andamento"

    qwen_ofc_manifest = carregar_manifesto_com_perfil(
        fase5_ofc["qwen_manifest_path"],
        perfil_execucao=PERFIL_EXECUCAO_OFICIAL,
        status_validos=("prepared", "completed"),
    )
    if checkpoint_qwen_existe(fase5_ofc["qwen_checkpoint_dir"]):
        qwen_ofc_status = "concluida"
    elif qwen_ofc_manifest is not None:
        qwen_ofc_status = "em_andamento"
    else:
        qwen_ofc_status = "pendente"

    pred_manifestos_expl: dict[str, dict[str, Any] | None] = {
        condicao_id: carregar_manifesto_com_perfil(
            path,
            perfil_execucao=PERFIL_EXECUCAO_EXPLORATORIO,
            status_validos=("completed",),
        )
        for condicao_id, path in fase7_expl["predicao_manifest_paths"].items()
    }
    pred_manifestos_ofc: dict[str, dict[str, Any] | None] = {
        condicao_id: carregar_manifesto_com_perfil(
            path,
            perfil_execucao=PERFIL_EXECUCAO_OFICIAL,
            status_validos=("completed",),
        )
        for condicao_id, path in fase7_ofc["predicao_manifest_paths"].items()
    }

    judge_expl_manifest, judge_expl_manifest_path = carregar_manifesto_juiz_exploratorio()
    judge_ofc_status, judge_ofc_manifest = status_oficial_por_manifesto(
        fase7_ofc["avaliacao_judge_manifest_path"],
        output_path=fase7_ofc["avaliacao_judge_path"],
        status_validos=("completed",),
    )

    protocolo_expl = carregar_manifesto_com_perfil(
        fase7_expl["protocolo_path"],
        perfil_execucao=PERFIL_EXECUCAO_EXPLORATORIO,
        status_validos=("completed", "prepared", "gerado"),
    ) or ler_json_se_existir(fase7_expl["protocolo_path"])
    protocolo_ofc = carregar_manifesto_com_perfil(
        fase7_ofc["protocolo_path"],
        perfil_execucao=PERFIL_EXECUCAO_OFICIAL,
        status_validos=("completed", "prepared", "gerado"),
    )

    human_report_expl = ler_json_se_existir(fase7_expl["relatorio_avaliacao_humana_path"])
    human_report_smoke = ler_json_se_existir(
        fase7_expl["fase7_dir"] / "relatorio_avaliacao_humana_smoke40.json"
    )
    estatistico_smoke = ler_json_se_existir(
        fase7_expl["fase7_dir"] / "relatorio_estatistico_smoke40.json"
    )

    metricas_smoke_path = fase7_expl["fase7_dir"] / "metricas_automaticas_smoke40.csv"

    fase5_componentes = [
        {
            "id": "gemini_ft",
            "rotulo": "Gemini FT",
            "modelo": "Gemini 2.5 Flash",
            "ambiente": "Vertex AI",
            "implementacao_status": "concluida",
            "validacao_exploratoria_status": "validada" if gemini_expl_manifest else "pendente",
            "execucao_oficial_status": gemini_ofc_status,
            "dataset_exploratorio": "real" if gemini_expl_manifest else None,
            "evidencia_exploratoria": (
                "prepare-only real com upload do dataset de treino para GCS"
                if gemini_expl_manifest
                else "nenhum artefato exploratório detectado"
            ),
            "artefato_exploratorio": (
                str(fase5_expl["gemini_manifest_path"]) if gemini_expl_manifest else None
            ),
            "observacao": "A execução oficial permanece pendente até o congelamento metodológico.",
        },
        {
            "id": "qwen_ft",
            "rotulo": "Qwen FT",
            "modelo": "Qwen 2.5 14B-Instruct",
            "ambiente": "RunPod / GPU de 80 GB",
            "implementacao_status": "concluida",
            "validacao_exploratoria_status": "validada" if qwen_expl_manifest else "pendente",
            "execucao_oficial_status": qwen_ofc_status,
            "dataset_exploratorio": (
                "misto (prepare-only com dataset real; smoke GPU com dataset sintético mínimo)"
                if qwen_expl_manifest
                else None
            ),
            "evidencia_exploratoria": (
                "prepare-only local + smoke real em RunPod H100 80GB documentado em scripts e runbook"
                if qwen_expl_manifest
                else "nenhum artefato exploratório detectado"
            ),
            "artefato_exploratorio": (
                str(fase5_expl["qwen_manifest_path"]) if qwen_expl_manifest else None
            ),
            "observacao": "O treino oficial com JSONL reais ainda não foi executado no perfil oficial.",
        },
    ]

    fase6_componentes: list[dict[str, Any]] = []
    rotulos_fase6 = {
        "gemini_zero_shot": "Gemini Zero-Shot",
        "gemini_ft": "Gemini FT",
        "qwen_zero_shot": "Qwen Zero-Shot",
        "qwen_ft": "Qwen FT",
    }
    for condicao_id in ("gemini_zero_shot", "gemini_ft", "qwen_zero_shot", "qwen_ft"):
        manifest_expl = pred_manifestos_expl[condicao_id]
        manifest_ofc = pred_manifestos_ofc[condicao_id]
        output_expl = fase7_expl["predicao_paths"][condicao_id]
        output_ofc = fase7_ofc["predicao_paths"][condicao_id]
        fase6_componentes.append(
            {
                "id": condicao_id,
                "rotulo": rotulos_fase6[condicao_id],
                "modelo": "Gemini 2.5 Flash" if condicao_id.startswith("gemini") else "Qwen 2.5 14B-Instruct",
                "implementacao_status": "concluida",
                "validacao_exploratoria_status": "validada" if manifest_expl else "pendente",
                "execucao_oficial_status": "concluida" if manifest_ofc and output_ofc.exists() else "pendente",
                "dataset_exploratorio": (
                    "sintético (smoke40 canônico)" if manifest_expl and manifest_expl.get("modo_inferencia") == "synthetic_smoke_test"
                    else "real"
                    if manifest_expl
                    else None
                ),
                "evidencia_exploratoria": (
                    (
                        f"{manifest_expl.get('predicoes_persistidas', '—')} predições persistidas em smoke exploratório"
                    )
                    if manifest_expl
                    else "condição ainda sem artefato exploratório canônico"
                ),
                "artefato_exploratorio": str(output_expl) if manifest_expl else None,
                "observacao": (
                    "Os artefatos exploratórios atuais servem para validação operacional. O perfil oficial permanece vazio."
                ),
            }
        )

    fase7_componentes = [
        {
            "id": "casos_protocolo",
            "rotulo": "Casos-base e protocolo",
            "implementacao_status": "concluida",
            "validacao_exploratoria_status": (
                "validada"
                if protocolo_expl and fase7_expl["casos_avaliacao_path"].exists()
                else "pendente"
            ),
            "execucao_oficial_status": (
                "concluida"
                if protocolo_ofc and fase7_ofc["casos_avaliacao_path"].exists()
                else "pendente"
            ),
            "dataset_exploratorio": "real" if fase7_expl["casos_avaliacao_path"].exists() else None,
            "evidencia_exploratoria": (
                "casos-base gerados no perfil exploratório e protocolo versionado persistido"
                if protocolo_expl and fase7_expl["casos_avaliacao_path"].exists()
                else "sem casos-base/protocolo exploratório detectados"
            ),
            "artefato_exploratorio": str(fase7_expl["protocolo_path"]) if protocolo_expl else None,
            "observacao": "A rodada oficial só deve começar após o congelamento metodológico.",
        },
        {
            "id": "judge",
            "rotulo": "Juiz automático",
            "implementacao_status": "concluida",
            "validacao_exploratoria_status": "validada" if judge_expl_manifest else "pendente",
            "execucao_oficial_status": judge_ofc_status,
            "dataset_exploratorio": "real (smoke curto)" if judge_expl_manifest else None,
            "evidencia_exploratoria": (
                f"{judge_expl_manifest.get('avaliacoes_persistidas', '—')} avaliações persistidas com `deepseek-chat`"
                if judge_expl_manifest
                else "sem smoke real do juiz detectado no workspace"
            ),
            "artefato_exploratorio": str(judge_expl_manifest_path) if judge_expl_manifest_path else None,
            "observacao": "As avaliações exploratórias do juiz não devem ser tratadas como análise final.",
        },
        {
            "id": "humana",
            "rotulo": "Avaliação humana",
            "implementacao_status": "concluida",
            "validacao_exploratoria_status": "validada" if human_report_smoke or human_report_expl else "pendente",
            "execucao_oficial_status": (
                "concluida" if fase7_ofc["relatorio_avaliacao_humana_path"].exists() else "pendente"
            ),
            "dataset_exploratorio": "sintético (smoke40)" if human_report_smoke else None,
            "evidencia_exploratoria": (
                f"{human_report_smoke.get('n_casos', '—')} casos, {human_report_smoke.get('n_registros_avaliacao', '—')} registros"
                if human_report_smoke
                else "infraestrutura cega implementada, sem relatório exploratório detectado"
            ),
            "artefato_exploratorio": (
                str(fase7_expl["fase7_dir"] / "relatorio_avaliacao_humana_smoke40.json")
                if human_report_smoke
                else None
            ),
            "observacao": "O relatório exploratório disponível é sintético e serve apenas para smoke test da infraestrutura.",
        },
        {
            "id": "metricas_estatistica",
            "rotulo": "Métricas e estatística",
            "implementacao_status": "concluida",
            "validacao_exploratoria_status": (
                "validada" if metricas_smoke_path.exists() and estatistico_smoke else "pendente"
            ),
            "execucao_oficial_status": (
                "concluida"
                if fase7_ofc["metricas_automaticas_path"].exists()
                and fase7_ofc["relatorio_estatistico_path"].exists()
                else "pendente"
            ),
            "dataset_exploratorio": "sintético (smoke40)" if metricas_smoke_path.exists() else None,
            "evidencia_exploratoria": (
                f"{contar_linhas(metricas_smoke_path) - 1 if contar_linhas(metricas_smoke_path) else '—'} linhas em metricas_automaticas_smoke40.csv"
                if metricas_smoke_path.exists() and estatistico_smoke
                else "sem consolidação exploratória detectada"
            ),
            "artefato_exploratorio": str(metricas_smoke_path) if metricas_smoke_path.exists() else None,
            "observacao": "Os números do smoke sintético não têm valor inferencial para a dissertação.",
        },
    ]

    fase5 = {
        "numero": 5,
        "nome": "Fine-Tuning Supervisionado",
        "descricao": (
            "Infraestrutura de preparação e treinamento dos modelos Gemini e Qwen."
        ),
        "implementacao_status": "concluida",
        "validacao_exploratoria_status": agregar_status_validacao(
            [item["validacao_exploratoria_status"] for item in fase5_componentes]
        ),
        "execucao_oficial_status": agregar_status_execucao_oficial(
            [item["execucao_oficial_status"] for item in fase5_componentes]
        ),
        "aviso_leitor": (
            "Esta seção documenta prontidão técnica e smoke tests. Nenhum modelo oficial foi treinado ainda."
        ),
        "bloqueio_principal": "Congelamento metodológico e decisão final sobre a rodada oficial.",
        "componentes": fase5_componentes,
    }
    fase6 = {
        "numero": 6,
        "nome": "Inferência e Predições",
        "descricao": (
            "Runners canônicos das quatro condições experimentais, com retomada incremental e manifests próprios."
        ),
        "implementacao_status": "concluida",
        "validacao_exploratoria_status": agregar_status_validacao(
            [item["validacao_exploratoria_status"] for item in fase6_componentes]
        ),
        "execucao_oficial_status": agregar_status_execucao_oficial(
            [item["execucao_oficial_status"] for item in fase6_componentes]
        ),
        "aviso_leitor": (
            "As predições mostradas no perfil exploratório servem para validação operacional. Não são resultados finais."
        ),
        "bloqueio_principal": "Rodada oficial ainda não iniciada nas quatro condições.",
        "componentes": fase6_componentes,
    }
    fase7 = {
        "numero": 7,
        "nome": "Avaliação Final",
        "descricao": (
            "Casos-base, protocolo, juiz automático, avaliação humana, métricas e inferência estatística."
        ),
        "implementacao_status": "concluida",
        "validacao_exploratoria_status": agregar_status_validacao(
            [item["validacao_exploratoria_status"] for item in fase7_componentes]
        ),
        "execucao_oficial_status": agregar_status_execucao_oficial(
            [item["execucao_oficial_status"] for item in fase7_componentes]
        ),
        "aviso_leitor": (
            "Os artefatos exploratórios desta fase foram usados apenas para smoke tests sintéticos ou integrações curtas."
        ),
        "bloqueio_principal": "Execução oficial depende das quatro condições completas e do congelamento do protocolo.",
        "componentes": fase7_componentes,
    }

    payload = {
        "meta": {
            "gerado_em": datetime.now(UTC).replace(microsecond=0).isoformat(),
            "fonte_script": "python3 -m pipeline.ferramentas.status_fases57",
            "titulo": "Prontidão Experimental das Fases 5–7",
            "aviso_principal": (
                "Implementação concluída e smoke tests exploratórios disponíveis; execução oficial ainda pendente."
            ),
            "criterio_conservador": (
                "Artefatos antigos sem `perfil_execucao` explícito são ignorados para não sugerir falsamente uma rodada oficial."
            ),
        },
        "sumario": {
            "implementacao_status": "concluida",
            "validacao_exploratoria_status": agregar_status_validacao(
                [
                    fase5["validacao_exploratoria_status"],
                    fase6["validacao_exploratoria_status"],
                    fase7["validacao_exploratoria_status"],
                ]
            ),
            "execucao_oficial_status": agregar_status_execucao_oficial(
                [
                    fase5["execucao_oficial_status"],
                    fase6["execucao_oficial_status"],
                    fase7["execucao_oficial_status"],
                ]
            ),
            "mensagem": (
                "O painel abaixo mostra prontidão técnica, não resultados científicos consolidados."
            ),
        },
        "fases": {
            "fase5": fase5,
            "fase6": fase6,
            "fase7": fase7,
        },
    }
    return payload


def main() -> None:
    """Gera o snapshot estático para o dashboard de documentação."""
    escrever_json_atomico(STATUS_FASES_5_7_PATH, gerar_status_fases_5_7())
    print(f"Status das Fases 5–7 atualizado em {STATUS_FASES_5_7_PATH}")


if __name__ == "__main__":
    main()
