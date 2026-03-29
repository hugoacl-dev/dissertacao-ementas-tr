"""
protocolo.py — Protocolo executável da Fase 7

Versiona o prompt do LLM-as-a-Judge, define os contratos mínimos dos
artefatos da Fase 7 e gera um manifesto JSON reprodutível para a etapa
de avaliação.
"""
from __future__ import annotations

import hashlib
import logging
import sys
from pathlib import Path
from typing import Any

from pipeline.core.artefato_utils import escrever_json_atomico
from pipeline.core.project_paths import (
    FASE7_AMOSTRA_HUMANA_PATH,
    FASE7_AVALIACAO_HUMANA_PATH,
    FASE7_AVALIACAO_JUDGE_PATH,
    FASE7_CASOS_AVALIACAO_PATH,
    FASE7_METRICAS_AUTOMATICAS_PATH,
    FASE7_PREDICAO_PATHS,
    FASE7_PROTOCOLO_PATH,
    FASE7_RELATORIO_ESTATISTICO_PATH,
    LLM_JUDGE_PROMPT_PATH,
)

log = logging.getLogger(__name__)

VERSAO_PROTOCOLO_FASE7 = "2026-03-29"
MODELO_JUIZ = "DeepSeek V3"

DIMENSOES_JUIZ = (
    "pertinencia_tematica",
    "completude_dispositiva",
    "fidelidade_factual",
    "concisao",
    "adequacao_terminologica",
)

CRITERIOS_AVALIACAO_HUMANA = (
    "adequacao",
    "completude",
    "concisao",
    "fluencia",
)

CONDICOES_EXPERIMENTAIS = (
    {
        "id": "gemini_ft",
        "modelo": "Gemini 2.5 Flash",
        "familia": "gemini",
        "condicao": "fine_tuned",
    },
    {
        "id": "gemini_zero_shot",
        "modelo": "Gemini 2.5 Flash",
        "familia": "gemini",
        "condicao": "zero_shot",
    },
    {
        "id": "qwen_ft",
        "modelo": "Qwen 2.5 14B-Instruct",
        "familia": "qwen",
        "condicao": "fine_tuned",
    },
    {
        "id": "qwen_zero_shot",
        "modelo": "Qwen 2.5 14B-Instruct",
        "familia": "qwen",
        "condicao": "zero_shot",
    },
)


def ler_prompt_llm_judge() -> str:
    """Lê o prompt versionado do LLM-as-a-Judge."""
    return LLM_JUDGE_PROMPT_PATH.read_text(encoding="utf-8").strip()


def calcular_sha256_texto(texto: str) -> str:
    """Calcula o SHA-256 do conteúdo textual fornecido."""
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def schema_resposta_llm_judge() -> dict[str, Any]:
    """Retorna o schema canônico da resposta do juiz automático."""
    propriedades = {
        dimensao: {
            "type": "object",
            "additionalProperties": False,
            "required": ["score", "justificativa"],
            "properties": {
                "score": {"type": "integer", "minimum": 1, "maximum": 5},
                "justificativa": {"type": "string", "minLength": 1},
            },
        }
        for dimensao in DIMENSOES_JUIZ
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(DIMENSOES_JUIZ),
        "properties": propriedades,
    }


def schema_registro_caso_avaliacao() -> dict[str, Any]:
    """Retorna o schema canônico de um caso da Fase 7."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["caso_id", "indice_teste", "fundamentacao", "ementa_referencia"],
        "properties": {
            "caso_id": {"type": "string", "minLength": 1},
            "indice_teste": {"type": "integer", "minimum": 0},
            "fundamentacao": {"type": "string", "minLength": 1},
            "ementa_referencia": {"type": "string", "minLength": 1},
        },
    }


def schema_registro_predicao() -> dict[str, Any]:
    """Retorna o schema canônico de uma predição da Fase 7."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["caso_id", "condicao_id", "ementa_gerada"],
        "properties": {
            "caso_id": {"type": "string", "minLength": 1},
            "condicao_id": {"type": "string", "minLength": 1},
            "ementa_gerada": {"type": "string", "minLength": 1},
        },
    }


def validar_resposta_llm_judge(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Valida o payload do LLM-as-a-Judge."""
    if not isinstance(payload, dict):
        raise ValueError("A resposta do juiz deve ser um objeto JSON.")

    chaves_esperadas = set(DIMENSOES_JUIZ)
    chaves_recebidas = set(payload)
    faltantes = chaves_esperadas - chaves_recebidas
    extras = chaves_recebidas - chaves_esperadas
    if faltantes:
        raise ValueError(f"Dimensões ausentes na resposta do juiz: {sorted(faltantes)}")
    if extras:
        raise ValueError(f"Dimensões inesperadas na resposta do juiz: {sorted(extras)}")

    normalizado: dict[str, dict[str, Any]] = {}
    for dimensao in DIMENSOES_JUIZ:
        bloco = payload[dimensao]
        if not isinstance(bloco, dict):
            raise ValueError(f"A dimensão '{dimensao}' deve ser um objeto.")
        if set(bloco) != {"score", "justificativa"}:
            raise ValueError(
                f"A dimensão '{dimensao}' deve conter apenas 'score' e 'justificativa'."
            )
        score = bloco.get("score")
        justificativa = bloco.get("justificativa")
        if not isinstance(score, int) or not (1 <= score <= 5):
            raise ValueError(f"O score de '{dimensao}' deve ser inteiro entre 1 e 5.")
        if not isinstance(justificativa, str) or not justificativa.strip():
            raise ValueError(
                f"A justificativa de '{dimensao}' deve ser texto não vazio."
            )
        normalizado[dimensao] = {
            "score": score,
            "justificativa": justificativa.strip(),
        }
    return normalizado


def validar_registro_caso_avaliacao(payload: dict[str, Any]) -> dict[str, Any]:
    """Valida um registro de caso-base da Fase 7."""
    if not isinstance(payload, dict):
        raise ValueError("O caso de avaliação deve ser um objeto JSON.")
    if set(payload) != {"caso_id", "indice_teste", "fundamentacao", "ementa_referencia"}:
        raise ValueError(
            "O caso de avaliação deve conter apenas "
            "`caso_id`, `indice_teste`, `fundamentacao` e `ementa_referencia`."
        )

    caso_id = payload.get("caso_id")
    indice_teste = payload.get("indice_teste")
    fundamentacao = payload.get("fundamentacao")
    ementa_referencia = payload.get("ementa_referencia")

    if not isinstance(caso_id, str) or not caso_id.strip():
        raise ValueError("`caso_id` deve ser texto não vazio.")
    if not isinstance(indice_teste, int) or indice_teste < 0:
        raise ValueError("`indice_teste` deve ser inteiro não negativo.")
    if not isinstance(fundamentacao, str) or not fundamentacao.strip():
        raise ValueError("`fundamentacao` deve ser texto não vazio.")
    if not isinstance(ementa_referencia, str) or not ementa_referencia.strip():
        raise ValueError("`ementa_referencia` deve ser texto não vazio.")

    return {
        "caso_id": caso_id.strip(),
        "indice_teste": indice_teste,
        "fundamentacao": fundamentacao.strip(),
        "ementa_referencia": ementa_referencia.strip(),
    }


def validar_registro_predicao(
    payload: dict[str, Any],
    *,
    condicao_id_esperada: str | None = None,
) -> dict[str, Any]:
    """Valida um registro de predição da Fase 7."""
    if not isinstance(payload, dict):
        raise ValueError("A predição deve ser um objeto JSON.")
    if set(payload) != {"caso_id", "condicao_id", "ementa_gerada"}:
        raise ValueError(
            "A predição deve conter apenas `caso_id`, `condicao_id` e `ementa_gerada`."
        )

    caso_id = payload.get("caso_id")
    condicao_id = payload.get("condicao_id")
    ementa_gerada = payload.get("ementa_gerada")
    condicoes_validas = {item["id"] for item in CONDICOES_EXPERIMENTAIS}

    if not isinstance(caso_id, str) or not caso_id.strip():
        raise ValueError("`caso_id` da predição deve ser texto não vazio.")
    if not isinstance(condicao_id, str) or not condicao_id.strip():
        raise ValueError("`condicao_id` da predição deve ser texto não vazio.")
    if condicao_id not in condicoes_validas:
        raise ValueError(f"`condicao_id` inválido na predição: {condicao_id}")
    if condicao_id_esperada is not None and condicao_id != condicao_id_esperada:
        raise ValueError(
            f"`condicao_id` divergente do arquivo esperado: {condicao_id} != {condicao_id_esperada}"
        )
    if not isinstance(ementa_gerada, str) or not ementa_gerada.strip():
        raise ValueError("`ementa_gerada` deve ser texto não vazio.")

    return {
        "caso_id": caso_id.strip(),
        "condicao_id": condicao_id,
        "ementa_gerada": ementa_gerada.strip(),
    }


def calcular_score_global_llm_judge(
    payload: dict[str, dict[str, Any]],
) -> float:
    """Calcula o score global médio a partir do payload validado."""
    scores = [payload[dimensao]["score"] for dimensao in DIMENSOES_JUIZ]
    return sum(scores) / len(scores)


def contrato_artefatos_fase7() -> dict[str, Any]:
    """Retorna o contrato mínimo dos artefatos da Fase 7."""
    return {
        "manifesto": str(FASE7_PROTOCOLO_PATH),
        "casos_avaliacao": str(FASE7_CASOS_AVALIACAO_PATH),
        "predicoes": {nome: str(path) for nome, path in FASE7_PREDICAO_PATHS.items()},
        "metricas_automaticas": str(FASE7_METRICAS_AUTOMATICAS_PATH),
        "avaliacao_llm_judge": str(FASE7_AVALIACAO_JUDGE_PATH),
        "amostra_humana": str(FASE7_AMOSTRA_HUMANA_PATH),
        "avaliacao_humana": str(FASE7_AVALIACAO_HUMANA_PATH),
        "relatorio_estatistico": str(FASE7_RELATORIO_ESTATISTICO_PATH),
    }


def gerar_manifesto_fase7() -> dict[str, Any]:
    """Gera o manifesto versionado do protocolo da Fase 7."""
    prompt = ler_prompt_llm_judge()
    return {
        "versao_protocolo": VERSAO_PROTOCOLO_FASE7,
        "co_desfechos_primarios": [
            "bertscore_f1",
            "judge_score_global",
        ],
        "condicoes_experimentais": list(CONDICOES_EXPERIMENTAIS),
        "llm_judge": {
            "modelo": MODELO_JUIZ,
            "prompt_path": str(LLM_JUDGE_PROMPT_PATH),
            "prompt_sha256": calcular_sha256_texto(prompt),
            "schema_resposta": schema_resposta_llm_judge(),
            "dimensoes": list(DIMENSOES_JUIZ),
            "score_global": "media_aritmetica_simples",
        },
        "schema_caso_avaliacao": schema_registro_caso_avaliacao(),
        "schema_predicao": schema_registro_predicao(),
        "avaliacao_humana": {
            "casos_amostrados": 40,
            "avaliadores": 2,
            "criterios": list(CRITERIOS_AVALIACAO_HUMANA),
            "escala": "likert_1_5",
            "cegamento": "ordem_aleatoria_sem_identificacao_de_modelo",
            "concordancia": "weighted_cohen_kappa_quadratico_por_criterio",
        },
        "inferencia": {
            "unidade": "caso_pareado",
            "bootstrap_pareado_iteracoes": 10000,
            "permutacao_pareada_iteracoes": 10000,
            "ajuste_primario": "holm_bonferroni",
            "ajuste_secundario": "benjamini_hochberg",
        },
        "artefatos": contrato_artefatos_fase7(),
    }


def escrever_manifesto_fase7(path: Path = FASE7_PROTOCOLO_PATH) -> Path:
    """Escreve o manifesto versionado da Fase 7 em disco."""
    manifesto = gerar_manifesto_fase7()
    escrever_json_atomico(path, manifesto, indent=2)
    return path


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    output_path = escrever_manifesto_fase7()
    log.info("Manifesto da Fase 7 gerado em %s", output_path)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, OSError, ValueError) as exc:
        log.critical("Execução interrompida: %s", exc)
        sys.exit(1)
