"""
06_baseline_gemini.py — Baseline zero-shot do Gemini na Fase 6

Gera as ementas zero-shot do Gemini 2.5 Flash para os casos de avaliação,
carregados do artefato canônico `casos_avaliacao.jsonl`, com retomada
incremental e persistência no schema canônico de predições.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from pipeline.core.artefato_utils import escrever_json_atomico
from pipeline.fase7.predicoes_utils import (
    carregar_casos_predicao,
    carregar_predicoes_existentes,
    filtrar_casos_pendentes,
    normalizar_ementa_gerada,
    persistir_predicoes,
)
from pipeline.core.project_paths import (
    FASE7_CASOS_AVALIACAO_PATH,
    FASE7_PREDICAO_MANIFEST_PATHS,
    FASE7_PREDICAO_PATHS,
    PERFIL_EXECUCAO_CLI_PADRAO,
    PERFIL_EXECUCAO_OFICIAL,
    PERFIS_EXECUCAO,
    SYSTEM_PROMPT_PATH,
    resolver_manifestos_predicoes_fase7,
    resolver_predicoes_fase7,
    validar_perfil_execucao,
)
from pipeline.fase7.protocolo import CONDICOES_EXPERIMENTAIS, calcular_sha256_texto

log = logging.getLogger(__name__)

CONDICAO_ID = "gemini_zero_shot"
MODELO_PADRAO = "gemini-2.5-flash"


def _construir_cliente_gemini() -> Any:
    try:
        from google import genai
        from google.genai.types import HttpOptions
    except ImportError as exc:
        raise ImportError(
            "Dependência ausente: instale `google-genai` no ambiente das fases avançadas."
        ) from exc

    obrigatorias = ["GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION", "GOOGLE_GENAI_USE_VERTEXAI"]
    faltantes = [nome for nome in obrigatorias if not os.getenv(nome)]
    if faltantes:
        raise EnvironmentError(
            "Variáveis de ambiente obrigatórias ausentes para Vertex AI: "
            + ", ".join(faltantes)
        )
    return genai.Client(http_options=HttpOptions(api_version="v1"))


def _extrair_texto_resposta_gemini(resposta: Any) -> str:
    texto = getattr(resposta, "text", None)
    if isinstance(texto, str) and texto.strip():
        return texto

    candidatos = getattr(resposta, "candidates", None) or []
    for candidato in candidatos:
        content = getattr(candidato, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str) and part_text.strip():
                return part_text

    raise ValueError("Resposta do Gemini sem texto utilizável.")


def gerar_ementa_gemini(
    cliente: Any,
    *,
    model_id: str,
    system_prompt: str,
    fundamentacao: str,
    temperature: float,
    top_p: float,
    max_output_tokens: int,
) -> str:
    """Executa uma geração zero-shot no Gemini."""
    from google.genai.types import GenerateContentConfig

    resposta = cliente.models.generate_content(
        model=model_id,
        contents=fundamentacao,
        config=GenerateContentConfig(
            system_instruction=[system_prompt],
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
        ),
    )
    return normalizar_ementa_gerada(_extrair_texto_resposta_gemini(resposta))


def _validar_condicao_gemini(condicao_id: str) -> str:
    condicoes_validas = {
        item["id"]
        for item in CONDICOES_EXPERIMENTAIS
        if item["familia"] == "gemini"
    }
    if condicao_id not in condicoes_validas:
        raise ValueError(
            f"`condicao_id` inválido para o runner Gemini: {condicao_id}. "
            f"Use uma dentre {sorted(condicoes_validas)}."
        )
    return condicao_id


def _validar_modelo_gemini_para_condicao(*, condicao_id: str, model_id: str) -> None:
    """Impede rotulagem fine-tuned com o modelo base padrão."""
    if condicao_id == "gemini_ft" and model_id == MODELO_PADRAO:
        raise ValueError(
            "A condição `gemini_ft` exige informar explicitamente o identificador "
            "do modelo ajustado gerado na Fase 5."
        )


def _gerar_ementa_gemini_com_retry(
    cliente: Any,
    *,
    model_id: str,
    system_prompt: str,
    fundamentacao: str,
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    max_retries: int,
    retry_backoff_seconds: float,
) -> str:
    ultima_exc: Exception | None = None
    for tentativa in range(1, max_retries + 1):
        try:
            return gerar_ementa_gemini(
                cliente,
                model_id=model_id,
                system_prompt=system_prompt,
                fundamentacao=fundamentacao,
                temperature=temperature,
                top_p=top_p,
                max_output_tokens=max_output_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - tolerância operacional a falhas transitórias
            ultima_exc = exc
            if tentativa == max_retries:
                break
            espera = retry_backoff_seconds * (2 ** (tentativa - 1))
            log.warning(
                "Falha transitória no Gemini (%s/%s): %s. Nova tentativa em %.1fs.",
                tentativa,
                max_retries,
                exc,
                espera,
            )
            time.sleep(espera)
    assert ultima_exc is not None
    raise RuntimeError(
        f"Falha ao gerar ementa no Gemini após {max_retries} tentativas."
    ) from ultima_exc


def executar_baseline_gemini(
    *,
    casos_path: Path = FASE7_CASOS_AVALIACAO_PATH,
    output_path: Path | None = None,
    model_id: str = MODELO_PADRAO,
    condicao_id: str = CONDICAO_ID,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_output_tokens: int = 256,
    limit: int | None = None,
    flush_every: int = 20,
    max_retries: int = 3,
    retry_backoff_seconds: float = 2.0,
    perfil_execucao: str = PERFIL_EXECUCAO_OFICIAL,
    manifest_path: Path | None = None,
) -> Path:
    """Executa inferência do Gemini com retomada incremental.

    Pode ser usado tanto para a condição zero-shot quanto para a condição
    fine-tuned, desde que `condicao_id` e `model_id` sejam consistentes.
    """
    condicao_id = _validar_condicao_gemini(condicao_id)
    perfil_execucao = validar_perfil_execucao(perfil_execucao)
    _validar_modelo_gemini_para_condicao(condicao_id=condicao_id, model_id=model_id)
    if output_path is None:
        output_path = FASE7_PREDICAO_PATHS[condicao_id]
    if manifest_path is None:
        manifest_path = FASE7_PREDICAO_MANIFEST_PATHS[condicao_id]
    if flush_every <= 0:
        raise ValueError("`flush_every` deve ser inteiro positivo.")
    if limit is not None and limit <= 0:
        raise ValueError("`limit` deve ser positivo quando informado.")
    if max_retries <= 0:
        raise ValueError("`max_retries` deve ser inteiro positivo.")
    if retry_backoff_seconds < 0:
        raise ValueError("`retry_backoff_seconds` não pode ser negativo.")

    casos_df = carregar_casos_predicao(casos_path)
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()

    existentes = carregar_predicoes_existentes(output_path, condicao_id=condicao_id)
    registros = list(existentes)
    pendentes = filtrar_casos_pendentes(casos_df, existentes)
    if limit is not None:
        pendentes = pendentes[:limit]

    manifesto: dict[str, Any] = {
        "condicao_id": condicao_id,
        "perfil_execucao": perfil_execucao,
        "familia_modelo": "gemini",
        "model_id": model_id,
        "casos_path": str(casos_path),
        "output_path": str(output_path),
        "system_prompt_path": str(SYSTEM_PROMPT_PATH),
        "system_prompt_sha256": calcular_sha256_texto(system_prompt),
        "temperature": temperature,
        "top_p": top_p,
        "max_output_tokens": max_output_tokens,
        "flush_every": flush_every,
        "max_retries": max_retries,
        "retry_backoff_seconds": retry_backoff_seconds,
        "total_casos_base": int(len(casos_df)),
        "predicoes_existentes": len(existentes),
        "predicoes_pendentes_planejadas": len(pendentes),
        "status": "running",
    }
    escrever_json_atomico(manifest_path, manifesto, indent=2)

    if not pendentes:
        manifesto["status"] = "completed"
        manifesto["predicoes_persistidas"] = len(registros)
        manifesto["predicoes_geradas_nesta_execucao"] = 0
        escrever_json_atomico(manifest_path, manifesto, indent=2)
        return output_path

    cliente = _construir_cliente_gemini()

    try:
        for indice, caso in enumerate(pendentes, start=1):
            ementa = _gerar_ementa_gemini_com_retry(
                cliente,
                model_id=model_id,
                system_prompt=system_prompt,
                fundamentacao=caso["fundamentacao"],
                temperature=temperature,
                top_p=top_p,
                max_output_tokens=max_output_tokens,
                max_retries=max_retries,
                retry_backoff_seconds=retry_backoff_seconds,
            )
            registros.append(
                {
                    "caso_id": caso["caso_id"],
                    "condicao_id": condicao_id,
                    "ementa_gerada": ementa,
                }
            )
            if indice % flush_every == 0:
                persistir_predicoes(output_path, condicao_id=condicao_id, registros=registros)
                log.info("Gemini %s: %s registros persistidos", condicao_id, len(registros))

        persistir_predicoes(output_path, condicao_id=condicao_id, registros=registros)
    except Exception as exc:  # noqa: BLE001 - manifesto de falha
        manifesto["status"] = "failed"
        manifesto["erro"] = str(exc)
        manifesto["predicoes_persistidas"] = len(registros)
        escrever_json_atomico(manifest_path, manifesto, indent=2)
        raise

    manifesto["status"] = "completed"
    manifesto["predicoes_persistidas"] = len(registros)
    manifesto["predicoes_geradas_nesta_execucao"] = len(registros) - len(existentes)
    escrever_json_atomico(manifest_path, manifesto, indent=2)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inferência do Gemini 2.5 Flash para condições zero-shot ou fine-tuned."
    )
    parser.add_argument(
        "--perfil-execucao",
        choices=PERFIS_EXECUCAO,
        default=PERFIL_EXECUCAO_CLI_PADRAO,
    )
    parser.add_argument("--casos-path", type=Path, default=FASE7_CASOS_AVALIACAO_PATH)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--model-id", default=MODELO_PADRAO)
    parser.add_argument("--condicao-id", default=CONDICAO_ID)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--flush-every", type=int, default=20)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-backoff-seconds", type=float, default=2.0)
    parser.add_argument("--manifest-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = _parse_args()
    predicao_paths = resolver_predicoes_fase7(args.perfil_execucao)
    manifest_paths = resolver_manifestos_predicoes_fase7(args.perfil_execucao)
    output_path = executar_baseline_gemini(
        casos_path=args.casos_path,
        output_path=args.output_path or predicao_paths[args.condicao_id],
        model_id=args.model_id,
        condicao_id=args.condicao_id,
        temperature=args.temperature,
        top_p=args.top_p,
        max_output_tokens=args.max_output_tokens,
        limit=args.limit,
        flush_every=args.flush_every,
        max_retries=args.max_retries,
        retry_backoff_seconds=args.retry_backoff_seconds,
        perfil_execucao=args.perfil_execucao,
        manifest_path=args.manifest_path or manifest_paths[args.condicao_id],
    )
    log.info("Predições do baseline Gemini persistidas em %s", output_path)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, OSError, ValueError, ImportError, EnvironmentError) as exc:
        log.critical("Execução interrompida: %s", exc)
        sys.exit(1)
