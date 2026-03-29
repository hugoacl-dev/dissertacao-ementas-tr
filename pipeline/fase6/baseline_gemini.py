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
from pathlib import Path
from typing import Any

from pipeline.fase7.predicoes_utils import (
    carregar_casos_predicao,
    carregar_predicoes_existentes,
    filtrar_casos_pendentes,
    normalizar_ementa_gerada,
    persistir_predicoes,
)
from pipeline.core.project_paths import FASE7_CASOS_AVALIACAO_PATH, FASE7_PREDICAO_PATHS, SYSTEM_PROMPT_PATH

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


def executar_baseline_gemini(
    *,
    casos_path: Path = FASE7_CASOS_AVALIACAO_PATH,
    output_path: Path = FASE7_PREDICAO_PATHS[CONDICAO_ID],
    model_id: str = MODELO_PADRAO,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_output_tokens: int = 256,
    limit: int | None = None,
    flush_every: int = 20,
) -> Path:
    """Executa o baseline zero-shot do Gemini com retomada incremental."""
    casos_df = carregar_casos_predicao(casos_path)
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    cliente = _construir_cliente_gemini()

    existentes = carregar_predicoes_existentes(output_path, condicao_id=CONDICAO_ID)
    registros = list(existentes)
    pendentes = filtrar_casos_pendentes(casos_df, existentes)
    if limit is not None:
        pendentes = pendentes[:limit]

    for indice, caso in enumerate(pendentes, start=1):
        ementa = gerar_ementa_gemini(
            cliente,
            model_id=model_id,
            system_prompt=system_prompt,
            fundamentacao=caso["fundamentacao"],
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
        )
        registros.append(
            {
                "caso_id": caso["caso_id"],
                "condicao_id": CONDICAO_ID,
                "ementa_gerada": ementa,
            }
        )
        if indice % flush_every == 0:
            persistir_predicoes(output_path, condicao_id=CONDICAO_ID, registros=registros)
            log.info("Gemini baseline: %s registros persistidos", len(registros))

    persistir_predicoes(output_path, condicao_id=CONDICAO_ID, registros=registros)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline zero-shot do Gemini 2.5 Flash.")
    parser.add_argument("--casos-path", type=Path, default=FASE7_CASOS_AVALIACAO_PATH)
    parser.add_argument("--output-path", type=Path, default=FASE7_PREDICAO_PATHS[CONDICAO_ID])
    parser.add_argument("--model-id", default=MODELO_PADRAO)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--flush-every", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = _parse_args()
    output_path = executar_baseline_gemini(
        casos_path=args.casos_path,
        output_path=args.output_path,
        model_id=args.model_id,
        temperature=args.temperature,
        top_p=args.top_p,
        max_output_tokens=args.max_output_tokens,
        limit=args.limit,
        flush_every=args.flush_every,
    )
    log.info("Predições do baseline Gemini persistidas em %s", output_path)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, OSError, ValueError, ImportError, EnvironmentError) as exc:
        log.critical("Execução interrompida: %s", exc)
        sys.exit(1)
