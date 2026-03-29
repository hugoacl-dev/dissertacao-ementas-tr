"""
06_baseline_qwen.py — Baseline zero-shot do Qwen na Fase 6

Gera as ementas zero-shot do Qwen 2.5 14B-Instruct para os casos de
avaliação, carregados do artefato canônico `casos_avaliacao.jsonl`, com
retomada incremental e persistência no schema canônico de predições.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from pipeline.fase7.predicoes_utils import (
    carregar_casos_predicao,
    carregar_predicoes_existentes,
    filtrar_casos_pendentes,
    normalizar_ementa_gerada,
    persistir_predicoes,
)
from pipeline.core.project_paths import FASE7_CASOS_AVALIACAO_PATH, FASE7_PREDICAO_PATHS, SYSTEM_PROMPT_PATH

log = logging.getLogger(__name__)

CONDICAO_ID = "qwen_zero_shot"
MODELO_PADRAO = "Qwen/Qwen2.5-14B-Instruct"


def _carregar_modelo_qwen(
    *,
    model_id: str,
    trust_remote_code: bool = False,
) -> tuple[object, object]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Dependências ausentes: instale `torch`, `transformers` e `accelerate` no ambiente das fases avançadas."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    return model, tokenizer


def gerar_ementa_qwen(
    model: object,
    tokenizer: object,
    *,
    system_prompt: str,
    fundamentacao: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    """Executa uma geração zero-shot no Qwen usando chat template."""
    import torch

    mensagens = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": fundamentacao},
    ]
    entradas = tokenizer.apply_chat_template(
        mensagens,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    entradas = {chave: valor.to(model.device) for chave, valor in entradas.items()}

    kwargs_geracao = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature <= 0:
        kwargs_geracao["do_sample"] = False
    else:
        kwargs_geracao["do_sample"] = True
        kwargs_geracao["temperature"] = temperature
        kwargs_geracao["top_p"] = top_p

    with torch.no_grad():
        saida = model.generate(**entradas, **kwargs_geracao)
    ids_gerados = saida[:, entradas["input_ids"].shape[1] :]
    texto = tokenizer.batch_decode(ids_gerados, skip_special_tokens=True)[0]
    return normalizar_ementa_gerada(texto)


def executar_baseline_qwen(
    *,
    casos_path: Path = FASE7_CASOS_AVALIACAO_PATH,
    output_path: Path = FASE7_PREDICAO_PATHS[CONDICAO_ID],
    model_id: str = MODELO_PADRAO,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    limit: int | None = None,
    flush_every: int = 20,
    trust_remote_code: bool = False,
) -> Path:
    """Executa o baseline zero-shot do Qwen com retomada incremental."""
    casos_df = carregar_casos_predicao(casos_path)
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    model, tokenizer = _carregar_modelo_qwen(
        model_id=model_id,
        trust_remote_code=trust_remote_code,
    )

    existentes = carregar_predicoes_existentes(output_path, condicao_id=CONDICAO_ID)
    registros = list(existentes)
    pendentes = filtrar_casos_pendentes(casos_df, existentes)
    if limit is not None:
        pendentes = pendentes[:limit]

    for indice, caso in enumerate(pendentes, start=1):
        ementa = gerar_ementa_qwen(
            model,
            tokenizer,
            system_prompt=system_prompt,
            fundamentacao=caso["fundamentacao"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
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
            log.info("Qwen baseline: %s registros persistidos", len(registros))

    persistir_predicoes(output_path, condicao_id=CONDICAO_ID, registros=registros)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline zero-shot do Qwen 2.5 14B-Instruct.")
    parser.add_argument("--casos-path", type=Path, default=FASE7_CASOS_AVALIACAO_PATH)
    parser.add_argument("--output-path", type=Path, default=FASE7_PREDICAO_PATHS[CONDICAO_ID])
    parser.add_argument("--model-id", default=MODELO_PADRAO)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--flush-every", type=int, default=20)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = _parse_args()
    output_path = executar_baseline_qwen(
        casos_path=args.casos_path,
        output_path=args.output_path,
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        limit=args.limit,
        flush_every=args.flush_every,
        trust_remote_code=args.trust_remote_code,
    )
    log.info("Predições do baseline Qwen persistidas em %s", output_path)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, OSError, ValueError, ImportError) as exc:
        log.critical("Execução interrompida: %s", exc)
        sys.exit(1)
