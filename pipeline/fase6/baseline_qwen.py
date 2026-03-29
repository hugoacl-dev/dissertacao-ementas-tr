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
    resolver_artefatos_fase7,
    resolver_manifestos_predicoes_fase7,
    resolver_predicoes_fase7,
    validar_perfil_execucao,
)
from pipeline.fase7.protocolo import CONDICOES_EXPERIMENTAIS, calcular_sha256_texto

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

    model_path = Path(model_id)
    usa_adapter_peft = model_path.exists() and model_path.is_dir() and (model_path / "adapter_config.json").exists()
    if usa_adapter_peft:
        try:
            from peft import AutoPeftModelForCausalLM
        except ImportError as exc:
            raise ImportError(
                "Checkpoint LoRA detectado, mas `peft` não está disponível no ambiente."
            ) from exc
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
    else:
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
    output_path: Path | None = None,
    model_id: str = MODELO_PADRAO,
    condicao_id: str = CONDICAO_ID,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    limit: int | None = None,
    flush_every: int = 20,
    trust_remote_code: bool = False,
    perfil_execucao: str = PERFIL_EXECUCAO_OFICIAL,
    manifest_path: Path | None = None,
) -> Path:
    """Executa inferência do Qwen com retomada incremental.

    Pode ser usado tanto para a condição zero-shot quanto para a condição
    fine-tuned, incluindo checkpoints LoRA locais.
    """
    condicoes_validas = {
        item["id"]
        for item in CONDICOES_EXPERIMENTAIS
        if item["familia"] == "qwen"
    }
    if condicao_id not in condicoes_validas:
        raise ValueError(
            f"`condicao_id` inválido para o runner Qwen: {condicao_id}. "
            f"Use uma dentre {sorted(condicoes_validas)}."
        )
    perfil_execucao = validar_perfil_execucao(perfil_execucao)
    if condicao_id == "qwen_ft":
        model_path = Path(model_id)
        if not model_path.exists() or not model_path.is_dir():
            raise ValueError(
                "A condição `qwen_ft` exige um diretório local de checkpoint gerado na Fase 5."
            )
        if not (model_path / "adapter_config.json").exists():
            raise ValueError(
                "A condição `qwen_ft` exige checkpoint LoRA válido com `adapter_config.json`."
            )
    if output_path is None:
        output_path = FASE7_PREDICAO_PATHS[condicao_id]
    if manifest_path is None:
        manifest_path = FASE7_PREDICAO_MANIFEST_PATHS[condicao_id]
    if flush_every <= 0:
        raise ValueError("`flush_every` deve ser inteiro positivo.")
    if limit is not None and limit <= 0:
        raise ValueError("`limit` deve ser positivo quando informado.")

    casos_df = carregar_casos_predicao(casos_path)
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()

    existentes = carregar_predicoes_existentes(output_path, condicao_id=condicao_id)
    registros = list(existentes)
    pendentes = filtrar_casos_pendentes(casos_df, existentes)
    if limit is not None:
        pendentes = pendentes[:limit]

    usa_adapter_peft = Path(model_id).exists() and (Path(model_id) / "adapter_config.json").exists()
    manifesto: dict[str, Any] = {
        "condicao_id": condicao_id,
        "perfil_execucao": perfil_execucao,
        "familia_modelo": "qwen",
        "model_id": model_id,
        "modo_inferencia": "peft_adapter_local" if usa_adapter_peft else "modelo_base_ou_mergeado",
        "casos_path": str(casos_path),
        "output_path": str(output_path),
        "system_prompt_path": str(SYSTEM_PROMPT_PATH),
        "system_prompt_sha256": calcular_sha256_texto(system_prompt),
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "flush_every": flush_every,
        "trust_remote_code": trust_remote_code,
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

    model, tokenizer = _carregar_modelo_qwen(
        model_id=model_id,
        trust_remote_code=trust_remote_code,
    )

    try:
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
                    "condicao_id": condicao_id,
                    "ementa_gerada": ementa,
                }
            )
            if indice % flush_every == 0:
                persistir_predicoes(output_path, condicao_id=condicao_id, registros=registros)
                log.info("Qwen %s: %s registros persistidos", condicao_id, len(registros))

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
        description="Inferência do Qwen 2.5 14B-Instruct para condições zero-shot ou fine-tuned."
    )
    parser.add_argument(
        "--perfil-execucao",
        choices=PERFIS_EXECUCAO,
        default=PERFIL_EXECUCAO_CLI_PADRAO,
    )
    parser.add_argument("--casos-path", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--model-id", default=MODELO_PADRAO)
    parser.add_argument("--condicao-id", default=CONDICAO_ID)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--flush-every", type=int, default=20)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--manifest-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = _parse_args()
    artefatos_fase7 = resolver_artefatos_fase7(args.perfil_execucao)
    predicao_paths = resolver_predicoes_fase7(args.perfil_execucao)
    manifest_paths = resolver_manifestos_predicoes_fase7(args.perfil_execucao)
    output_path = executar_baseline_qwen(
        casos_path=args.casos_path or artefatos_fase7["casos_avaliacao_path"],
        output_path=args.output_path or predicao_paths[args.condicao_id],
        model_id=args.model_id,
        condicao_id=args.condicao_id,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        limit=args.limit,
        flush_every=args.flush_every,
        trust_remote_code=args.trust_remote_code,
        perfil_execucao=args.perfil_execucao,
        manifest_path=args.manifest_path or manifest_paths[args.condicao_id],
    )
    log.info("Predições do baseline Qwen persistidas em %s", output_path)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, OSError, ValueError, ImportError) as exc:
        log.critical("Execução interrompida: %s", exc)
        sys.exit(1)
