"""
05_finetuning_qwen.py — Fine-tuning supervisionado do Qwen na Fase 5

Prepara o dataset conversacional a partir do conjunto de treino e, em ambiente
com GPU/Unsloth, executa o fine-tuning LoRA do Qwen 2.5 14B-Instruct com
persistência de manifesto e checkpoint final.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path
from typing import Any

from pipeline.fase5.tuning_utils import (
    calcular_batch_size_efetivo,
    carregar_amostras_treino_sft,
    escrever_manifesto_tuning,
    gerar_nome_experimento,
)
from pipeline.core.project_paths import (
    DATASET_TREINO_PATH,
    FASE5_QWEN_CHECKPOINT_DIR,
    FASE5_QWEN_MANIFEST_PATH,
    SYSTEM_PROMPT_PATH,
)

log = logging.getLogger(__name__)

MODELO_BASE_PADRAO = "Qwen/Qwen2.5-14B-Instruct"
TARGET_MODULES_PADRAO = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def preparar_dataset_qwen(dataset_path: Path = DATASET_TREINO_PATH) -> list[dict[str, Any]]:
    """Converte o dataset de treino para o formato conversacional do SFTTrainer."""
    return carregar_amostras_treino_sft(dataset_path)


def executar_finetuning_qwen(
    *,
    dataset_path: Path = DATASET_TREINO_PATH,
    model_id: str = MODELO_BASE_PADRAO,
    output_dir: Path = FASE5_QWEN_CHECKPOINT_DIR,
    max_seq_length: int = 8192,
    lora_rank: int = 16,
    lora_alpha: int = 16,
    learning_rate: float = 1e-4,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 8,
    num_train_epochs: int = 3,
    warmup_steps: int = 10,
    logging_steps: int = 10,
    save_strategy: str = "epoch",
    save_total_limit: int = 2,
    load_in_4bit: bool = True,
    prepare_only: bool = False,
    seed: int = 3407,
    target_modules: tuple[str, ...] = TARGET_MODULES_PADRAO,
    manifest_path: Path = FASE5_QWEN_MANIFEST_PATH,
) -> Path:
    """Prepara e opcionalmente executa o fine-tuning LoRA do Qwen."""
    amostras = preparar_dataset_qwen(dataset_path)
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    effective_batch_size = calcular_batch_size_efetivo(
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    run_name = gerar_nome_experimento("qwen_sft")

    manifesto: dict[str, Any] = {
        "plataforma": "unsloth_trl",
        "model_id": model_id,
        "dataset_path_local": str(dataset_path),
        "system_prompt_path": str(SYSTEM_PROMPT_PATH),
        "system_prompt_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
        "output_dir": str(output_dir),
        "max_seq_length": max_seq_length,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "learning_rate": learning_rate,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": effective_batch_size,
        "num_train_epochs": num_train_epochs,
        "warmup_steps": warmup_steps,
        "logging_steps": logging_steps,
        "save_strategy": save_strategy,
        "save_total_limit": save_total_limit,
        "load_in_4bit": load_in_4bit,
        "seed": seed,
        "target_modules": list(target_modules),
        "train_samples": len(amostras),
        "run_name": run_name,
        "status": "prepared" if prepare_only else "training",
    }

    if prepare_only:
        escrever_manifesto_tuning(manifest_path, manifesto)
        return manifest_path

    try:
        from datasets import Dataset
        from trl import SFTConfig, SFTTrainer
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise ImportError(
            "Dependências ausentes: instale `datasets`, `trl` e `unsloth` no ambiente da Fase 5."
        ) from exc

    dataset = Dataset.from_list(amostras)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=list(target_modules),
        lora_alpha=lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=str(output_dir),
            dataset_text_field=None,
            max_seq_length=max_seq_length,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            save_strategy=save_strategy,
            save_total_limit=save_total_limit,
            seed=seed,
            report_to="none",
            run_name=run_name,
            packing=False,
        ),
    )
    resultado = trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    manifesto["status"] = "completed"
    manifesto["training_loss"] = getattr(resultado, "training_loss", None)
    manifesto["global_step"] = getattr(resultado, "global_step", None)
    manifesto["checkpoint_dir"] = str(output_dir)
    escrever_manifesto_tuning(manifest_path, manifesto)
    return manifest_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tuning LoRA do Qwen 2.5 14B-Instruct.")
    parser.add_argument("--dataset-path", type=Path, default=DATASET_TREINO_PATH)
    parser.add_argument("--model-id", default=MODELO_BASE_PADRAO)
    parser.add_argument("--output-dir", type=Path, default=FASE5_QWEN_CHECKPOINT_DIR)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-train-epochs", type=int, default=3)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-strategy", default="epoch")
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = _parse_args()
    output_path = executar_finetuning_qwen(
        dataset_path=args.dataset_path,
        model_id=args.model_id,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        load_in_4bit=not args.no_4bit,
        prepare_only=args.prepare_only,
        seed=args.seed,
    )
    log.info("Manifesto Qwen SFT persistido em %s", output_path)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, OSError, ValueError, ImportError) as exc:
        log.critical("Execução interrompida: %s", exc)
        sys.exit(1)
