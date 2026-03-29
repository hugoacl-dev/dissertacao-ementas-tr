"""
05_finetuning_gemini.py — Fine-tuning supervisionado do Gemini na Fase 5

Prepara e, opcionalmente, submete um job de supervised fine-tuning no Vertex AI
para o Gemini 2.5 Flash. O script registra um manifesto local da execução e
persiste o nome do modelo gerado quando o job conclui.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

from pipeline.fase5.tuning_utils import (
    construir_uri_gcs,
    escrever_manifesto_tuning,
    gerar_nome_experimento,
)
from pipeline.core.project_paths import (
    DATASET_TREINO_PATH,
    FASE5_GEMINI_MANIFEST_PATH,
    FASE5_GEMINI_MODELO_PATH,
)

log = logging.getLogger(__name__)

MODELO_BASE_PADRAO = "gemini-2.5-flash"
REGIAO_PADRAO = "us-central1"


def _upload_para_gcs(local_path: Path, *, bucket: str, objeto: str) -> str:
    try:
        from google.cloud import storage
    except ImportError as exc:
        raise ImportError(
            "Dependência ausente: instale `google-cloud-storage` no ambiente da Fase 5."
        ) from exc

    bucket_nome = bucket.removeprefix("gs://").strip("/")
    cliente = storage.Client()
    blob = cliente.bucket(bucket_nome).blob(objeto.lstrip("/"))
    blob.upload_from_filename(str(local_path))
    return construir_uri_gcs(bucket_nome, objeto)


def _submeter_tuning_gemini(
    *,
    project_id: str,
    location: str,
    source_model: str,
    train_dataset_gcs_uri: str,
    tuned_model_display_name: str,
    validation_dataset_gcs_uri: str | None = None,
    epochs: int | None = None,
    adapter_size: int | None = None,
    learning_rate_multiplier: float | None = None,
) -> Any:
    try:
        import vertexai
        from vertexai.tuning import sft
    except ImportError as exc:
        raise ImportError(
            "Dependência ausente: instale `google-cloud-aiplatform` no ambiente da Fase 5."
        ) from exc

    vertexai.init(project=project_id, location=location)

    kwargs: dict[str, Any] = {
        "source_model": source_model,
        "train_dataset": train_dataset_gcs_uri,
        "tuned_model_display_name": tuned_model_display_name,
    }
    if validation_dataset_gcs_uri:
        kwargs["validation_dataset"] = validation_dataset_gcs_uri
    if epochs is not None:
        kwargs["epochs"] = epochs
    if adapter_size is not None:
        kwargs["adapter_size"] = adapter_size
    if learning_rate_multiplier is not None:
        kwargs["learning_rate_multiplier"] = learning_rate_multiplier

    return sft.train(**kwargs)


def executar_finetuning_gemini(
    *,
    project_id: str,
    location: str = REGIAO_PADRAO,
    dataset_path: Path = DATASET_TREINO_PATH,
    source_model: str = MODELO_BASE_PADRAO,
    train_dataset_gcs_uri: str | None = None,
    validation_dataset_gcs_uri: str | None = None,
    staging_bucket: str | None = None,
    gcs_prefix: str = "dissertacao-ementas-tr/fase5",
    tuned_model_display_name: str | None = None,
    epochs: int | None = None,
    adapter_size: int | None = None,
    learning_rate_multiplier: float | None = None,
    wait: bool = False,
    poll_interval_seconds: int = 60,
    prepare_only: bool = False,
    manifest_path: Path = FASE5_GEMINI_MANIFEST_PATH,
    output_model_name_path: Path = FASE5_GEMINI_MODELO_PATH,
) -> Path:
    """Prepara e opcionalmente submete o job SFT do Gemini."""
    if train_dataset_gcs_uri is None:
        if staging_bucket is None:
            raise ValueError(
                "Informe `train_dataset_gcs_uri` ou `staging_bucket` para disponibilizar o dataset ao Vertex AI."
            )
        objeto = f"{gcs_prefix.strip('/')}/{dataset_path.name}"
        train_dataset_gcs_uri = _upload_para_gcs(dataset_path, bucket=staging_bucket, objeto=objeto)

    display_name = tuned_model_display_name or gerar_nome_experimento("gemini_sft")
    manifesto: dict[str, Any] = {
        "plataforma": "vertex_ai",
        "source_model": source_model,
        "project_id": project_id,
        "location": location,
        "dataset_path_local": str(dataset_path),
        "train_dataset_gcs_uri": train_dataset_gcs_uri,
        "validation_dataset_gcs_uri": validation_dataset_gcs_uri,
        "tuned_model_display_name": display_name,
        "epochs": epochs,
        "adapter_size": adapter_size,
        "learning_rate_multiplier": learning_rate_multiplier,
        "status": "prepared" if prepare_only else "submitted",
    }

    if prepare_only:
        escrever_manifesto_tuning(manifest_path, manifesto)
        return manifest_path

    job = _submeter_tuning_gemini(
        project_id=project_id,
        location=location,
        source_model=source_model,
        train_dataset_gcs_uri=train_dataset_gcs_uri,
        validation_dataset_gcs_uri=validation_dataset_gcs_uri,
        tuned_model_display_name=display_name,
        epochs=epochs,
        adapter_size=adapter_size,
        learning_rate_multiplier=learning_rate_multiplier,
    )
    manifesto["job_resource_name"] = getattr(job, "resource_name", str(job))

    if wait:
        while not job.has_ended:
            time.sleep(poll_interval_seconds)
            job.refresh()
        manifesto["job_state"] = str(getattr(job, "state", "unknown"))
        manifesto["tuned_model_name"] = getattr(job, "tuned_model_name", None)
        manifesto["tuned_model_endpoint_name"] = getattr(job, "tuned_model_endpoint_name", None)
        if manifesto["tuned_model_name"]:
            output_model_name_path.parent.mkdir(parents=True, exist_ok=True)
            output_model_name_path.write_text(str(manifesto["tuned_model_name"]), encoding="utf-8")

    escrever_manifesto_tuning(manifest_path, manifesto)
    return manifest_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tuning supervisionado do Gemini 2.5 Flash.")
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--location", default=REGIAO_PADRAO)
    parser.add_argument("--dataset-path", type=Path, default=DATASET_TREINO_PATH)
    parser.add_argument("--source-model", default=MODELO_BASE_PADRAO)
    parser.add_argument("--train-dataset-gcs-uri", default=None)
    parser.add_argument("--validation-dataset-gcs-uri", default=None)
    parser.add_argument("--staging-bucket", default=None)
    parser.add_argument("--gcs-prefix", default="dissertacao-ementas-tr/fase5")
    parser.add_argument("--tuned-model-display-name", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--adapter-size", type=int, default=None)
    parser.add_argument("--learning-rate-multiplier", type=float, default=None)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-interval-seconds", type=int, default=60)
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = _parse_args()
    output_path = executar_finetuning_gemini(
        project_id=args.project_id,
        location=args.location,
        dataset_path=args.dataset_path,
        source_model=args.source_model,
        train_dataset_gcs_uri=args.train_dataset_gcs_uri,
        validation_dataset_gcs_uri=args.validation_dataset_gcs_uri,
        staging_bucket=args.staging_bucket,
        gcs_prefix=args.gcs_prefix,
        tuned_model_display_name=args.tuned_model_display_name,
        epochs=args.epochs,
        adapter_size=args.adapter_size,
        learning_rate_multiplier=args.learning_rate_multiplier,
        wait=args.wait,
        poll_interval_seconds=args.poll_interval_seconds,
        prepare_only=args.prepare_only,
    )
    log.info("Manifesto Gemini SFT persistido em %s", output_path)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, OSError, ValueError, ImportError) as exc:
        log.critical("Execução interrompida: %s", exc)
        sys.exit(1)
