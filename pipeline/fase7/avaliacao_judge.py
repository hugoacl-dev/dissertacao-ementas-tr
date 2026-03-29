"""
avaliacao_judge.py — Executor canônico do LLM-as-a-Judge da Fase 7

Lê os casos-base e as predições disponíveis, chama o modelo juiz com prompt
versionado, valida a resposta JSON e persiste `avaliacao_llm_judge.jsonl` com
retomada incremental por par caso-condição.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any
from urllib import request

from pipeline.core.artefato_utils import escrever_json_atomico, escrever_jsonl_atomico
from pipeline.core.project_paths import (
    FASE7_AVALIACAO_JUDGE_MANIFEST_PATH,
    FASE7_AVALIACAO_JUDGE_PATH,
    FASE7_AVALIACAO_JUDGE_BRUTA_PATH,
    FASE7_CASOS_AVALIACAO_PATH,
    PERFIL_EXECUCAO_CLI_PADRAO,
    PERFIL_EXECUCAO_OFICIAL,
    PERFIS_EXECUCAO,
    FASE7_PREDICAO_PATHS,
    resolver_artefatos_fase7,
    resolver_predicoes_fase7,
    validar_perfil_execucao,
)

from .metricas import carregar_casos_avaliacao, carregar_predicoes_condicao
from .protocolo import (
    CONDICOES_EXPERIMENTAIS,
    MODELO_JUIZ,
    MODELO_JUIZ_API_PADRAO,
    calcular_sha256_texto,
    ler_prompt_llm_judge,
    validar_registro_avaliacao_judge,
    validar_resposta_llm_judge,
)

log = logging.getLogger(__name__)

API_BASE_PADRAO = "https://api.deepseek.com"


def _validar_registro_avaliacao_judge_bruta(payload: dict[str, Any]) -> dict[str, Any]:
    """Valida minimamente um registro bruto do juiz automático."""
    if not isinstance(payload, dict):
        raise ValueError("O registro bruto do juiz deve ser um objeto JSON.")
    if set(payload) != {
        "caso_id",
        "condicao_id",
        "model_id_api",
        "mensagem_usuario",
        "resposta_bruta",
    }:
        raise ValueError(
            "O registro bruto do juiz deve conter apenas "
            "`caso_id`, `condicao_id`, `model_id_api`, `mensagem_usuario` e `resposta_bruta`."
        )
    if not isinstance(payload.get("caso_id"), str) or not payload["caso_id"].strip():
        raise ValueError("`caso_id` do registro bruto do juiz deve ser texto não vazio.")
    if not isinstance(payload.get("condicao_id"), str) or not payload["condicao_id"].strip():
        raise ValueError("`condicao_id` do registro bruto do juiz deve ser texto não vazio.")
    if not isinstance(payload.get("model_id_api"), str) or not payload["model_id_api"].strip():
        raise ValueError("`model_id_api` do registro bruto do juiz deve ser texto não vazio.")
    if not isinstance(payload.get("mensagem_usuario"), str) or not payload["mensagem_usuario"].strip():
        raise ValueError("`mensagem_usuario` do registro bruto do juiz deve ser texto não vazio.")
    if not isinstance(payload.get("resposta_bruta"), dict):
        raise ValueError("`resposta_bruta` do registro bruto do juiz deve ser objeto JSON.")
    return payload


def _ler_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo JSONL não encontrado: {path}")
    registros: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for numero, linha in enumerate(f, start=1):
            conteudo = linha.strip()
            if not conteudo:
                continue
            try:
                registros.append(json.loads(conteudo))
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON inválido em {path}:{numero}: {exc.msg}") from exc
    return registros


def carregar_avaliacoes_judge_existentes(
    path: Path = FASE7_AVALIACAO_JUDGE_PATH,
) -> list[dict[str, Any]]:
    """Carrega avaliações já persistidas e verifica duplicidade por par."""
    if not path.exists():
        return []
    registros = [validar_registro_avaliacao_judge(registro) for registro in _ler_jsonl(path)]
    contagem = Counter((registro["caso_id"], registro["condicao_id"]) for registro in registros)
    duplicados = sorted(par for par, quantidade in contagem.items() if quantidade > 1)
    if duplicados:
        raise ValueError(
            "Avaliações do juiz duplicadas detectadas. "
            f"Exemplos: {duplicados[:5]}"
        )
    return registros


def persistir_avaliacoes_judge(
    path: Path,
    registros: list[dict[str, Any]],
) -> None:
    """Valida e persiste o arquivo consolidado de avaliações do juiz."""
    normalizados = [validar_registro_avaliacao_judge(registro) for registro in registros]
    contagem = Counter((registro["caso_id"], registro["condicao_id"]) for registro in normalizados)
    duplicados = sorted(par for par, quantidade in contagem.items() if quantidade > 1)
    if duplicados:
        raise ValueError(
            "Avaliações do juiz duplicadas detectadas. "
            f"Exemplos: {duplicados[:5]}"
    )
    escrever_jsonl_atomico(path, normalizados)


def carregar_avaliacoes_judge_brutas_existentes(
    path: Path = FASE7_AVALIACAO_JUDGE_BRUTA_PATH,
) -> list[dict[str, Any]]:
    """Carrega o artefato bruto do juiz e verifica duplicidade por par."""
    if not path.exists():
        return []
    registros = [_validar_registro_avaliacao_judge_bruta(registro) for registro in _ler_jsonl(path)]
    contagem = Counter((registro["caso_id"], registro["condicao_id"]) for registro in registros)
    duplicados = sorted(par for par, quantidade in contagem.items() if quantidade > 1)
    if duplicados:
        raise ValueError(
            "Avaliações brutas do juiz duplicadas detectadas. "
            f"Exemplos: {duplicados[:5]}"
        )
    return registros


def persistir_avaliacoes_judge_brutas(
    path: Path,
    registros: list[dict[str, Any]],
) -> None:
    """Persiste o artefato bruto do juiz com verificação de duplicidade por par."""
    normalizados = [_validar_registro_avaliacao_judge_bruta(registro) for registro in registros]
    contagem = Counter((registro["caso_id"], registro["condicao_id"]) for registro in normalizados)
    duplicados = sorted(par for par, quantidade in contagem.items() if quantidade > 1)
    if duplicados:
        raise ValueError(
            "Avaliações brutas do juiz duplicadas detectadas. "
            f"Exemplos: {duplicados[:5]}"
        )
    escrever_jsonl_atomico(path, normalizados)


def carregar_predicoes_disponiveis_para_judge(
    *,
    casos_path: Path = FASE7_CASOS_AVALIACAO_PATH,
    predicao_paths: dict[str, Path] = FASE7_PREDICAO_PATHS,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Retorna pares caso-condição disponíveis para avaliação do juiz."""
    casos_df = carregar_casos_avaliacao(casos_path)
    observacoes: list[dict[str, Any]] = []
    condicoes_ausentes: list[str] = []

    caso_ids = set(casos_df["caso_id"])
    for condicao in CONDICOES_EXPERIMENTAIS:
        condicao_id = condicao["id"]
        path = predicao_paths[condicao_id]
        if not path.exists():
            condicoes_ausentes.append(condicao_id)
            continue

        predicoes_df = carregar_predicoes_condicao(path, condicao_id=condicao_id)
        ids_predicoes = set(predicoes_df["caso_id"])
        extras = sorted(ids_predicoes - caso_ids)
        if extras:
            raise ValueError(
                f"A condição '{condicao_id}' contém casos inexistentes na base. "
                f"Exemplos extras: {extras[:5]}"
            )

        merged = casos_df.merge(predicoes_df, on="caso_id", how="inner")
        observacoes.extend(merged.to_dict(orient="records"))

    if not observacoes:
        raise FileNotFoundError(
            "Nenhum arquivo de predição disponível para avaliação do juiz. "
            "Gere primeiro pelo menos uma condição experimental."
        )
    return observacoes, condicoes_ausentes


def filtrar_observacoes_pendentes(
    observacoes: list[dict[str, Any]],
    avaliacoes_existentes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Remove pares caso-condição já avaliados."""
    avaliados = {(item["caso_id"], item["condicao_id"]) for item in avaliacoes_existentes}
    return [
        observacao
        for observacao in observacoes
        if (observacao["caso_id"], observacao["condicao_id"]) not in avaliados
    ]


def construir_mensagem_usuario_judge(
    *,
    fundamentacao: str,
    ementa_gerada: str,
) -> str:
    """Monta a entrada factual do juiz automático."""
    return (
        "Fundamentação original:\n"
        f"{fundamentacao.strip()}\n\n"
        "Ementa candidata:\n"
        f"{ementa_gerada.strip()}"
    )


def _executar_requisicao_chat_json(
    *,
    api_key: str,
    api_base: str,
    model_id: str,
    system_prompt: str,
    user_message: str,
    temperature: float,
    max_output_tokens: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    url = api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": temperature,
        "max_tokens": max_output_tokens,
        "response_format": {"type": "json_object"},
    }
    req = request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with request.urlopen(req, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def _extrair_avaliacao_da_resposta_chat(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        escolha = payload["choices"][0]
        finish_reason = escolha.get("finish_reason")
        conteudo = escolha["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Resposta do juiz em formato inesperado.") from exc

    if finish_reason == "length":
        raise ValueError("Resposta do juiz truncada por limite de tokens.")
    if not isinstance(conteudo, str) or not conteudo.strip():
        raise ValueError("Resposta do juiz sem conteúdo textual utilizável.")
    try:
        return json.loads(conteudo)
    except json.JSONDecodeError as exc:
        raise ValueError("Resposta do juiz não retornou JSON válido.") from exc


def avaliar_observacao_com_judge(
    *,
    model_id: str,
    system_prompt: str,
    fundamentacao: str,
    ementa_gerada: str,
    temperature: float,
    max_output_tokens: int,
    api_base: str = API_BASE_PADRAO,
    api_key_env_var: str = "DEEPSEEK_API_KEY",
    timeout_seconds: float = 120.0,
    max_retries: int = 3,
    retry_backoff_seconds: float = 2.0,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Avalia uma única ementa candidata com retry/backoff."""
    api_key = os.getenv(api_key_env_var)
    if not api_key:
        raise EnvironmentError(
            f"Variável de ambiente obrigatória ausente para o juiz automático: {api_key_env_var}"
        )

    ultima_exc: Exception | None = None
    user_message = construir_mensagem_usuario_judge(
        fundamentacao=fundamentacao,
        ementa_gerada=ementa_gerada,
    )
    for tentativa in range(1, max_retries + 1):
        try:
            resposta = _executar_requisicao_chat_json(
                api_key=api_key,
                api_base=api_base,
                model_id=model_id,
                system_prompt=system_prompt,
                user_message=user_message,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                timeout_seconds=timeout_seconds,
            )
            avaliacao = _extrair_avaliacao_da_resposta_chat(resposta)
            return (
                validar_resposta_llm_judge(avaliacao),
                {
                    "model_id_api": model_id,
                    "mensagem_usuario": user_message,
                    "resposta_bruta": resposta,
                },
            )
        except Exception as exc:  # noqa: BLE001 - tolerância operacional a falhas transitórias
            ultima_exc = exc
            if tentativa == max_retries:
                break
            espera = retry_backoff_seconds * (2 ** (tentativa - 1))
            log.warning(
                "Falha transitória no juiz automático (%s/%s): %s. Nova tentativa em %.1fs.",
                tentativa,
                max_retries,
                exc,
                espera,
            )
            time.sleep(espera)

    assert ultima_exc is not None
    raise RuntimeError(
        f"Falha ao avaliar com o juiz automático após {max_retries} tentativas."
    ) from ultima_exc


def executar_avaliacao_judge(
    *,
    casos_path: Path = FASE7_CASOS_AVALIACAO_PATH,
    predicao_paths: dict[str, Path] = FASE7_PREDICAO_PATHS,
    output_path: Path = FASE7_AVALIACAO_JUDGE_PATH,
    raw_output_path: Path = FASE7_AVALIACAO_JUDGE_BRUTA_PATH,
    manifest_path: Path = FASE7_AVALIACAO_JUDGE_MANIFEST_PATH,
    model_id: str = MODELO_JUIZ_API_PADRAO,
    api_base: str = API_BASE_PADRAO,
    api_key_env_var: str = "DEEPSEEK_API_KEY",
    temperature: float = 0.0,
    max_output_tokens: int = 1024,
    timeout_seconds: float = 120.0,
    limit: int | None = None,
    flush_every: int = 20,
    max_retries: int = 3,
    retry_backoff_seconds: float = 2.0,
    perfil_execucao: str = PERFIL_EXECUCAO_OFICIAL,
) -> Path:
    """Executa o LLM-as-a-Judge com retomada incremental."""
    perfil_execucao = validar_perfil_execucao(perfil_execucao)
    if flush_every <= 0:
        raise ValueError("`flush_every` deve ser inteiro positivo.")
    if limit is not None and limit <= 0:
        raise ValueError("`limit` deve ser positivo quando informado.")
    if max_retries <= 0:
        raise ValueError("`max_retries` deve ser inteiro positivo.")
    if retry_backoff_seconds < 0:
        raise ValueError("`retry_backoff_seconds` não pode ser negativo.")

    system_prompt = ler_prompt_llm_judge()
    observacoes, condicoes_ausentes = carregar_predicoes_disponiveis_para_judge(
        casos_path=casos_path,
        predicao_paths=predicao_paths,
    )
    existentes = carregar_avaliacoes_judge_existentes(output_path)
    existentes_brutos = carregar_avaliacoes_judge_brutas_existentes(raw_output_path)
    pares_existentes = {(item["caso_id"], item["condicao_id"]) for item in existentes}
    pares_existentes_brutos = {(item["caso_id"], item["condicao_id"]) for item in existentes_brutos}
    if pares_existentes != pares_existentes_brutos:
        raise ValueError(
            "Os artefatos normalizado e bruto do juiz estão desalinhados. "
            "Reexecute a etapa para restaurar a rastreabilidade completa."
        )
    registros = list(existentes)
    registros_brutos = list(existentes_brutos)
    pendentes = filtrar_observacoes_pendentes(observacoes, existentes)
    if limit is not None:
        pendentes = pendentes[:limit]

    manifesto: dict[str, Any] = {
        "perfil_execucao": perfil_execucao,
        "modelo_juiz_logico": MODELO_JUIZ,
        "model_id": model_id,
        "model_id_api": model_id,
        "api_base": api_base,
        "api_key_env_var": api_key_env_var,
        "casos_path": str(casos_path),
        "output_path": str(output_path),
        "raw_output_path": str(raw_output_path),
        "prompt_sha256": calcular_sha256_texto(system_prompt),
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "timeout_seconds": timeout_seconds,
        "flush_every": flush_every,
        "max_retries": max_retries,
        "retry_backoff_seconds": retry_backoff_seconds,
        "observacoes_disponiveis": len(observacoes),
        "avaliacoes_existentes": len(existentes),
        "avaliacoes_pendentes_planejadas": len(pendentes),
        "condicoes_ausentes": condicoes_ausentes,
        "status": "running",
    }
    escrever_json_atomico(manifest_path, manifesto, indent=2)

    try:
        for indice, observacao in enumerate(pendentes, start=1):
            avaliacao, bruto = avaliar_observacao_com_judge(
                model_id=model_id,
                system_prompt=system_prompt,
                fundamentacao=observacao["fundamentacao"],
                ementa_gerada=observacao["ementa_gerada"],
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                api_base=api_base,
                api_key_env_var=api_key_env_var,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                retry_backoff_seconds=retry_backoff_seconds,
            )
            registros.append(
                {
                    "caso_id": observacao["caso_id"],
                    "condicao_id": observacao["condicao_id"],
                    "avaliacao": avaliacao,
                }
            )
            registros_brutos.append(
                {
                    "caso_id": observacao["caso_id"],
                    "condicao_id": observacao["condicao_id"],
                    **bruto,
                }
            )
            if indice % flush_every == 0:
                persistir_avaliacoes_judge(output_path, registros)
                persistir_avaliacoes_judge_brutas(raw_output_path, registros_brutos)
                log.info("Juiz automático: %s avaliações persistidas", len(registros))

        persistir_avaliacoes_judge(output_path, registros)
        persistir_avaliacoes_judge_brutas(raw_output_path, registros_brutos)
    except Exception as exc:  # noqa: BLE001 - manifesto de falha
        manifesto["status"] = "failed"
        manifesto["erro"] = str(exc)
        manifesto["avaliacoes_persistidas"] = len(registros)
        escrever_json_atomico(manifest_path, manifesto, indent=2)
        raise

    manifesto["status"] = "completed"
    manifesto["avaliacoes_persistidas"] = len(registros)
    manifesto["avaliacoes_geradas_nesta_execucao"] = len(registros) - len(existentes)
    escrever_json_atomico(manifest_path, manifesto, indent=2)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Executor canônico do LLM-as-a-Judge.")
    parser.add_argument(
        "--perfil-execucao",
        choices=PERFIS_EXECUCAO,
        default=PERFIL_EXECUCAO_CLI_PADRAO,
    )
    parser.add_argument("--casos-path", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--raw-output-path", type=Path, default=None)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--model-id", default=MODELO_JUIZ_API_PADRAO)
    parser.add_argument("--api-base", default=API_BASE_PADRAO)
    parser.add_argument("--api-key-env-var", default="DEEPSEEK_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--flush-every", type=int, default=20)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-backoff-seconds", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = _parse_args()
    artefatos = resolver_artefatos_fase7(args.perfil_execucao)
    output_path = executar_avaliacao_judge(
        casos_path=args.casos_path or artefatos["casos_avaliacao_path"],
        predicao_paths=resolver_predicoes_fase7(args.perfil_execucao),
        output_path=args.output_path or artefatos["avaliacao_judge_path"],
        raw_output_path=args.raw_output_path or artefatos["avaliacao_judge_bruta_path"],
        manifest_path=args.manifest_path or artefatos["avaliacao_judge_manifest_path"],
        model_id=args.model_id,
        api_base=args.api_base,
        api_key_env_var=args.api_key_env_var,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        timeout_seconds=args.timeout_seconds,
        limit=args.limit,
        flush_every=args.flush_every,
        max_retries=args.max_retries,
        retry_backoff_seconds=args.retry_backoff_seconds,
        perfil_execucao=args.perfil_execucao,
    )
    log.info("Avaliações do juiz automático persistidas em %s", output_path)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, OSError, ValueError, ImportError, EnvironmentError) as exc:
        log.critical("Execução interrompida: %s", exc)
        sys.exit(1)
