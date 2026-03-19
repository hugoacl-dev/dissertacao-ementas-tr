"""
01_ingestao.py — Fase 1: Ingestão e Construção da Base de Dados Local

Converte o dump binário PostgreSQL (custom format) do sistema sistema judicial em:
  - Um banco SQLite local (data/banco_sistema_judicial.sqlite) para consultas futuras.
  - Um arquivo JSON compacto (data/dados_brutos.json) com os pares
    {id, fundamentacao, ementa, data_cadastro} válidos, como entrada para a Fase 2.
    O campo data_cadastro é preservado para viabilizar a divisão cronológica na Fase 3.

Dependência externa: pg_restore (instalável via `brew install postgresql@16`)
Executar a partir da raiz do projeto: python3 pipeline/01_ingestao.py
"""
from __future__ import annotations

import json
import logging
import sqlite3
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterator

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

log = logging.getLogger(__name__)

DUMP_PATH = Path("dump_sistema_judicial.sql")
DB_PATH = Path("data/banco_sistema_judicial.sqlite")
JSON_PATH = Path("data/dados_brutos.json")

TARGET_TABLE = "turmarecursal_processo"
COMMIT_BATCH_SIZE = 5_000  # Faz commit a cada N registros para segurança transacional


# ---------------------------------------------------------------------------
# Modelos de dados
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RegistroProcesso:
    """Representa um par fundamentação/ementa extraído do dump."""

    id: str
    fundamentacao: str
    ementa: str
    data_cadastro: str


@dataclass
class ExtractionStats:
    """Contadores coletados durante a extração para logging/auditoria."""

    total_lidos: int = 0
    descartados_nulos: int = 0
    exportados: int = 0

    @property
    def taxa_aproveitamento(self) -> float:
        if self.total_lidos == 0:
            return 0.0
        return self.exportados / self.total_lidos * 100


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------


def verificar_pg_restore() -> None:
    """Garante que o binário `pg_restore` está disponível no PATH.

    Raises:
        RuntimeError: se `pg_restore` não for encontrado.
    """
    try:
        subprocess.run(
            ["pg_restore", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError(
            "pg_restore não encontrado.\n"
            "O arquivo dump_sistema_judicial.sql é um PostgreSQL Custom Dump (binário comprimido).\n"
            "Instale no Mac via: brew install postgresql@16"
        ) from exc


@contextmanager
def abrir_sqlite(path: Path) -> Generator[sqlite3.Connection, None, None]:
    """Context manager para conexão SQLite com configurações de performance."""
    conn = sqlite3.connect(path)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        yield conn
    finally:
        conn.close()


def inicializar_schema(conn: sqlite3.Connection) -> None:
    """Cria a tabela alvo no SQLite se ainda não existir."""
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TARGET_TABLE} (
            id           TEXT PRIMARY KEY,
            votoementa   TEXT,
            ementa       TEXT,
            data_cadastro TEXT
        )
        """
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Lógica de extração
# ---------------------------------------------------------------------------


def _stream_pg_restore(dump_path: Path) -> Iterator[str]:
    """Abre um subprocesso pg_restore e itera pelas linhas do stdout.

    Yields:
        Linhas brutas (sem newline) extraídas do pg_restore.

    Raises:
        RuntimeError: se o pg_restore retornar código de saída diferente de 0.
    """
    cmd = [
        "pg_restore",
        "--data-only",
        f"--table={TARGET_TABLE}",
        "--file=-",
        str(dump_path),
    ]
    log.info("Iniciando pg_restore: %s", " ".join(cmd))

    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    ) as proc:
        if proc.stdout is None:
            raise RuntimeError("pg_restore não produziu saída (stdout é None)")
        yield from (linha.rstrip("\n") for linha in proc.stdout)

        _, stderr_output = proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"pg_restore falhou (código {proc.returncode}):\n{stderr_output.strip()}"
            )


def _parse_copy_header(linha: str) -> list[str] | None:
    """Extrai a lista de colunas de uma declaração COPY do pg_restore.

    Exemplo de entrada:
        COPY public.turmarecursal_processo (id, votoementa, ementa, data_cadastro) FROM stdin;

    Returns:
        Lista de nomes de coluna, ou None se a linha não for um cabeçalho COPY da tabela alvo.
    """
    prefix = f"COPY public.{TARGET_TABLE} ("
    if not linha.startswith(prefix):
        return None
    cols_str = linha[len(prefix) : linha.index(")")]
    return [c.strip() for c in cols_str.split(",")]


def _desescapar_tsv(valor: str | None) -> str | None:
    r"""Converte o valor de uma célula TSV do pg_dump para string Python.

    pg_dump representa NULL como \N e embute \n e \r nos campos de texto.

    Returns:
        String tratada, ou None se o valor for NULL.
    """
    if valor is None or valor == r"\N":
        return None
    return valor.replace(r"\n", "\n").replace(r"\r", "\r")


def extrair_registros(dump_path: Path) -> tuple[list[RegistroProcesso], ExtractionStats]:
    """Lê o dump PostgreSQL via pg_restore e retorna os registros válidos.

    Um registro é considerado válido quando `votoementa` e `ementa` são
    ambos não-nulos e não-vazios.

    Returns:
        Tupla (registros_validos, estatísticas).
    """
    stats = ExtractionStats()
    registros: list[RegistroProcesso] = []
    colunas: list[str] = []
    lendo_dados = False

    for linha in _stream_pg_restore(dump_path):
        # --- Início do bloco de dados ---
        if not lendo_dados:
            cols = _parse_copy_header(linha)
            if cols is not None:
                colunas = cols
                lendo_dados = True
            continue

        # --- Fim do bloco de dados ---
        if linha == r"\.":
            break

        # --- Linha de dado ---
        stats.total_lidos += 1
        valores = linha.split("\t")

        if len(valores) != len(colunas):
            log.warning("Linha com número inesperado de colunas — ignorada: %r", linha[:80])
            stats.descartados_nulos += 1
            continue

        row = dict(zip(colunas, valores))
        votoementa = _desescapar_tsv(row.get("votoementa"))
        ementa = _desescapar_tsv(row.get("ementa"))

        if not votoementa or not votoementa.strip() or not ementa or not ementa.strip():
            stats.descartados_nulos += 1
            continue

        registros.append(
            RegistroProcesso(
                id=row.get("id", ""),
                fundamentacao=votoementa,
                ementa=ementa,
                data_cadastro=row.get("data_cadastro", ""),
            )
        )
        stats.exportados += 1

        if stats.total_lidos % 10_000 == 0:
            log.info("%d registros lidos até agora...", stats.total_lidos)

    return registros, stats


# ---------------------------------------------------------------------------
# Persistência
# ---------------------------------------------------------------------------


def popular_sqlite(conn: sqlite3.Connection, registros: list[RegistroProcesso]) -> None:
    """Insere os registros no SQLite em batches transacionais.

    Usa INSERT OR REPLACE para idempotência (re-execução segura).
    """
    sql = f"""
        INSERT OR REPLACE INTO {TARGET_TABLE} (id, votoementa, ementa, data_cadastro)
        VALUES (?, ?, ?, ?)
    """
    batch: list[tuple[str, str, str, str]] = []

    for i, reg in enumerate(registros, start=1):
        batch.append((reg.id, reg.fundamentacao, reg.ementa, reg.data_cadastro))
        if i % COMMIT_BATCH_SIZE == 0:
            conn.executemany(sql, batch)
            conn.commit()
            batch.clear()
            log.info("SQLite: %d/%d registros persistidos.", i, len(registros))

    if batch:
        conn.executemany(sql, batch)
        conn.commit()


def exportar_json(registros: list[RegistroProcesso], path: Path) -> None:
    """Serializa os registros para JSON compacto (sem indentação) em UTF-8.

    Inclui o campo 'data_cadastro' para viabilizar a divisão cronológica
    treino/teste na Fase 3 (03_anonimizacao.py).
    """
    payload = [
        {
            "id": r.id,
            "fundamentacao": r.fundamentacao,
            "ementa": r.ementa,
            "data_cadastro": r.data_cadastro,
        }
        for r in registros
    ]
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    log.info("JSON exportado para %s (%s registros).", path, len(payload))



# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    """Pipeline principal da Fase 1."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log.info("=== Fase 1: Ingestão e Construção da Base de Dados ===")

    # Pré-condições
    verificar_pg_restore()
    if not DUMP_PATH.exists():
        raise FileNotFoundError(f"Dump não encontrado: {DUMP_PATH}")

    # Extração
    registros, stats = extrair_registros(DUMP_PATH)

    log.info(
        "Extração concluída. Total lido: %d | Exportados: %d | Descartados: %d (%.1f%% aproveitamento)",
        stats.total_lidos,
        stats.exportados,
        stats.descartados_nulos,
        stats.taxa_aproveitamento,
    )

    if not registros:
        log.warning("Nenhum registro válido extraído. Verifique o dump e o schema.")
        return

    # Persistência SQLite
    with abrir_sqlite(DB_PATH) as conn:
        inicializar_schema(conn)
        popular_sqlite(conn, registros)
    log.info("SQLite salvo em %s.", DB_PATH)

    # Exportação JSON
    exportar_json(registros, JSON_PATH)
    log.info("=== Fase 1 finalizada com sucesso. ===")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, FileNotFoundError) as exc:
        log.critical("Execução interrompida: %s", exc)
        sys.exit(1)
