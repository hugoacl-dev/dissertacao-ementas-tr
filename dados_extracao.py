import os
import sqlite3
import subprocess
import json
import logging
from pathlib import Path

# Configuração de Logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DUMP_PATH = 'dump_tr_one.sql'
DB_PATH = 'banco_tr_one.sqlite'
JSON_PATH = 'dados_brutos.json'

def verificar_dependencias():
    """Verifica se o pg_restore está instalado, essencial para ler o formato custom binário (.sql) do Postgres."""
    try:
        subprocess.run(['pg_restore', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.error("ERRO: 'pg_restore' não encontrado!")
        logging.error("O arquivo 'dump_tr_one.sql' não é um SQL em texto plano (apesar da extensão).")
        logging.error("Ele é um PostgreSQL Custom Database Dump (binário comprimido).")
        logging.error("Para processá-lo no Mac, instale via Homebrew: brew install postgresql")
        exit(1)

def criar_sqlite():
    """Inicializa o banco de dados SQLite com a estrutura necessária."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS turmarecursal_processo (
            id INTEGER PRIMARY KEY,
            votoementa TEXT,
            ementa TEXT,
            data_cadastro TEXT
        )
    ''')
    conn.commit()
    return conn

def extrair_e_popular_dados(conn):
    """
    Usa pg_restore para ler de forma em stream apenas os dados da tabela alvo,
    sem precisar instanciar um servidor PostgreSQL inteiro.
    O pg_restore vai gerar comandos COPY e dados delimitados por tabulação (TSV).
    """
    logging.info(f"Lendo do arquivo binário {DUMP_PATH} via pg_restore...")
    
    # Extrai apenas os dados (-a) da tabela turmarecursal_processo
    comando = ['pg_restore', '--data-only', '--table=turmarecursal_processo', '-f', '-', DUMP_PATH]
    
    processo = subprocess.Popen(comando, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
    
    cursor = conn.cursor()
    _colunas = []
    lendo_dados = False
    registros_inseridos = 0
    dados_json = []

    for linha in processo.stdout:
        linha = linha.strip()
        
        # Início dos dados (Encontra a declaração do COPY)
        if linha.startswith("COPY public.turmarecursal_processo ("):
            # Ex: COPY public.turmarecursal_processo (id, partes, ..., votoementa, ..., ementa) FROM stdin;
            colunas_str = linha[linha.find('(')+1:linha.find(')')]
            _colunas = [c.strip() for c in colunas_str.split(',')]
            lendo_dados = True
            continue
        
        # Fim dos dados representados pelo \.
        if lendo_dados and linha == r'\.':
            lendo_dados = False
            break
            
        if lendo_dados and _colunas:
            valores = linha.split('\t')
            row_dict = dict(zip(_colunas, valores))
            
            # Alguns NULLs no COPY são representados como \N
            votoementa = None if row_dict.get('votoementa') == r'\N' else row_dict.get('votoementa')
            ementa = None if row_dict.get('ementa') == r'\N' else row_dict.get('ementa')
            _id = row_dict.get('id')
            data_cadastro = row_dict.get('data_cadastro')
            
            # Desescapa os retornos de linha embutidos num TSV gerado pelo pg_dump
            if votoementa:
                votoementa = votoementa.replace(r'\n', '\n').replace(r'\r', '\r')
            if ementa:
                ementa = ementa.replace(r'\n', '\n').replace(r'\r', '\r')
            
            # Insere no SQLite
            cursor.execute(
                'INSERT OR REPLACE INTO turmarecursal_processo (id, votoementa, ementa, data_cadastro) VALUES (?, ?, ?, ?)',
                (_id, votoementa, ementa, data_cadastro)
            )
            
            # Acumula para o JSONL apenas se houver os dois (Target/Label e Feature/Voto) válidos.
            if votoementa and ementa and votoementa.strip() and ementa.strip():
                dados_json.append({
                    "id": _id,
                    "fundamentacao": votoementa,
                    "ementa": ementa
                })
            
            registros_inseridos += 1
            if registros_inseridos % 1000 == 0:
                logging.info(f"{registros_inseridos} registros processados...")

    conn.commit()
    return dados_json

def exportar_json(dados_json):
    """Exporta para o arquivo dados_brutos.json."""
    if not dados_json:
        logging.warning("Não há dados válidos para exportar (tudo vazio).")
        return
        
    logging.info(f"Exportando {len(dados_json)} pares válidos para {JSON_PATH}...")
    with open(JSON_PATH, 'w', encoding='utf-8') as f:
        # Array contínuo do JSON
        json.dump(dados_json, f, ensure_ascii=False, indent=2)
    logging.info("Processo de extração da Fase 1 finalizado com sucesso!")

def pipeline_principal():
    verificar_dependencias()
    conn = criar_sqlite()
    dados_json = extrair_e_popular_dados(conn)
    exportar_json(dados_json)
    conn.close()

if __name__ == '__main__':
    pipeline_principal()
