import json
import re
import html
import logging
from pathlib import Path

# Configuração de Logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

INPUT_PATH = 'dados_brutos.json'
OUTPUT_PATH = 'dados_limpos.json'

def limpar_texto(texto):
    """
    Função core da Fase 2. Aplica saneamento via Regex e NLP básica
    sobre o texto bruto da fundamentação ou ementa.
    """
    if not texto:
        return ""

    # 1. Resolver HTML Entities (ex: &nbsp; -> espaço, &quot; -> ")
    texto = html.unescape(texto)

    # 2. Remoção de Tags HTML
    # Ex: <p style="...">, <strong>, <ol>, <li>
    texto = re.sub(r'<[^>]+>', ' ', texto)

    # 3. Remoção de Metadados Processuais e de Protocolo
    # Ex: "Processo nº 0500148-54.2016.4.05.8200" ou "Autos: 1234..."
    texto = re.sub(r'(?i)(processo|protocolo|autos)[\s\w]*n[º°o]?[\s:]*[\d\.\-]+(?:/\d+)?', ' ', texto)
    
    # Remoção de IDs de documentos PJe (incluindo com parênteses ex: "(id. 48772689)")
    texto = re.sub(r'(?i)\(?id\.?[\s:]*\d{6,}\)?', ' ', texto)

    # 4. Remoção de Datas Soltas / Extensas, Carimbos DJe e Ações Civis Públicas (ACP) irrelevantes
    # Ex: "PROCESSO ELETRÔNICO DJe-s/n DIVULG 23-05-2024" ou "DJe 26.09.2012"
    texto = re.sub(r'(?i)(Data de Julgamento.*?|PROCESSO ELETRÔNICO.*?DJe-s/n.*?(PUBLIC|DIVULG).*?\d{4})', ' ', texto)
    texto = re.sub(r'(?i)DJe\s+\d{1,2}[\./]\d{1,2}[\./]\d{2,4}', ' ', texto)
    
    # Ex: "Ação Civil Pública - ACP nº 1012072-89.2018.401.3400 -- DPU"
    texto = re.sub(r'(?i)Ação Civil Pública[\s\-]+ACP\s*n[º°o]?\s*[\d\.\-]+[^\.]*DPU', ' ', texto)
    
    cidades = r'(joão pessoa|campina grande|guarabira|patos|monteiro|sousa)'
    meses = r'(janeiro|fevereiro|março|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro)'
    texto = re.sub(fr'(?i){cidades}[\s,]+(\d{{1,2}})[\sde]+{meses}[\sde]+(\d{{4}})', ' ', texto)
    
    # Remover menções soltas relativas a datas do tipo dia.mês.ano ou dia/mês/ano com prefixos
    texto = re.sub(r'(?i)(atualizado|falecido)[\s]+em[\s]+\d{1,2}[\./]\d{1,2}[\./]\d{2,4}', ' ', texto)
    texto = re.sub(r'(?i)falecido[\s]+em[\s]+\d{4}', ' ', texto)
    texto = re.sub(r'(?i)\d{1,2}[\./]\d{1,2}[\./]\d{2,4}', ' ', texto) # Datas dd/mm/yyyy ou dd.mm.yyyy

    # 5. Remoção de Assinaturas, Saudações Honoríficas e Praxe do Tribunal
    # Remove blocos como: "A Turma Recursal dos Juizados Especiais... DEU PROVIMENTO..."
    # que costumam aparecer no final do voto poluindo a ratio decidendi.
    texto = re.sub(r'(?i)A Turma Recursal dos Juizados Especiais.*?(votos?|juiz[- ]relator|honorários).*?(?=\.|$)', ' ', texto)
    
    # Destroi a "Súmula de Julgamento" e blocos finais com Nomes de Juízes soltos em CAPSLOCK
    texto = re.sub(r'(?i)Súmula de Julgamento:?.*$', ' ', texto)
    texto = re.sub(r'(?i)(Juiz Federal|Juíza Federal|Relator|Relatora|Excelentíssimo|Desembargador)[\w\s]*?$', ' ', texto)
    
    # Ex: "BIANOR ARRUDA BEZERRA NETO" jogado no final da string
    texto = re.sub(r'\.?[ \n]*[A-ZÀ-Ÿ\s]{10,}$', ' ', texto)

    # 6. Normalização de Espaços e Quebras de Linha
    # O PJe gera muitas quebras duplas \n e tabs \t
    texto = re.sub(r'[\r\n\t]+', ' ', texto)
    # Garante que múltiplos espaços virarão 1 só
    texto = re.sub(r'\s{2,}', ' ', texto)

    return texto.strip()

def processar_base():
    if not Path(INPUT_PATH).exists():
        logging.error(f"Arquivo {INPUT_PATH} não encontrado. Execute a Fase 1 primeiro.")
        return

    logging.info(f"Lendo base bruta de {INPUT_PATH}...")
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        dados = json.load(f)

    dados_limpos = []
    total = len(dados)

    logging.info(f"Iniciando saneamento de {total} registros via Regex...")
    
    for i, item in enumerate(dados):
        id_reg = item.get("id")
        fundamentacao = limpar_texto(item.get("fundamentacao", ""))
        ementa = limpar_texto(item.get("ementa", ""))

        # Filtra registros que, após o parser HTML/Regex do PJe, ficaram vazios
        if fundamentacao and ementa and len(fundamentacao) > 50 and len(ementa) > 20:
            dados_limpos.append({
                "id": id_reg,
                "fundamentacao": fundamentacao,
                "ementa": ementa
            })
            
        if (i + 1) % 5000 == 0:
            logging.info(f"{i + 1}/{total} registros limpos...")

    logging.info(f"Saneamento concluído! Registros válidos após limpeza: {len(dados_limpos)}")
    logging.info(f"Exportando para {OUTPUT_PATH}...")
    
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(dados_limpos, f, ensure_ascii=False, indent=2)

    logging.info("Fase 2 finalizada com sucesso!")

if __name__ == '__main__':
    processar_base()
