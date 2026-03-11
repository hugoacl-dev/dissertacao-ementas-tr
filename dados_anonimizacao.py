import json
import re
import random
import logging
from pathlib import Path

# Configuração de Logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

INPUT_PATH = 'dados_limpos.json'
TRAIN_PATH = 'dataset_treino.jsonl'
TEST_PATH = 'dataset_teste.jsonl'
TEST_SIZE = 0.10 # 10% da base para testes da banca, 90% para treinar a IA

def anonimizar_texto(texto):
    """
    Substitui Informações Pessoalmente Identificáveis (PII) por tokens genéricos.
    Atende aos requisitos rigorosos da LGPD antes do envio à nuvem.
    """
    if not texto: return ""

    # 1. CPFs (11 dígitos formatados ou não)
    texto = re.sub(r'\b\d{3}\.\d{3}\.\d{3}-\d{2}\b', '[CPF]', texto)
    
    # 2. CNPJs (14 dígitos formatados)
    texto = re.sub(r'\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b', '[CNPJ]', texto)
    
    # 3. Ocultar nomes próprios (Após honoríficos ou em formato Nome e Sobrenome Maiúsculos)
    honorificos = r'(?:Sr\.|Sra\.|Dr\.|Dra\.|advogado|advogada|autor|autora|réu|ré|juiz|juíza|relator|relatora|desembargador|desembargadora)'
    texto = re.sub(fr'(?i){honorificos}\s+([A-ZÀ-Ÿ][a-zà-ÿ]+\s+){{1,4}}[A-ZÀ-Ÿ][a-zà-ÿ]+', r'\g<0> ([NOME_OCULTADO])', texto)
    
    # Nomes Próprios Isolados: 3 ou mais palavras Capitalizadas seguidas (Ex: Geraldo Ferreira da Silva)
    texto = re.sub(r'\b([A-ZÀ-Ÿ][a-zà-ÿ]+\s+){2,5}[A-ZÀ-Ÿ][a-zà-ÿ]+\b', '[NOME_PESSOA]', texto)
    
    # 4. Endereços, Municípios da Paraíba Frequentes e Empresas
    # Municípios, Zonas e Logradouros
    locais = r'(?i)(em\s+João Pessoa|em\s+S\.\s*Miguel|em\s+São Miguel|no\s+Conde|Sítio Corredor|zona rural.*?(?=[,\.]))'
    texto = re.sub(locais, ' [LOCAL_OCULTADO] ', texto)
    texto = re.sub(r'(?i)(rua|avenida|praça|sítio|bairro) [a-zà-ÿ\s]+', '[ENDEREÇO]', texto)
    
    # Empresas (Razão Social contendo Ltda, S/A, S.A, ME)
    texto = re.sub(r'\b[A-ZÀ-Ÿa-zà-ÿ\s]+\s+(Ltda\.?|LTDA\.?|S/A|S\.A\.?|ME|EPP)\b', '[EMPRESA]', texto)

    # 5. Mascarar números de contas/agências comuns e CEPs numéricos puros
    texto = re.sub(r'\b\d{4,5}-\d{1}\b', '[CONTA-DIGITO]', texto)
    texto = re.sub(r'\b\d{5}-\d{3}\b', '[CEP]', texto)

    return texto

def formatar_prompt_gemini(fundamentacao, ementa):
    """
    Empacota o par {fundamentacao, ementa} no formato multiturmo conversacional
    exigido pelo SDK google-generativeai para Supervised Fine-Tuning de Gemini.
    """
    instrucao_sistema = (
        "Você é um assistente jurídico experiente que auxilia juízes a escreverem "
        "Ementas Judiciais, que são resumos curtos, estruturados e objetivos do que "
        "foi decidido numa fundamentação (voto). Ao ser fornecida a fundamentação de um "
        "Recurso, você deve responder unica e exclusivamente com o texto da Ementa correspondente."
    )

    return {
        "contents": [
            {
                "role": "system",
                "parts": [{"text": instrucao_sistema}]
            },
            {
                "role": "user",
                "parts": [{"text": f"Gere a ementa para a seguinte fundamentação: {fundamentacao}"}]
            },
            {
                "role": "model",
                "parts": [{"text": ementa}]
            }
        ]
    }

def gerar_datasets():
    if not Path(INPUT_PATH).exists():
        logging.error(f"Arquivo {INPUT_PATH} não encontrado. Execute a Fase 2 primeiro.")
        return

    logging.info(f"Lendo base limpa de {INPUT_PATH}...")
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        dados = json.load(f)

    # Aplica anonimização
    logging.info("Aplicando filtros rigorosos de LGPD (anonimização) em todos os textos...")
    
    dados_formatados = []
    
    for item in dados:
        fund_anonima = anonimizar_texto(item["fundamentacao"])
        ementa_anonima = anonimizar_texto(item["ementa"])
        
        # Converte pro dicionário do Gemini JSONL
        linha_conversacional = formatar_prompt_gemini(fund_anonima, ementa_anonima)
        dados_formatados.append(linha_conversacional)

    logging.info(f"Total de exemplos válidos e anonimizados: {len(dados_formatados)}")
    
    # Shuffle reproduzível para divisão Treino/Teste
    random.seed(42)
    random.shuffle(dados_formatados)
    
    qtd_teste = int(len(dados_formatados) * TEST_SIZE)
    dataset_teste = dados_formatados[:qtd_teste]
    dataset_treino = dados_formatados[qtd_teste:]
    
    logging.info(f"Divisão concluída: {len(dataset_treino)} para TREINO (Fine-Tuning), {len(dataset_teste)} para TESTE (Colab)")

    # Escrevendo JSONL (Linhas JSON puras, não arrays, exigência da Google)
    logging.info(f"Gravando {TRAIN_PATH}...")
    with open(TRAIN_PATH, 'w', encoding='utf-8') as f:
        for ex in dataset_treino:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')

    logging.info(f"Gravando {TEST_PATH}...")
    with open(TEST_PATH, 'w', encoding='utf-8') as f:
        for ex in dataset_teste:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')

    logging.info("Fase 3 finalizada com sucesso! A base está legalizada (LGPD) e engatilhada para a nuvem.")

if __name__ == '__main__':
    gerar_datasets()
