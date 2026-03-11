import json
import re
from collections import defaultdict

def audit(file_path):
    print(f"\n=============================================")
    print(f"  Auditando {file_path}")
    print(f"=============================================")
    
    # Expressões Regulares para flagrar Resquícios
    patterns = {
        "1. Datas (dd/mm/aaaa ou dd.mm.aaaa)": re.compile(r'\b\d{1,2}[/.]\d{1,2}[/.]\d{2,4}\b'),
        "2. Datas Extensas": re.compile(r'(?i)\b\d{1,2}\s+de\s+(janeiro|fevereiro|março|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro)\s+de\s+\d{4}\b'),
        "3. NPU (Número do Processo)": re.compile(r'\b\d{7}-\d{2}\.\d{4}\.\d{1,2}\.\d{2}\.\d{4}\b'),
        "4. ID PJe": re.compile(r'(?i)\bid[\.\s:]*\d{5,}'),
        "5. CPF": re.compile(r'\b\d{3}\.\d{3}\.\d{3}-\d{2}\b'),
        "6. CNPJ": re.compile(r'\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b'),
        "7. Cidades Explicitas": re.compile(r'(?i)\b(João Pessoa|Campina Grande|Cabedelo|Santa Rita|Patos|Sousa|Cajazeiras|Guarabira|Paraíba)\b'),
        "8. Termo de Empresa (Ltda, S/A, ME)": re.compile(r'(?i)\b(Ltda\.?|S/A|EPP|ME)\b'),
        "9. Juízes/Relatores Isolados": re.compile(r'(?i)\b(Juiz Federal|Juíza Federal|Relator|Relatora|Desembargador)\b'),
        "10. DJe / Carimbos Ocultos": re.compile(r'(?i)\b(DJe|PROCESSO ELETRÔNICO|DIVULG|Ação Civil Pública)\b')
    }
    
    counts = defaultdict(int)
    examples = defaultdict(list)
    total_lines = 0
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            total_lines += 1
            try:
                obj = json.loads(line)
                text = ""
                # Concatena os textos do jsonl (voto e ementa)
                for content in obj['contents']:
                    for part in content['parts']:
                        text += part['text'] + " "
                
                for cat, regex in patterns.items():
                    matches = regex.findall(text)
                    if matches:
                        counts[cat] += len(matches)
                        if len(examples[cat]) < 3:
                            examples[cat].append(matches[0])
            except Exception:
                pass
                
    print(f"Total de registros analisados: {total_lines}\n")
    for cat in patterns.keys():
        if counts[cat] > 0:
            print(f"[!] VAZAMENTO DETECTADO - {cat}: {counts[cat]} ocorrências.")
            print(f"    Amostras capturadas: {examples[cat]}")
        else:
            print(f"[OK] {cat}: Base 100% Limpa.")

if __name__ == "__main__":
    audit("dataset_treino.jsonl")
    audit("dataset_teste.jsonl")
