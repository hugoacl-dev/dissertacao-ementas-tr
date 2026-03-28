# AGENTS.md — Dissertação: Geração Abstrativa de Ementas Judiciais

Arquivo de referência para agentes de código que trabalham neste projeto.

---

## Visão Geral do Projeto

Este é um projeto de pesquisa acadêmica (Mestrado) para **geração abstrativa de ementas judiciais** via Fine-Tuning de LLMs. O objetivo é desenvolver um pipeline computacional completo que, a partir de votos (fundamentações) de um Juizado Especial Federal, gere automaticamente ementas judiciais em linguagem natural.

**Corpus:** 32.312 pares {fundamentação, ementa} extraídos do sistema judicial.

**Modelos utilizados:**
- Gemini 2.5 Flash (Google, via Vertex AI)
- Qwen 2.5 14B-Instruct (Alibaba, via LoRA/Unsloth)

**Autor:** Hugo Andrade Correia Lima Filho

---

## Estrutura do Projeto

```
/Users/nti/dissertacao-ementas-tr/
├── pipeline/              # Scripts Python do pipeline de dados
│   ├── 01_ingestao.py     # Fase 1: Extração do dump PostgreSQL
│   ├── 02_higienizacao.py # Fase 2: Limpeza de ruído estrutural (regex)
│   ├── 03_anonimizacao.py # Fase 3: LGPD + formatação JSONL
│   ├── 04_estatisticas.py # Fase 4: Estatísticas descritivas do corpus
│   ├── audit.py           # Auditoria pós-anonimização (verificação LGPD)
│   ├── ver_registro.py    # Utilitário para inspecionar registros
│   ├── run_all.sh         # Script bash para executar Fases 1–4
│   └── system_prompt.txt  # Prompt de sistema para o modelo
├── data/                  # Artefatos gerados (NÃO versionados)
│   ├── dados_brutos.json
│   ├── dados_limpos.json
│   ├── dataset_treino.jsonl
│   ├── dataset_teste.jsonl
│   ├── estatisticas_corpus.json
│   └── banco_sistema_judicial.sqlite
├── docs/                  # Dashboard interativo (GitHub Pages)
│   ├── index.html
│   ├── css/styles.css
│   └── js/{app.js,charts.js}
├── pesquisa/              # Documentos de pesquisa (NÃO versionados)
├── requirements.txt       # Dependências Python
├── .gitignore            # Arquivos/diretórios ignorados pelo Git
└── README.md             # Documentação principal
```

---

## Stack Tecnológico

| Componente | Tecnologia |
|------------|------------|
| Linguagem | Python 3.10+ |
| Fases 1–4 | pandas ≥ 2.2 + biblioteca padrão Python |
| Banco local | SQLite |
| Fonte de dados | PostgreSQL (dump binário custom format) |
| Fine-tuning Gemini | Google Cloud AI Platform (Vertex AI) |
| Fine-tuning Qwen | Unsloth + LoRA (GPU RunPod A100 80GB) |
| Dashboard | HTML/CSS/JS vanilla + Chart.js |
| Deploy | GitHub Pages (pasta `docs/`) |

---

## Pipeline de Dados

O projeto é dividido em **7 fases**. As Fases 1–4 são sequenciais; após a Fase 4, as Fases 5 e 6 executam em paralelo, convergindo na Fase 7.

### Fases Implementadas (1–4)

| Fase | Script | Descrição |
|------|--------|-----------|
| 1 | `pipeline/01_ingestao.py` | Ingestão do dump PostgreSQL e exportação dos pares válidos com `data_cadastro` para divisão cronológica |
| 2 | `pipeline/02_higienizacao.py` | Remoção de ruído estrutural via Regex (HTML, IDs PJe, carimbos DJe, assinaturas). Preserva datas e conteúdo de mérito |
| 3 | `pipeline/03_anonimizacao.py` | Anonimização LGPD: CPF, CNPJ, NPU, e-mail, telefone, nomes de partes privadas → tokens genéricos. Agentes públicos preservados (Art. 93, IX CF). **Split cronológico 90/10** |
| — | `pipeline/audit.py` | Auditoria pós-Fase 3: verifica ausência de dados pessoais residuais (7 categorias) |
| 4 | `pipeline/04_estatisticas.py` | Estatísticas descritivas: funil de attrition, distribuições, novel n-grams, word cloud, histogramas |

### Fases em Desenvolvimento (5–7)

| Fase | Scripts | Descrição |
|------|---------|-----------|
| 5 | `05_finetuning_gemini.py`, `05_finetuning_qwen.py` | Fine-tuning supervisionado dos dois LLMs |
| 6 | `06_baseline_gemini.py`, `06_baseline_qwen.py` | Baseline zero-shot para comparação |
| 7 | Notebook Colab | Avaliação das 4 condições experimentais (ROUGE + BERTScore + LLM-as-a-Judge + bootstrap + avaliação humana) |

---

## Comandos de Build e Execução

### Executar todo o pipeline (Fases 1–4 + auditoria)

```bash
bash pipeline/run_all.sh
```

**Pré-requisitos:**
- `pg_restore` (instalável via `brew install postgresql@16` no macOS)
- Python 3.10+

### Executar fases individualmente

```bash
python3 pipeline/01_ingestao.py
python3 pipeline/02_higienizacao.py
python3 pipeline/03_anonimizacao.py
python3 pipeline/audit.py
python3 pipeline/04_estatisticas.py
```

### Inspecionar registros do dataset

```bash
# Registro 42 do conjunto de teste
python3 pipeline/ver_registro.py 42

# Registro 10 do conjunto de treino
python3 pipeline/ver_registro.py 10 treino
```

### Pipeline completo + atualização do dashboard

```bash
bash pipeline/run_all.sh && git add docs/data/estatisticas_corpus.json && git commit -m "dados: atualizar estatísticas" && git push
```

---

## Convenções de Código

### Idioma
- Código, variáveis e docstrings em **português brasileiro**
- A linguagem da documentação reflete o domínio jurídico brasileiro

### Estilo
- Type hints obrigatórios (`from __future__ import annotations`)
- Docstrings Google-style em todas as funções públicas
- Logging estruturado com timestamps
- Dataclasses para estatísticas e modelos de dados

### Padrão de módulos

```python
"""
nome_modulo.py — Descrição breve

Descrição detalhada do propósito, entradas e saídas.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

# Configuração de logging
log = logging.getLogger(__name__)

# Constantes de caminho (Path objects, não strings)
INPUT_PATH = Path("data/entrada.json")
OUTPUT_PATH = Path("data/saida.json")

# Dataclass para estatísticas
@dataclass
class Stats:
    contador: int = 0
    
    @property
    def taxa(self) -> float:
        return self.contador / 100

# Funções com type hints
def processar(dados: list[dict]) -> Stats:
    """Processa os dados e retorna estatísticas."""
    ...

# Entrypoint padronizado
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    processar()

if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, OSError) as exc:
        log.critical("Execução interrompida: %s", exc)
        sys.exit(1)
```

### Regex
- Padrões compilados no módulo (evita recompilação)
- Uso de raw strings (`r"..."`)
- Flags explícitas (`re.IGNORECASE`)

```python
_RE_CPF = re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b")
```

### Arquivos de dados
- JSON: compacto (`separators=(",", ":")`), UTF-8, `ensure_ascii=False`
- JSONL: uma linha por registro, sem indentação
- SQLite: modo WAL para melhor performance

---

## Estratégia de Testes

**Não há suite de testes automatizados** neste projeto. A validação é feita via:

1. **Auditoria LGPD:** `pipeline/audit.py` verifica 7 categorias de dados pessoais
2. **Estatísticas descritivas:** Fase 4 gera métricas para validação humana
3. **Dashboard interativo:** Visualização em `docs/index.html`
4. **Inspeção manual:** `pipeline/ver_registro.py` para amostragem

---

## Considerações de Segurança e LGPD

### Dados pessoais tratados

| Categoria | Token de substituição |
|-----------|----------------------|
| CPF | `[CPF]` |
| CNPJ | `[CNPJ]` |
| NPU (Número do Processo) | `[NPU]` |
| Conta bancária | `[CONTA-DIGITO]` |
| E-mail | `[EMAIL]` |
| Telefone | `[TELEFONE]` |
| Nome próprio | `[NOME_PESSOA]` / `[NOME_OCULTADO]` |
| Endereço | `[ENDEREÇO_COMPLETO]` |
| Data com cidade | `[DATA]` |

### O que NÃO é anonimizado (conforme LGPD)
- Cidades, estados, CEPs (dados públicos)
- Razões sociais (pessoa jurídica)
- Nomes de agentes públicos (Art. 93, IX CF)
- Termos jurídicos (Tribunal Regional Federal, etc.)

### Divisão treino/teste
- **Critério:** Cronológico por `data_cadastro` (90% treino, 10% teste)
- **NÃO é feito shuffle aleatório** — evita temporal leakage
- As decisões mais antigas vão para treino; as mais recentes para teste

---

## Artefatos de Dados

### Entrada
- `dump_sistema_judicial.sql` — Dump binário PostgreSQL (463MB, não versionado)

### Intermediários (gerados)
- `data/dados_brutos.json` — Pós-Fase 1 (com `data_cadastro`)
- `data/dados_limpos.json` — Pós-Fase 2
- `data/banco_sistema_judicial.sqlite` — Banco SQLite local

### Finais (gerados)
- `data/dataset_treino.jsonl` — Dataset treino formatado (90%)
- `data/dataset_teste.jsonl` — Dataset teste formatado (10%)
- `data/estatisticas_corpus.json` — Métricas e estatísticas

### JSONL Format
Formato compatível com Gemini Supervised Fine-Tuning e Unsloth/Qwen:

```json
{"contents": [
  {"role": "user", "parts": [{"text": "instrução + fundamentação"}]},
  {"role": "model", "parts": [{"text": "ementa"}]}
]}
```

Nota: A API de tuning do Gemini NÃO suporta role `"system"` — a instrução é embutida no turno `"user"`.

---

## Dependências

### Produção (Fases 1–4)
- `pandas >= 2.2.0`
- `pg_restore` (externo, para Fase 1)

### Desenvolvimento/Fases avançadas
```
# Fase 5a (Gemini)
google-cloud-aiplatform>=1.40.0

# Fase 5b (Qwen — executar em RunPod)
unsloth[cu124-torch260]

# Fases 6–7 (Colab)
rouge-score, bert-score, transformers, deepseek-sdk
```

---

## Git e Versionamento

### O que é versionado
- Todo código em `pipeline/`
- Dashboard em `docs/` (exceto dados)
- `README.md`, `requirements.txt`

### O que NÃO é versionado (`.gitignore`)
- `data/*` — Artefatos gerados (pesados)
- `pesquisa/` — Documentos de pesquisa privados
- `dump_sistema_judicial.sql` — Fonte de dados
- Arquivos de credenciais (`.env*`, `credentials*.json`)

### Workflow recomendado
```bash
# Após executar pipeline
bash pipeline/run_all.sh

# Commit apenas do JSON de estatísticas (para o dashboard)
git add docs/data/estatisticas_corpus.json
git commit -m "dados: atualizar estatísticas"
git push
```

---

## Contatos e Referências

- **Autor:** Hugo Andrade Correia Lima Filho
- **Instituição:** IFPB / PPGTI
- **Pesquisa:** Geração Abstrativa de Ementas Judiciais via Fine-Tuning de LLM

