# Geração Abstrativa de Ementas Judiciais via Fine-Tuning de LLM

Pipeline de dados e treinamento para Processamento de Linguagem Natural aplicado ao domínio jurídico.

Para agentes de código, a fonte canônica de instruções do projeto é `AGENTS.md`. A pasta `.claude/` é mantida apenas como camada de compatibilidade para Claude Code.

**Modelos:** `Gemini 2.5 Flash` (Google, Vertex AI) · `Qwen 2.5 14B-Instruct` (Alibaba, LoRA/Unsloth)

---

## Objetivo

Desenvolver e validar estatisticamente um pipeline computacional completo para a **geração abstrativa de ementas judiciais** a partir de votos (fundamentações) de um juizado especial federal, combinando extração segura de dados (LGPD e Resolução CNJ 615/2025), fine-tuning supervisionado de dois LLMs de naturezas distintas (proprietário e open-source) e validação em quatro eixos: léxico-semântico, qualidade jurídica (LLM-as-a-Judge), estatístico e humano.

## Corpus

- **32.321 pares** {fundamentação, ementa} extraídos do sistema judicial
- Razão de compressão média: **23,41:1** (649,4 palavras → 30,1 palavras)
- Novel n-grams: **39,2%** de unigramas, **73,5%** de bigramas e **87,5%** de trigramas inéditos, confirmando a natureza **abstrativa** das ementas

## Pipeline

O projeto é dividido em **7 fases**. As Fases 1–4 são sequenciais; após a Fase 4, as Fases 5 e 6 executam em **paralelo**, convergindo na Fase 7:

| Fase | Script(s) | Descrição |
|---:|---|---|
| 1 | `pipeline/01_ingestao.py` | Ingestão do dump PostgreSQL e exportação dos pares válidos com `data_cadastro` para divisão cronológica |
| 2 | `pipeline/02_higienizacao.py` | Remoção de ruído estrutural via Regex (HTML, IDs PJe, carimbos DJe, assinaturas). Exclusão de registros corrompidos (fund. idêntica à ementa). Datas e conteúdo de mérito são preservados |
| 3 | `pipeline/03_anonimizacao.py` | Anonimização LGPD: CPF, CNPJ, NPU, e-mail, telefone, nomes de partes privadas → tokens genéricos. Agentes públicos preservados (Art. 93, IX CF). **Split cronológico** 90/10 |
| — | `pipeline/audit.py` | Auditoria pós-Fase 3: verifica ausência de dados pessoais residuais (8 categorias, incluindo nomes privados em contexto residual) |
| 4 | `pipeline/04_estatisticas.py` | Estatísticas descritivas: funil de attrition, distribuições, novel n-grams |
| 5 | `pipeline/05_finetuning_gemini.py` | Fine-tuning supervisionado (SFT) do **Gemini 2.5 Flash** via API do Vertex AI, com manifesto local e modo de preparação prévia do job |
| 5 | `pipeline/05_finetuning_qwen.py` | Fine-tuning (LoRA via Unsloth) do **Qwen 2.5 14B** em GPU RunPod A100 80GB, com manifesto local e modo de preparação prévia do treino |
| 6 | `pipeline/06_baseline_gemini.py` | Baseline zero-shot do Gemini 2.5 Flash para comparação |
| 6 | `pipeline/06_baseline_qwen.py` | Baseline zero-shot do Qwen 2.5 14B para comparação |
| 7 | `pipeline/fase7_casos.py` + `pipeline/fase7_protocolo.py` + `pipeline/fase7_metricas.py` + `pipeline/fase7_estatisticas.py` + Notebook Colab | Geração dos casos-base, protocolo versionado, consolidação de métricas e inferência estatística pareada das 4 condições experimentais: ROUGE + **BERTScore F1** + LLM-as-a-Judge + avaliação humana *(em desenvolvimento)* |

### Execução

```bash
# Executar todo o pipeline (Fases 1–4 + auditoria LGPD)
bash pipeline/run_all.sh

# Atualizar pipeline e dashboard de uma vez
bash pipeline/run_all.sh && git add docs/data/estatisticas_corpus.json && git commit -m "dados: atualizar estatísticas" && git push
```

### Utilitários

```bash
# Inspecionar um registro do dataset (índice 0 do teste, por padrão)
python3 pipeline/ver_registro.py 42          # registro 42 do teste
python3 pipeline/ver_registro.py 10 treino   # registro 10 do treino

# Rodar a suíte mínima de regressão
pytest -q

# Gerar os casos-base da Fase 7 a partir do dataset de teste
python3 pipeline/fase7_casos.py

# Preparar o job SFT do Gemini sem submeter ao Vertex AI
python3 pipeline/05_finetuning_gemini.py --project-id SEU_PROJECT_ID --staging-bucket gs://SEU_BUCKET --prepare-only

# Preparar o treino LoRA do Qwen sem executar localmente
python3 pipeline/05_finetuning_qwen.py --prepare-only

# Executar baseline zero-shot do Gemini
python3 pipeline/06_baseline_gemini.py

# Executar baseline zero-shot do Qwen
python3 pipeline/06_baseline_qwen.py --model-id Qwen/Qwen2.5-14B-Instruct

# Gerar o manifesto versionado da Fase 7
python3 pipeline/fase7_protocolo.py

# Consolidar ROUGE, BERTScore e scores do juiz em metricas_automaticas.csv
python3 pipeline/fase7_metricas.py

# Gerar o relatório estatístico da Fase 7 a partir da tabela consolidada
python3 pipeline/fase7_estatisticas.py
```

## Dashboard

Dashboard interativo para visualização das estatísticas do corpus, disponível em [`docs/`](docs/index.html).

- Funil de attrition, histogramas de comprimento, novel n-grams, distribuição temporal
- Tema claro (projetor) e escuro
- Dados consumidos de `docs/data/estatisticas_corpus.json`

## Avaliação (Fase 7)

A Fase 7 compara **quatro condições experimentais** (Gemini FT, Gemini Zero-Shot, Qwen FT, Qwen Zero-Shot) contra as ementas oficiais (referência humana). A validação é conduzida em quatro eixos:

1. **Léxico-Semântico:** ROUGE-1/2/L + **BERTScore F1** com `xlm-roberta-large` e `rescale_with_baseline=True`
2. **Qualidade Jurídica:** LLM-as-a-Judge via DeepSeek V3 em 5 dimensões adaptadas à Recomendação CNJ 154/2024, com **score global** calculado como média aritmética das dimensões
3. **Estatístico:** inferência **pareada por exemplo**, com bootstrap pareado (**10.000** reamostragens) para IC 95%, teste de permutação pareado (**10.000** iterações) para `p-value` e ajustes de multiplicidade
4. **Humano:** **40 casos**, 2 avaliadores, design cego, escala Likert `1–5` e **weighted Cohen's kappa quadrático** por critério

Os **co-desfechos primários** são `BERTScore F1` médio e `score global` médio do LLM-as-a-Judge. A inferência confirmatória é conduzida separadamente para Gemini e Qwen; a consistência entre arquiteturas é analisada de forma **exploratória**.

Infraestrutura já versionada no repositório:

- Casos-base da Fase 7: `pipeline/fase7_casos.py`
- Helper compartilhado das predições: `pipeline/fase7_predicoes_utils.py`
- Prompt do juiz: `pipeline/prompts/llm_judge_prompt.txt`
- Contratos e manifesto da Fase 7: `pipeline/fase7_protocolo.py`
- Consolidação das métricas da Fase 7: `pipeline/fase7_metricas.py`
- Núcleo estatístico da Fase 7: `pipeline/fase7_estatisticas.py`
- Tabela consolidada esperada de entrada: `data/fase7/metricas_automaticas.csv`
- Schema de `casos_avaliacao.jsonl`: `caso_id`, `indice_teste`, `fundamentacao`, `ementa_referencia`
- Schema de cada arquivo de predição: `caso_id`, `condicao_id`, `ementa_gerada`

## Reprodutibilidade

| Artefato | Descrição |
|---|---|
| `requirements.txt` | Dependências do pipeline local e da suíte de testes |
| `requirements_fases_avancadas.txt` | Dependências opcionais das Fases 5–7, para ambientes específicos |
| `tests/` | Suíte mínima de regressão e smoke test sintético das Fases 2–4 |
| `.github/workflows/testes.yml` | CI executando `pytest -q` em `push` para `main` e em PRs |
| `pipeline/05_finetuning_gemini.py` | Preparação e submissão do SFT do Gemini, com manifesto em `data/fase5/gemini_sft_manifest.json` |
| `pipeline/05_finetuning_qwen.py` | Preparação e execução do SFT LoRA do Qwen, com manifesto em `data/fase5/qwen_sft_manifest.json` |
| `pipeline/fase5_tuning_utils.py` | Carregamento do dataset conversacional, nomes de experimento e persistência dos manifestos da Fase 5 |
| `pipeline/06_baseline_gemini.py` | Geração das predições `gemini_zero_shot.jsonl` via Vertex AI |
| `pipeline/06_baseline_qwen.py` | Geração das predições `qwen_zero_shot.jsonl` via `transformers` |
| `pipeline/fase7_casos.py` | Geração de `data/fase7/casos_avaliacao.jsonl` a partir de `data/dataset_teste.jsonl` |
| `pipeline/fase7_predicoes_utils.py` | Leitura, retomada incremental e persistência canônica das predições |
| `pipeline/prompts/llm_judge_prompt.txt` | Prompt versionado do LLM-as-a-Judge |
| `pipeline/fase7_protocolo.py` | Geração do manifesto e contratos mínimos dos artefatos da Fase 7 |
| `pipeline/fase7_metricas.py` | Consolidação de ROUGE, BERTScore, score global do juiz e dimensões individuais em `data/fase7/metricas_automaticas.csv` |
| `pipeline/fase7_estatisticas.py` | Inferência estatística pareada e geração de `data/fase7/relatorio_estatistico.json` |
| **Divisão cronológica** | Por `data_cadastro` em `pipeline/03_anonimizacao.py` (sem shuffle aleatório) |
| Versão dos modelos base | Registrada em `modelo_gemini_nome.txt` e `modelo_qwen_checkpoint/` |
| `.gitignore` | Exclui dados brutos, banco SQLite e documentos privados |

## Licença

Este repositório contém código desenvolvido para fins acadêmicos. Os dados judiciais utilizados são de acesso restrito institucional.
