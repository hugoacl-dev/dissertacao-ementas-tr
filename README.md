# Geração Abstrativa de Ementas Judiciais via Fine-Tuning de LLM

Pipeline de dados e treinamento para Processamento de Linguagem Natural aplicado ao domínio jurídico.

Para agentes de código neste workspace, a fonte de instruções mantida localmente está em `.claude/AGENTS.md`.

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
| 1 | `pipeline.fase1_4.fase01_ingestao` | Ingestão do dump PostgreSQL e exportação dos pares válidos com `data_cadastro` para divisão cronológica |
| 2 | `pipeline.fase1_4.fase02_higienizacao` | Remoção de ruído estrutural via Regex (HTML, IDs PJe, carimbos DJe, assinaturas). Exclusão de registros corrompidos (fund. idêntica à ementa). Datas e conteúdo de mérito são preservados |
| 3 | `pipeline.fase1_4.fase03_anonimizacao` | Anonimização LGPD: CPF, CNPJ, NPU, e-mail, telefone, nomes de partes privadas → tokens genéricos. Agentes públicos preservados (Art. 93, IX CF). **Split cronológico** 90/10 |
| — | `pipeline.ferramentas.auditoria` | Auditoria pós-Fase 3: verifica ausência de dados pessoais residuais (8 categorias, incluindo nomes privados em contexto residual) |
| 4 | `pipeline.fase1_4.fase04_estatisticas` | Estatísticas descritivas: funil de attrition, distribuições, novel n-grams |
| 5 | `pipeline.fase5.finetuning_gemini` | Fine-tuning supervisionado (SFT) do **Gemini 2.5 Flash** via API do Vertex AI, com manifesto local e modo de preparação prévia do job |
| 5 | `pipeline.fase5.finetuning_qwen` | Fine-tuning (LoRA via Unsloth) do **Qwen 2.5 14B** em GPU RunPod A100 80GB, com manifesto local e modo de preparação prévia do treino |
| 6 | `pipeline.fase6.baseline_gemini` | Runner canônico de inferência do Gemini 2.5 Flash, com retomada incremental, manifesto de execução e suporte às condições `gemini_zero_shot` e `gemini_ft` |
| 6 | `pipeline.fase6.baseline_qwen` | Runner canônico de inferência do Qwen 2.5 14B, com retomada incremental, manifesto de execução e suporte às condições `qwen_zero_shot` e `qwen_ft` (incluindo checkpoint LoRA local) |
| 7 | `pipeline.fase7.casos_avaliacao` + `pipeline.fase7.protocolo` + `pipeline.fase7.avaliacao_judge` + `pipeline.fase7.avaliacao_humana` + `pipeline.fase7.metricas` + `pipeline.fase7.estatisticas` | Geração dos casos-base, protocolo versionado, executor canônico do LLM-as-a-Judge, amostragem/relatório da avaliação humana, consolidação de métricas e inferência estatística pareada das 4 condições experimentais |

### Execução

```bash
# Executar todo o pipeline (Fases 1–4 + auditoria LGPD)
bash scripts/run_all.sh

# Atualizar pipeline e dashboard de uma vez
bash scripts/run_all.sh && git add docs/data/estatisticas_corpus.json && git commit -m "dados: atualizar estatísticas" && git push
```

### Utilitários

```bash
# Inspecionar um registro do dataset (índice 0 do teste, por padrão)
python3 -m pipeline.ferramentas.ver_registro 42          # registro 42 do teste
python3 -m pipeline.ferramentas.ver_registro 10 treino   # registro 10 do treino

# Rodar a suíte mínima de regressão
pytest -q

# Gerar os casos-base da Fase 7 a partir do dataset de teste
python3 -m pipeline.fase7.casos_avaliacao

# Preparar o job SFT do Gemini sem submeter ao Vertex AI
# Por padrão, a CLI grava artefatos exploratórios em data/exploratorio/
python3 -m pipeline.fase5.finetuning_gemini --project-id SEU_PROJECT_ID --staging-bucket gs://SEU_BUCKET --prepare-only

# Preparar o treino LoRA do Qwen sem executar localmente
python3 -m pipeline.fase5.finetuning_qwen --prepare-only

# Executar baseline zero-shot do Gemini
python3 -m pipeline.fase6.baseline_gemini

# Executar baseline zero-shot do Qwen
python3 -m pipeline.fase6.baseline_qwen --model-id Qwen/Qwen2.5-14B-Instruct

# Rodar explicitamente no perfil oficial após congelamento metodológico
python3 -m pipeline.fase5.finetuning_gemini --project-id SEU_PROJECT_ID --staging-bucket gs://SEU_BUCKET --prepare-only --perfil-execucao oficial
python3 -m pipeline.fase6.baseline_gemini --perfil-execucao oficial

# Gerar predições fine-tuned do Gemini usando o modelo ajustado da Fase 5 oficial
python3 -m pipeline.fase6.baseline_gemini --perfil-execucao oficial --condicao-id gemini_ft --model-id "$(cat data/fase5/modelo_gemini_nome.txt)"

# Gerar predições fine-tuned do Qwen usando o checkpoint LoRA local da Fase 5 oficial
python3 -m pipeline.fase6.baseline_qwen --perfil-execucao oficial --condicao-id qwen_ft --model-id data/fase5/modelo_qwen_checkpoint

# Gerar o manifesto versionado da Fase 7
# Por padrão, a CLI grava artefatos exploratórios em data/exploratorio/
python3 -m pipeline.fase7.protocolo

# Executar o LLM-as-a-Judge e persistir avaliacao_llm_judge.jsonl
python3 -m pipeline.fase7.avaliacao_judge

# Preparar a amostra cega, o gabarito separado e o CSV da avaliação humana
python3 -m pipeline.fase7.avaliacao_humana --modo preparar

# Consolidar a planilha preenchida da avaliação humana
python3 -m pipeline.fase7.avaliacao_humana --modo analisar

# Consolidar ROUGE, BERTScore e scores do juiz em metricas_automaticas.csv
python3 -m pipeline.fase7.metricas

# Gerar o relatório estatístico da Fase 7 a partir da tabela consolidada
python3 -m pipeline.fase7.estatisticas

# Quando a execução passar a ser oficial, explicite o perfil
python3 -m pipeline.fase7.casos_avaliacao --perfil-execucao oficial
python3 -m pipeline.fase7.protocolo --perfil-execucao oficial
python3 -m pipeline.fase7.avaliacao_judge --perfil-execucao oficial
python3 -m pipeline.fase7.avaliacao_humana --perfil-execucao oficial --modo preparar
python3 -m pipeline.fase7.metricas --perfil-execucao oficial
python3 -m pipeline.fase7.estatisticas --perfil-execucao oficial
```

### Perfis de Execução

As CLIs das Fases 5, 6 e 7 agora distinguem dois perfis:

- `exploratorio` (padrão): artefatos locais em `data/exploratorio/` e, no Gemini SFT, prefixo GCS padrão `testes/fase5`
- `oficial`: artefatos nos caminhos canônicos `data/fase5/` e `data/fase7/`, a serem usados somente após o congelamento do pipeline, prompt e parâmetros

Essa separação evita que smoke tests e rodadas exploratórias contaminem os artefatos destinados ao resultado final.

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

- Casos-base da Fase 7: `pipeline/fase7/casos_avaliacao.py`
- Helper compartilhado das predições: `pipeline/fase7/predicoes_utils.py`
- Prompt do juiz: `pipeline/prompts/llm_judge_prompt.txt`
- Contratos e manifesto da Fase 7: `pipeline/fase7/protocolo.py`
- Executor canônico do LLM-as-a-Judge: `pipeline/fase7/avaliacao_judge.py`
- Amostragem cega, gabarito separado e relatório da avaliação humana: `pipeline/fase7/avaliacao_humana.py`
- Consolidação das métricas da Fase 7: `pipeline/fase7/metricas.py`
- Núcleo estatístico da Fase 7: `pipeline/fase7/estatisticas.py`
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
| `pipeline/fase5/finetuning_gemini.py` | Preparação e submissão do SFT do Gemini, com perfis `exploratorio`/`oficial`; a CLI grava em `data/exploratorio/fase5/` por padrão e usa `data/fase5/` apenas com `--perfil-execucao oficial` |
| `pipeline/fase5/finetuning_qwen.py` | Preparação e execução do SFT LoRA do Qwen, com separação entre artefatos exploratórios e oficiais |
| `pipeline/fase5/tuning_utils.py` | Carregamento do dataset conversacional, nomes de experimento e persistência dos manifestos da Fase 5 |
| `pipeline/fase6/baseline_gemini.py` | Geração das predições `gemini_zero_shot.jsonl` e `gemini_ft.jsonl`, com perfis `exploratorio`/`oficial`; a CLI usa `data/exploratorio/fase7/predicoes/` por padrão |
| `pipeline/fase6/baseline_qwen.py` | Geração das predições `qwen_zero_shot.jsonl` e `qwen_ft.jsonl`, com suporte a checkpoint LoRA local e separação entre artefatos exploratórios e oficiais |
| `pipeline/fase7/casos_avaliacao.py` | Geração de `data/fase7/casos_avaliacao.jsonl` a partir de `data/dataset_teste.jsonl` |
| `pipeline/fase7/casos_avaliacao.py` | Geração dos casos-base com perfis `exploratorio`/`oficial`, mantendo os testes em `data/exploratorio/fase7/` por padrão |
| `pipeline/fase7/avaliacao_judge.py` | Execução incremental do DeepSeek V3 / `deepseek-chat` com separação entre artefatos exploratórios e oficiais |
| `pipeline/fase7/avaliacao_humana.py` | Geração da amostra estratificada cega (`amostra_humana.json`), gabarito separado (`gabarito_cegamento_humano.json`), template `avaliacao_humana.csv` e relatório humano consolidado, também separados por perfil |
| `pipeline/fase7/predicoes_utils.py` | Leitura, retomada incremental e persistência canônica das predições |
| `pipeline/prompts/llm_judge_prompt.txt` | Prompt versionado do LLM-as-a-Judge |
| `pipeline/fase7/protocolo.py` | Geração do manifesto e contratos mínimos dos artefatos da Fase 7, em perfil exploratório por padrão |
| `pipeline/fase7/metricas.py` | Consolidação de ROUGE, BERTScore, score global do juiz e dimensões individuais em caminhos separados por perfil |
| `pipeline/fase7/estatisticas.py` | Inferência estatística pareada e geração do relatório estatístico em caminhos separados por perfil |
| **Divisão cronológica** | Por `data_cadastro` em `pipeline/fase1_4/fase03_anonimizacao.py` (sem shuffle aleatório) |
| Versão dos modelos base | Registrada em `modelo_gemini_nome.txt` e `modelo_qwen_checkpoint/` |
| `.gitignore` | Exclui dados brutos, banco SQLite e documentos privados |

## Licença

Este repositório contém código desenvolvido para fins acadêmicos. Os dados judiciais utilizados são de acesso restrito institucional.
