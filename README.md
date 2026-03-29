# Geração Abstrativa de Ementas Judiciais via Fine-Tuning de LLM

Pipeline de dados e treinamento para Processamento de Linguagem Natural aplicado ao domínio jurídico.

Para agentes de código, a fonte canônica de instruções do projeto é `AGENTS.md`. A pasta `.claude/` é mantida apenas como camada de compatibilidade para Claude Code.

**Autor:** Hugo Andrade Correia Lima Filho
**Modelos:** `Gemini 2.5 Flash` (Google, Vertex AI) · `Qwen 2.5 14B-Instruct` (Alibaba, LoRA/Unsloth)

---

## Objetivo

Desenvolver e validar estatisticamente um pipeline computacional completo para a **geração abstrativa de ementas judiciais** a partir de votos (fundamentações) de um juizado especial federal, combinando extração segura de dados (LGPD e Resolução CNJ 615/2025), fine-tuning supervisionado de dois LLMs de naturezas distintas (proprietário e open-source) e validação em quatro eixos: léxico-semântico, qualidade jurídica (LLM-as-a-Judge), estatístico e humano.

## Corpus

- **32.312 pares** {fundamentação, ementa} extraídos do sistema judicial
- Razão de compressão média: **23,8:1** (660 palavras → 30 palavras)
- Novel n-grams: **38,9%** de unigramas e **87,2%** de trigramas inéditos, confirmando a natureza **genuinamente abstrativa** das ementas

## Pipeline

O projeto é dividido em **7 fases**. As Fases 1–4 são sequenciais; após a Fase 4, as Fases 5 e 6 executam em **paralelo**, convergindo na Fase 7:

| Fase | Script(s) | Descrição |
|---:|---|---|
| 1 | `pipeline/01_ingestao.py` | Ingestão do dump PostgreSQL e exportação dos pares válidos com `data_cadastro` para divisão cronológica |
| 2 | `pipeline/02_higienizacao.py` | Remoção de ruído estrutural via Regex (HTML, IDs PJe, carimbos DJe, assinaturas). Exclusão de registros corrompidos (fund. idêntica à ementa). Datas e conteúdo de mérito são preservados |
| 3 | `pipeline/03_anonimizacao.py` | Anonimização LGPD: CPF, CNPJ, NPU, e-mail, telefone, nomes de partes privadas → tokens genéricos. Agentes públicos preservados (Art. 93, IX CF). **Split cronológico** 90/10 |
| — | `pipeline/audit.py` | Auditoria pós-Fase 3: verifica ausência de dados pessoais residuais (7 categorias) |
| 4 | `pipeline/04_estatisticas.py` | Estatísticas descritivas: funil de attrition, distribuições, novel n-grams |
| 5 | `pipeline/05_finetuning_gemini.py` | Fine-tuning supervisionado (SFT) do **Gemini 2.5 Flash** via API do Vertex AI *(em desenvolvimento)* |
| 5 | `pipeline/05_finetuning_qwen.py` | Fine-tuning (LoRA via Unsloth) do **Qwen 2.5 14B** em GPU RunPod A100 80GB *(em desenvolvimento)* |
| 6 | `pipeline/06_baseline_gemini.py` | Baseline zero-shot do Gemini 2.5 Flash para comparação *(em desenvolvimento)* |
| 6 | `pipeline/06_baseline_qwen.py` | Baseline zero-shot do Qwen 2.5 14B para comparação *(em desenvolvimento)* |
| 7 | Notebook Colab | Avaliação das 4 condições experimentais: ROUGE + BERTScore + LLM-as-a-Judge + bootstrap + avaliação humana *(em desenvolvimento)* |

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
```

## Dashboard

Dashboard interativo para visualização das estatísticas do corpus, disponível em [`docs/`](docs/index.html).

- Funil de attrition, histogramas de comprimento, novel n-grams, distribuição temporal
- Tema claro (projetor) e escuro
- Dados consumidos de `data/estatisticas_corpus.json`

## Avaliação (Fase 7)

A Fase 7 compara **quatro condições experimentais** (Gemini FT, Gemini Zero-Shot, Qwen FT, Qwen Zero-Shot) contra as ementas oficiais (referência humana). A validação é conduzida em quatro eixos:

1. **Léxico-Semântico:** ROUGE-1/2/L + BERTScore com `xlm-roberta-large` (captura sinonímia jurídica que ROUGE ignora)
2. **Qualidade Jurídica:** LLM-as-a-Judge via DeepSeek V3 — 5 dimensões adaptadas à Recomendação CNJ 154/2024 (pertinência, completude, fidelidade, concisão, adequação terminológica)
3. **Estatístico:** Bootstrap resampling (1.000 iterações, IC 95%, p-value formal) para cada modelo separadamente
4. **Humano:** 30–50 amostras, 2 avaliadores, design cego, Likert 1–5, Cohen's Kappa (κ ≥ 0,6)

## Reprodutibilidade

| Artefato | Descrição |
|---|---|
| `requirements.txt` | Dependências do pipeline local (Fases 1–4) |
| **Divisão cronológica** | Por `data_cadastro` em `pipeline/03_anonimizacao.py` (sem shuffle aleatório) |
| Versão dos modelos base | Registrada em `modelo_gemini_nome.txt` e `modelo_qwen_checkpoint/` |
| `.gitignore` | Exclui dados brutos, banco SQLite e documentos privados |

## Licença

Este repositório contém código desenvolvido para fins acadêmicos. Os dados judiciais utilizados são de acesso restrito institucional.
