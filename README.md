# Geração Abstrativa de Ementas Judiciais via Fine-Tuning de LLM

Pipeline de dados e treinamento para Processamento de Linguagem Natural aplicado ao domínio jurídico.

**Autor:** Hugo Andrade Correia Lima Filho
**Modelo Base:** `gemini-3.1-pro` (Google DeepMind)

---

## Objetivo

Desenvolver e validar um pipeline computacional completo para a **geração abstrativa de ementas judiciais** a partir de votos (fundamentações) da Turma Recursal da Justiça Federal na Paraíba (JFPB), utilizando fine-tuning supervisionado de LLMs.

## Corpus

- **32.325 pares** {fundamentação, ementa} extraídos do sistema TR-ONE
- Razão de compressão média: **23,6:1** (651 palavras → 30 palavras)
- Novel n-grams: **38,7%** de unigramas e **87,0%** de trigramas inéditos, confirmando a natureza **genuinamente abstrativa** das ementas

## Pipeline

O projeto é dividido em **7 fases sequenciais**:

| Fase | Script | Descrição |
|---:|---|---|
| 1 | `pipeline/01_ingestao.py` | Ingestão do dump PostgreSQL e exportação dos pares válidos |
| 2 | `pipeline/02_higienizacao.py` | Limpeza de ruídos processuais via Regex (IDs PJe, carimbos DJe, assinaturas) |
| 3 | `pipeline/03_anonimizacao.py` | Anonimização LGPD (CPF, nomes, locais → tokens genéricos) + split treino/teste |
| 4 | `pipeline/04_estatisticas.py` | Estatísticas descritivas: funil de attrition, distribuições, novel n-grams |
| 5 | `pipeline/05_finetuning.py` | Fine-tuning supervisionado via API Google AI Studio *(em desenvolvimento)* |
| 6 | `pipeline/06_baseline.py` | Baseline zero-shot para comparação *(em desenvolvimento)* |
| 7 | Notebook Colab | Avaliação: ROUGE, BERTScore, NLI, bootstrap resampling *(em desenvolvimento)* |

### Execução

```bash
# Executar todo o pipeline (Fases 1–4)
bash pipeline/run_all.sh
```

### Auditoria LGPD

```bash
# Verifica ausência de PII residual nos datasets
python3 pipeline/audit.py
```

## Dashboard

Dashboard interativo para visualização das estatísticas do corpus, disponível em [`docs/`](docs/index.html).

- Funil de attrition, histogramas de comprimento, novel n-grams, distribuição temporal
- Tema claro (projetor) e escuro
- Dados consumidos de `data/estatisticas_corpus.json`

## Avaliação (Fase 7)

A validação será conduzida em três eixos:

1. **Sintático-Semântico:** ROUGE-1/2/L + BERTScore
2. **Factual:** NLI via `xlm-roberta-large-xnli` (detecção de alucinações)
3. **Estatístico:** Bootstrap resampling (1.000 iterações, IC 95%)

## Reprodutibilidade

| Artefato | Descrição |
|---|---|
| `random.seed(42)` | Seed para divisão treino/teste determinística |
| `.gitignore` | Exclui dados brutos e banco SQLite do versionamento |

## Licença

Este repositório contém código desenvolvido para fins acadêmicos. Os dados judiciais utilizados são de acesso restrito institucional.
