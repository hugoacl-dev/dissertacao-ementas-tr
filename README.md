# Geração Abstrativa de Ementas Judiciais via Fine-Tuning de LLM

Pipeline de dados e treinamento para Processamento de Linguagem Natural aplicado ao domínio jurídico.

**Autor:** Hugo Andrade Correia Lima Filho
**Modelo Base:** `gemini-3.1-pro` (Google DeepMind)

---

## Objetivo

Desenvolver e validar um pipeline computacional completo para a **geração abstrativa de ementas judiciais** a partir de votos (fundamentações) da juizado especial federal (JEF), utilizando fine-tuning supervisionado de LLMs.

## Corpus

- **32.325 pares** {fundamentação, ementa} extraídos do sistema sistema judicial
- Razão de compressão média: **23,9:1** (660 palavras → 30 palavras)
- Novel n-grams: **38,3%** de unigramas e **86,6%** de trigramas inéditos, confirmando a natureza **genuinamente abstrativa** das ementas

## Pipeline

O projeto é dividido em **7 fases**. As Fases 1–4 são sequenciais; após a Fase 4, as Fases 5 e 6 executam em **paralelo**, convergindo na Fase 7:

| Fase | Script | Descrição |
|---:|---|---|
| 1 | `pipeline/01_ingestao.py` | Ingestão do dump PostgreSQL e exportação dos pares válidos |
| 2 | `pipeline/02_higienizacao.py` | Remoção de ruído estrutural via Regex (HTML, IDs PJe, carimbos DJe, assinaturas). Datas e conteúdo de mérito são preservados |
| 3 | `pipeline/03_anonimizacao.py` | Anonimização LGPD: CPF, CNPJ, NPU, e-mail, telefone, nomes de partes privadas → tokens genéricos. Nomes de agentes públicos (juízes, relatores) são preservados (Art. 93, IX CF). Split treino/teste 90/10 |
| — | `pipeline/audit.py` | Auditoria pós-Fase 3: verifica ausência de dados pessoais residuais (7 categorias) |
| 4 | `pipeline/04_estatisticas.py` | Estatísticas descritivas: funil de attrition, distribuições, novel n-grams |
| 5 | `pipeline/05_finetuning.py` | Fine-tuning supervisionado via API Google AI Studio *(em desenvolvimento)* |
| 6 | `pipeline/06_baseline.py` | Baseline zero-shot para comparação *(em desenvolvimento)* |
| 7 | Notebook Colab | Avaliação: ROUGE, BERTScore, NLI, bootstrap, avaliação humana (Likert + Kappa) *(em desenvolvimento)* |

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

A Fase 7 gera ementas com o modelo fine-tuned e, para cada amostra, compara a `ementa_baseline` (Fase 6) e a `ementa_finetuned` contra a `ementa_oficial` (referência humana). A validação é conduzida em quatro eixos:

1. **Sintático-Semântico:** ROUGE-1/2/L + BERTScore com `xlm-roberta-large`
2. **Factual:** NLI via `xlm-roberta-large-xnli` (detecção de alucinações)
3. **Estatístico:** Bootstrap resampling (1.000 iterações, IC 95%, p-value)
4. **Humano:** 30-50 amostras, 2 avaliadores, design cego, Likert 1-5, Cohen's Kappa (κ ≥ 0.6)

## Reprodutibilidade

| Artefato | Descrição |
|---|---|
| `requirements.txt` | Dependência externa do pipeline (Fases 1–4 usam apenas stdlib) |
| `random.seed(42)` | Seed para divisão treino/teste determinística |
| `.gitignore` | Exclui dados brutos, banco SQLite e documentos privados |

## Licença

Este repositório contém código desenvolvido para fins acadêmicos. Os dados judiciais utilizados são de acesso restrito institucional.
