# Geração Abstrativa de Ementas Judiciais via Fine-Tuning de LLM

**Autor:** Hugo Andrade Correia Lima Filho
**Modelos:** Gemini 2.5 Flash (Google, Vertex AI) · Qwen 2.5 14B-Instruct (Alibaba, LoRA/Unsloth)
**Idioma:** Todo o projeto é em **português do Brasil** — código, docs, commits, interações.
**Status:** Fases 1–4 ✅ concluídas e validadas | Fases 5–7 ⏳ em desenvolvimento (scripts ainda não criados)

---

## O que é este projeto

Dissertação de mestrado: pipeline completo para gerar **ementas judiciais** (resumos abstrativos) a partir de votos/fundamentações da Turma Recursal da Justiça Federal da Paraíba. 32.312 pares {fundamentação, ementa}, razão de compressão 23,8:1, natureza genuinamente abstrativa (38,9% novel unigrams).

## Glossário

| Termo | Definição |
|---|---|
| **Fundamentação (voto)** | Texto longo do juiz justificando a decisão (~660 palavras méd.). É o **input** do modelo. A coluna de origem no TR-ONE é `votoementa`. |
| **Ementa** | Resumo curto e estruturado do que foi decidido (~30 palavras méd.). É o **target/output**. Segue o **formato condensado**: `ÁREA. TEMA. FUNDAMENTO. RESULTADO.` |
| **Turma Recursal (TR)** | Órgão colegiado dos Juizados Especiais Federais (JEF) que julga recursos. |
| **Voto-ementa** | Formato simplificado de decisão (Art. 39, Regimento Interno, Res. 01/2024-TR/PB) onde relatório+voto+ementa são condensados numa peça única. Afeta o **input** (voto mais curto), não o **output** (ementa sempre registrada como campo independente). ~0,6% das ementas carregam marcação explícita. |
| **NPU** | Número Único do Processo (formato CNJ: `NNNNNNN-DD.AAAA.J.TR.OOOO`). Dado anonimizado. |
| **Novel n-grams** | % de n-grams da ementa ausentes na fundamentação — mede abstratividade (See et al., 2017). |
| **TR-ONE** | Sistema judicial de origem dos dados. |
| **CNJ 154/2024** | Recomendação que propõe ementas em 5 partes (cabeçalho, caso, questão, razões, dispositivo). As ementas da TR **não seguem** este formato — usam o condensado. A CNJ 154/2024 **inspira os critérios de qualidade** (5 dimensões do LLM-as-a-Judge), **não o formato de saída**. |

## Estrutura do repositório

```
pipeline/                 # Scripts Python do pipeline (Fases 1–7)
  01_ingestao.py          # Fase 1: dump PostgreSQL → JSON
  02_higienizacao.py      # Fase 2: limpeza regex (HTML, IDs PJe, assinaturas)
  03_anonimizacao.py      # Fase 3: LGPD + split cronológico + JSONL
  04_estatisticas.py      # Fase 4: estatísticas descritivas do corpus
  system_prompt.txt       # System prompt canônico (lido por 03, 05, 06)
  audit.py                # Auditoria pós-anonimização (7 categorias)
  ver_registro.py         # Utilitário: inspecionar registro do dataset
  run_all.sh              # Executa Fases 1–4 + auditoria
data/                     # ⚠️ GITIGNORED — dados brutos, limpos, datasets JSONL
docs/                     # Dashboard interativo (Chart.js, tema claro/escuro)
pesquisa/                 # ⚠️ GITIGNORED — documentação de pesquisa
  PLANO_ARQUITETURAL.md   # Plano técnico detalhado das 7 fases
  NOTAS_PESQUISA.md       # Log empírico de execução e justificativas
  Hugo - Compartilhado.md # Pre-projeto completo da dissertação
  fluxos/                 # Diagramas Mermaid de cada fase
  REFERENCIAS.md          # Índice bibliográfico completo (fonte canônica: Zotero local)
  orientacoes/            # Reuniões com os orientadores e passos definidos
```

> ⚠️ `data/` e `pesquisa/` existem **apenas localmente** (gitignored). Se não encontrar esses diretórios, solicite ao usuário que os forneça.

## Pipeline (7 fases)

| Fase | Script | Descrição |
|---:|---|---|
| 1 | `01_ingestao.py` | Dump PostgreSQL → pares {fundamentação, ementa, data_cadastro} |
| 2 | `02_higienizacao.py` | Limpeza regex + filtros de qualidade |
| 3 | `03_anonimizacao.py` | Anonimização LGPD + split cronológico 90/10 → JSONL |
| 4 | `04_estatisticas.py` | Funil, distribuições, novel n-grams, compressão |
| 5 | `05_finetuning_*.py` | SFT: Gemini (Vertex AI) e Qwen (LoRA/Unsloth) *(não criado)* |
| 6 | `06_baseline_*.py` | Zero-shot com mesmos modelos base *(não criado)* |
| 7 | Notebook Colab | Avaliação: ROUGE + BERTScore + LLM-as-a-Judge + bootstrap + humana |

## Dashboard

`docs/index.html` — Chart.js puro, tema claro/escuro (otimizado para projetor).
Consome `docs/data/estatisticas_corpus.json` (copiado da Fase 4).
Testar localmente: `python3 -m http.server -d docs`

## Composição temática do corpus

O corpus é multitemático (~47% previdenciário, ~17% seguridade social, ~10% assistencial, ~9% processual, ~5% administrativo, ~2,5% FGTS, ~2% civil, tributário, constitucional e outros). O prompt e a avaliação **não devem restringir-se** a um subdomínio.

## Comandos essenciais

```bash
bash pipeline/run_all.sh                          # Pipeline completo (Fases 1–4)
python3 pipeline/ver_registro.py 42                # Inspecionar registro 42 do teste
python3 pipeline/ver_registro.py 10 treino         # Registro 10 do treino
python3 -m http.server -d docs                     # Servir dashboard localmente
```

## Regras obrigatórias

Leia as regras em `.claude/rules/` **antes** de qualquer modificação:
- **`rules/lgpd.md`** — Restrições de dados pessoais (inegociáveis)
- **`rules/codigo.md`** — Convenções de código Python
- **`rules/pesquisa.md`** — Rigor científico e metodológico

## Documentação detalhada

1. `pesquisa/REFERENCIAS.md` — Índice bibliográfico com classificação e notas de relevância (fonte canônica: Zotero)
2. `pesquisa/PLANO_ARQUITETURAL.md` — Arquitetura técnica de cada fase
3. `pesquisa/NOTAS_PESQUISA.md` — Log empírico e justificativas metodológicas
4. `pesquisa/Hugo - Compartilhado.md` — Pre-projeto completo (seções 1–6, referências)
5. `pesquisa/fluxos/` — Diagramas Mermaid de cada fase
