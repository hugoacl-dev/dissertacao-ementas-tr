# Rigor Científico e Metodológico

## Princípios invioláveis

- **Divisão treino/teste é CRONOLÓGICA** — por `data_cadastro`. Jamais sugerir `random.shuffle()` ou `train_test_split(random_state=...)`. Protocolo: SLDS (Rolshoven et al., 2025).
- **Unidade de medida é palavras** — `len(texto.split())`. Não usar tokens de subword, a não ser quando exigido pela API do modelo.
- **Quatro eixos de avaliação** são obrigatórios, não opcionais:
  1. Léxico-semântico: ROUGE-1/2/L + BERTScore (`xlm-roberta-large`)
  2. Qualidade jurídica: LLM-as-a-Judge via DeepSeek V3, 5 dimensões (ver seção abaixo)
  3. Estatístico: Bootstrap resampling (1.000 it., IC 95%, p-value)
  4. Humano: 30–50 amostras, 2 avaliadores, design cego, Likert 1–5, Cohen's κ ≥ 0,6

## Design experimental

- **4 condições**: Gemini FT, Gemini ZS, Qwen FT, Qwen ZS
- Variável isolada: **fine-tuning** (mesmos dados, mesmo prompt, mesma temperatura=0)
- Modelo juiz (DeepSeek V3) é de família distinta dos avaliados — elimina viés de família
- Dimensões do LLM-as-a-Judge: pertinência temática, completude dispositiva, fidelidade factual, concisão, adequação terminológica

## CNJ 154/2024 — escopo de aplicação

A Recomendação CNJ nº 154/2024 propõe um modelo de ementa em **5 partes** (cabeçalho, caso em exame, questão em discussão, razões de decidir, dispositivo/tese). Contudo, as ementas reais da Turma Recursal da JFPB seguem o **formato condensado** típico dos JEFs: `ÁREA. TEMA. FUNDAMENTO. RESULTADO.` — em uma única peça contínua, compatível com o voto-ementa autorizado pelo Art. 39 do Regimento Interno (Resolução nº 01/2024-TR/PB).

Portanto:
- A CNJ 154/2024 **inspira os critérios de qualidade** (as 5 dimensões do LLM-as-a-Judge), **não o formato de saída**.
- A dimensão "adequação terminológica" avalia o **uso correto da terminologia jurídica**, não conformidade com a estrutura de 5 partes.
- O prompt do juiz (Fase 7) deve ser calibrado para o formato condensado de turma recursal.

## Composição temática do corpus

O corpus é multitemático, não exclusivamente previdenciário:
- ~47% previdenciário, ~17% seguridade social, ~10% assistencial, ~9% processual, ~5% administrativo, ~2,5% FGTS, ~2% civil, e diversas outras áreas (tributário, constitucional, financiamento habitacional, etc.).
- O prompt e a avaliação não devem restringir-se a um subdomínio.

## Ao sugerir mudanças metodológicas

- Citar trade-offs explicitamente (ex: custo computacional vs. ganho em métrica)
- Considerar impacto na reprodutibilidade
- Verificar alinhamento com o SLDS (Rolshoven et al., 2025) — referência primária
- Alterações no funil de dados **devem** atualizar `pesquisa/NOTAS_PESQUISA.md` (Seção 2)

## Citações

- Formato ABNT: (SOBRENOME, ano) no texto, referência completa na seção de referências
- Ao mencionar trabalhos relacionados, verificar se já consta em `pesquisa/NOTAS_PESQUISA.md` (Seção 6)

## Alucinação jurídica

Atenção especial ao conceito de **alucinação jurídica**: geração de texto fluente e verossímil, mas factualmente inconsistente com o voto original. Exemplos: inversão de desfecho, omissão de fundamento legal, atribuição de tese inexistente. O Eixo 2 (LLM-as-a-Judge) e o Eixo 4 (validação humana) são os mecanismos de detecção.

## System prompt canônico

O prompt de instrução usado no dataset de treino e nos baselines está em `pipeline/system_prompt.txt`:

> "Você é o relator de uma Turma Recursal de Juizado Especial Federal. Sua tarefa é redigir a ementa do acórdão a partir do voto a seguir. A ementa deve ser um resumo curto, objetivo e autossuficiente do que foi decidido, no formato condensado utilizado pelas Turmas Recursais: área do direito, tema, fundamento determinante e resultado do julgamento. Responda exclusivamente com o texto da ementa, sem explicações adicionais."

Este prompt é carregado por `pipeline/03_anonimizacao.py` e embutido no turno `user` do JSONL (a API de SFT do Gemini não suporta role `system`). Qualquer script de Fase 5, 6 ou 7 **deve ler o mesmo arquivo** para manter consistência experimental.

> ⚠️ Alterar o prompt exige re-execução de `pipeline/run_all.sh` para regenerar os datasets JSONL.
