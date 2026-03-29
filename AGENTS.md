# AGENTS.md — Fonte Canônica para Agentes

Guia normativo para agentes de código que trabalham neste projeto. Este arquivo é a fonte principal de contexto, regras e convenções do repositório.

## Papel Deste Arquivo

- `AGENTS.md` é a fonte canônica para agentes.
- A pasta `.claude/` existe apenas como camada de compatibilidade para Claude Code.
- Configurações específicas de ferramenta, como permissões locais ou preferências de branch, não são política geral do projeto.

## Visão Geral do Projeto

Este repositório implementa a dissertação de mestrado sobre **geração abstrativa de ementas judiciais** a partir de votos/fundamentações de uma Turma Recursal da Justiça Federal da Paraíba.

- **Corpus atual:** 32.321 pares `{fundamentação, ementa}`
- **Modelos-alvo:** Gemini 2.5 Flash e Qwen 2.5 14B-Instruct
- **Objetivo:** construir e validar um pipeline completo de extração, higienização, anonimização, estatísticas, fine-tuning, baseline e avaliação
- **Idioma do projeto:** português brasileiro em código, documentação, commits e interações

## Estado Atual do Repositório

### Fases implementadas

| Fase | Script | Estado | Saída principal |
|---|---|---|---|
| 1 | `pipeline.fase1_4.fase01_ingestao` | concluída | `data/dados_brutos.json`, `data/banco_sistema_judicial.sqlite` |
| 2 | `pipeline.fase1_4.fase02_higienizacao` | concluída | `data/dados_limpos.json` |
| 3 | `pipeline.fase1_4.fase03_anonimizacao` | concluída | `data/dataset_treino.jsonl`, `data/dataset_teste.jsonl` |
| — | `pipeline.ferramentas.auditoria` | concluída | auditoria LGPD dos JSONL |
| 4 | `pipeline.fase1_4.fase04_estatisticas` | concluída | `data/estatisticas_corpus.json`, `docs/data/estatisticas_corpus.json` |

### Fases implementadas no código, dependentes de ambiente específico

| Fase | Script | Estado |
|---|---|---|
| 5 | `pipeline.fase5.finetuning_gemini` | implementado; requer Vertex AI + `google-cloud-aiplatform` + `google-cloud-storage` |
| 5 | `pipeline.fase5.finetuning_qwen` | implementado; requer GPU + `unsloth` + `datasets` + `trl` |
| 6 | `pipeline.fase6.baseline_gemini` | implementado; requer Vertex AI + `google-genai` |
| 6 | `pipeline.fase6.baseline_qwen` | implementado; requer `torch` + `transformers` |
| 7 | Notebook Colab de avaliação humana | em desenvolvimento |

### Garantias de engenharia já implementadas

- suíte mínima de regressão em `tests/`
- smoke test sintético do pipeline para as Fases 2–4
- CI no GitHub Actions executando `pytest -q` em `push` para `main` e em `pull_request`
- helpers compartilhados para parsing do JSONL, validação de `data_cadastro`, paths do projeto e escrita atômica de artefatos JSON
- utilitários compartilhados da Fase 5 em `pipeline/fase5/tuning_utils.py`
- preparação e submissão do SFT do Gemini em `pipeline/fase5/finetuning_gemini.py`
- preparação e execução do SFT LoRA do Qwen em `pipeline/fase5/finetuning_qwen.py`
- prompt versionado do LLM-as-a-Judge em `pipeline/prompts/llm_judge_prompt.txt`
- geração dos casos-base da Fase 7 em `pipeline/fase7/casos_avaliacao.py`
- helper compartilhado de leitura e persistência das predições em `pipeline/fase7/predicoes_utils.py`
- manifesto versionado e contratos mínimos da Fase 7 em `pipeline/fase7/protocolo.py`
- consolidação de métricas da Fase 7 em `pipeline/fase7/metricas.py`
- núcleo de inferência estatística pareada da Fase 7 em `pipeline/fase7/estatisticas.py`

## Estrutura Relevante do Projeto

```text
pipeline/                 # Código do pipeline de dados e preparação experimental
docs/                     # Dashboard estático consumindo docs/data/estatisticas_corpus.json
data/                     # Artefatos gerados, gitignored
pesquisa/                 # Documentação metodológica local, gitignored
.claude/                  # Compatibilidade com Claude Code
README.md                 # Documentação humana de alto nível
requirements.txt          # Dependências registradas
```

### Documentos locais de pesquisa

A pasta `pesquisa/` existe neste ambiente local e é uma fonte importante de contexto metodológico. Sempre que uma mudança tocar desenho experimental, justificativas metodológicas, referências ou decisões LGPD, consulte:

1. `pesquisa/PLANO_ARQUITETURAL.md`
2. `pesquisa/NOTAS_PESQUISA.md`
3. documento compartilhado principal da dissertação em `pesquisa/`
4. `pesquisa/REFERENCIAS.md`
5. `pesquisa/fluxos/`

## Stack Real do Projeto

| Componente | Tecnologia |
|---|---|
| Linguagem principal | Python 3.10+ |
| Fases 1–4 | `pandas`, `numpy` e biblioteca padrão |
| Banco local | SQLite |
| Fonte de dados | PostgreSQL custom dump + `pg_restore` |
| Dashboard | HTML, CSS e JavaScript vanilla + Chart.js |
| Fine-tuning Gemini | Google Cloud Vertex AI |
| Fine-tuning Qwen | Unsloth + LoRA em GPU RunPod |

### Dependências registradas

- Produção local e testes: `requirements.txt`
- Estatísticas: `numpy` já é usado pelo código de `pipeline/fase1_4/fase04_estatisticas.py`
- Fases 5–7: `requirements_fases_avancadas.txt` e ambientes próprios, conforme a documentação da pesquisa
- Fase 5 Gemini: `google-cloud-aiplatform`, `google-cloud-storage`
- Fase 5 Qwen: `datasets`, `trl`, `unsloth`

## Convenções de Código

### Regras gerais

- Usar `from __future__ import annotations` no topo dos módulos Python.
- Escrever código, variáveis, docstrings e logs em português brasileiro.
- Usar type hints em funções públicas.
- Preferir `logging` a `print()` para informações de execução em scripts do pipeline.
- Usar `pathlib.Path` para caminhos de arquivo.
- Abrir arquivos com `encoding="utf-8"`.
- Compilar regex no nível do módulo.
- Usar `dataclasses` para estruturas de estatística e modelos simples.
- Serializar JSON de dados em UTF-8 sem indentação desnecessária.

### Estrutura esperada dos scripts do pipeline

1. Docstring de módulo com objetivo, entradas, saídas e forma de execução
2. Configuração e constantes
3. Padrões pré-compilados
4. Lógica de negócio
5. I/O e orquestração
6. `main()` com bloco `if __name__ == "__main__"`

### Tratamento de erros

- Erros fatais devem terminar com `log.critical(...)` e `sys.exit(1)` no entrypoint.
- Dependências entre fases devem gerar `FileNotFoundError` com instrução explícita da fase anterior.

## Invariantes Metodológicos

Estas regras são centrais para a pesquisa e não devem ser alteradas incidentalmente.

### Split treino/teste

- A divisão é **cronológica** por `data_cadastro`.
- As decisões mais antigas vão para treino; as mais recentes vão para teste.
- Não usar `shuffle`, `random_state` ou `train_test_split` aleatório para este protocolo.
- `data_cadastro` deve ser válida e parseável; registros com data nula, vazia ou inválida devem abortar a execução em vez de serem tolerados silenciosamente.

### Unidade de medida

- A unidade padrão é **palavras**, via `len(texto.split())`.
- Tokens de subword só entram quando a API do modelo exigir explicitamente.

### Quatro eixos de avaliação

As condições experimentais finais devem manter os quatro eixos previstos:

1. ROUGE-1/2/L + **BERTScore F1** com `xlm-roberta-large` e `rescale_with_baseline=True`
2. LLM-as-a-Judge via **DeepSeek V3**, com cinco dimensões e **score global** definido como média aritmética das dimensões
3. Inferência estatística **pareada por exemplo**, separada por modelo:
   bootstrap pareado com **10.000** reamostragens para intervalos de confiança, teste de permutação pareado com **10.000** iterações para `p-value`, ajuste **Holm-Bonferroni** para os co-desfechos primários e **Benjamini-Hochberg** para análises secundárias
4. Avaliação humana cega com **40 casos**, amostragem estratificada por quartis do comprimento da fundamentação, **2 avaliadores**, escala Likert `1–5` e **weighted Cohen's kappa quadrático** por critério

### Protocolo versionado da Fase 7

- O prompt canônico do juiz automático deve ser lido de `pipeline/prompts/llm_judge_prompt.txt`.
- Os casos-base da Fase 7 devem ser gerados por `python3 -m pipeline.fase7.casos_avaliacao`, a partir de `data/dataset_teste.jsonl`.
- Os baselines zero-shot devem gravar suas saídas em `data/fase7/predicoes/gemini_zero_shot.jsonl` e `data/fase7/predicoes/qwen_zero_shot.jsonl`.
- Os scripts de baseline devem usar retomada incremental e respeitar o schema canônico de predição.
- O manifesto e os contratos mínimos dos artefatos da Fase 7 devem ser gerados por `python3 -m pipeline.fase7.protocolo`.
- A consolidação de ROUGE, BERTScore, `judge_score_global` e dimensões individuais deve ser executada por `python3 -m pipeline.fase7.metricas`.
- A inferência estatística da Fase 7 deve ser executada por `python3 -m pipeline.fase7.estatisticas`, consumindo `data/fase7/metricas_automaticas.csv`.
- O schema da resposta do LLM-as-a-Judge exige exatamente cinco dimensões, cada uma com `score` inteiro de 1 a 5 e `justificativa` textual.
- O schema de `casos_avaliacao.jsonl` exige exatamente `caso_id`, `indice_teste`, `fundamentacao` e `ementa_referencia`.
- O schema dos arquivos de predição exige exatamente `caso_id`, `condicao_id` e `ementa_gerada`.
- O `score_global` do juiz não deve ser solicitado ao modelo; ele é calculado a posteriori como média aritmética simples.
- A tabela consolidada de métricas da Fase 7 deve conter, no mínimo, as colunas `caso_id`, `condicao_id`, `metrica` e `score`.
- O pareamento entre FT e zero-shot é obrigatório por `caso_id`; inconsistências de pareamento devem abortar a execução.

### Escopo inferencial atual

- Os **co-desfechos primários** são `BERTScore F1` médio e `score global` médio do LLM-as-a-Judge.
- A inferência confirmatória é conduzida **separadamente para Gemini e Qwen**.
- A consistência do efeito entre modelos é **exploratória**, não hipótese confirmatória adicional.
- ROUGE e as cinco dimensões individuais do juiz são **desfechos secundários**.

### Formato das ementas

- O corpus real usa formato condensado típico das Turmas Recursais: `ÁREA. TEMA. FUNDAMENTO. RESULTADO.`
- A Recomendação CNJ 154/2024 inspira critérios de qualidade, não a estrutura de saída em cinco partes.

### Consistência entre fine-tuning, baseline e avaliação

- O prompt canônico está em `pipeline/prompts/system_prompt.txt`.
- Scripts das Fases 5, 6 e 7 devem ler o mesmo arquivo para preservar consistência experimental.
- Alterar o prompt exige reexecução de `bash scripts/run_all.sh` para regenerar os JSONL.

## Segurança, LGPD e Dados Sensíveis

### Nunca expor dados pessoais reais

- Nunca incluir em exemplos, logs, documentação ou commits: CPF, NPU real, e-mail, telefone, endereço completo ou nomes de partes privadas.
- Em exemplos, usar apenas tokens como `[CPF]`, `[CNPJ]`, `[NPU]`, `[EMAIL]`, `[TELEFONE]`, `[CONTA-DIGITO]`, `[NOME_PESSOA]`, `[NOME_OCULTADO]` e `[ENDEREÇO_COMPLETO]`.

### O que é anonimizado

| Categoria | Token |
|---|---|
| CPF | `[CPF]` |
| CNPJ | `[CNPJ]` |
| NPU | `[NPU]` |
| Conta bancária | `[CONTA-DIGITO]` |
| E-mail | `[EMAIL]` |
| Telefone | `[TELEFONE]` |
| Nome de parte privada | `[NOME_PESSOA]` / `[NOME_OCULTADO]` |
| Endereço | `[ENDEREÇO_COMPLETO]` |
| Data com cidade | `[DATA]` |

### O que não é anonimizado

- Nomes de agentes públicos, conforme Art. 93, IX da CF
- Razões sociais de pessoa jurídica
- Cidades, estados e CEPs
- Termos jurídicos e nomes institucionais

### Arquivos sensíveis

Não commitar:

- `data/*.json`
- `data/*.jsonl`
- `data/*.sqlite`
- `dump_sistema_judicial.sql`
- conteúdo privado em `pesquisa/`
- credenciais como `.env*`, `credentials*.json`, `*.key`

### Auditoria LGPD

- Novos padrões de anonimização devem ser validados com `python3 -m pipeline.ferramentas.auditoria`.
- A auditoria verifica 8 categorias de vazamento residual, incluindo nomes próprios em contexto privado residual.
- Decisões metodológicas de anonimização devem ser refletidas em `pesquisa/NOTAS_PESQUISA.md` quando alterarem comportamento experimental.

## Contrato Atual do JSONL

- O prompt canônico atual é o de `pipeline/prompts/system_prompt.txt`.
- O turno `user` do JSONL embute a instrução e a fundamentação no formato:
  `{system_prompt}\n\nGere a ementa para a seguinte fundamentação:\n{fundamentação}`.
- A extração do conteúdo real do turno `user` foi centralizada em `pipeline/core/jsonl_utils.py`.
- Novos consumidores do JSONL devem reutilizar esse helper, em vez de replicar parsing local.

## Comandos Operacionais Essenciais

### Pipeline completo

```bash
bash scripts/run_all.sh
```

Executa Fase 1, Fase 2, Fase 3, auditoria LGPD e Fase 4.

### Fases individuais

```bash
python3 -m pipeline.fase1_4.fase01_ingestao
python3 -m pipeline.fase1_4.fase02_higienizacao
python3 -m pipeline.fase1_4.fase03_anonimizacao
python3 -m pipeline.ferramentas.auditoria
python3 -m pipeline.fase1_4.fase04_estatisticas
```

### Inspeção manual

```bash
python3 -m pipeline.ferramentas.ver_registro 42
python3 -m pipeline.ferramentas.ver_registro 10 treino
```

### Testes de regressão

```bash
pytest -q
```

### Fluxo operacional das Fases 5–7

```bash
python3 -m pipeline.fase7.casos_avaliacao
python3 -m pipeline.fase5.finetuning_gemini --project-id SEU_PROJECT_ID --staging-bucket gs://SEU_BUCKET --prepare-only
python3 -m pipeline.fase5.finetuning_qwen --prepare-only
python3 -m pipeline.fase6.baseline_gemini
python3 -m pipeline.fase6.baseline_qwen --model-id Qwen/Qwen2.5-14B-Instruct
python3 -m pipeline.fase7.protocolo
python3 -m pipeline.fase7.metricas
python3 -m pipeline.fase7.estatisticas
```

### CI

- Workflow versionado em `.github/workflows/testes.yml`
- Executa `pytest -q` em `push` para `main` e em `pull_request`

### Dashboard local

```bash
python3 -m http.server -d docs
```

### Atualização do dashboard versionado

`pipeline/fase1_4/fase04_estatisticas.py` já copia automaticamente `data/estatisticas_corpus.json` para `docs/data/estatisticas_corpus.json`. Após rodar o pipeline ou a Fase 4, basta versionar o artefato em `docs/data/`.

## Artefatos de Dados

### Entrada principal

- `dump_sistema_judicial.sql`

### Intermediários

- `data/dados_brutos.json`
- `data/dados_limpos.json`
- `data/banco_sistema_judicial.sqlite`

### Finais

- `data/dataset_treino.jsonl`
- `data/dataset_teste.jsonl`
- `data/estatisticas_corpus.json`
- `docs/data/estatisticas_corpus.json`
- `data/fase5/gemini_sft_manifest.json`
- `data/fase5/qwen_sft_manifest.json`
- `data/fase5/modelo_gemini_nome.txt`
- `data/fase5/modelo_qwen_checkpoint/`

### Formato JSONL

```json
{"contents":[
  {"role":"user","parts":[{"text":"instrução + fundamentação"}]},
  {"role":"model","parts":[{"text":"ementa"}]}
]}
```

Observação: a API de tuning do Gemini não aceita `role: "system"` em `contents`; por isso a instrução é embutida no turno `user`.

## Git e Versionamento

### É esperado versionar

- código em `pipeline/`
- dashboard em `docs/`
- `AGENTS.md`
- `README.md`
- `requirements.txt`
- `requirements_fases_avancadas.txt`

### Workflow comum após regenerar estatísticas

```bash
bash scripts/run_all.sh
git add docs/data/estatisticas_corpus.json
git commit -m "dados: atualizar estatísticas"
git push
```

## Relação com a Pasta `.claude`

- `.claude/CLAUDE.md` deve apenas apontar para este arquivo.
- `.claude/rules/*.md` devem funcionar como ponte leve para as seções daqui.
- `.claude/commands/*.md` podem existir como atalhos operacionais, desde que não contradigam este documento.
- `.claude/settings.json` é opcional e específico de Claude Code.

## Referências Internas Mais Importantes

- `README.md` para visão humana de alto nível
- `pipeline/prompts/system_prompt.txt` para o prompt canônico
- `pipeline/fase6/baseline_gemini.py` e `pipeline/fase6/baseline_qwen.py` para as condições zero-shot
- `pipeline/prompts/llm_judge_prompt.txt` para o prompt canônico do LLM-as-a-Judge
- `pipeline/fase7/casos_avaliacao.py` para geração dos casos-base da avaliação
- `pipeline/fase7/predicoes_utils.py` para retomada incremental e persistência das predições
- `pipeline/fase7/protocolo.py` para contratos e manifesto da Fase 7
- `pipeline/fase7/metricas.py` para consolidação das métricas automáticas e do juiz
- `pipeline/fase7/estatisticas.py` para a inferência estatística pareada da Fase 7
- `tests/` para a suíte mínima de regressão e o smoke test das Fases 2–4
- `.github/workflows/testes.yml` para a automação de CI
- `pesquisa/PLANO_ARQUITETURAL.md` para o desenho técnico das 7 fases
- `pesquisa/NOTAS_PESQUISA.md` para números observados e justificativas metodológicas
- documento compartilhado principal da dissertação em `pesquisa/` para o enquadramento acadêmico
