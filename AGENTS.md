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
| 1 | `pipeline/01_ingestao.py` | concluída | `data/dados_brutos.json`, `data/banco_sistema_judicial.sqlite` |
| 2 | `pipeline/02_higienizacao.py` | concluída | `data/dados_limpos.json` |
| 3 | `pipeline/03_anonimizacao.py` | concluída | `data/dataset_treino.jsonl`, `data/dataset_teste.jsonl` |
| — | `pipeline/audit.py` | concluída | auditoria LGPD dos JSONL |
| 4 | `pipeline/04_estatisticas.py` | concluída | `data/estatisticas_corpus.json`, `docs/data/estatisticas_corpus.json` |

### Fases ainda não implementadas no código

| Fase | Scripts esperados | Estado |
|---|---|---|
| 5 | `pipeline/05_finetuning_gemini.py`, `pipeline/05_finetuning_qwen.py` | em desenvolvimento |
| 6 | `pipeline/06_baseline_gemini.py`, `pipeline/06_baseline_qwen.py` | em desenvolvimento |
| 7 | Notebook Colab de avaliação | em desenvolvimento |

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
3. `pesquisa/documento_compartilhado.md`
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
- Estatísticas: `numpy` já é usado pelo código de `pipeline/04_estatisticas.py`
- Fases 5–7: `requirements_fases_avancadas.txt` e ambientes próprios, conforme a documentação da pesquisa

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

1. ROUGE-1/2/L + BERTScore
2. LLM-as-a-Judge via DeepSeek V3
3. Bootstrap com 1.000 iterações, IC 95% e p-value
4. Avaliação humana cega com 2 avaliadores

### Formato das ementas

- O corpus real usa formato condensado típico das Turmas Recursais: `ÁREA. TEMA. FUNDAMENTO. RESULTADO.`
- A Recomendação CNJ 154/2024 inspira critérios de qualidade, não a estrutura de saída em cinco partes.

### Consistência entre fine-tuning, baseline e avaliação

- O prompt canônico está em `pipeline/system_prompt.txt`.
- Scripts das Fases 5, 6 e 7 devem ler o mesmo arquivo para preservar consistência experimental.
- Alterar o prompt exige reexecução de `bash pipeline/run_all.sh` para regenerar os JSONL.

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

- Novos padrões de anonimização devem ser validados com `python3 pipeline/audit.py`.
- A auditoria verifica 8 categorias de vazamento residual, incluindo nomes próprios em contexto privado residual.
- Decisões metodológicas de anonimização devem ser refletidas em `pesquisa/NOTAS_PESQUISA.md` quando alterarem comportamento experimental.

## Contrato Atual do JSONL

- O prompt canônico atual é o de `pipeline/system_prompt.txt`.
- O turno `user` do JSONL embute a instrução e a fundamentação no formato:
  `{system_prompt}\n\nGere a ementa para a seguinte fundamentação:\n{fundamentação}`.
- A extração do conteúdo real do turno `user` foi centralizada em `pipeline/jsonl_utils.py`.
- Novos consumidores do JSONL devem reutilizar esse helper, em vez de replicar parsing local.

## Comandos Operacionais Essenciais

### Pipeline completo

```bash
bash pipeline/run_all.sh
```

Executa Fase 1, Fase 2, Fase 3, auditoria LGPD e Fase 4.

### Fases individuais

```bash
python3 pipeline/01_ingestao.py
python3 pipeline/02_higienizacao.py
python3 pipeline/03_anonimizacao.py
python3 pipeline/audit.py
python3 pipeline/04_estatisticas.py
```

### Inspeção manual

```bash
python3 pipeline/ver_registro.py 42
python3 pipeline/ver_registro.py 10 treino
```

### Testes de regressão

```bash
pytest -q
```

### Dashboard local

```bash
python3 -m http.server -d docs
```

### Atualização do dashboard versionado

`pipeline/04_estatisticas.py` já copia automaticamente `data/estatisticas_corpus.json` para `docs/data/estatisticas_corpus.json`. Após rodar o pipeline ou a Fase 4, basta versionar o artefato em `docs/data/`.

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
bash pipeline/run_all.sh
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
- `pipeline/system_prompt.txt` para o prompt canônico
- `pesquisa/PLANO_ARQUITETURAL.md` para o desenho técnico das 7 fases
- `pesquisa/NOTAS_PESQUISA.md` para números observados e justificativas metodológicas
- `pesquisa/documento_compartilhado.md` para o enquadramento acadêmico da dissertação
