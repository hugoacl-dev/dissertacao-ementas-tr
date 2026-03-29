# Compatibilidade com Claude Code

Esta pasta existe apenas para compatibilidade com Claude Code.

## Fonte principal

Leia primeiro `../AGENTS.md`. Ele é a fonte canônica de:

- contexto do projeto
- estado atual das fases
- convenções de código
- regras LGPD
- invariantes metodológicos
- comandos operacionais

## Como usar esta pasta

- `rules/` contém ponte curta para as seções correspondentes de `AGENTS.md`
- `commands/` contém runbooks enxutos para operações frequentes
- `settings.json` contém preferências opcionais do Claude Code, não política geral do repositório

## Observações específicas

- A pasta `pesquisa/` existe neste ambiente local e deve ser consultada quando a tarefa tocar metodologia ou justificativas acadêmicas.
- O prompt canônico do projeto está em `pipeline/prompts/system_prompt.txt`.
- A extração da fundamentação a partir do turno `user` do JSONL foi centralizada em `pipeline/jsonl_utils.py`.
