# Regras LGPD

Fonte canônica: `../AGENTS.md`.

Consulte especialmente:

- `## Segurança, LGPD e Dados Sensíveis`
- `## Divergência Conhecida Entre Documentação e Código`

Resumo mínimo:

- nunca exponha dados pessoais reais em exemplos, logs, commits ou documentação;
- use apenas tokens genéricos ao demonstrar dados;
- valide novos padrões de anonimização com `python3 pipeline/audit.py`;
- trate `pipeline/system_prompt.txt` como prompt canônico, não o prefixo legado ainda citado por `audit.py`.
