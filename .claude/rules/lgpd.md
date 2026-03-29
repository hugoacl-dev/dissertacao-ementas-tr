# Regras LGPD

Fonte canônica: `../AGENTS.md`.

Consulte especialmente:

- `## Segurança, LGPD e Dados Sensíveis`
- `## Contrato Atual do JSONL`

Resumo mínimo:

- nunca exponha dados pessoais reais em exemplos, logs, commits ou documentação;
- use apenas tokens genéricos ao demonstrar dados;
- valide novos padrões de anonimização com `python3 pipeline/audit.py`;
- trate `pipeline/prompts/system_prompt.txt` como prompt canônico e `pipeline/jsonl_utils.py` como helper compartilhado para extrair o conteúdo real do turno `user`.
