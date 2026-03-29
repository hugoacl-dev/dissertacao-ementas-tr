# Regras LGPD

Fonte local: `../AGENTS.md`.

Consulte especialmente:

- `## Hierarquia de Decisão`
- `## Segurança, LGPD e Dados Sensíveis`
- `## Contrato Atual do JSONL`
- `## Checklist Mínimo de Entrega`

Resumo mínimo:

- nunca exponha dados pessoais reais em exemplos, logs, commits ou documentação;
- use apenas tokens genéricos ao demonstrar dados;
- valide novos padrões de anonimização com `python3 -m pipeline.ferramentas.auditoria`;
- em caso de dúvida, favoreça a proteção de dados e a rastreabilidade da transformação;
- trate `pipeline/prompts/system_prompt.txt` como prompt canônico e `pipeline/core/jsonl_utils.py` como helper compartilhado para extrair o conteúdo real do turno `user`.
