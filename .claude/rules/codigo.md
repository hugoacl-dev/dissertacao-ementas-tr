# Regras de Código

Fonte local: `../AGENTS.md`.

Consulte especialmente:

- `## Mandato do Agente`
- `## Hierarquia de Decisão`
- `## Stack Real do Projeto`
- `## Convenções de Código`
- `## Comandos Operacionais Essenciais`
- `## Git e Versionamento`

Resumo operacional:

- preserve a coerência arquitetural e os contratos já existentes antes de propor refatorações amplas;
- prefira mudanças pequenas, auditáveis, testáveis e fáceis de justificar;
- trate testes, validação, logging e reprodutibilidade como requisitos, não como acabamento;
- não altere silenciosamente split, prompt, métrica, schema ou protocolo experimental ao mexer no código.

Notas para usuários de Claude Code:

- As Fases 1–4 usam `pandas`, e a Fase 4 também usa `numpy`; não trate o pipeline local como stdlib-only.
- Preferências locais de ferramenta em `.claude/settings.json` não substituem as regras do projeto.
