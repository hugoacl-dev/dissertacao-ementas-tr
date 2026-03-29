---
description: Regenerar estatísticas descritivas do corpus (Fase 4)
---

Runbook curto. Fonte local: `.claude/AGENTS.md`.

Consulte especialmente:

- `## Mandato do Agente`
- `## Hierarquia de Decisão`
- `## Invariantes Metodológicos`
- `## Comandos Operacionais Essenciais`

Use este comando quando as Fases 1–3 já estiverem consistentes e o objetivo for apenas regenerar as estatísticas descritivas.

Antes de executar, verificar:

- se os artefatos de entrada já refletem o estado correto do corpus;
- se a atualização dos números terá impacto em dashboard, README, notas de pesquisa ou texto da dissertação;
- se a mudança observada é explicável e metodologicamente consistente.

```bash
python3 -m pipeline.fase1_4.fase04_estatisticas
```

Artefatos relevantes:

- `data/estatisticas_corpus.json`
- `docs/data/estatisticas_corpus.json`

Observação:

- o script já copia o JSON final para `docs/data/`, então não é necessário `cp` manual antes do commit.
- alterações nesses artefatos podem exigir revisão de números citados em documentação e material acadêmico.
