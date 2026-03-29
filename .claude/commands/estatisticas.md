---
description: Regenerar estatísticas descritivas do corpus (Fase 4)
---

Runbook curto. Fonte canônica: `AGENTS.md`.

```bash
python3 -m pipeline.fase1_4.fase04_estatisticas
```

Artefatos relevantes:

- `data/estatisticas_corpus.json`
- `docs/data/estatisticas_corpus.json`

Observação:

- o script já copia o JSON final para `docs/data/`, então não é necessário `cp` manual antes do commit.
