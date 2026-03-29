---
description: Executar o pipeline completo (Fases 1–4 + auditoria LGPD)
---

Runbook curto. Fonte local: `.claude/AGENTS.md`.

```bash
bash scripts/run_all.sh
```

Ordem executada:

1. `python3 -m pipeline.fase1_4.fase01_ingestao`
2. `python3 -m pipeline.fase1_4.fase02_higienizacao`
3. `python3 -m pipeline.fase1_4.fase03_anonimizacao`
4. `python3 -m pipeline.ferramentas.auditoria`
5. `python3 -m pipeline.fase1_4.fase04_estatisticas`

Observação:

- a Fase 4 já atualiza `docs/data/estatisticas_corpus.json` automaticamente.
