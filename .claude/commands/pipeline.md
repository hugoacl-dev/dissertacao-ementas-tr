---
description: Executar o pipeline completo (Fases 1–4 + auditoria LGPD)
---

Runbook curto. Fonte canônica: `AGENTS.md`.

```bash
bash pipeline/run_all.sh
```

Ordem executada:

1. `pipeline/01_ingestao.py`
2. `pipeline/02_higienizacao.py`
3. `pipeline/03_anonimizacao.py`
4. `pipeline/audit.py`
5. `pipeline/04_estatisticas.py`

Observação:

- a Fase 4 já atualiza `docs/data/estatisticas_corpus.json` automaticamente.
