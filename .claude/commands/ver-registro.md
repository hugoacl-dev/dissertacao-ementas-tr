---
description: Inspecionar um registro específico do dataset (teste ou treino)
argument-hint: "[índice] [teste|treino]"
---

Runbook curto. Fonte canônica: `AGENTS.md`.

```bash
python3 pipeline/ver_registro.py 42
python3 pipeline/ver_registro.py 10 treino
```

Notas:

- `índice` é opcional no script e o padrão é `0`
- `split` pode ser `teste` ou `treino`
