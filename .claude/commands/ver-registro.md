---
description: Inspecionar um registro específico do dataset (teste ou treino)
argument-hint: "[índice] [teste|treino]"
---

Exibe fundamentação e ementa de um registro do dataset para inspeção visual.

**Uso:**
```bash
# Registro 42 do dataset de teste (padrão)
python3 pipeline/ver_registro.py 42

# Registro 10 do dataset de treino
python3 pipeline/ver_registro.py 10 treino
```

**Argumentos:**
- `ÍNDICE` (obrigatório): número do registro (0-indexed)
- `SPLIT` (opcional): `teste` (padrão) ou `treino`

Útil para verificar qualidade da limpeza, anonimização e inspeção qualitativa de exemplos.
