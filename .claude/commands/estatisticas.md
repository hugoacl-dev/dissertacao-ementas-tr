---
description: Regenerar estatísticas descritivas do corpus (Fase 4)
---

Regenera as estatísticas descritivas a partir dos datasets já processados (pós-anonimização).

```bash
python3 pipeline/04_estatisticas.py
```

**Artefato gerado:** `data/estatisticas_corpus.json`

Este JSON é consumido pelo dashboard em `docs/` e pelo Notebook de avaliação (Fase 7).

**Métricas calculadas:**
- Funil de attrition (brutos → limpos → final)
- Distribuição de comprimento em palavras (média, mediana, P5/P25/P75/P95)
- Razão de compressão (fundamentação / ementa)
- Novel n-grams (uni/bi/tri) — evidência de abstratividade
- Período temporal coberto
- Total de palavras por split

Para atualizar o dashboard após regenerar:
```bash
cp data/estatisticas_corpus.json docs/data/ && git add docs/data/estatisticas_corpus.json && git commit -m "dados: atualizar estatísticas" && git push
```
