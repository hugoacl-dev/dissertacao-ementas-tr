---
description: Executar o pipeline completo (Fases 1–4 + auditoria LGPD)
---

Executa o pipeline de dados na ordem correta:

```bash
bash pipeline/run_all.sh
```

Este comando executa sequencialmente:
1. **Fase 1** — Ingestão (`01_ingestao.py`): dump SQL → `dados_brutos.json`
2. **Fase 2** — Higienização (`02_higienizacao.py`): limpeza regex → `dados_limpos.json`
3. **Fase 3** — Anonimização (`03_anonimizacao.py`): LGPD + split cronológico → `dataset_treino.jsonl` + `dataset_teste.jsonl`
4. **Auditoria** (`audit.py`): verificação de dados pessoais residuais (7 categorias)
5. **Fase 4** — Estatísticas (`04_estatisticas.py`): métricas descritivas → `estatisticas_corpus.json`

Para atualizar pipeline **e** dashboard de uma vez:
```bash
bash pipeline/run_all.sh && git add docs/data/estatisticas_corpus.json && git commit -m "dados: atualizar estatísticas" && git push
```
