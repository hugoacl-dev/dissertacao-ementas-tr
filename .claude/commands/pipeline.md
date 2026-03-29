---
description: Executar o pipeline completo (Fases 1–4 + auditoria LGPD)
---

Runbook curto. Fonte local: `.claude/AGENTS.md`.

Consulte especialmente:

- `## Mandato do Agente`
- `## Hierarquia de Decisão`
- `## Invariantes Metodológicos`
- `## Segurança, LGPD e Dados Sensíveis`
- `## Comandos Operacionais Essenciais`

Use este comando quando a intenção for recomputar o pipeline base inteiro, e não apenas inspecionar um artefato específico.

Antes de executar, verificar:

- se a reexecução completa é realmente necessária;
- se a entrada principal e os artefatos sensíveis estão no lugar esperado;
- se qualquer mudança resultante exigirá revisão metodológica, atualização de documentação ou atualização do dashboard.

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
- se a saída do pipeline mudar, trate isso como potencial impacto experimental, não apenas como alteração operacional.
