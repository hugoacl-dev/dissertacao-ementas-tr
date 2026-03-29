---
description: Inspecionar um registro específico do dataset (teste ou treino)
argument-hint: "[índice] [teste|treino]"
---

Runbook curto. Fonte local: `.claude/AGENTS.md`.

Consulte especialmente:

- `## Mandato do Agente`
- `## Segurança, LGPD e Dados Sensíveis`
- `## Contrato Atual do JSONL`
- `## Comandos Operacionais Essenciais`

Use este comando para inspeção manual pontual de registros, sem alterar o protocolo experimental.

```bash
python3 -m pipeline.ferramentas.ver_registro 42
python3 -m pipeline.ferramentas.ver_registro 10 treino
```

Notas:

- `índice` é opcional no script e o padrão é `0`
- `split` pode ser `teste` ou `treino`
- a inspeção não autoriza copiar dados reais para logs, documentação, commits ou mensagens externas;
- se a inspeção revelar falha de higienização ou anonimização, a correção deve ser acompanhada de nova auditoria LGPD.
