# Regras de Proteção de Dados (LGPD)

> Estas regras são **inegociáveis**. Violações comprometem compliance regulatória e podem invalidar a pesquisa.

## Dados pessoais — NUNCA expor

- **NUNCA** incluir em outputs, exemplos, logs ou commits: CPFs, nomes de partes privadas, NPUs reais, e-mails, telefones ou endereços de pessoas naturais.
- Ao criar exemplos ou demonstrações, usar **exclusivamente** tokens genéricos:
  - `[NOME_PESSOA]`, `[NOME_OCULTADO]`, `[CPF]`, `[CNPJ]`, `[NPU]`
  - `[EMAIL]`, `[TELEFONE]`, `[CONTA-DIGITO]`, `[ENDEREÇO_COMPLETO]`
- Nomes de **agentes públicos** (juízes, relatores, ministros) **podem** ser preservados — Art. 93, IX da Constituição Federal.
- Razões sociais (pessoa jurídica), cidades e CEPs **NÃO** são dados pessoais (LGPD Art. 5º, I).

## Arquivos sensíveis — NUNCA commitar

Os seguintes dados são gitignored e devem permanecer assim:
- `data/*.json`, `data/*.jsonl`, `data/*.sqlite` — datasets brutos e processados
- `dump_sistema_judicial.sql` — dump PostgreSQL completo
- `pesquisa/` — documentação interna da pesquisa
- Credenciais (`.env*`, `credentials*.json`, `*.key`)

## Anonimização

- Qualquer **novo padrão regex** de anonimização deve ser validado com `pipeline/audit.py`.
- O audit verifica **7 categorias** de dados pessoais residuais após processamento.
- Sempre documentar decisões de anonimização em `pesquisa/NOTAS_PESQUISA.md` (Seção 4).

## Base regulatória

- LGPD — Lei nº 13.709/2018
- Resolução CNJ nº 615/2025 — IA no Poder Judiciário
- Recomendação CNJ nº 154/2024 — Modelo-padrão de ementas
