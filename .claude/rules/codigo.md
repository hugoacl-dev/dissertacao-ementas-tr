# Convenções de Código Python

## Versão e imports

- Python ≥ 3.10
- Usar `from __future__ import annotations` no topo de cada módulo
- Fases 1–4 usam **exclusivamente a biblioteca padrão** (sem dependências externas)
- Fase 5+: dependências externas permitidas, documentadas em `requirements.txt`

## Estilo

- Docstrings no padrão Google, em **português do Brasil**
- Type hints completos em todas as funções públicas
- Logging via módulo `logging` — jamais `print()` para informações de execução
- Caminhos via `pathlib.Path` — nunca strings hardcoded
- Encoding sempre `utf-8` ao abrir arquivos
- Regex pré-compilados (`re.compile()`) no nível do módulo para performance
- `dataclasses` para contadores e estruturas de dados imutáveis
- JSON de dados: `json.dump(data, f, ensure_ascii=False, separators=(",",":"))` — sem indentação (arquivos chegam a 217MB)

## Estrutura dos scripts do pipeline

Cada script segue o padrão:
1. Docstring de módulo com fase, entradas, saídas e como executar
2. Seção de configuração (constantes, paths)
3. Padrões pré-compilados (se aplicável)
4. Lógica de negócio (funções puras)
5. I/O e pipeline (orquestração)
6. Entrypoint (`main()` + `if __name__ == "__main__"`)

## Execução

- Scripts são executados **a partir da raiz do projeto**: `python3 pipeline/XX.py`
- Pipeline completo: `bash pipeline/run_all.sh`
- Novos scripts devem ser integrados ao `run_all.sh`

## Error handling

- Erros fatais: `log.critical()` + `sys.exit(1)` no bloco `if __name__ == "__main__"`
- Dependências sequenciais: usar `FileNotFoundError` com mensagem indicando qual fase executar antes
- Pattern: `try/except (FileNotFoundError, OSError)` no entrypoint

## Commits

- Mensagens em português, formato: `tipo: descrição` (ex: `dados: atualizar estatísticas`)
- Nunca commitar arquivos de `data/` ou `pesquisa/` (ver `.gitignore`)
