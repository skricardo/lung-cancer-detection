# CLAUDE.md — Contexto do Projeto

## Regras de Estilo (obrigatorio para todos os agentes)

- Nunca usar emojis em nenhum arquivo, comentario ou output.
- Em notebooks Jupyter, nao usar secoes numeradas (ex: "1. Introducao", "2. Dados"). Usar apenas titulos descritivos.
- Nunca separar secoes com linhas horizontais ("---") em notebooks ou markdown.

## Objetivo

Desenvolvimento de um sistema de deteccao de cancer de pulmao (nodulos pulmonares) utilizando Deep Learning, com base no dataset **LUNA16**.

## Estrutura

| Pasta        | Descricao                                  |
|--------------|--------------------------------------------|
| `src/`       | Codigo-fonte (modelos, pipeline, utils)    |
| `notebooks/` | Notebooks de exploracao e prototipos       |
| `tests/`     | Testes unitarios e de integracao           |
| `data/luna/` | Dataset LUNA16 (nao versionado)            |
| `data/raw/`  | Dados auxiliares brutos (nao versionado)   |
| `docs/`      | Documentacao tecnica e de referencia       |

## Stack

- **Python** >= 3.12
- **PyTorch** (modelo principal)
- **SimpleITK** (leitura de imagens medicas .mhd/.raw)
- **scikit-learn / scikit-image** (pre e pos processamento)
- **Gradio** (interface de demonstracao)
- **Jupyter** (experimentacao)
- **Manim** (visualizacoes/animacoes)
- **uv** (gerenciador de ambiente)
- **pytest** (testes)

## Demandas / TODO

- [x] Setup do projeto (pastas, ambiente, git)
- [ ] Download e organizacao do dataset LUNA16
- [ ] EDA — Analise exploratoria dos CT scans
- [ ] Pipeline de pre-processamento (windowing, resampling, normalizacao)
- [ ] Definicao e treino do modelo (arquitetura)
- [ ] Avaliacao (metricas, curvas FROC)
- [ ] Interface Gradio para demonstracao
- [ ] Documentacao final e portfolio

## Decisoes de Design

_A ser preenchido conforme o projeto avanca._

## Notas Importantes

- O dataset LUNA16 **nao e versionado** no Git (esta no `.gitignore`).
- Modelos treinados (`.pth`, `.pt`) tambem sao ignorados pelo Git.
- Variaveis sensiveis ficam no `.env` (nao versionado).
- Python 3.12 usado no venv (3.14 nao tem wheels pre-compiladas para moderngl/manim).
