# CLAUDE.md — Contexto do Projeto

## 🎯 Objetivo

Desenvolvimento de um sistema de detecção de câncer de pulmão (nódulos pulmonares) utilizando Deep Learning, com base no dataset **LUNA16**.

## 📁 Estrutura

| Pasta        | Descrição                                  |
|--------------|--------------------------------------------|
| `src/`       | Código-fonte (modelos, pipeline, utils)    |
| `notebooks/` | Notebooks de exploração e protótipos       |
| `tests/`     | Testes unitários e de integração           |
| `data/luna/` | Dataset LUNA16 (não versionado)            |
| `data/raw/`  | Dados auxiliares brutos (não versionado)   |
| `docs/`      | Documentação técnica e de referência       |

## 🛠️ Stack

- **Python** >= 3.11.3
- **PyTorch** (modelo principal)
- **SimpleITK** (leitura de imagens médicas .mhd/.raw)
- **scikit-learn / scikit-image** (pré e pós processamento)
- **Gradio** (interface de demonstração)
- **Jupyter** (experimentação)
- **Manim** (visualizações/animações)
- **uv** (gerenciador de ambiente)
- **pytest** (testes)

## 📋 Demandas / TODO

- [ ] 🔧 Setup do projeto (pastas, ambiente, git)
- [ ] 📥 Download e organização do dataset LUNA16
- [ ] 🔍 EDA — Análise exploratória dos CT scans
- [ ] 🏗️ Pipeline de pré-processamento (windowing, resampling, normalização)
- [ ] 🧠 Definição e treino do modelo (arquitetura)
- [ ] 📊 Avaliação (métricas, curvas FROC)
- [ ] 🖥️ Interface Gradio para demonstração
- [ ] 📝 Documentação final e portfólio

## 📌 Decisões de Design

_A ser preenchido conforme o projeto avança._

## ⚠️ Notas Importantes

- O dataset LUNA16 **não é versionado** no Git (está no `.gitignore`).
- Modelos treinados (`.pth`, `.pt`) também são ignorados pelo Git.
- Variáveis sensíveis ficam no `.env` (não versionado).
