# Lung Cancer Detection — LUNA16

Deteccao de nodulos pulmonares usando Deep Learning com o dataset [LUNA16](https://luna16.grand-challenge.org/).

## Estrutura do Projeto

```
├── data/
│   ├── luna/          # Dataset LUNA16 (CT scans, annotations)
│   └── raw/           # Dados brutos auxiliares
├── docs/              # Documentacao do projeto
├── notebooks/         # Jupyter notebooks (EDA, prototipos)
├── src/               # Codigo-fonte do projeto
├── tests/             # Testes automatizados
├── .env               # Variaveis de ambiente (nao versionado)
├── .gitignore
├── pyproject.toml     # Dependencias e configuracao do projeto
└── README.md
```

## Setup

### Pre-requisitos

- Python >= 3.11.3
- [uv](https://docs.astral.sh/uv/) (gerenciador de pacotes)

### Instalacao

```bash
# Clonar o repositorio
git clone https://github.com/skricardo/lung-cancer-detection.git
cd lung-cancer-detection

# Criar ambiente virtual e instalar dependencias
uv venv
uv sync
```

### Ativacao do Ambiente

```bash
# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

## Testes

```bash
pytest
```

## Dataset

O projeto utiliza o **LUNA16** (Lung Nodule Analysis 2016), que contem:
- 888 CT scans
- Anotacoes de nodulos validadas por radiologistas
- Formato: `.mhd` / `.raw`

> O dataset nao e versionado. Faca o download em [luna16.grand-challenge.org](https://luna16.grand-challenge.org/) e coloque os arquivos em `data/luna/`.

## Licenca

Este projeto e para fins educacionais e de pesquisa.
