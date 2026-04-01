# 🫁 Lung Cancer Detection — LUNA16

Detecção de nódulos pulmonares usando Deep Learning com o dataset [LUNA16](https://luna16.grand-challenge.org/).

## 📂 Estrutura do Projeto

```
├── data/
│   ├── luna/          # Dataset LUNA16 (CT scans, annotations)
│   └── raw/           # Dados brutos auxiliares
├── docs/              # Documentação do projeto
├── notebooks/         # Jupyter notebooks (EDA, protótipos)
├── src/               # Código-fonte do projeto
├── tests/             # Testes automatizados
├── .env               # Variáveis de ambiente (não versionado)
├── .gitignore
├── pyproject.toml     # Dependências e configuração do projeto
└── README.md
```

## ⚙️ Setup

### Pré-requisitos

- Python >= 3.11.3
- [uv](https://docs.astral.sh/uv/) (gerenciador de pacotes)

### Instalação

```bash
# Clonar o repositório
git clone https://github.com/skricardo/lung-cancer-detection.git
cd lung-cancer-detection

# Criar ambiente virtual e instalar dependências
uv venv
uv sync
```

### Ativação do Ambiente

```bash
# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

## 🧪 Testes

```bash
pytest
```

## 📊 Dataset

O projeto utiliza o **LUNA16** (Lung Nodule Analysis 2016), que contém:
- 888 CT scans
- Anotações de nódulos validadas por radiologistas
- Formato: `.mhd` / `.raw`

> ⚠️ O dataset não é versionado. Faça o download em [luna16.grand-challenge.org](https://luna16.grand-challenge.org/) e coloque os arquivos em `data/luna/`.

## 📝 Licença

Este projeto é para fins educacionais e de pesquisa.
