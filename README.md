# Lung Cancer Detection — LUNA16

Este projeto implementa um modelo de Deep Learning (CNN 3D) para a detecção automatizada de nódulos pulmonares, utilizando o dataset [LUNA16](https://luna16.grand-challenge.org/).

## Status do Projeto: Fase 2 Concluída

Atualmente, o projeto concluiu sua segunda fase de treinamento, atingindo marcos significativos de performance em um ambiente de dados altamente desbalanceado.

### Resultados Atuais (Época 10)
| Métrica | Valor | Observação |
| :--- | :--- | :--- |
| **Recall (Sensibilidade)** | **~95%** | Essencial para não perder nódulos reais em exames médicos. |
| **F1-Score** | **0.2495** | Recorde atingido com ajuste fino de Learning Rate (Fase 2). |
| **Acurácia de Validação** | **98.5%** | Alta estabilidade na classificação de candidatos. |

---

## Diferenciais Técnicos

### 1. Robust Data Loading & Cache Recovery
Implementamos um sistema de carregamento de dados resiliente que resolve problemas de corrupção de arquivos de forma autônoma. Se um arquivo de cache (`.pt`) for detectado como corrompido, o sistema:
1. Detecta o `RuntimeError`.
2. Exclui o arquivo inválido.
3. Regenera o crop 3D diretamente do scan MHD original.
4. Salva uma nova cópia íntegra e prossegue com o treinamento sem interrupções.

### 2. Balanced Training Strategy
Utilizamos uma abordagem de amostragem balanceada (`ratio_int=2`) para garantir que o modelo veja nódulos suficientes durante o treino, compensando o desbalanceamento natural do dataset (onde apenas ~0.25% dos candidatos são nódulos).

---

## Estrutura do Projeto

```
├── checkpoints/       # Modelos treinados (.pt)
├── data/luna/         # Dataset LUNA16 (MHD/RAW)
├── notebooks/         # Analise de resultados e avaliacao:
│   ├── 07_results_analysis.ipynb   # Graficos de Loss/F1
│   └── 08_model_evaluation.ipynb   # Validacao do modelo final
├── scripts/           # Scripts de execucao:
│   ├── run_training.py          # Treino inicial (Fase 1)
│   └── run_training_phase2.py   # Refinamento (Phase 2)
├── src/               # Core do projeto (Model, Data, Inference)
└── pyproject.toml     # Dependencias (uv)
```

## Como Executar

### Pré-requisitos
- Python >= 3.11
- [uv](https://docs.astral.sh/uv/)

### Instalação
```bash
git clone https://github.com/skricardo/lung-cancer-detection.git
cd lung-cancer-detection
uv sync
```

### Treinamento
Para iniciar o treinamento do zero ou retomar as fases:
```bash
# Fase 1 (Epocas 1-5)
python scripts/run_training.py

# Fase 2 (Epocas 6-10 - Fine Tuning)
python scripts/run_training_phase2.py
```

---

## Licença e Aviso Legal

Este projeto tem caráter estritamente educacional e de pesquisa. O modelo aqui apresentado é uma ferramenta de estudo e **não substitui, sob nenhuma circunstância, o diagnóstico, aconselhamento ou tratamento realizado por um profissional médico qualificado**. Os resultados obtidos não devem ser utilizados para fins clínicos ou de diagnóstico em pacientes reais.
