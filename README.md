# Lung Cancer Detection -- LUNA16

Deteccao automatizada de nodulos pulmonares em tomografias computadorizadas (CT scans) utilizando uma CNN 3D treinada no dataset [LUNA16](https://luna16.grand-challenge.org/).

---

## Status do Projeto

O projeto concluiu duas fases de treinamento (10 epocas), atingindo marcos significativos de performance em um cenario de dados altamente desbalanceado (apenas ~0.25% dos candidatos sao nodulos reais).

### Resultados -- Fase 2 (Epoca 10)

| Metrica | Valor | Observacao |
|:---|:---|:---|
| **Recall (Sensibilidade)** | **~95%** | Prioridade maxima: detectar o maior numero possivel de nodulos |
| **F1-Score** | **0.2495** | Recorde atingido com fine-tuning na Fase 2 |
| **Acuracia de Validacao** | **98.5%** | Alta estabilidade na classificacao |
| **AUC-ROC** | Consultar notebook 11 | Avaliacao completa com curvas ROC e PR |

---

## Arquitetura do Modelo

O `LunaModel` e uma CNN 3D com 222K parametros, projetada para classificar crops volumetricos de CT scans.

```
Entrada: (1, 32, 48, 48) -- crop 3D de CT scan

BatchNorm3d(1)
    |
LunaBlock 1: Conv3d(1->8)   + ReLU, Conv3d(8->8)   + ReLU + MaxPool3d
LunaBlock 2: Conv3d(8->16)  + ReLU, Conv3d(16->16)  + ReLU + MaxPool3d
LunaBlock 3: Conv3d(16->32) + ReLU, Conv3d(32->32)  + ReLU + MaxPool3d
LunaBlock 4: Conv3d(32->64) + ReLU, Conv3d(64->64)  + ReLU + MaxPool3d
    |
Flatten -> Linear(1152, 2) -> Softmax

Saida: (prob_nao_nodulo, prob_nodulo)
```

**Caracteristicas:**
- Inicializacao Kaiming para convergencia estavel
- BatchNorm na entrada para normalizacao de valores Hounsfield (HU)
- 4 blocos convolucionais com aumento progressivo de canais

---

## Estrategia de Treinamento

| Fase | Epocas | Learning Rate | Descricao |
|:---:|:---:|:---:|:---|
| **Fase 1** | 1-5 | 0.001 | Treinamento inicial com augmentation completa |
| **Fase 2** | 6-10 | 0.0005 | Fine-tuning com LR reduzido |

**Otimizador:** SGD com Momentum 0.99

**Data Augmentation 3D:**
- Flip horizontal/vertical
- Offset (0.1)
- Scale (0.2)
- Rotacao aleatoria
- Ruido Gaussiano (std=25)

**Balanceamento:** Amostragem balanceada com `ratio_int=2` para compensar o desbalanceamento extremo do dataset.

---

## Diferenciais Tecnicos

### Robust Data Loading e Cache Recovery
Sistema de carregamento resiliente que resolve corrupcao de arquivos de forma autonoma:
1. Detecta o `RuntimeError` ao carregar um cache `.pt` corrompido.
2. Exclui o arquivo invalido.
3. Regenera o crop 3D diretamente do scan MHD original.
4. Salva uma nova copia integra e prossegue sem interrupcoes.

### Offline Disk Caching
Crops 3D sao pre-processados e salvos em disco como tensores `.pt`, eliminando reprocessamento de CT scans durante o treinamento e reduzindo drasticamente o tempo de I/O.

---

## Estrutura do Projeto

```
lung-cancer-detection/
|
|-- checkpoints/                    # Modelos treinados e resultados
|   |-- luna_model_best.pt          # Melhor checkpoint (F1-Score)
|   +-- val_results_phase2.pth      # Probabilidades e labels da validacao
|
|-- data/luna/                      # Dataset LUNA16 (MHD/RAW + CSVs)
|
|-- notebooks/                      # Notebooks de analise e desenvolvimento
|   |-- 00_duvidas_e_respostas.ipynb    # Fundamentos teoricos (ReLU, XOR, etc.)
|   |-- 01_download_luna16.ipynb        # Download e organizacao do dataset
|   |-- 02_explore_csv_data.ipynb       # Exploracao dos metadados CSV
|   |-- 03_candidates_vs_annotations.ipynb  # Cruzamento candidatos x anotacoes
|   |-- 03_unify_data_sources.ipynb     # Unificacao das fontes de dados
|   |-- 04_ct_scan_to_dataset.ipynb     # Conversao CT scan -> dataset de crops
|   |-- 05_build_cache.ipynb            # Construcao do cache offline
|   |-- 05_model_architecture.ipynb     # Design e teste da arquitetura CNN 3D
|   |-- 06_training.ipynb               # Loop de treinamento e metricas
|   |-- 07_results_analysis.ipynb       # Graficos de evolucao (Loss/F1/Acc)
|   |-- 08_model_evaluation.ipynb       # Avaliacao do modelo final
|   |-- 10_visual_report.ipynb          # Relatorio visual base
|   +-- 11_resultado_final.ipynb        # Showcase final de resultados
|
|-- scripts/                        # Scripts de execucao
|   |-- run_training.py             # Treinamento Fase 1 (epocas 1-5)
|   +-- run_training_phase2.py      # Fine-tuning Fase 2 (epocas 6-10)
|
|-- src/                            # Codigo-fonte principal
|   |-- model.py                    # Arquitetura LunaModel (CNN 3D)
|   |-- luna_data.py                # Dataset, cache e carregamento de CT scans
|   |-- training.py                 # Loop de treino, metricas e checkpoints
|   +-- inference.py                # Inferencia em lote sobre candidatos
|
|-- scratch/                        # Scripts auxiliares e utilitarios
|-- tests/                          # Testes
+-- pyproject.toml                  # Dependencias (gerenciadas via uv)
```

---

## Como Executar

### Pre-requisitos
- Python >= 3.11
- [uv](https://docs.astral.sh/uv/)
- Dataset LUNA16 (CT scans em formato MHD/RAW)

### Instalacao
```bash
git clone https://github.com/skricardo/lung-cancer-detection.git
cd lung-cancer-detection
uv sync
```

### Treinamento
```bash
# Fase 1 (Epocas 1-5)
python scripts/run_training.py

# Fase 2 - Fine Tuning (Epocas 6-10)
python scripts/run_training_phase2.py
```

### Avaliacao
Abra o notebook `notebooks/11_resultado_final.ipynb` para visualizar:
- Evolucao do treinamento (Loss, F1-Score, Accuracy)
- Curvas ROC e Precision-Recall
- Matriz de Confusao
- Dashboard de KPIs
- Galeria visual de diagnostico (True Positives, False Positives, False Negatives)
- Distribuicao de probabilidades

---

## Proximos Passos

- [ ] Implementar **Focal Loss** para priorizar exemplos dificeis
- [ ] Experimentar **ratio_int=3 ou 4** no balanceamento
- [ ] Testar **learning rate schedulers** (cosine annealing)
- [ ] Adicionar **dropout** para regularizacao
- [ ] Avaliar **segmentacao** para localizacao de nodulos no CT completo

---

## Stack Tecnologica

| Componente | Tecnologia |
|:---|:---|
| Framework | PyTorch |
| Imagens Medicas | SimpleITK |
| Metricas | scikit-learn |
| Visualizacao | Matplotlib |
| Gerenciamento | uv + pyproject.toml |

---

## Licenca e Aviso Legal

Este projeto tem carater estritamente educacional e de pesquisa. O modelo aqui apresentado e uma ferramenta de estudo e **nao substitui, sob nenhuma circunstancia, o diagnostico, aconselhamento ou tratamento realizado por um profissional medico qualificado**. Os resultados obtidos nao devem ser utilizados para fins clinicos ou de diagnostico em pacientes reais.
