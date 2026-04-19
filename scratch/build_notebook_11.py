"""
Script para gerar o notebook 11 - Showcase de Resultados
Um notebook criativo e visualmente impactante com os principais resultados do projeto.
Sem emojis. Apenas exemplos com CTs verificados no disco.
"""
import nbformat as nbf
import os


def build_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": ".venv",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12.12"
        }
    }

    cells = []

    # =====================================================================
    # CELL 1 - Title / Hero Section (Markdown)
    # =====================================================================
    cells.append(nbf.v4.new_markdown_cell(
        '# Deteccao Automatizada de Nodulos Pulmonares com Deep Learning\n'
        '\n'
        '---\n'
        '\n'
        '<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); '
        'padding: 30px; border-radius: 15px; color: white; margin-bottom: 20px;">\n'
        '\n'
        '### Resumo Executivo do Projeto\n'
        '\n'
        '| Item | Detalhe |\n'
        '|:---|:---|\n'
        '| **Dataset** | LUNA16 -- 888 CT scans, 551.065 candidatos |\n'
        '| **Modelo** | CNN 3D customizada (`LunaModel`) -- 222K parametros |\n'
        '| **Treinamento** | 10 epocas (2 fases), SGD + Momentum 0.99 |\n'
        '| **Recall (Sensibilidade)** | **~95%** -- essencial para nao perder nodulos reais |\n'
        '| **F1-Score** | **0.2495** -- recorde com fine-tuning na Fase 2 |\n'
        '| **Acuracia de Validacao** | **98.5%** -- alta estabilidade |\n'
        '\n'
        '</div>\n'
        '\n'
        '> **Aviso Legal**: Este projeto tem carater estritamente educacional e de pesquisa. '
        'Os resultados **nao substituem** o diagnostico de um profissional medico qualificado.\n'
    ))

    # =====================================================================
    # CELL 2 - Setup / Imports (Code)
    # =====================================================================
    cells.append(nbf.v4.new_code_cell(
        '# ============================================================\n'
        '# Setup e Importacoes\n'
        '# ============================================================\n'
        'import sys, os\n'
        'import torch\n'
        'import numpy as np\n'
        'import matplotlib.pyplot as plt\n'
        'import matplotlib.gridspec as gridspec\n'
        'from matplotlib.patches import FancyBboxPatch\n'
        'import matplotlib.patheffects as path_effects\n'
        'from sklearn.metrics import (\n'
        '    precision_recall_curve, roc_curve, auc,\n'
        '    confusion_matrix, ConfusionMatrixDisplay\n'
        ')\n'
        '\n'
        '# Adiciona o diretorio src ao path\n'
        'sys.path.insert(0, os.path.abspath("../src"))\n'
        'from luna_data import get_ct, load_candidates\n'
        'from model import LunaModel\n'
        '\n'
        '# Estilo visual premium (dark theme)\n'
        'plt.rcParams.update({\n'
        '    "figure.facecolor": "#0d1117",\n'
        '    "axes.facecolor": "#161b22",\n'
        '    "axes.edgecolor": "#30363d",\n'
        '    "axes.labelcolor": "#c9d1d9",\n'
        '    "text.color": "#c9d1d9",\n'
        '    "xtick.color": "#8b949e",\n'
        '    "ytick.color": "#8b949e",\n'
        '    "grid.color": "#21262d",\n'
        '    "font.family": "sans-serif",\n'
        '    "font.size": 12,\n'
        '    "axes.titlesize": 14,\n'
        '    "axes.titleweight": "bold",\n'
        '})\n'
        '\n'
        'device = torch.device("cuda" if torch.cuda.is_available() else "cpu")\n'
        'print(f"Dispositivo: {device}")\n'
        'print(f"PyTorch {torch.__version__}")\n'
    ))

    # =====================================================================
    # CELL 3 - Section: Arquitetura do Modelo (Markdown)
    # =====================================================================
    cells.append(nbf.v4.new_markdown_cell(
        '---\n'
        '## 1. Arquitetura do Modelo -- LunaModel\n'
        '\n'
        'O `LunaModel` e uma **CNN 3D** projetada para classificar crops volumetricos de CT scans.\n'
        'A entrada e um tensor `(1, 32, 48, 48)` representando uma regiao de interesse do pulmao.\n'
        '\n'
        '```\n'
        '+--------------------------------------------------------------+\n'
        '|                    LunaModel (222K params)                    |\n'
        '+--------------------------------------------------------------+\n'
        '|  Input: (1, 32, 48, 48) -- crop 3D de CT scan                |\n'
        '|                                                               |\n'
        '|  +----------------------------------------------+            |\n'
        '|  | BatchNorm3d(1)                               |            |\n'
        '|  +----------------------------------------------+            |\n'
        '|           |                                                   |\n'
        '|  +--------v-----------------------------------------+        |\n'
        '|  | LunaBlock 1: Conv3d(1->8) + ReLU                |        |\n'
        '|  |              Conv3d(8->8) + ReLU + MaxPool3d     |        |\n'
        '|  +--------------------------------------------------+        |\n'
        '|           |                                                   |\n'
        '|  +--------v-----------------------------------------+        |\n'
        '|  | LunaBlock 2: Conv3d(8->16) + ReLU               |        |\n'
        '|  |              Conv3d(16->16) + ReLU + MaxPool3d   |        |\n'
        '|  +--------------------------------------------------+        |\n'
        '|           |                                                   |\n'
        '|  +--------v-----------------------------------------+        |\n'
        '|  | LunaBlock 3: Conv3d(16->32) + ReLU              |        |\n'
        '|  |              Conv3d(32->32) + ReLU + MaxPool3d   |        |\n'
        '|  +--------------------------------------------------+        |\n'
        '|           |                                                   |\n'
        '|  +--------v-----------------------------------------+        |\n'
        '|  | LunaBlock 4: Conv3d(32->64) + ReLU              |        |\n'
        '|  |              Conv3d(64->64) + ReLU + MaxPool3d   |        |\n'
        '|  +--------------------------------------------------+        |\n'
        '|           |                                                   |\n'
        '|  +--------v-----------------------------------------+        |\n'
        '|  | Flatten -> Linear(1152, 2) -> Softmax            |        |\n'
        '|  +--------------------------------------------------+        |\n'
        '|                                                               |\n'
        '|  Output: (nao_nodulo_prob, nodulo_prob)                      |\n'
        '+--------------------------------------------------------------+\n'
        '```\n'
        '\n'
        '**Diferenciais da arquitetura:**\n'
        '- Inicializacao Kaiming para melhor convergencia\n'
        '- BatchNorm na entrada (tail) para normalizacao dos valores HU\n'
        '- 4 blocos convolucionais com aumento progressivo de canais (8 -> 16 -> 32 -> 64)\n'
    ))

    # =====================================================================
    # CELL 4 - Info do Modelo (Code)
    # =====================================================================
    cells.append(nbf.v4.new_code_cell(
        '# ============================================================\n'
        '# Carregando o Modelo e Info\n'
        '# ============================================================\n'
        'ckpt = torch.load("../checkpoints/luna_model_best.pt", map_location="cpu", weights_only=False)\n'
        'model = LunaModel()\n'
        'model.load_state_dict(ckpt["model_state_dict"])\n'
        'model.eval()\n'
        '\n'
        'total_params = sum(p.numel() for p in model.parameters())\n'
        'trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)\n'
        '\n'
        'print(f"Total de parametros: {total_params:,}")\n'
        'print(f"Parametros treinaveis: {trainable_params:,}")\n'
        'print(f"Melhor F1 registrado: {ckpt.get(\'f1\', \'N/A\'):.4f}")\n'
        'print(f"Epoca do melhor checkpoint: {ckpt.get(\'epoch\', \'N/A\')}")\n'
    ))

    # =====================================================================
    # CELL 5 - Section: Evolucao do Treinamento (Markdown)
    # =====================================================================
    cells.append(nbf.v4.new_markdown_cell(
        '---\n'
        '## 2. Evolucao do Treinamento -- Fase 1 & Fase 2\n'
        '\n'
        'O treinamento foi dividido em **duas fases**:\n'
        '\n'
        '| Fase | Epocas | Learning Rate | Augmentation |\n'
        '|:---:|:---:|:---:|:---|\n'
        '| **Fase 1** | 1-5 | 0.001 | Flip, Offset(0.1), Scale(0.2), Rotate, Noise(25) |\n'
        '| **Fase 2** | 6-10 | 0.0005 | Mesma + LR reduzido para fine-tuning |\n'
    ))

    # =====================================================================
    # CELL 6 - Graficos de Evolucao (Code)
    # =====================================================================
    cells.append(nbf.v4.new_code_cell(
        '# ============================================================\n'
        '# Evolucao do Treinamento (10 Epocas)\n'
        '# ============================================================\n'
        'history = ckpt.get("history", {})\n'
        '\n'
        'epochs = list(range(1, len(history.get("train_loss", [])) + 1))\n'
        'train_loss = history.get("train_loss", [])\n'
        'val_loss = history.get("val_loss", [])\n'
        'train_f1 = history.get("train_f1", [])\n'
        'val_f1 = history.get("val_f1", [])\n'
        'train_acc = history.get("train_acc", [])\n'
        'val_acc = history.get("val_acc", [])\n'
        '\n'
        'fig, axes = plt.subplots(1, 3, figsize=(20, 6))\n'
        '\n'
        '# --- Loss ---\n'
        'ax = axes[0]\n'
        'ax.plot(epochs, train_loss, "o-", color="#58a6ff", linewidth=2, markersize=6, label="Train")\n'
        'ax.plot(epochs, val_loss, "s-", color="#f0883e", linewidth=2, markersize=6, label="Val")\n'
        'ax.axvline(x=5.5, color="#8b949e", linestyle="--", alpha=0.7, label="Fase 1 -> 2")\n'
        'ax.set_title("Loss", fontsize=15, fontweight="bold", color="#58a6ff")\n'
        'ax.set_xlabel("Epoca")\n'
        'ax.set_ylabel("Loss")\n'
        'ax.legend(framealpha=0.3)\n'
        'ax.grid(True, alpha=0.3)\n'
        '\n'
        '# --- F1-Score ---\n'
        'ax = axes[1]\n'
        'ax.plot(epochs, train_f1, "o-", color="#3fb950", linewidth=2, markersize=6, label="Train")\n'
        'ax.plot(epochs, val_f1, "s-", color="#f85149", linewidth=2, markersize=6, label="Val")\n'
        'ax.axvline(x=5.5, color="#8b949e", linestyle="--", alpha=0.7, label="Fase 1 -> 2")\n'
        'if val_f1:\n'
        '    best_idx = np.argmax(val_f1)\n'
        '    ax.annotate(f"Best: {val_f1[best_idx]:.4f}",\n'
        '                xy=(epochs[best_idx], val_f1[best_idx]),\n'
        '                xytext=(epochs[best_idx]+0.5, val_f1[best_idx]+0.05),\n'
        '                arrowprops=dict(arrowstyle="->", color="#f85149", lw=1.5),\n'
        '                fontsize=11, color="#f85149", fontweight="bold")\n'
        'ax.set_title("F1-Score", fontsize=15, fontweight="bold", color="#3fb950")\n'
        'ax.set_xlabel("Epoca")\n'
        'ax.set_ylabel("F1")\n'
        'ax.legend(framealpha=0.3)\n'
        'ax.grid(True, alpha=0.3)\n'
        '\n'
        '# --- Accuracy ---\n'
        'ax = axes[2]\n'
        'ax.plot(epochs, train_acc, "o-", color="#bc8cff", linewidth=2, markersize=6, label="Train")\n'
        'ax.plot(epochs, val_acc, "s-", color="#f778ba", linewidth=2, markersize=6, label="Val")\n'
        'ax.axvline(x=5.5, color="#8b949e", linestyle="--", alpha=0.7, label="Fase 1 -> 2")\n'
        'ax.set_title("Accuracy", fontsize=15, fontweight="bold", color="#bc8cff")\n'
        'ax.set_xlabel("Epoca")\n'
        'ax.set_ylabel("Accuracy")\n'
        'ax.legend(framealpha=0.3)\n'
        'ax.grid(True, alpha=0.3)\n'
        '\n'
        'plt.suptitle("Evolucao do Treinamento -- 10 Epocas", fontsize=18, fontweight="bold", color="white", y=1.02)\n'
        'plt.tight_layout()\n'
        'plt.show()\n'
    ))

    # =====================================================================
    # CELL 7 - Section: Metricas Avancadas (Markdown)
    # =====================================================================
    cells.append(nbf.v4.new_markdown_cell(
        '---\n'
        '## 3. Analise de Metricas Avancadas\n'
        '\n'
        'Alem da acuracia simples, analisamos metricas que revelam o comportamento do modelo\n'
        'em diferentes **pontos de operacao** (thresholds).\n'
    ))

    # =====================================================================
    # CELL 8 - ROC & PR Curves (Code)
    # =====================================================================
    cells.append(nbf.v4.new_code_cell(
        '# ============================================================\n'
        '# Curvas ROC e Precision-Recall\n'
        '# ============================================================\n'
        'results = torch.load("../checkpoints/val_results_phase2.pth", weights_only=False)\n'
        'probs = results["probs"]\n'
        'labels = results["labels"]\n'
        '\n'
        '# Metricas\n'
        'precision, recall, thresholds = precision_recall_curve(labels, probs)\n'
        'f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)\n'
        'best_f1_idx = np.argmax(f1_scores)\n'
        'best_threshold = thresholds[best_f1_idx]\n'
        'best_f1 = f1_scores[best_f1_idx]\n'
        'best_prec = precision[best_f1_idx]\n'
        'best_rec = recall[best_f1_idx]\n'
        '\n'
        'fpr, tpr, _ = roc_curve(labels, probs)\n'
        'roc_auc = auc(fpr, tpr)\n'
        '\n'
        'fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 7))\n'
        '\n'
        '# --- ROC Curve ---\n'
        'ax1.fill_between(fpr, tpr, alpha=0.15, color="#58a6ff")\n'
        'ax1.plot(fpr, tpr, color="#58a6ff", lw=2.5, label=f"AUC = {roc_auc:.4f}")\n'
        'ax1.plot([0, 1], [0, 1], color="#8b949e", lw=1.5, linestyle="--", alpha=0.7, label="Random")\n'
        'ax1.set_xlabel("Taxa de Falsos Positivos (FPR)", fontsize=12)\n'
        'ax1.set_ylabel("Taxa de Verdadeiros Positivos (TPR)", fontsize=12)\n'
        'ax1.set_title("Curva ROC", fontsize=15, fontweight="bold", color="#58a6ff")\n'
        'ax1.legend(loc="lower right", fontsize=11, framealpha=0.3)\n'
        'ax1.grid(True, alpha=0.2)\n'
        '\n'
        '# --- PR Curve ---\n'
        'ax2.fill_between(recall, precision, alpha=0.15, color="#3fb950")\n'
        'ax2.plot(recall, precision, color="#3fb950", lw=2.5, label="PR Curve")\n'
        'ax2.scatter([best_rec], [best_prec], color="#f85149", s=120, zorder=5,\n'
        '            edgecolors="white", linewidths=1.5, label=f"Melhor F1={best_f1:.4f}")\n'
        'ax2.set_xlabel("Recall", fontsize=12)\n'
        'ax2.set_ylabel("Precisao", fontsize=12)\n'
        'ax2.set_title("Curva Precision-Recall", fontsize=15, fontweight="bold", color="#3fb950")\n'
        'ax2.legend(loc="upper right", fontsize=11, framealpha=0.3)\n'
        'ax2.grid(True, alpha=0.2)\n'
        '\n'
        '# --- Confusion Matrix ---\n'
        'preds = (probs >= best_threshold).astype(int)\n'
        'cm = confusion_matrix(labels, preds)\n'
        'im = ax3.imshow(cm, interpolation="nearest", cmap="Blues")\n'
        'for i in range(2):\n'
        '    for j in range(2):\n'
        '        text_color = "white" if cm[i, j] > cm.max()/2 else "#c9d1d9"\n'
        '        ax3.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",\n'
        '                 fontsize=16, fontweight="bold", color=text_color)\n'
        'ax3.set_xticks([0, 1]); ax3.set_xticklabels(["Nao-Nodulo", "Nodulo"])\n'
        'ax3.set_yticks([0, 1]); ax3.set_yticklabels(["Nao-Nodulo", "Nodulo"])\n'
        'ax3.set_xlabel("Predicao", fontsize=12)\n'
        'ax3.set_ylabel("Real", fontsize=12)\n'
        'ax3.set_title("Matriz de Confusao", fontsize=15, fontweight="bold", color="#bc8cff")\n'
        '\n'
        'plt.suptitle("Analise de Performance do Modelo (Validacao Completa)",\n'
        '             fontsize=18, fontweight="bold", color="white", y=1.02)\n'
        'plt.tight_layout()\n'
        'plt.show()\n'
        '\n'
        'sep = "=" * 60\n'
        'print(f"\\n{sep}")\n'
        'print(f"  METRICAS CONSOLIDADAS (threshold={best_threshold:.4f})")\n'
        'print(f"{sep}")\n'
        'print(f"  Melhor F1-Score:  {best_f1:.4f}")\n'
        'print(f"  Precisao:         {best_prec:.4f}")\n'
        'print(f"  Recall:           {best_rec:.4f}")\n'
        'print(f"  AUC-ROC:          {roc_auc:.4f}")\n'
        'print(f"{sep}")\n'
    ))

    # =====================================================================
    # CELL 9 - Section: KPIs Dashboard (Markdown)
    # =====================================================================
    cells.append(nbf.v4.new_markdown_cell(
        '---\n'
        '## 4. Dashboard de KPIs\n'
        '\n'
        'Os principais indicadores de performance do modelo em um visual de impacto.\n'
    ))

    # =====================================================================
    # CELL 10 - KPI Cards (Code)
    # =====================================================================
    cells.append(nbf.v4.new_code_cell(
        '# ============================================================\n'
        '# Dashboard Visual de KPIs\n'
        '# ============================================================\n'
        'fig, axes = plt.subplots(1, 4, figsize=(22, 5))\n'
        '\n'
        'kpis = [\n'
        '    ("AUC-ROC", f"{roc_auc:.4f}", "#58a6ff", "AUC"),\n'
        '    ("Melhor F1", f"{best_f1:.4f}", "#3fb950", "F1"),\n'
        '    ("Recall", f"{best_rec:.4f}", "#f0883e", "REC"),\n'
        '    ("Precisao", f"{best_prec:.4f}", "#bc8cff", "PRE"),\n'
        ']\n'
        '\n'
        'for ax, (title, value, color, icon) in zip(axes, kpis):\n'
        '    ax.set_xlim(0, 1)\n'
        '    ax.set_ylim(0, 1)\n'
        '    ax.set_aspect("equal")\n'
        '    ax.axis("off")\n'
        '    \n'
        '    # Card background\n'
        '    fancy = FancyBboxPatch((0.05, 0.05), 0.9, 0.9,\n'
        '                          boxstyle="round,pad=0.05",\n'
        '                          facecolor=color, alpha=0.15,\n'
        '                          edgecolor=color, linewidth=2)\n'
        '    ax.add_patch(fancy)\n'
        '    \n'
        '    # Icon label\n'
        '    ax.text(0.5, 0.78, icon, ha="center", va="center",\n'
        '            fontsize=22, fontweight="bold", color=color,\n'
        '            alpha=0.5, transform=ax.transAxes)\n'
        '    # Value\n'
        '    txt = ax.text(0.5, 0.48, value, ha="center", va="center",\n'
        '                  fontsize=28, fontweight="bold", color=color,\n'
        '                  transform=ax.transAxes)\n'
        '    txt.set_path_effects([path_effects.withStroke(linewidth=1, foreground="#0d1117")])\n'
        '    # Title\n'
        '    ax.text(0.5, 0.18, title, ha="center", va="center",\n'
        '            fontsize=14, color="#c9d1d9", transform=ax.transAxes)\n'
        '\n'
        'plt.suptitle("Indicadores-Chave de Performance",\n'
        '             fontsize=18, fontweight="bold", color="white", y=1.0)\n'
        'plt.tight_layout()\n'
        'plt.show()\n'
    ))

    # =====================================================================
    # CELL 11 - Section: Galeria Visual (Markdown)
    # =====================================================================
    cells.append(nbf.v4.new_markdown_cell(
        '---\n'
        '## 5. Galeria de Diagnostico Visual\n'
        '\n'
        'Visualizacoes de crops 3D reais onde o modelo:\n'
        '- **Acertou com confianca** (True Positives)\n'
        '- **Gerou alarmes falsos** (False Positives)\n'
        '- **Perdeu nodulos** (False Negatives)\n'
        '\n'
        'Cada imagem mostra o **slice central axial** do crop de 32x48x48 voxels.\n'
    ))

    # =====================================================================
    # CELL 12 - Visual Gallery (Code) -- USING VERIFIED UIDs
    # =====================================================================
    cells.append(nbf.v4.new_code_cell(
        '# ============================================================\n'
        '# Galeria de Diagnostico Visual\n'
        '# ============================================================\n'
        'def plot_gallery(examples, title, color, ax_row):\n'
        '    """Plota uma linha de crops com estilo premium."""\n'
        '    for idx, (ax, ex) in enumerate(zip(ax_row, examples)):\n'
        '        ct = get_ct(ex["uid"])\n'
        '        crop, _ = ct.extract_crop(ex["xyz"])\n'
        '        center_slice = crop[crop.shape[0]//2]\n'
        '        \n'
        '        ax.imshow(center_slice, cmap="gray", aspect="auto")\n'
        '        \n'
        '        # Borda colorida\n'
        '        for spine in ax.spines.values():\n'
        '            spine.set_edgecolor(color)\n'
        '            spine.set_linewidth(2.5)\n'
        '        \n'
        '        prob_text = f"P = {ex[\'prob\']:.4f}"\n'
        '        label_text = "NODULO" if ex["label"] == 1 else "NAO-NODULO"\n'
        '        ax.set_title(f"{prob_text}\\n{label_text}",\n'
        '                     fontsize=10, color=color, fontweight="bold")\n'
        '        ax.set_xticks([]); ax.set_yticks([])\n'
        '        get_ct.cache_clear()\n'
        '\n'
        '# Exemplos verificados (CTs disponiveis no disco)\n'
        'tps = [\n'
        '    {"uid": "1.3.6.1.4.1.14519.5.2.1.6279.6001.287966244644280690737019247886",\n'
        '     "xyz": (67.61451718, 85.02525992, -109.8084416), "prob": 1.0000, "label": 1},\n'
        '    {"uid": "1.3.6.1.4.1.14519.5.2.1.6279.6001.246758220302211646532176593724",\n'
        '     "xyz": (102.42, 65.3, 1562.32), "prob": 1.0000, "label": 1},\n'
        '    {"uid": "1.3.6.1.4.1.14519.5.2.1.6279.6001.986011151772797848993829243183",\n'
        '     "xyz": (45.13, 84.05, -211.55), "prob": 1.0000, "label": 1}\n'
        ']\n'
        'fps = [\n'
        '    {"uid": "1.3.6.1.4.1.14519.5.2.1.6279.6001.161002239822118346732951898613",\n'
        '     "xyz": (-121.828333494, 18.933116703, -280.116923295), "prob": 1.0000, "label": 0},\n'
        '    {"uid": "1.3.6.1.4.1.14519.5.2.1.6279.6001.324290109423920971676288828329",\n'
        '     "xyz": (103.725866116, -120.844854799, -197.90130835), "prob": 1.0000, "label": 0},\n'
        '    {"uid": "1.3.6.1.4.1.14519.5.2.1.6279.6001.199171741859530285887752432478",\n'
        '     "xyz": (115.364846498, 157.926101981, -565.796715546), "prob": 1.0000, "label": 0}\n'
        ']\n'
        'fns = [\n'
        '    {"uid": "1.3.6.1.4.1.14519.5.2.1.6279.6001.701514276942509393419164159551",\n'
        '     "xyz": (-85.63, 29.12, -162.49), "prob": 0.0056, "label": 1},\n'
        '    {"uid": "1.3.6.1.4.1.14519.5.2.1.6279.6001.674809958213117379592437424616",\n'
        '     "xyz": (-74.7508318473, 57.6862016943, -175.315789672), "prob": 0.1075, "label": 1},\n'
        '    {"uid": "1.3.6.1.4.1.14519.5.2.1.6279.6001.802595762867498341201607992711",\n'
        '     "xyz": (39.99, 34.27, -171.69), "prob": 0.1585, "label": 1}\n'
        ']\n'
        '\n'
        'fig, axes = plt.subplots(3, 3, figsize=(16, 16))\n'
        '\n'
        'row_labels = [\n'
        '    ("Nodulos Detectados (True Positives)", "#3fb950"),\n'
        '    ("Alarmes Falsos (False Positives)", "#f85149"),\n'
        '    ("Omissoes (False Negatives)", "#f0883e"),\n'
        ']\n'
        'examples_list = [tps, fps, fns]\n'
        '\n'
        'for row_idx, (title, color) in enumerate(row_labels):\n'
        '    plot_gallery(examples_list[row_idx], title, color, axes[row_idx])\n'
        '    axes[row_idx][0].set_ylabel(title, fontsize=12, fontweight="bold",\n'
        '                                color=color, labelpad=15)\n'
        '\n'
        'plt.suptitle("Galeria de Diagnostico Visual -- Exemplos Reais do Modelo",\n'
        '             fontsize=18, fontweight="bold", color="white", y=1.01)\n'
        'plt.tight_layout()\n'
        'plt.show()\n'
    ))

    # =====================================================================
    # CELL 13 - Section: Distribuicao de Probabilidades (Markdown)
    # =====================================================================
    cells.append(nbf.v4.new_markdown_cell(
        '---\n'
        '## 6. Distribuicao de Probabilidades\n'
        '\n'
        'Como o modelo distribui as probabilidades entre nodulos e nao-nodulos?\n'
        'Uma boa separacao indica que o modelo aprendeu a distinguir as duas classes.\n'
    ))

    # =====================================================================
    # CELL 14 - Probability Distribution (Code)
    # =====================================================================
    cells.append(nbf.v4.new_code_cell(
        '# ============================================================\n'
        '# Distribuicao de Probabilidades\n'
        '# ============================================================\n'
        'fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))\n'
        '\n'
        'pos_probs = probs[labels == 1]\n'
        'neg_probs = probs[labels == 0]\n'
        '\n'
        '# Histograma sobreposto\n'
        'ax1.hist(neg_probs, bins=80, alpha=0.6, color="#58a6ff", label=f"Nao-Nodulo (n={len(neg_probs):,})",\n'
        '         edgecolor="none", density=True)\n'
        'ax1.hist(pos_probs, bins=40, alpha=0.7, color="#f85149", label=f"Nodulo (n={len(pos_probs):,})",\n'
        '         edgecolor="none", density=True)\n'
        'ax1.axvline(x=best_threshold, color="#3fb950", linestyle="--", lw=2,\n'
        '             label=f"Threshold otimo = {best_threshold:.4f}")\n'
        'ax1.set_xlabel("Probabilidade de Nodulo", fontsize=12)\n'
        'ax1.set_ylabel("Densidade", fontsize=12)\n'
        'ax1.set_title("Distribuicao de Probabilidades", fontsize=15,\n'
        '               fontweight="bold", color="white")\n'
        'ax1.legend(fontsize=11, framealpha=0.3)\n'
        'ax1.grid(True, alpha=0.2)\n'
        '\n'
        '# Box plot\n'
        'bp = ax2.boxplot([neg_probs, pos_probs],\n'
        '                  labels=["Nao-Nodulo", "Nodulo"],\n'
        '                  patch_artist=True, widths=0.5,\n'
        '                  medianprops=dict(color="white", linewidth=2))\n'
        'bp["boxes"][0].set_facecolor("#58a6ff")\n'
        'bp["boxes"][0].set_alpha(0.5)\n'
        'bp["boxes"][1].set_facecolor("#f85149")\n'
        'bp["boxes"][1].set_alpha(0.5)\n'
        'ax2.set_ylabel("Probabilidade de Nodulo", fontsize=12)\n'
        'ax2.set_title("Comparacao Box Plot", fontsize=15,\n'
        '               fontweight="bold", color="white")\n'
        'ax2.grid(True, alpha=0.2)\n'
        '\n'
        'plt.suptitle("Analise da Distribuicao de Probabilidades",\n'
        '             fontsize=18, fontweight="bold", color="white", y=1.02)\n'
        'plt.tight_layout()\n'
        'plt.show()\n'
    ))

    # =====================================================================
    # CELL 15 - Section: Desafios e Proximos Passos (Markdown)
    # =====================================================================
    cells.append(nbf.v4.new_markdown_cell(
        '---\n'
        '## 7. Desafios e Estrategias Futuras\n'
        '\n'
        '<div style="background: #161b22; padding: 25px; border-radius: 12px; '
        'border-left: 4px solid #58a6ff; margin: 15px 0;">\n'
        '\n'
        '### Diferenciais Tecnicos Implementados\n'
        '\n'
        '1. **Robust Cache Recovery** -- Sistema de carregamento resiliente que detecta e '
        'regenera automaticamente arquivos de cache corrompidos\n'
        '2. **Balanced Training** -- Amostragem balanceada (`ratio_int=2`) para compensar o '
        'desbalanceamento extremo (apenas 0.25% dos candidatos sao nodulos)\n'
        '3. **3D Data Augmentation** -- Flip, offset, scale, rotacao e ruido aplicados via '
        'transformacao afim (`affine_grid` + `grid_sample`)\n'
        '4. **Two-Phase Training** -- Learning rate decay entre Fase 1 e Fase 2 para fine-tuning\n'
        '\n'
        '</div>\n'
        '\n'
        '<div style="background: #161b22; padding: 25px; border-radius: 12px; '
        'border-left: 4px solid #f0883e; margin: 15px 0;">\n'
        '\n'
        '### Analise dos Erros\n'
        '\n'
        '| Tipo de Erro | Causa Provavel | Estrategia de Melhoria |\n'
        '|:---|:---|:---|\n'
        '| **Falsos Positivos** | Vasos sanguineos em corte transversal | Aumentar profundidade do crop; augmentations mais agressivas |\n'
        '| **Falsos Negativos** | Nodulos muito pequenos ou colados na parede | Focal Loss para exemplos dificeis |\n'
        '| **Precision baixa** | Desbalanceamento extremo do dataset | Ajustar `ratio_int`; tecnicas de hard negative mining |\n'
        '\n'
        '</div>\n'
        '\n'
        '<div style="background: #161b22; padding: 25px; border-radius: 12px; '
        'border-left: 4px solid #3fb950; margin: 15px 0;">\n'
        '\n'
        '### Proximos Passos\n'
        '\n'
        '- [ ] Implementar **Focal Loss** para focar em exemplos dificeis\n'
        '- [ ] Experimentar **ratio_int=3 ou 4** no balanceamento\n'
        '- [ ] Testar **learning rate schedulers** (cosine annealing)\n'
        '- [ ] Adicionar **dropout** para regularizacao\n'
        '- [ ] Avaliar **segmentacao** para localizar nodulos no CT completo\n'
        '\n'
        '</div>\n'
        '\n'
        '---\n'
        '\n'
        '<div style="text-align: center; padding: 20px; color: #8b949e;">\n'
        '\n'
        '**Projeto**: Lung Cancer Detection -- LUNA16<br>\n'
        '**Stack**: Python - PyTorch - SimpleITK - scikit-learn - Matplotlib<br>\n'
        '**Dataset**: <a href="https://luna16.grand-challenge.org/" style="color: #58a6ff;">LUNA16 Grand Challenge</a>\n'
        '\n'
        '</div>\n'
    ))

    nb.cells = cells

    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                "notebooks", "11_resultado_final.ipynb")
    with open(output_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"[OK] Notebook criado: {output_path}")


if __name__ == "__main__":
    build_notebook()
