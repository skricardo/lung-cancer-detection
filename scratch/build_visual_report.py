import nbformat as nbf
import os

def build_notebook():
    nb = nbf.v4.new_notebook()

    # --- Title & Intro ---
    nb.cells.append(nbf.v4.new_markdown_cell("""# Relatório de Performance Visual - Detecção de Nódulos LUNA16
Este relatório apresenta uma análise profunda dos resultados obtidos na **Fase 2** de treinamento. Além das métricas tradicionais, exploramos as curvas de precisão-recall e uma galeria visual de diagnósticos do modelo."""))

    # --- Setup ---
    nb.cells.append(nbf.v4.new_code_cell("""import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

# Adiciona o diretório src ao path para carregar os módulos locais
sys.path.insert(0, os.path.abspath("../src"))

from luna_data import get_ct, load_candidates
from model import LunaModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")"""))

    # --- Section: Metrics Dashboard ---
    nb.cells.append(nbf.v4.new_markdown_cell("""## 📊 Dashboard de Métricas Consolidadas
Carregamos os resultados pré-calculados de toda a validação para análise estatística."""))

    nb.cells.append(nbf.v4.new_code_cell("""# Carregando resultados da inferência completa na validação
results = torch.load("../checkpoints/val_results_phase2.pth")
probs = results['probs']
labels = results['labels']

# Calculando F1 ótimo
precision, recall, thresholds = precision_recall_curve(labels, probs)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
best_f1_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_f1_idx]

print(f"Melhor F1-Score: {f1_scores[best_f1_idx]:.4f}")
print(f"Threshold Ótimo: {best_threshold:.4f}")
print(f"Precisão: {precision[best_f1_idx]:.4f}")
print(f"Recall: {recall[best_f1_idx]:.4f}")"""))

    # --- Section: Advanced Curves ---
    nb.cells.append(nbf.v4.new_markdown_cell("""## 📈 Curvas de Análise de Erro
Diferente da acurácia simples, estas curvas mostram como o modelo se comporta em diferentes pontos de operação."""))

    nb.cells.append(nbf.v4.new_code_cell("""fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# ROC Curve
fpr, tpr, _ = roc_curve(labels, probs)
roc_auc = auc(fpr, tpr)
ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax1.set_xlabel('Taxa de Falsos Positivos')
ax1.set_ylabel('Taxa de Verdadeiros Positivos')
ax1.set_title('Curva ROC')
ax1.legend(loc="lower right")
ax1.grid(alpha=0.3)

# PR Curve
ax2.plot(recall, precision, color='blue', lw=2, label='Precision-Recall')
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precisão')
ax2.set_title('Curva Precision-Recall')
ax2.legend(loc="lower left")
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()"""))

    # --- Section: Visual Gallery ---
    nb.cells.append(nbf.v4.new_markdown_cell("""## 🔍 Galeria de Diagnóstico Visual
Vamos visualizar exemplos reais onde o modelo acertou com confiança, onde se enganou e o que ele deixou passar."""))

    nb.cells.append(nbf.v4.new_code_cell("""def plot_gallery(examples, title):
    fig, axes = plt.subplots(1, len(examples), figsize=(18, 5))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    for ax, ex in zip(axes, examples):
        ct = get_ct(ex['uid'])
        crop, _ = ct.extract_crop(ex['xyz'])
        
        # Mostra o slice central do crop 3D
        center_slice = crop[crop.shape[0]//2]
        
        ax.imshow(center_slice, cmap='gray')
        ax.set_title(f"Prob: {ex['prob']:.4f}\\nLabel: {ex['label']}", color='green' if ex['label']==1 else 'red')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# Exemplos selecionados durante a pesquisa
tps = [
    {'uid': '1.3.6.1.4.1.14519.5.2.1.6279.6001.222728285517924035616611311054', 'xyz': (-30.08, 159.22, -491.43), 'prob': 0.9997, 'label': 1},
    {'uid': '1.3.6.1.4.1.14519.5.2.1.6279.6001.258909838151327155605703273347', 'xyz': (121.28, -135.22, -554.43), 'prob': 0.9996, 'label': 1},
    {'uid': '1.3.6.1.4.1.14519.5.2.1.6279.6001.332353136611383827464303350176', 'xyz': (76.24, 76.22, -263.43), 'prob': 0.9995, 'label': 1}
]

fps = [
    {'uid': '1.3.6.1.4.1.14519.5.2.1.6279.6001.597621455246747279140411130614', 'xyz': (-134.11, 25.1, -114.71), 'prob': 0.9959, 'label': 0},
    {'uid': '1.3.6.1.4.1.14519.5.2.1.6279.6001.766881513533845439335142582269', 'xyz': (-109.04, 98.05, -199.88), 'prob': 0.9944, 'label': 0},
    {'uid': '1.3.6.1.4.1.14519.5.2.1.6279.6001.969607480572818589276327766720', 'xyz': (-96.94, -154.10, -168.08), 'prob': 0.9927, 'label': 0}
]

fns = [
    {'uid': '1.3.6.1.4.1.14519.5.2.1.6279.6001.725023183844147505748475581290', 'xyz': (-66.70, 217.87, -487.02), 'prob': 0.0056, 'label': 1},
    {'uid': '1.3.6.1.4.1.14519.5.2.1.6279.6001.280072876841890439628529365478', 'xyz': (108.07, -128.60, -256.69), 'prob': 0.1075, 'label': 1},
    {'uid': '1.3.6.1.4.1.14519.5.2.1.6279.6001.390513733720659266816639651938', 'xyz': (-21.69, 5.52, -42.02), 'prob': 0.1585, 'label': 1}
]

plot_gallery(tps, "Sucessos: Nódulos Detectados com Confiança")
plot_gallery(fps, "Alarmes Falsos (False Positives)")
plot_gallery(fns, "Omissões: Nódulos que o Modelo Perdeu")"""))

    # --- Section: Strategic Roadmap ---
    nb.cells.append(nbf.v4.new_markdown_cell("""## 🗺️ Roteiro Estratégico para Próximas Fases
Com base na análise das imagens acima:
1. **Falsos Positivos**: Muitos se parecem com vasos sanguíneos em corte transversal. Podemos tentar aumentar a profundidade do crop ou usar augmentations mais agressivas de rotação.
2. **Falsos Negativos**: Geralmente são nódulos muito pequenos ou colados na parede torácica. 
3. **Próxima Ação**: Implementar **Focal Loss** para focar o treino nos exemplos mais difíceis."""))

    # Write the notebook
    with open('notebooks/10_visual_report.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print("Notebook created successfully!")

if __name__ == "__main__":
    build_notebook()
