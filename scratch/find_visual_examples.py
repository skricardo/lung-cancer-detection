import sys
import os
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader

# Adicionar src ao path
sys.path.insert(0, str(Path(os.getcwd()) / "src"))

from luna_data import load_candidates, get_ct
from model import LunaModel
from inference import run_inference

def find_examples():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = 'checkpoints/luna_model_best.pt'
    
    # Carregar modelo
    checkpoint = torch.load(ckpt_path, map_location=device)
    model = LunaModel().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Pegar todos os candidatos de validacao
    all_cands = load_candidates()
    val_cands = all_cands[::10]
    
    # Priorizar nódulos para garantir que temos exemplos
    pos_cands = [c for c in val_cands if c.is_nodule]
    neg_cands = [c for c in val_cands if not c.is_nodule]
    
    # Pegar todos os pos e uma amostra de neg
    test_cands = pos_cands + neg_cands[:5000]
    
    print(f"Running inference on {len(test_cands)} candidates...")
    results = run_inference(test_cands, model, device, batch_size=64, print_every=0)
    
    probs = results['probs']
    labels = results['labels']
    uids = results['series_uids']
    xyzs = results['center_xyzs']
    
    # True Positives (Nódulos com maior prob)
    tp_indices = [i for i, (p, l) in enumerate(zip(probs, labels)) if l == 1]
    tp_indices = sorted(tp_indices, key=lambda i: probs[i], reverse=True)
    
    # False Positives (Não-nódulos com maior prob)
    fp_indices = [i for i, (p, l) in enumerate(zip(probs, labels)) if l == 0]
    fp_indices = sorted(fp_indices, key=lambda i: probs[i], reverse=True)
    
    # False Negatives (Nódulos com menor prob)
    fn_indices = [i for i, (p, l) in enumerate(zip(probs, labels)) if l == 1]
    fn_indices = sorted(fn_indices, key=lambda i: probs[i])

    print("\nTOP TRUE POSITIVES:")
    for i in tp_indices[:5]:
        print(f"Prob: {probs[i]:.4f} | UID: {uids[i]} | XYZ: {xyzs[i]}")

    print("\nTOP FALSE POSITIVES:")
    for i in fp_indices[:5]:
        print(f"Prob: {probs[i]:.4f} | UID: {uids[i]} | XYZ: {xyzs[i]}")

    print("\nTOP FALSE NEGATIVES (Misses):")
    for i in fn_indices[:5]:
        print(f"Prob: {probs[i]:.4f} | UID: {uids[i]} | XYZ: {xyzs[i]}")

if __name__ == "__main__":
    find_examples()
