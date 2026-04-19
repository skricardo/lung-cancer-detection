import sys
import os
from pathlib import Path
import torch
import numpy as np

# Adicionar src ao path
sys.path.insert(0, str(Path(os.getcwd()) / "src"))

from luna_data import load_candidates
from model import LunaModel
from inference import run_inference

def generate_validation_data():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = 'checkpoints/luna_model_best.pt'
    output_path = 'checkpoints/val_results_phase2.pth'
    
    # Carregar modelo
    checkpoint = torch.load(ckpt_path, map_location=device)
    model = LunaModel().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Pegar todos os candidatos de validacao
    print("Loading candidates...")
    all_cands = load_candidates()
    val_cands = all_cands[::10]
    
    print(f"Running full inference on {len(val_cands)} validation candidates...")
    results = run_inference(val_cands, model, device, batch_size=128, print_every=100)
    
    # Salvar apenas o necessário para os gráficos
    data_to_save = {
        'probs': results['probs'],
        'labels': results['labels']
    }
    
    torch.save(data_to_save, output_path)
    print(f"Validation results saved to {output_path}")

if __name__ == "__main__":
    generate_validation_data()
