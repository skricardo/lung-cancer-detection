import torch
import os
import sys

# Adicionar src/ ao path para que o torch.load encontre as classes se necessário
sys.path.insert(0, "src")

ckpt_path = r'd:\Deep-learning - Deteccao de nodulo\checkpoints\luna_model_last.pt'

if os.path.exists(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    epoch = checkpoint.get('epoch', 'N/A')
    f1 = checkpoint.get('f1', 'N/A')
    history = checkpoint.get('history', {})
    
    print(f"Checkpoint Epoch: {epoch}")
    print(f"Checkpoint F1: {f1}")
    print("\nHistory:")
    for key, values in history.items():
        print(f"  {key}: {values}")
else:
    print("Checkpoint not found.")
