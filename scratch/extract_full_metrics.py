import sys
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Adicionar src ao path
sys.path.insert(0, str(Path(os.getcwd()) / "src"))

from model import LunaModel
from training import LunaTrainingDataset, validate

def get_full_metrics(ckpt_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    model = LunaModel().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    val_ds = LunaTrainingDataset(val_stride=10, is_val=True)
    val_loader = DataLoader(val_ds, batch_size=32, num_workers=0)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    
    print(f"\nValidating {ckpt_path}...")
    metrics = validate(model, val_loader, loss_fn, device)
    return metrics

if __name__ == "__main__":
    best_metrics = get_full_metrics('checkpoints/luna_model_best.pt')
    print(f"Best Metrics (Epoch 4): {best_metrics}")
    
    last_metrics = get_full_metrics('checkpoints/luna_model_last.pt')
    print(f"Last Metrics (Epoch 5): {last_metrics}")
