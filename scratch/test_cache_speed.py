import sys
import time
from pathlib import Path
import torch

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from training import LunaTrainingDataset

def test_cache_performance():
    print("Iniciando teste de performance do Dataset...")
    ds = LunaTrainingDataset(val_stride=10, is_val=False, ratio_int=1)
    
    start_time = time.time()
    # Carregar as primeiras 5 amostras
    for i in range(5):
        crop_t, label_t, series_uid, _ = ds[i]
        print(f"  Amostra {i} carregada. Shape: {crop_t.shape}, UID: {series_uid[:10]}...")
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTempo total para 5 amostras: {total_time:.4f} segundos")
    
    if total_time < 1.0:
        print(">>> SUCESSO: Cache está ativo e voando! <<<")
    else:
        print(">>> ALERTA: Ainda parece estar carregando do disco original. <<<")

if __name__ == "__main__":
    test_cache_performance()
