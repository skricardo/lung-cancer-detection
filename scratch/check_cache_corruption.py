import torch
import os
from pathlib import Path
from tqdm import tqdm

cache_dir = Path(r'd:\Deep-learning - Deteccao de nodulo\data\luna\cache')

corrupted_files = []
files = list(cache_dir.glob('*.pt'))

print(f"Checking {len(files)} files...")

for f in tqdm(files):
    try:
        torch.load(f, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"Error in {f}: {e}")
        corrupted_files.append(f)

print(f"\nTotal corrupted files: {len(corrupted_files)}")
for f in corrupted_files:
    print(f)
