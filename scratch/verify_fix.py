import torch
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(os.getcwd()) / "src"))

from luna_data import LunaDataset, get_cache_path
from training import LunaTrainingDataset

def test_corruption_recovery():
    datasets = [
        LunaDataset(val_stride=10, is_val=True),
        LunaTrainingDataset(val_stride=10, is_val=True)
    ]
    
    for ds in datasets:
        print(f"\nTesting {ds.__class__.__name__}...")
        # Pick a candidate and its cache path
        candidate = ds.pos_list[0] if hasattr(ds, 'pos_list') else ds.candidates[0]
        cache_path = get_cache_path(candidate.series_uid, candidate.center_xyz)
        
        print(f"Testing recovery for: {cache_path}")
        
        # 1. Ensure file exists or create it
        if not cache_path.exists():
            print("Creating initial cache file...")
            _ = ds[0] 
        
        # 2. Corrupt the file
        print("Corrupting the file manually...")
        with open(cache_path, 'wb') as f:
            f.write(b"NOT A PYTORCH FILE AT ALL")
        
        # 3. Try access via Dataset
        print(f"Accessing corrupted file via {ds.__class__.__name__}...")
        try:
            crop_t, label_t, series_uid, center_irc = ds[0]
            print(f"Success! {ds.__class__.__name__} recovered from corrupted file.")
            assert crop_t is not None
            assert cache_path.exists()
            # Check if it was saved correctly now
            torch.load(cache_path, weights_only=False)
            print("Verified: New cache file is valid.")
        except Exception as e:
            print(f"Failed to recover for {ds.__class__.__name__}: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

if __name__ == "__main__":
    test_corruption_recovery()
