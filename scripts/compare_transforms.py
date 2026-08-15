import os
import glob
import numpy as np
import torch
import sys

# Add root to sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.models.nafnet import NAFNetSR

def main():
    # Find sample_test images
    sample_dir = os.path.join(ROOT, "sample_test")
    npy_files = glob.glob(os.path.join(sample_dir, "*.npy"))
    if not npy_files:
        print(f"No .npy files found in {sample_dir}")
        return

    # Initialize models
    model_log = NAFNetSR(in_ch=1, input_transform="log", use_log_channel=True).eval()
    model_asinh = NAFNetSR(in_ch=1, input_transform="asinh", use_log_channel=True).eval()

    print("=== Input Transform Comparison on sample_test ===")
    print(f"Found {len(npy_files)} files.\n")

    for f in npy_files[:5]: # just do the first few
        arr = np.load(f)
        if arr.ndim == 3:
            arr = arr[..., 0]
        
        arr = arr.astype(np.float32)
        x = torch.from_numpy(arr)[None, None, :, :]
        
        # Calculate base statistics
        neg_pct = (arr < 0).mean() * 100
        clamp_pct = (arr < 1e-3).mean() * 100
        
        # Run through stems
        with torch.no_grad():
            out_log = model_log._stem(x)
            out_asinh = model_asinh._stem(x)
            
        log_channel_log = out_log[0, 1].numpy()
        log_channel_asinh = out_asinh[0, 1].numpy()
        
        print(f"File: {os.path.basename(f)}")
        print(f"  Shape: {arr.shape}")
        print(f"  Inputs < 0:      {neg_pct:>5.1f}%")
        print(f"  Inputs < 0.001:  {clamp_pct:>5.1f}%")
        print(f"  -- Log Transform (log(x.clamp(1e-3))) --")
        print(f"     Min:  {log_channel_log.min():.4f}")
        print(f"     Max:  {log_channel_log.max():.4f}")
        print(f"     Mean: {log_channel_log.mean():.4f}")
        print(f"     Std:  {log_channel_log.std():.4f}")
        print(f"  -- Asinh Transform (asinh(x/0.1)) --")
        print(f"     Min:  {log_channel_asinh.min():.4f}")
        print(f"     Max:  {log_channel_asinh.max():.4f}")
        print(f"     Mean: {log_channel_asinh.mean():.4f}")
        print(f"     Std:  {log_channel_asinh.std():.4f}")
        print("-" * 60)

if __name__ == "__main__":
    main()
