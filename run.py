#!/usr/bin/env python
"""KLA PS01 — AI-Based Restoration of Degraded Images.
Main submission evaluation entry script: run.py

Usage:
    python run.py <input-dir> <output-dir>
    python run.py --input_dir <input-dir> --output_dir <output-dir>

Guarantees:
  - Reads all .npy files (and supported image files) from <input-dir>
  - Creates <output-dir> if it does not already exist
  - Generates one restored .npy file for every input file with identical filename
  - Outputs are 2D grayscale float32 arrays of shape (2H, 2W)
  - Output values are strictly in [0.0, 1.0] with zero NaN / Inf values
  - Restored images have the correct 2x target resolution
  - 100% self-contained: runs offline on NVIDIA GPU (or CPU fallback) without
    requiring internet, API keys, or manual configuration
"""
import argparse
import glob
import os
import queue
import sys
import threading
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# MODEL ARCHITECTURES (Self-contained, inline definition)
# =====================================================================

class LayerNorm2d(nn.Module):
    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class SimpleGate(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFBlock(nn.Module):
    def __init__(self, c, dw_expand=2, ffn_expand=2):
        super().__init__()
        dw = c * dw_expand
        self.conv1 = nn.Conv2d(c, dw, 1)
        self.conv2 = nn.Conv2d(dw, dw, 3, padding=1, groups=dw)
        self.conv3 = nn.Conv2d(dw // 2, c, 1)
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(dw // 2, dw // 2, 1))
        self.sg = SimpleGate()
        ffn = c * ffn_expand
        self.conv4 = nn.Conv2d(c, ffn, 1)
        self.conv5 = nn.Conv2d(ffn // 2, c, 1)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.drop1 = nn.Identity()
        self.drop2 = nn.Identity()
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, inp):
        x = self.conv2(self.conv1(self.norm1(inp)))
        x = self.sg(x)
        x = self.conv3(x * self.sca(x))
        y = inp + x * self.beta
        x = self.conv5(self.sg(self.conv4(self.norm2(y))))
        return y + x * self.gamma


class NAFNetSR(nn.Module):
    def __init__(self, in_ch=1, width=32, middle_blk_num=8,
                 enc_blk_nums=(2, 2, 4), dec_blk_nums=(2, 2, 2), scale=2,
                 use_log_channel=True):
        super().__init__()
        self.scale = scale
        self.use_log_channel = use_log_channel
        stem_in = in_ch * (2 if use_log_channel else 1)
        self.intro = nn.Conv2d(stem_in, width, 3, padding=1)
        self.encoders, self.downs = nn.ModuleList(), nn.ModuleList()
        self.decoders, self.ups = nn.ModuleList(), nn.ModuleList()
        chan = width
        for n in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(n)]))
            self.downs.append(nn.Conv2d(chan, chan * 2, 2, stride=2))
            chan *= 2
        self.middle = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])
        for n in dec_blk_nums:
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False),
                                          nn.PixelShuffle(2)))
            chan //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(n)]))
        self.sr_head = nn.Conv2d(width, in_ch * scale * scale, 3, padding=1)
        self.shuffle = nn.PixelShuffle(scale)
        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        anchor = F.interpolate(inp, scale_factor=self.scale, mode="bicubic",
                               align_corners=False)
        _, _, h, w = inp.shape
        ph = (self.padder_size - h % self.padder_size) % self.padder_size
        pw = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(inp, (0, pw, 0, ph), mode="reflect") if (ph or pw) else inp
        if self.use_log_channel:
            x = torch.cat([x, torch.log(x.clamp_min(1e-3))], dim=1)
        x = self.intro(x)
        skips = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x)
            skips.append(x)
            x = down(x)
        x = self.middle(x)
        for dec, up, skip in zip(self.decoders, self.ups, skips[::-1]):
            x = dec(up(x) + skip)
        x = self.shuffle(self.sr_head(x))
        return anchor + x[..., : h * self.scale, : w * self.scale]


class SAFMNSR(nn.Module):
    """Fast variant for SAFMN architecture."""
    class _CCM(nn.Module):
        def __init__(self, dim, growth=2.0):
            super().__init__()
            h = int(dim * growth)
            self.net = nn.Sequential(nn.Conv2d(dim, h, 3, padding=1), nn.GELU(),
                                     nn.Conv2d(h, dim, 1))

        def forward(self, x):
            return self.net(x)

    class _SAFM(nn.Module):
        def __init__(self, dim, n_levels=4):
            super().__init__()
            self.n_levels = n_levels
            c = dim // n_levels
            self.mfr = nn.ModuleList([nn.Conv2d(c, c, 3, padding=1, groups=c)
                                      for _ in range(n_levels)])
            self.aggr = nn.Conv2d(dim, dim, 1)
            self.act = nn.GELU()

        def forward(self, x):
            h, w = x.shape[-2:]
            xc = x.chunk(self.n_levels, dim=1)
            out = []
            for i in range(self.n_levels):
                if i > 0:
                    p = 2 ** i
                    s = F.adaptive_max_pool2d(xc[i], (max(h // p, 1), max(w // p, 1)))
                    s = F.interpolate(self.mfr[i](s), size=(h, w), mode="nearest")
                else:
                    s = self.mfr[i](xc[i])
                out.append(s)
            return self.act(self.aggr(torch.cat(out, 1))) * x

    class _Blk(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.n1, self.n2 = LayerNorm2d(dim), LayerNorm2d(dim)
            self.safm = SAFMNSR._SAFM(dim)
            self.ccm = SAFMNSR._CCM(dim)

        def forward(self, x):
            x = x + self.safm(self.n1(x))
            return x + self.ccm(self.n2(x))

    def __init__(self, in_ch=1, dim=36, n_blocks=8, scale=2,
                 use_log_channel=True):
        super().__init__()
        self.scale = scale
        self.use_log_channel = use_log_channel
        stem_in = in_ch * (2 if use_log_channel else 1)
        self.to_feat = nn.Conv2d(stem_in, dim, 3, padding=1)
        self.feats = nn.Sequential(*[SAFMNSR._Blk(dim) for _ in range(n_blocks)])
        self.sr_head = nn.Conv2d(dim, in_ch * scale * scale, 3, padding=1)
        self.shuffle = nn.PixelShuffle(scale)

    def forward(self, inp):
        anchor = F.interpolate(inp, scale_factor=self.scale, mode="bicubic",
                               align_corners=False)
        x = inp
        if self.use_log_channel:
            x = torch.cat([x, torch.log(x.clamp_min(1e-3))], dim=1)
        x = self.feats(self.to_feat(x))
        return anchor + self.shuffle(self.sr_head(x))


ARCHS = {
    "nafnet_w48": lambda **k: NAFNetSR(width=48, middle_blk_num=12,
                                        enc_blk_nums=(2, 2, 4),
                                        dec_blk_nums=(2, 2, 2), **k),
    "nafnet_w32": lambda **k: NAFNetSR(width=32, middle_blk_num=8,
                                        enc_blk_nums=(2, 2, 4),
                                        dec_blk_nums=(2, 2, 2), **k),
    "nafnet_w16": lambda **k: NAFNetSR(width=16, middle_blk_num=4,
                                        enc_blk_nums=(1, 1, 2),
                                        dec_blk_nums=(1, 1, 1), **k),
    "safmn": lambda **k: SAFMNSR(dim=36, n_blocks=8, **k),
}


# =====================================================================
# I/O UTILITIES
# =====================================================================

def read_image(path):
    """Read degraded input image. Preserves unclipped float32 values."""
    if path.lower().endswith(".npy"):
        arr = np.load(path)
        if arr.ndim == 3:
            arr = arr[..., 0] if arr.shape[-1] <= 4 else arr[0]
        return arr.astype(np.float32, copy=False)
    # Lazy import for non-numpy formats if present
    import cv2
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Unable to read image at {path}")
    if img.ndim == 3:
        img = img[..., 0]
    return img.astype(np.float32) / (65535.0 if img.dtype == np.uint16 else 255.0)


def sanitize_and_save_npy(path, arr):
    """Ensure array is 2D float32 strictly in [0, 1] without NaN/Inf, and save."""
    # Ensure 2D
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr.squeeze(-1)
    elif arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr.squeeze(0)
    # Sanitize NaNs and Infs
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    # Strict clamping to [0, 1]
    arr = np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)
    # Atomic save
    np.save(path, arr)


def header_shape(path):
    """Extract (H, W) without full file decode."""
    if path.lower().endswith(".npy"):
        with open(path, "rb") as fh:
            ver = np.lib.format.read_magic(fh)
            rd = (np.lib.format.read_array_header_1_0 if ver[0] == 1
                  else np.lib.format.read_array_header_2_0)
            return tuple(rd(fh)[0][:2])
    import cv2
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Cannot read header for {path}")
    return tuple(img.shape[:2])


def scan_shapes(files):
    """Scan shapes concurrently."""
    import concurrent.futures as cf
    shapes, buckets, errors = {}, {}, []
    with cf.ThreadPoolExecutor(max_workers=min(16, (os.cpu_count() or 4) * 2)) as ex:
        def _read_shp(p):
            try:
                return (p, header_shape(p), None)
            except Exception as e:
                return (p, None, e)
        for p, shp, err in ex.map(_read_shp, files):
            if err is not None:
                errors.append(f"{os.path.basename(p)}: {err}")
            else:
                shapes[p] = shp
                buckets.setdefault(shp, []).append(p)
    return shapes, buckets, errors


def find_weights(explicit_path=None):
    """Find trained model checkpoint path automatically."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        explicit_path,
        os.path.join(here, "models", "model_fp16.pt"),
        os.path.join(here, "weights", "model_fp16.pt"),
        os.path.join(here, "models", "best_model.pt"),
        os.path.join(here, "weights", "best_model.pt"),
    ]
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    # Search models/ and weights/ for any .pt
    for search_dir in [os.path.join(here, "models"), os.path.join(here, "weights"), here]:
        if os.path.isdir(search_dir):
            pts = glob.glob(os.path.join(search_dir, "*.pt"))
            if pts:
                return sorted(pts)[0]
    raise FileNotFoundError("Model weights checkpoint not found in models/ or weights/.")


class ImageBucket(torch.utils.data.Dataset):
    def __init__(self, paths, shapes):
        self.paths = paths
        self.shapes = shapes

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        try:
            arr = read_image(p)
        except Exception:
            arr = np.zeros(self.shapes[p], dtype=np.float32)
        return torch.from_numpy(arr)[None], i


# =====================================================================
# MAIN RUNNER
# =====================================================================

def main():
    t_start = time.perf_counter()

    parser = argparse.ArgumentParser(
        description="KLA PS01: AI-Based Restoration of Degraded Images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Positional arguments (primary interface)
    parser.add_argument("input_pos", nargs="?", default=None,
                        help="Input directory containing degraded .npy files")
    parser.add_argument("output_pos", nargs="?", default=None,
                        help="Output directory to save restored .npy files")

    # Optional named arguments
    parser.add_argument("-i", "--input_dir", "--input-dir", dest="input_flag",
                        default=None, help="Input directory")
    parser.add_argument("-o", "--output_dir", "--output-dir", dest="output_flag",
                        default=None, help="Output directory")
    parser.add_argument("-w", "--weights", "--model", default=None,
                        help="Path to model weights checkpoint (.pt)")
    parser.add_argument("--batch_size", "--batch-size", type=int, default=16,
                        help="Batch size for inference")
    parser.add_argument("--device", default=None,
                        help="Execution device ('cuda' or 'cpu')")
    parser.add_argument("--num_workers", "--num-workers", type=int, default=None,
                        help="DataLoader worker count")
    parser.add_argument("--cudnn_benchmark", "--cudnn-benchmark", action="store_true",
                        help="Enable cuDNN benchmark autotuning")

    args = parser.parse_args()

    input_dir = args.input_flag or args.input_pos
    output_dir = args.output_flag or args.output_pos

    if not input_dir or not output_dir:
        parser.print_help()
        print("\n[ERROR] Both input directory and output directory must be specified.", file=sys.stderr)
        print("Example: python run.py <input-dir> <output-dir>", file=sys.stderr)
        sys.exit(1)

    # 1. Output directory creation
    os.makedirs(output_dir, exist_ok=True)

    # 2. Discover all .npy files (and other image formats if present)
    files = []
    for ext in ("npy", "NPY", "png", "PNG", "tif", "tiff", "jpg", "jpeg"):
        files.extend(glob.glob(os.path.join(input_dir, f"*.{ext}")))
    files = sorted(set(files))

    if not files:
        print(f"[ERROR] No valid input files found in directory: {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Discovered {len(files)} input files in '{input_dir}'")

    # 3. Locate model weights
    weights_path = find_weights(args.weights)
    print(f"[INFO] Using model weights: {weights_path}")

    # 4. Device configuration
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = dev.startswith("cuda")
    if use_cuda:
        torch.backends.cudnn.benchmark = args.cudnn_benchmark
    print(f"[INFO] Execution device: {dev} (CUDA available: {torch.cuda.is_available()})")

    # 5. Load model checkpoint
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=True, mmap=True)
    arch = ckpt.get("arch", "nafnet_w48")
    in_ch = ckpt.get("in_ch", 1)
    scale = ckpt.get("scale", 2)
    use_log_channel = ckpt.get("use_log_channel", True)

    if arch not in ARCHS:
        arch = "nafnet_w48"
    model = ARCHS[arch](in_ch=in_ch, scale=scale, use_log_channel=use_log_channel)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(dev).eval()
    if use_cuda:
        model = model.to(memory_format=torch.channels_last).half()

    t_init = time.perf_counter()
    print(f"[INFO] Architecture: {arch} | Scale: {scale}x | Init time: {t_init - t_start:.2f}s")

    # 6. Group inputs by resolution bucket
    shapes, buckets, scan_errors = scan_shapes(files)
    for err in scan_errors:
        print(f"[WARN] Header error: {err}", file=sys.stderr)
    if not buckets:
        print("[ERROR] No readable images found.", file=sys.stderr)
        sys.exit(1)

    # 7. Asynchronous writer queue
    wq = queue.Queue(maxsize=256)
    n_written = [0]
    write_errors = []

    def writer_worker():
        while True:
            item = wq.get()
            if item is None:
                wq.task_done()
                return
            out_path, arr_2d = item
            try:
                sanitize_and_save_npy(out_path, arr_2d)
                n_written[0] += 1
            except Exception as e:
                write_errors.append(f"{os.path.basename(out_path)}: {e}")
            finally:
                wq.task_done()

    num_writers = min(4, max(2, os.cpu_count() or 2))
    writer_threads = [threading.Thread(target=writer_worker, daemon=True) for _ in range(num_writers)]
    for wt in writer_threads:
        wt.start()

    # 8. Worker configuration
    if args.num_workers is None:
        # Avoid process spawn overhead on Windows/macOS
        args.num_workers = 0 if sys.platform in ("win32", "darwin") else 4

    # 9. Main inference loop
    total_processed = 0
    with torch.inference_mode():
        for shp, bucket_paths in buckets.items():
            loader = torch.utils.data.DataLoader(
                ImageBucket(bucket_paths, shapes),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=use_cuda,
                persistent_workers=False,
            )

            for batch, idxs in loader:
                x = batch.to(dev, non_blocking=True)
                n_real = x.shape[0]
                if use_cuda:
                    x = x.half().to(memory_format=torch.channels_last)

                y = model(x)[:n_real]
                # Clamp on GPU first, then transfer to CPU
                y = y.clamp_(0.0, 1.0).float()
                out_np = y.cpu().numpy()

                idx_list = idxs.tolist() if torch.is_tensor(idxs) else idxs
                for k, gi in enumerate(idx_list):
                    src_file = bucket_paths[gi]
                    base_name = os.path.splitext(os.path.basename(src_file))[0]
                    # Exactly matching .npy output filename
                    out_filename = base_name + ".npy"
                    out_path = os.path.join(output_dir, out_filename)
                    # 2D array (H, W)
                    arr_2d = out_np[k, 0]
                    wq.put((out_path, arr_2d))

                total_processed += len(idx_list)
                if total_processed % 100 == 0:
                    print(f"[INFO] Processed {total_processed}/{len(files)} images...", flush=True)

    # 10. Wait for writer completion
    wq.join()
    for _ in writer_threads:
        wq.put(None)
    for wt in writer_threads:
        wt.join()

    t_end = time.perf_counter()
    total_time = t_end - t_start
    throughput = len(files) / max(total_time, 1e-9)

    print(f"[DONE] Successfully generated {n_written[0]}/{len(files)} restored .npy files in '{output_dir}'")
    print(f"[METRICS] Total time: {total_time:.2f}s | Throughput: {throughput:.1f} images/s")

    if write_errors:
        print(f"[WARN] {len(write_errors)} write errors occurred: {write_errors[:3]}", file=sys.stderr)


if __name__ == "__main__":
    main()
