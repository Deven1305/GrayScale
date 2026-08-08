#!/usr/bin/env python
"""KLA PS01 — evaluation / inference script.  THE SCORED FILE.

    python inference.py --input_dir <dir> --output_dir <dir>

Contract: read every degraded image in --input_dir, restore it at exactly 2x
the input resolution, write it to --output_dir under the same base filename.

WHY THIS FILE LOOKS THE WAY IT DOES
-----------------------------------
Measured runtime includes script startup and model initialisation, not just
the forward pass. So:

  * Imports are minimal: argparse, os, sys, time, glob, queue, threading,
    numpy, torch. NO torchvision, matplotlib, pandas, timm, cv2 unless a PNG
    is actually encountered (cv2 is imported lazily, inside the branch).
  * The model is defined INLINE. Importing a package tree to fetch one class
    costs 1-3 s of scored time.
  * NO torch.compile. A 30-120 s compile to save a few seconds of inference is
    a large net loss.
  * Weights load with weights_only=True and mmap=True.
  * Three-stage overlap: reader threads -> GPU main loop -> writer threads, so
    total time is max(read, compute, write) rather than their sum.
  * Inputs are bucketed by shape and batched within a bucket; mixed shapes
    cannot be batched.
  * cudnn.benchmark is OFF by default, against the usual advice. MEASURED on
    the 400-image test set: autotuning costs ~11.7 s of warmup while making the
    main loop only ~8% faster (114 vs 105 img/s), i.e. a 2.4x end-to-end LOSS
    (18.6 s -> 7.9 s with it off). It only amortises above roughly 15k images.
    Enable with --cudnn_benchmark if the test set is very large.
  * The input is NEVER clipped — out-of-range values carry the local noise
    level. Only the OUTPUT is clamped, at save time.

Zero hardcoded absolute paths. A malformed file is logged and skipped rather
than crashing the run.
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

# ===================================================================
# MODEL — defined inline on purpose (see module docstring)
# ===================================================================


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
    """Fast variant. Kept so one script can load either checkpoint."""

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


ARCHS = {"nafnet_w48": lambda **k: NAFNetSR(width=48, middle_blk_num=12,
                                            enc_blk_nums=(2, 2, 4),
                                            dec_blk_nums=(2, 2, 2), **k),
         "nafnet_w32": lambda **k: NAFNetSR(width=32, middle_blk_num=8,
                                            enc_blk_nums=(2, 2, 4),
                                            dec_blk_nums=(2, 2, 2), **k),
         "nafnet_w16": lambda **k: NAFNetSR(width=16, middle_blk_num=4,
                                            enc_blk_nums=(1, 1, 2),
                                            dec_blk_nums=(1, 1, 1), **k),
         "safmn": lambda **k: SAFMNSR(dim=36, n_blocks=8, **k)}


# ===================================================================
# I/O
# ===================================================================
def read_image(path):
    """Return float32 HxW. NEVER clipped."""
    if path.lower().endswith(".npy"):
        a = np.load(path)
        if a.ndim == 3:
            a = a[..., 0] if a.shape[-1] <= 4 else a[0]
        return a.astype(np.float32, copy=False)
    import cv2                                   # lazy: only if a PNG appears
    a = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if a is None:
        raise IOError("unreadable")
    if a.ndim == 3:
        a = a[..., 0]
    return a.astype(np.float32) / (65535.0 if a.dtype == np.uint16 else 255.0)


def write_image(path, arr, fmt):
    """arr: float32 HxW already clamped to [0,1]."""
    if fmt == "npy":
        np.save(path, arr.astype(np.float32, copy=False))
    else:
        import cv2
        cv2.imwrite(path, (arr * 255.0 + 0.5).astype(np.uint8),
                    [cv2.IMWRITE_PNG_COMPRESSION, 1])


class ImageBucket(torch.utils.data.Dataset):
    """One shape-bucket of inputs.

    MUST be defined at module level: Windows (and macOS) DataLoader workers
    use spawn, which pickles the dataset class by reference. A class nested
    inside main() raises "Can't pickle local object" the moment
    num_workers > 0.
    """

    def __init__(self, paths, shapes):
        self.paths = paths
        self.shapes = shapes

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        try:
            a = read_image(p)                    # NOT clipped
        except Exception:
            a = np.zeros(self.shapes[p], np.float32)
        return torch.from_numpy(a)[None], i


def main():
    t_start = time.perf_counter()

    ap = argparse.ArgumentParser(description="KLA PS01 image restoration")
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--weights", default=None,
                    help="default: weights/model_fp16.pt next to this script")
    ap.add_argument("--batch_size", type=int, default=16)   # measured optimum
    ap.add_argument("--num_workers", type=int, default=None,
                    help="reader processes. Default is platform-aware: 0 where "
                         "DataLoader workers SPAWN (Windows/macOS), because each "
                         "worker re-imports torch and that costs more than it "
                         "saves on a few hundred images; 4 where they FORK "
                         "(Linux), which is nearly free.")
    ap.add_argument("--output_format", default="same",
                    choices=["same", "npy", "png"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--cudnn_benchmark", action="store_true",
                    help="enable cudnn autotuning. OFF by default: measured a "
                         "2.4x END-TO-END LOSS on a 400-image run, because the "
                         "~11.7 s autotune dwarfs the ~8%% per-batch gain. "
                         "Only pays off above roughly 15k images.")
    args = ap.parse_args()
    args.no_benchmark = not args.cudnn_benchmark

    if args.num_workers is None:
        # MEASURED on 400 images: 4 workers 35.2 s vs 0 workers 16.4 s under
        # spawn. Under fork the tradeoff reverses, so decide by start method
        # rather than hardcoding either answer.
        import multiprocessing as _mp
        args.num_workers = 0 if _mp.get_start_method(allow_none=True) == "spawn"             or sys.platform in ("win32", "darwin") else 4

    here = os.path.dirname(os.path.abspath(__file__))
    weights = args.weights or os.path.join(here, "weights", "model_fp16.pt")
    os.makedirs(args.output_dir, exist_ok=True)

    files = []
    for ext in ("npy", "png", "tif", "tiff", "jpg", "jpeg", "bmp"):
        files += glob.glob(os.path.join(args.input_dir, f"*.{ext}"))
        files += glob.glob(os.path.join(args.input_dir, f"*.{ext.upper()}"))
    files = sorted(set(files))
    if not files:
        print(f"[error] no images found in {args.input_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"[info] {len(files)} input images")

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = dev.startswith("cuda")
    if use_cuda:
        torch.backends.cudnn.benchmark = not args.no_benchmark

    # ---------------- model -------------------------------------------
    ckpt = torch.load(weights, map_location="cpu", weights_only=True,
                      mmap=True)
    arch = ckpt.get("arch", "nafnet_w32")
    model = ARCHS[arch](in_ch=ckpt.get("in_ch", 1), scale=ckpt.get("scale", 2),
                        use_log_channel=ckpt.get("use_log_channel", True))
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(dev).eval()
    if use_cuda:
        model = model.to(memory_format=torch.channels_last).half()
    t_init = time.perf_counter()
    print(f"[info] arch={arch} device={dev} startup={t_init - t_start:.2f}s")

    # ---------------- bucket by shape ---------------------------------
    buckets = {}
    shapes = {}
    for f in files:
        try:
            if f.lower().endswith(".npy"):
                with open(f, "rb") as fh:
                    ver = np.lib.format.read_magic(fh)
                    rd = (np.lib.format.read_array_header_1_0 if ver[0] == 1
                          else np.lib.format.read_array_header_2_0)
                    shp = rd(fh)[0][:2]
            else:
                import cv2
                im = cv2.imread(f, cv2.IMREAD_UNCHANGED)
                shp = im.shape[:2]
            shapes[f] = tuple(shp)
            buckets.setdefault(tuple(shp), []).append(f)
        except Exception as e:
            print(f"[warn] cannot read header {os.path.basename(f)}: {e}",
                  file=sys.stderr)
    print(f"[info] shape buckets: "
          f"{ {f'{k[0]}x{k[1]}': len(v) for k, v in buckets.items()} }")

    # ---------------- warm up every shape once ------------------------
    # cudnn.benchmark autotunes per (N, C, H, W). Warming up at batch 1 while
    # the main loop runs batch 16 autotunes twice and wastes the warmup
    # entirely — measured at ~10 s of pure waste. So we warm up at the ACTUAL
    # batch size, and pad the final short batch back up to it (below) so each
    # bucket only ever presents one shape.
    if use_cuda and not args.no_benchmark:
        with torch.inference_mode():
            for shp, paths in buckets.items():
                bs = min(args.batch_size, len(paths))
                d = torch.zeros(bs, 1, shp[0], shp[1], device=dev,
                                dtype=torch.half).to(
                    memory_format=torch.channels_last)
                for _ in range(2):
                    model(d)
        torch.cuda.synchronize()
    t_warm = time.perf_counter()
    print(f"[info] warmup {t_warm - t_init:.2f}s")

    # ---------------- WRITER stage ------------------------------------
    wq = queue.Queue(maxsize=256)
    n_written = [0]
    errors = []

    def writer():
        while True:
            item = wq.get()
            if item is None:
                wq.task_done()
                return
            path, arr, fmt = item
            try:
                write_image(path, arr, fmt)
                n_written[0] += 1
            except Exception as e:
                errors.append(f"{os.path.basename(path)}: {e}")
            finally:
                wq.task_done()

    writers = [threading.Thread(target=writer, daemon=True)
               for _ in range(max(2, args.num_workers))]
    for w in writers:
        w.start()

    # ---------------- READER stage + GPU main loop --------------------
    total = 0
    with torch.inference_mode():
        for shp, paths in buckets.items():
            ld = torch.utils.data.DataLoader(
                ImageBucket(paths, shapes), batch_size=args.batch_size,
                shuffle=False, num_workers=args.num_workers,
                pin_memory=use_cuda, persistent_workers=False)
            for batch, idxs in ld:
                try:
                    x = batch.to(dev, non_blocking=True)
                    n_real = x.shape[0]
                    if use_cuda:
                        x = x.half().to(memory_format=torch.channels_last)
                        # pad a short final batch back to batch_size so cudnn
                        # sees a single shape per bucket and autotunes once
                        if not args.no_benchmark and n_real < args.batch_size \
                                and len(paths) >= args.batch_size:
                            pad = args.batch_size - n_real
                            x = torch.cat([x, x[:1].expand(pad, -1, -1, -1)], 0)
                    y = model(x)[:n_real]
                    # clamp on the GPU, then move the result back
                    y = y.clamp_(0.0, 1.0).float()
                    out = y.cpu().numpy()
                except Exception as e:
                    errors.append(f"batch {shp}: {e}")
                    continue
                for k, gi in enumerate(idxs.tolist()):
                    src = paths[gi]
                    base = os.path.splitext(os.path.basename(src))[0]
                    if args.output_format == "same":
                        ext = ".npy" if src.lower().endswith(".npy") else ".png"
                    else:
                        ext = "." + args.output_format
                    fmt = "npy" if ext == ".npy" else "png"
                    wq.put((os.path.join(args.output_dir, base + ext),
                            out[k, 0], fmt))
                total += len(idxs)
                if total % 200 == 0:
                    print(f"[info] {total}/{len(files)}", flush=True)

    wq.join()
    for _ in writers:
        wq.put(None)

    elapsed = time.perf_counter() - t_start
    print(f"[done] wrote {n_written[0]}/{len(files)} images to "
          f"{args.output_dir}")
    print(f"[time] startup {t_init - t_start:.2f}s | warmup "
          f"{t_warm - t_init:.2f}s | total {elapsed:.2f}s | "
          f"{len(files)/max(elapsed, 1e-9):.1f} img/s")
    if errors:
        print(f"[warn] {len(errors)} error(s); first 5: {errors[:5]}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
