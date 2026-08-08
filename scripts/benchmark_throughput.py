"""Stage-wise latency breakdown for the scored inference pipeline.

KLA measures END-TO-END wall clock including script startup and model
initialisation, so a forward-pass benchmark is the wrong measurement. This
reports every stage separately, and runs the real inference.py as a subprocess
so startup is captured honestly.

    python scripts/benchmark_throughput.py --input_dir data/Test_NoisyLR/NoisyLR

Cold vs warm cache is reported separately: the first pass reads from disk, the
second hits the OS page cache and looks artificially fast.
"""
import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def time_stages(input_dir, weights, device="cuda", batch=16, n=64):
    """Per-stage timing inside one process (read / H2D / compute / D2H / write)."""
    from src.models.registry import build_model

    files = sorted(Path(input_dir).glob("*.npy"))[:n]
    if not files:
        return {}

    ck = torch.load(weights, map_location="cpu", weights_only=True, mmap=True)
    model = build_model(ck["arch"], in_ch=ck["in_ch"], scale=ck["scale"],
                        use_log_channel=ck["use_log_channel"])
    model.load_state_dict({k: v.float() for k, v in ck["state_dict"].items()})
    model = model.to(device).eval()
    if device.startswith("cuda"):
        model = model.half().to(memory_format=torch.channels_last)

    t = {}
    t0 = time.perf_counter()
    arrs = [np.load(f) for f in files]
    t["read"] = time.perf_counter() - t0

    x = torch.from_numpy(np.stack(arrs))[:, None]
    if device.startswith("cuda"):
        # warm up before timing
        with torch.inference_mode():
            for _ in range(3):
                model(x[:batch].to(device).half().to(
                    memory_format=torch.channels_last))
        torch.cuda.synchronize()

    h2d = comp = d2h = 0.0
    outs = []
    with torch.inference_mode():
        for i in range(0, len(x), batch):
            b = x[i:i + batch]
            torch.cuda.synchronize() if device.startswith("cuda") else None
            s = time.perf_counter()
            g = b.to(device, non_blocking=True)
            if device.startswith("cuda"):
                g = g.half().to(memory_format=torch.channels_last)
                torch.cuda.synchronize()
            h2d += time.perf_counter() - s

            s = time.perf_counter()
            y = model(g)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            comp += time.perf_counter() - s

            s = time.perf_counter()
            y = y.clamp_(0, 1).float().cpu().numpy()
            d2h += time.perf_counter() - s
            outs.append(y)
    t["h2d"] = h2d
    t["compute"] = comp
    t["d2h"] = d2h

    with tempfile.TemporaryDirectory() as td:
        s = time.perf_counter()
        k = 0
        for blk in outs:
            for j in range(blk.shape[0]):
                np.save(Path(td) / f"{k:06d}.npy", blk[j, 0])
                k += 1
        t["write"] = time.perf_counter() - s

    t["n_images"] = len(files)
    return t


def time_end_to_end(input_dir, out_dir, extra=()):
    """Run the real script as a subprocess — this is what KLA measures."""
    cmd = [sys.executable, str(ROOT / "inference.py"),
           "--input_dir", str(input_dir), "--output_dir", str(out_dir), *extra]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    startup = warm = None
    for line in r.stdout.splitlines():
        if line.startswith("[time]"):
            for part in line.split("|"):
                if "startup" in part:
                    startup = float(part.split()[-1].rstrip("s"))
                if "warmup" in part:
                    warm = float(part.split()[-1].rstrip("s"))
    return {"total_s": dt, "startup_s": startup, "warmup_s": warm,
            "returncode": r.returncode, "stderr": r.stderr[-400:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="data/Test_NoisyLR/NoisyLR")
    ap.add_argument("--weights", default="weights/model_fp16.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="experiments/throughput.json")
    args = ap.parse_args()

    dev = args.device if torch.cuda.is_available() else "cpu"
    n_files = len(list(Path(args.input_dir).glob("*.npy")))
    print(f"[bench] {n_files} images, device={dev}")

    res = {"device": dev, "n_images": n_files}
    if torch.cuda.is_available():
        # record the VRAM class, not the model name — the artefact is published
        res["gpu_vram_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 1)

    # END-TO-END FIRST, on a clean GPU. Running the in-process stage breakdown
    # first leaves a model and a CUDA context resident in THIS process, which
    # then competes with the subprocess for the GPU and inflated the end-to-end
    # figure roughly 3x. Measure the number KLA actually measures before this
    # process touches the GPU at all.
    print("\n--- end-to-end (subprocess - what KLA measures) ---")
    with tempfile.TemporaryDirectory() as td:
        cold = time_end_to_end(args.input_dir, td)
        print(f"  cold cache: {cold['total_s']:.2f}s "
              f"(startup {cold['startup_s']}s, warmup {cold['warmup_s']}s)")
    runs = []
    for _ in range(3):
        with tempfile.TemporaryDirectory() as td:
            runs.append(time_end_to_end(args.input_dir, td))
    warm = min(runs, key=lambda r: r["total_s"])          # best of 3
    warm["all_warm_runs_s"] = [round(r["total_s"], 2) for r in runs]
    print(f"  warm cache: {warm['total_s']:.2f}s   (best of 3: "
          f"{warm['all_warm_runs_s']})")

    print("\n--- stage breakdown (in-process, 64 images) ---")
    st = time_stages(args.input_dir, args.weights, dev)
    if st:
        n = st.pop("n_images")
        tot = sum(st.values())
        for k, v in st.items():
            print(f"  {k:>8}: {v*1000:8.1f} ms  ({v/tot*100:5.1f}%)  "
                  f"{v/n*1000:6.2f} ms/img")
        print(f"  {'TOTAL':>8}: {tot*1000:8.1f} ms for {n} images")
        res["stages"] = st
        res["stage_images"] = n
    res["end_to_end_cold"] = cold
    res["end_to_end_warm"] = warm
    if n_files:
        res["ms_per_image_warm"] = warm["total_s"] / n_files * 1000
        print(f"\n  {warm['total_s']/n_files*1000:.1f} ms/image end-to-end "
              f"(warm), {n_files/warm['total_s']:.1f} img/s")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
