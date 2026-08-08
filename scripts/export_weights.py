"""Export a training checkpoint into the slim fp16 file inference.py loads.

The training checkpoint carries the optimizer, scheduler, EMA, full config and
history — useful for reproduction, useless at inference and expensive to load.
Startup time is scored, so we ship only what the forward pass needs, in fp16:
roughly a quarter the size and correspondingly faster to load.

    python scripts/export_weights.py \
        --ckpt experiments/runs/nafnet_w32/best.pt \
        --out weights/model_fp16.pt
"""
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="weights/model_fp16.pt")
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--fp32", action="store_true")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    sd = ck.get("ema_state_dict")
    which = "EMA"
    if sd is None or args.no_ema:
        sd = ck["model_state_dict"]
        which = "raw"

    dtype = torch.float32 if args.fp32 else torch.float16
    slim = {k: (v.to(dtype) if v.dtype.is_floating_point else v)
            for k, v in sd.items()}

    payload = {
        "state_dict": slim,
        "arch": cfg["model"]["arch"],
        "in_ch": cfg["model"]["in_ch"],
        "scale": cfg["model"]["scale"],
        "use_log_channel": cfg["model"]["use_log_channel"],
        # Provenance, so a shipped weight file can always be traced back.
        # Everything here must be a PLAIN python type: inference.py loads with
        # weights_only=True, which rejects arbitrary classes. torch.__version__
        # is a TorchVersion object, not a str, and silently breaks the load.
        "trained_epoch": int(ck.get("epoch", -1)),
        "git_commit": str(ck.get("git_commit", "")),
        "val_metrics": {k: float(v) for k, v in
                        (ck.get("metrics") or {}).items()
                        if isinstance(v, (int, float))},
        "seed": int(ck.get("seed", -1)),
        "torch_version": str(ck.get("torch_version", "")),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)

    src_mb = Path(args.ckpt).stat().st_size / 1e6
    dst_mb = out.stat().st_size / 1e6
    print(f"exported {which} weights as {dtype}")
    print(f"  {args.ckpt}  {src_mb:.1f} MB")
    print(f"  -> {out}     {dst_mb:.1f} MB   ({src_mb/max(dst_mb,1e-9):.1f}x smaller)")

    # verify the export actually loads under inference.py's contract
    ck2 = torch.load(out, map_location="cpu", weights_only=True, mmap=True)
    assert ck2["arch"] == payload["arch"]
    print("verified: loads with weights_only=True, mmap=True")


if __name__ == "__main__":
    main()
