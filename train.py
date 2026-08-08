"""Training entrypoint.

    python train.py --config configs/nafnet_w32.yaml
    python train.py --config configs/nafnet_w32.yaml --epochs 5 --tag smoke

Everything is config-driven; CLI flags only override for quick experiments and
the override is recorded in the checkpoint.
"""
import argparse
import json
import math
from pathlib import Path

import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from src.data.dataset import (ExternalHRPairs, KLAPairs, MixedDataset,
                              SyntheticPairs)
from src.data.degradation import DegradationConfig
from src.data.patterns import PatternPairs
from src.data.splits import split_by_source, summarise
from src.engine.trainer import Trainer
from src.losses.composite import CompositeLoss
from src.models.registry import build_model, count_params
from src.utils.seed import seed_worker, set_seed


def build_scheduler(opt, epochs, warmup, base_lr, min_lr):
    def fn(ep):
        if ep < warmup:
            return (ep + 1) / max(warmup, 1)
        t = (ep - warmup) / max(epochs - warmup, 1)
        cos = 0.5 * (1 + math.cos(math.pi * t))
        return (min_lr + (base_lr - min_lr) * cos) / base_lr
    return LambdaLR(opt, fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--epoch-len", type=int, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    if args.epochs:
        cfg["optim"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["data"]["batch_size"] = args.batch_size
    if args.epoch_len:
        cfg["data"]["epoch_len"] = args.epoch_len
    if args.tag:
        cfg["out_dir"] = f"{cfg['out_dir']}_{args.tag}"
    # Record overrides, but never an absolute path: the resolved config is
    # written into every checkpoint, and an absolute path there leaks the
    # machine's filesystem layout into a published artefact.
    _ov = {k: v for k, v in vars(args).items() if v is not None}
    if "config" in _ov:
        try:
            _ov["config"] = str(Path(_ov["config"]).resolve()
                                .relative_to(Path.cwd()).as_posix())
        except ValueError:
            _ov["config"] = Path(_ov["config"]).name
    cfg["cli_overrides"] = _ov

    set_seed(cfg["seed"], deterministic=cfg.get("deterministic", True))
    dev = args.device if torch.cuda.is_available() else "cpu"

    # ---------------- data: split BY SOURCE, never randomly ---------------
    tr_idx, va_idx = split_by_source(val_frac=cfg["data"]["val_frac"],
                                     seed=cfg["seed"])
    print("[split]", json.dumps(summarise(tr_idx, va_idx)))

    root = Path(cfg["data"]["root"])
    primary = KLAPairs(root, tr_idx, lr_patch=cfg["data"]["lr_patch"],
                       augment=True)

    dcfg = DegradationConfig(**cfg["degradation"])
    hr_patch = cfg["data"]["lr_patch"] * 2

    # ---- secondary pool: external photographs + procedural patterns -----
    pools, names = [], []
    ext_bundle = Path(cfg["data"]["root"]) / "processed/external_hr.npy"
    if ext_bundle.exists():
        # bundled path: no PNG decode, ~18x faster per item
        ds = ExternalHRPairs(ext_bundle, dcfg, hr_patch=hr_patch)
        pools.append(ds)
        names.append(f"external-bundle:{len(ds)}")
    elif cfg["data"].get("external_glob"):
        files = sorted(Path().glob(cfg["data"]["external_glob"]))
        if files:
            pools.append(SyntheticPairs(files, dcfg, hr_patch=hr_patch))
            names.append(f"external-png:{len(files)} (slow — run precrop_patches.py)")
        else:
            print(f"[warn] external_glob matched nothing: "
                  f"{cfg['data']['external_glob']}")

    if cfg["data"].get("pattern_frac", 0) > 0:
        # Procedural periodic structure. The KLA corpus is photographs and
        # contains almost none, which is why the model loses to bicubic on a
        # checkerboard. This manufactures the missing content.
        pools.append(PatternPairs(dcfg, hr_patch=hr_patch,
                                  length=cfg["data"]["epoch_len"]))
        names.append(f"patterns:{cfg['data']['pattern_frac']:.0%}")

    secondary = None
    if pools:
        if len(pools) == 1:
            secondary = pools[0]
        else:
            # weight patterns at pattern_frac of the SECONDARY pool
            pf = cfg["data"].get("pattern_frac", 0.0)
            ext_ratio = 1.0 - pf / max(1e-9, (1.0 - cfg["data"]["external_ratio"]))
            secondary = MixedDataset(pools[0], pools[1],
                                     ratio=max(0.0, min(1.0, ext_ratio)),
                                     epoch_len=cfg["data"]["epoch_len"])
        print(f"[data] secondary pool: {' + '.join(names)}")

    train_ds = MixedDataset(primary, secondary,
                            ratio=cfg["data"]["external_ratio"],
                            epoch_len=cfg["data"]["epoch_len"])
    val_ds = KLAPairs(root, va_idx, augment=False, full=True)

    g = torch.Generator()
    g.manual_seed(cfg["seed"])
    train_ld = DataLoader(train_ds, batch_size=cfg["data"]["batch_size"],
                          shuffle=True, num_workers=cfg["data"]["num_workers"],
                          pin_memory=True, drop_last=True,
                          persistent_workers=cfg["data"]["num_workers"] > 0,
                          worker_init_fn=seed_worker, generator=g)
    val_ld = DataLoader(val_ds, batch_size=8, shuffle=False,
                        num_workers=max(cfg["data"]["num_workers"] // 2, 0),
                        pin_memory=True)

    # ---------------- model / loss / optim --------------------------------
    model = build_model(cfg["model"]["arch"],
                        in_ch=cfg["model"]["in_ch"],
                        scale=cfg["model"]["scale"],
                        use_log_channel=cfg["model"]["use_log_channel"])
    n_par = count_params(model)
    print(f"[model] {cfg['model']['arch']}  params={n_par/1e6:.2f}M")
    cfg["param_count"] = n_par

    loss_fn = CompositeLoss(**cfg["loss"])
    opt = AdamW(model.parameters(), lr=cfg["optim"]["lr"],
                weight_decay=cfg["optim"]["weight_decay"],
                betas=tuple(cfg["optim"]["betas"]))
    sched = build_scheduler(opt, cfg["optim"]["epochs"],
                            cfg["optim"]["warmup_epochs"],
                            cfg["optim"]["lr"], cfg["optim"]["min_lr"])

    trainer = Trainer(model, loss_fn, opt, sched, cfg, device=dev,
                      out_dir=cfg["out_dir"],
                      ema_decay=cfg["optim"]["ema_decay"])
    Path(cfg["out_dir"]).mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(cfg, open(Path(cfg["out_dir"]) / "config.yaml", "w", encoding="utf-8"),
                   sort_keys=False)

    best = trainer.fit(train_ld, val_ld, epochs=cfg["optim"]["epochs"],
                       clip=cfg["optim"]["grad_clip"],
                       accum=cfg["data"]["accum_steps"],
                       val_every=cfg["eval"]["val_every"])
    print("[best]", json.dumps(best, indent=2))
    json.dump(best, open(Path(cfg["out_dir"]) / "best_metrics.json", "w"),
              indent=2)


if __name__ == "__main__":
    main()
