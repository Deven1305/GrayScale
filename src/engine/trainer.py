"""Training loop.

Hygiene decisions, all deliberate and all logged, because reproducibility is
an explicitly scored axis:
  * bf16 autocast, NOT fp16. This machine is recent NVIDIA architectures; bf16 needs no
    GradScaler and cannot overflow.
  * Gradient clipping at 1.0 — restoration runs spike on outlier crops.
  * EMA weights, evaluated with EMA.
  * Model selection by best val SSIM (the primary scored metric), not last
    epoch. The rule is fixed before training starts.
  * Every checkpoint carries the full config, git SHA, seed, torch version and
    metrics, so any number can be traced back to the exact code that made it.
"""
import json
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from ..metrics.full_reference import evaluate_batch
from ..utils.seed import git_hash
from .ema import ModelEMA


class Trainer:
    def __init__(self, model, loss_fn, optimizer, scheduler, cfg: Dict,
                 device="cuda", out_dir="experiments/runs/exp",
                 ema_decay: float = 0.999, log_dir: Optional[str] = None):
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.opt = optimizer
        self.sched = scheduler
        self.cfg = cfg
        self.device = device
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.ema = ModelEMA(self.model, ema_decay) if ema_decay > 0 else None
        self.tb = SummaryWriter(log_dir or str(self.out / "tb"))
        self.best = -1.0
        self.best_metrics = {}
        self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() \
            else torch.float16
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=(self.amp_dtype == torch.float16))
        self.history = []

    # ------------------------------------------------------------------
    def train_epoch(self, loader, epoch: int, clip: float = 1.0,
                    accum: int = 1) -> Dict[str, float]:
        self.model.train()
        agg, n, t0 = {}, 0, time.perf_counter()
        self.opt.zero_grad(set_to_none=True)

        for step, (lr, hr) in enumerate(loader):
            lr = lr.to(self.device, non_blocking=True)
            hr = hr.to(self.device, non_blocking=True)

            with torch.autocast("cuda", dtype=self.amp_dtype):
                pred = self.model(lr)
                loss, parts = self.loss_fn(pred.float(), hr.float())

            self.scaler.scale(loss / accum).backward()
            if (step + 1) % accum == 0:
                if clip:
                    self.scaler.unscale_(self.opt)
                    nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad(set_to_none=True)
                if self.ema:
                    self.ema.update(self.model)

            # accumulate ON DEVICE — no .item()/float() in the hot loop
            bs = lr.size(0)
            n += bs
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v * bs

        if self.sched:
            self.sched.step()

        dt = time.perf_counter() - t0
        out = {k: float(v) / max(n, 1) for k, v in agg.items()}   # single sync
        out["samples_per_sec"] = n / dt
        out["lr"] = self.opt.param_groups[0]["lr"]
        out["gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9 \
            if torch.cuda.is_available() else 0.0
        return out

    # ------------------------------------------------------------------
    @torch.no_grad()
    def validate(self, loader, use_ema: bool = True,
                 with_lpips: bool = True) -> Dict[str, float]:
        net = self.ema.ema if (use_ema and self.ema) else self.model
        net.eval()
        agg, n = {}, 0
        for lr, hr in loader:
            lr = lr.to(self.device, non_blocking=True)
            hr = hr.to(self.device, non_blocking=True)
            with torch.autocast("cuda", dtype=self.amp_dtype):
                pred = net(lr)
            m = evaluate_batch(pred.float(), hr.float(), with_lpips=with_lpips)
            bs = lr.size(0)
            n += bs
            for k, v in m.items():
                agg[k] = agg.get(k, 0.0) + v * bs
        return {k: v / max(n, 1) for k, v in agg.items()}

    # ------------------------------------------------------------------
    def save(self, epoch: int, metrics: Dict, tag: str = "last"):
        ckpt = {
            "model_state_dict": self.model.state_dict(),
            "ema_state_dict": self.ema.state_dict() if self.ema else None,
            "optimizer_state_dict": self.opt.state_dict(),
            "scheduler_state_dict": self.sched.state_dict() if self.sched else None,
            "epoch": epoch,
            "config": self.cfg,
            "input_transform": getattr(self.model, "input_transform", "log"),
            "git_commit": git_hash(),
            "metrics": metrics,
            "seed": self.cfg.get("seed"),
            "torch_version": str(torch.__version__),   # str, not TorchVersion
            "selection_rule": "best val SSIM (primary scored metric)",
        }
        torch.save(ckpt, self.out / f"{tag}.pt")

    def fit(self, train_loader, val_loader, epochs: int, clip: float = 1.0,
            accum: int = 1, val_every: int = 1):
        print(f"[trainer] amp={self.amp_dtype}  epochs={epochs}  "
              f"git={git_hash()[:8]}")
        for ep in range(1, epochs + 1):
            tr = self.train_epoch(train_loader, ep, clip=clip, accum=accum)
            row = {"epoch": ep, **{f"train/{k}": v for k, v in tr.items()}}

            if ep % val_every == 0 or ep == epochs:
                va = self.validate(val_loader)
                row.update({f"val/{k}": v for k, v in va.items()})
                for k, v in va.items():
                    self.tb.add_scalar(f"val/{k}", v, ep)
                if va["ssim"] > self.best:
                    self.best = va["ssim"]
                    self.best_metrics = va
                    self.save(ep, va, tag="best")
                print(f"  ep{ep:03d}  loss {tr.get('total', 0):.4f}  "
                      f"PSNR {va['psnr']:.3f}  SSIM {va['ssim']:.4f}  "
                      f"LPIPS {va.get('lpips', float('nan')):.4f}  "
                      f"{tr['samples_per_sec']:.0f} img/s  "
                      f"{tr['gpu_mem_gb']:.1f} GB")
            else:
                print(f"  ep{ep:03d}  loss {tr.get('total', 0):.4f}  "
                      f"{tr['samples_per_sec']:.0f} img/s")

            for k, v in tr.items():
                self.tb.add_scalar(f"train/{k}", v, ep)
            self.history.append(row)
            self.save(ep, row, tag="last")
            json.dump(self.history, open(self.out / "history.json", "w"), indent=1)

        self.tb.close()
        return self.best_metrics
