"""Generate docs/ablation_results.md and docs/ood_report.md from JSON artefacts.

Generated rather than hand-written so the documents can never drift from the
numbers that were actually produced.

    python scripts/make_reports.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(p):
    p = ROOT / p
    return json.load(open(p)) if p.exists() else None


# --------------------------------------------------------------- ablations
def ablation_report():
    res = load("experiments/ablations.json")
    if not res:
        return "  (no ablations.json — run scripts/run_ablations.py)"

    base = res.get("002", {}).get("metrics")
    lines = [
        "# LOSS ABLATION RESULTS", "",
        "One variable per run. Every row is identical apart from the loss term",
        "named in its title: same backbone, same seed, same data mix (KLA pairs",
        "only, so the loss is isolated from the content change), same epoch",
        "budget, same source-disjoint validation split.", "",
        "Purpose is **attribution**, not peak score. Without this table we could",
        "not claim that any individual loss term helped.", "",
        "| ID | Loss change | PSNR ↑ | SSIM ↑ | LPIPS ↓ | vs base |",
        "|---|---|---|---|---|---|",
    ]
    for aid, r in res.items():
        m = r["metrics"]
        d = "—" if not base or aid == "002" else \
            f"{m['psnr']-base['psnr']:+.3f} dB / {m['ssim']-base['ssim']:+.4f} SSIM"
        lines.append(f"| **{aid}** | {r['label']} | {m['psnr']:.3f} | "
                     f"{m['ssim']:.4f} | {m.get('lpips', float('nan')):.4f} | {d} |")

    ep = next(iter(res.values())).get("epochs", "?")
    lines += ["", f"*{ep} epochs per run — short by design; the comparison "
                  "between rows is what matters, not the absolute values.*", ""]

    # ---- interpretation, derived from the numbers -----------------------
    if base:
        lines += ["## What the table says", ""]
        best_ssim = max(res.items(), key=lambda kv: kv[1]["metrics"]["ssim"])
        best_lpips = min(res.items(),
                         key=lambda kv: kv[1]["metrics"].get("lpips", 9))
        lines += [
            f"* **Best SSIM:** `{best_ssim[0]}` — {best_ssim[1]['label']} "
            f"({best_ssim[1]['metrics']['ssim']:.4f})",
            f"* **Best LPIPS:** `{best_lpips[0]}` — {best_lpips[1]['label']} "
            f"({best_lpips[1]['metrics'].get('lpips', float('nan')):.4f})",
        ]
        if "007" in res:
            m7 = res["007"]["metrics"]
            lines += ["", f"* **Control (007, L2 instead of Charbonnier):** "
                          f"PSNR {m7['psnr']:.3f}, SSIM {m7['ssim']:.4f}, "
                          f"LPIPS {m7.get('lpips', float('nan')):.4f}. "]
            if base and m7.get("lpips", 9) > base.get("lpips", 0):
                lines.append("  Its LPIPS is worse than the Charbonnier base, "
                             "which is the over-smoothing the spec warns "
                             "against, reproduced deliberately.")
    return "\n".join(lines)


# --------------------------------------------------------------------- OOD
def ood_report():
    ev = load("experiments/eval_results.json")
    bl = load("experiments/baselines.json")
    cons = load("experiments/consistency.json")
    if not ev:
        return "  (no eval_results.json — run evaluate.py)"

    lines = [
        "# OUT-OF-DISTRIBUTION REPORT", "",
        "KLA withholds the test ground truth, so generalisation cannot be",
        "measured directly on the scored set. Instead we **manufacture** ground",
        "truth: because Phase 0 reconstructed the degradation to 0.982 histogram",
        "overlap, any clean image can be turned into a labelled pair.", "",
        "That is what makes every number below possible.", "",
        "## Per-family results", "",
        "| Family | Content | PSNR ↑ | SSIM ↑ | LPIPS ↓ | n |",
        "|---|---|---|---|---|---|",
    ]

    desc = {
        "val_in_distribution": ("KLA held-out val", "photographs (in-distribution)"),
        "proxy_ood_tonal_extremes": ("Tonal extremes", "darkest + brightest KLA sources"),
        "ood/Urban100": ("Urban100", "**buildings / cityscapes** — the OOD case KLA named"),
        "ood/BSD100": ("BSD100", "natural scenes"),
        "ood/Set14": ("Set14", "classic SR benchmark"),
    }
    base_row = ev.get("val_in_distribution")
    for k, v in ev.items():
        if not isinstance(v, dict) or "psnr" not in v:
            continue
        label, content = desc.get(k, (k, ""))
        lines.append(f"| {label} | {content} | {v['psnr']:.3f} | {v['ssim']:.4f} "
                     f"| {v.get('lpips', float('nan')):.4f} | {v.get('n_images','—')} |")

    if bl:
        lines += ["", "## Against the in-distribution baselines", "",
                  "| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |", "|---|---|---|---|"]
        for k, m in bl.items():
            lines.append(f"| {k} | {m['psnr']:.3f} | {m['ssim']:.4f} | "
                         f"{m.get('lpips', float('nan')):.4f} |")
        if base_row:
            lines.append(f"| **ours** | **{base_row['psnr']:.3f}** | "
                         f"**{base_row['ssim']:.4f}** | "
                         f"**{base_row.get('lpips', float('nan')):.4f}** |")

    # ---- honest reading -------------------------------------------------
    lines += ["", "## Honest reading", ""]
    if base_row:
        for k, v in ev.items():
            if not isinstance(v, dict) or "psnr" not in v or k == "val_in_distribution":
                continue
            label = desc.get(k, (k, ""))[0]
            d = v["psnr"] - base_row["psnr"]
            if d > 0:
                lines.append(f"* **{label} scores {d:+.2f} dB vs in-distribution "
                             f"— i.e. it is EASIER, not harder.** It is therefore "
                             f"weak evidence of generalisation and we do not "
                             f"present it as an OOD win.")
            else:
                lines.append(f"* **{label}: {d:+.2f} dB vs in-distribution.** "
                             f"A genuine drop, which is what an OOD family "
                             f"should show.")

    if cons:
        s = cons["summary"]
        lines += ["", "## Degradation consistency on the REAL test set (no GT)", "",
                  "```", "x̂ = model(y);  ŷ = degradation(x̂);  error = ‖ŷ − y‖",
                  "```", "",
                  "| | RMSE |", "|---|---|",
                  f"| median | {s['median_rmse']:.4f} |",
                  f"| p95 | {s['p95_rmse']:.4f} |",
                  f"| max | {s['max_rmse']:.4f} |", "",
                  "⚠️ Necessary, not sufficient: an over-smoothed output can still",
                  "re-degrade convincingly, because the degradation destroys high",
                  "frequencies anyway. Use it to catch gross failure, not to rank",
                  "good models.", ""]

    lines += ["", "## Where the model still degrades", "",
              "See `docs/ANALYSIS.md` §5 for the measured failure modes:",
              "over-smoothing of high-frequency texture, and fine periodic",
              "structure where it can score below bicubic.", ""]
    return "\n".join(lines)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    for name, fn in (("docs/ablation_results.md", ablation_report),
                     ("docs/ood_report.md", ood_report)):
        body = fn()
        if body.strip().startswith("("):
            print(f"skip {name}: {body.strip()}")
            continue
        (ROOT / name).write_text(body + "\n", encoding="utf-8")
        print(f"wrote {name}  ({len(body.splitlines())} lines)")


if __name__ == "__main__":
    main()
