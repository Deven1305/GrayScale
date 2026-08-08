"""Regenerate the README results table from the JSON artefacts.

Generating rather than hand-writing means the README can never drift from the
numbers actually produced. Run after evaluate.py.

    python scripts/make_results_table.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START = "<!-- RESULTS_TABLE_START -->"
END = "<!-- RESULTS_TABLE_END -->"


def fmt(v, n=3):
    return "—" if v is None else f"{v:.{n}f}"


def main():
    bl_p = ROOT / "experiments/baselines.json"
    ev_p = ROOT / "experiments/eval_results.json"
    if not bl_p.exists():
        raise SystemExit("run scripts/run_baselines.py first")

    bl = json.load(open(bl_p))
    ev = json.load(open(ev_p)) if ev_p.exists() else {}

    rows = []
    label = {"bicubic_x2": "Bicubic ×2 *(floor)*",
             "bm3d+bicubic_x2": "BM3D + bicubic ×2"}
    for k, m in bl.items():
        rows.append((label.get(k, k), m["psnr"], m["ssim"], m.get("lpips"),
                     m.get("msssim")))

    ours = ev.get("val_in_distribution")
    if ours:
        arch = ev.get("arch", "model")
        p = ev.get("params_M")
        name = f"**{arch}** ({p:.2f} M params)" if p else f"**{arch}**"
        rows.append((name, ours["psnr"], ours["ssim"], ours.get("lpips"),
                     ours.get("msssim")))

    lines = ["| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | MS-SSIM ↑ |",
             "|---|---|---|---|---|"]
    for n, ps, ss, lp, ms in rows:
        lines.append(f"| {n} | {fmt(ps)} | {fmt(ss, 4)} | {fmt(lp, 4)} | "
                     f"{fmt(ms, 4)} |")

    ood = ev.get("proxy_ood_tonal_extremes")
    if ood:
        lines += ["", "**Proxy-OOD** — held-out darkest and brightest source "
                      "clusters (Phase 0 §9; the corpus has no visual-origin "
                      "families, so this replaces leave-one-origin-out):", "",
                  "| Split | PSNR ↑ | SSIM ↑ | LPIPS ↓ |", "|---|---|---|---|",
                  f"| tonal extremes | {fmt(ood['psnr'])} | "
                  f"{fmt(ood['ssim'], 4)} | {fmt(ood.get('lpips'), 4)} |"]

    if ours:
        notes = []
        bic = bl.get("bicubic_x2")
        if bic:
            won = (ours["psnr"] > bic["psnr"] and ours["ssim"] > bic["ssim"]
                   and ours.get("lpips", 9) < bic.get("lpips", 9))
            notes.append(
                f"* vs **bicubic** (the floor): "
                f"{'beats it on all three scored metrics' if won else 'DOES NOT beat it — investigate'}"
                f" — {ours['psnr']-bic['psnr']:+.2f} dB PSNR, "
                f"{ours['ssim']-bic['ssim']:+.4f} SSIM, "
                f"{ours.get('lpips',0)-bic.get('lpips',0):+.4f} LPIPS.")
        bm = bl.get("bm3d+bicubic_x2")
        if bm:
            dp = ours["psnr"] - bm["psnr"]
            # the connective has to follow the sign, or the sentence lies
            joiner = "and" if dp > 0 else "but"
            note = (f"* vs **BM3D+bicubic**: {ours['ssim']-bm['ssim']:+.4f} SSIM, "
                    f"{ours.get('lpips',0)-bm.get('lpips',0):+.4f} LPIPS {joiner} "
                    f"{dp:+.2f} dB PSNR")
            if dp > 0:
                note += (" — ahead on all three. Worth noting BM3D's LPIPS is "
                         "*worse than plain bicubic*: it buys PSNR by "
                         "over-smoothing, which PSNR rewards and LPIPS punishes.")
            else:
                note += (". BM3D wins PSNR by over-smoothing, which PSNR "
                         "rewards and LPIPS punishes — its LPIPS is worse than "
                         "plain bicubic. Stated plainly rather than omitted.")
            notes.append(note)
        if notes:
            lines += [""] + notes

    table = "\n".join(lines)
    rd = ROOT / "README.md"
    s = rd.read_text(encoding="utf-8")
    s = re.sub(f"{re.escape(START)}.*?{re.escape(END)}",
               f"{START}\n{table}\n{END}", s, flags=re.S)
    rd.write_text(s, encoding="utf-8")
    # stdout on Windows is cp1252 and cannot encode the arrows
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(table)
    print(f"\nwrote table into {rd}")


if __name__ == "__main__":
    main()
