"""P010's headline figure, parsed out of the readout rather than retyped.

The numbers already exist, published, in `artifacts/reports/P010/readout.md`
§17.1b and §14.6 -- so the only way a figure of them can be wrong is a
transcription error, and the way to make that impossible is to not transcribe.
This parses those two tables and plots them; if either moves or changes shape,
the job exits instead of drawing a stale picture.  Same contract as P009's
`plot_heldout --verify`.

Two panels, one message each
----------------------------
A  The five arms on the training objective, ranked, against the seed floor.
   The floor is a band rather than a caption, because the whole point is that
   `shuffle` and `decoy` sit inside it: an effect inside the floor is not an
   effect.
B  What `oracle-v2`'s total splits into, at both checkpoints.  The capacity
   part is itself inside the floor, so the conformation-specific part is the
   whole story -- and it grows 69% -> 74% as training goes on, which is the
   §6.1 requirement that a decomposition be read at two checkpoints.

The earlier version of this figure put four sigma bands side by side with a
fourth group labelled "overall".  That was wrong twice over: "overall" is the
sampling-weighted *aggregate* of the other three, so standing it beside them
implies a fourth band; and sixteen bars each carrying a printed value is a
table drawn as a chart.  Both panels here carry five marks or fewer.

Signs stay as the readout writes them -- negative = better, because it is a
relative change in a loss -- with the axis saying so, rather than being
flipped to make longer bars mean better.

Run from the workspace root::

    sbatch repos/research/kineidos-v3-diag/kineidos/slurm/p010_plot.sbatch
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# Categorical slots of the validated palette, plus a muted slot for the arms
# that are controls rather than treatments.
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#9aa7b6"
INK, BODY, MUTE, FAINT, GRID = "#1a2332", "#3c4a5c", "#5c7390", "#8795a8", "#e7edf4"
SURFACE = "#fcfcfc"

#: Row label in §17.1b -> (short name for the axis, what it is, colour).
#: Order here is the order the readout lists them; the panel sorts by value.
ARMS = [
    ("oracle", "linear oracle", "a 2-D shadow after LN", GREY),
    ("oracle-v2-shuffle", "shuffle", "mis-paired — the capacity floor", GREY),
    ("oracle-v2-decoy", "decoy", "right molecule, wrong conformation", AQUA),
    ("oracle-v3", "oracle-v3", "frame-bearing, frames unaligned", BLUE),
    ("oracle-v2", "oracle-v2", "anchor-distance Fourier, invariant", ORANGE),
]


def _f(tok: str) -> float:
    return float(tok.replace("−", "-").replace("%", "").replace("*", "").strip())


def parse(readout: Path) -> dict:
    text = readout.read_text()

    # --- §17.1b: five arms x two checkpoints, plus the seed floor -----------
    i = text.find("### 17.1b")
    if i < 0:
        raise SystemExit("§17.1b not found; the figure has no source")
    block = text[i:i + 2500]
    num = r"\*{0,2}([-+−]?[\d.]+)%\*{0,2}"
    arms: dict[str, dict[str, float]] = {}
    for key, _short, _what, _c in ARMS:
        m = re.search(rf"^\|[^|\n]*`{re.escape(key)}`[^|\n]*\|\s*{num}\s*\|\s*{num}\s*\|",
                      block, re.M)
        if not m:
            raise SystemExit(f"§17.1b: row for {key!r} did not parse")
        arms[key] = {"499": _f(m.group(1)), "999": _f(m.group(2))}
    m = re.search(rf"^\|[^|\n]*种子底[^|\n]*\|\s*{num}\s*\|\s*{num}\s*\|", block, re.M)
    if not m:
        raise SystemExit("§17.1b: the seed-floor row did not parse")
    floor = {"499": _f(m.group(1)), "999": _f(m.group(2))}

    # --- §14.6: what oracle-v2's total splits into -------------------------
    j = text.find("### 14.6")
    if j < 0:
        raise SystemExit("§14.6 not found; the figure has no source")
    dec = text[j:j + 2500]
    pp = r"\*{0,2}([\d.]+)\s*pp"
    m = re.search(rf"^\|[^|\n]*容量[^|\n]*\|\s*{pp}[^|]*\|\s*{pp}[^|]*\|", dec, re.M)
    if not m:
        raise SystemExit("§14.6: the capacity row did not parse")
    cap = {"499": _f(m.group(1)), "999": _f(m.group(2))}
    m = re.search(rf"^\|[^|\n]*构象特异[^|\n]*\|\s*{pp}\s*（(\d+)%）[^|]*\|\s*{pp}\s*（(\d+)%）[^|]*\|",
                  dec, re.M)
    if not m:
        raise SystemExit("§14.6: the conformation row did not parse")
    conf = {"499": _f(m.group(1)), "999": _f(m.group(3))}
    share = {"499": int(m.group(2)), "999": int(m.group(4))}

    return {"arms": arms, "floor": floor, "capacity": cap,
            "conformation": conf, "share": share, "source": str(readout)}


def draw(d: dict, out_png: Path, step: str = "999") -> Path:
    plt.rcParams.update({
        "font.size": 12, "axes.edgecolor": GRID, "axes.labelcolor": BODY,
        "xtick.color": MUTE, "ytick.color": MUTE, "axes.titlecolor": INK,
        "font.family": "DejaVu Sans",
    })
    FW, FH = 18.4, 4.6
    fig = plt.figure(figsize=(FW, FH))
    fig.patch.set_facecolor(SURFACE)
    axA = fig.add_axes([0.158, 0.215, 0.462, 0.575])
    axB = fig.add_axes([0.737, 0.215, 0.238, 0.575])
    for ax in (axA, axB):
        ax.set_facecolor("#ffffff")
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    # ---------- A: the five arms, ranked ----------
    floor = d["floor"][step]
    rows = sorted(ARMS, key=lambda a: d["arms"][a[0]][step])      # best (most negative) first
    y = np.arange(len(rows))[::-1]
    vals = [d["arms"][k][step] for k, *_ in rows]
    axA.axvspan(-floor, floor, color=BLUE, alpha=0.13, lw=0, zorder=1)
    axA.axvline(0, color=FAINT, lw=1.1, zorder=2)
    axA.grid(True, axis="x", color=GRID, lw=0.9)
    axA.barh(y, vals, 0.56, color=[r[3] for r in rows], zorder=3)
    for yi, v, r in zip(y, vals, rows):
        axA.annotate(f"{v:+.2f}%", xy=(v, yi), xytext=(-9 if v < 0 else 9, 0),
                     textcoords="offset points", va="center",
                     ha="right" if v < 0 else "left", fontsize=12.5,
                     color=INK if r[3] is ORANGE else BODY,
                     fontweight="bold" if r[3] is ORANGE else "normal")
    axA.set_yticks(y)
    axA.set_yticklabels([r[1] for r in rows], fontsize=13, color=INK)
    for yi, r in zip(y, rows):
        axA.annotate(r[2], xy=(0, yi), xycoords=("axes fraction", "data"),
                     xytext=(-12, -16), textcoords="offset points",
                     ha="right", va="center", fontsize=10.5, color=MUTE)
    axA.tick_params(axis="y", length=0, pad=6)
    axA.set_xlabel("change in the training objective vs random-1gpu   ·   negative = better",
                   fontsize=11.5, labelpad=7)
    lo = min(vals)
    axA.set_xlim(lo * 1.22, max(max(vals), floor) * 2.6 + 0.8)
    axA.set_ylim(-0.6, len(rows) - 0.4)

    # ---------- B: what the winner's total is made of ----------
    steps = ["499", "999"]
    x = np.arange(len(steps))
    cap = [d["capacity"][s] for s in steps]
    conf = [d["conformation"][s] for s in steps]
    axB.grid(True, axis="y", color=GRID, lw=0.9)
    axB.bar(x, cap, 0.5, color=GREY, zorder=3)
    axB.bar(x, conf, 0.5, bottom=cap, color=ORANGE, zorder=3)
    for xi, c, f in zip(x, cap, conf):
        axB.annotate(f"{c:.2f} pp", xy=(xi, c / 2), ha="center", va="center",
                     fontsize=11, color="#ffffff")
        axB.annotate(f"{f:.2f} pp", xy=(xi, c + f / 2), ha="center", va="center",
                     fontsize=12.5, color="#ffffff", fontweight="bold")
        axB.annotate(f"{d['share'][steps[xi]]}%", xy=(xi, c + f), xytext=(0, 8),
                     textcoords="offset points", ha="center", fontsize=13,
                     color=INK, fontweight="bold")
    for xi, s in zip(x, steps):
        axB.axhline(d["floor"][s], xmin=0.08 + 0.5 * xi, xmax=0.42 + 0.5 * xi,
                    color=BLUE, lw=2.0, ls=(0, (3, 2)), zorder=4)
    axB.set_xticks(x)
    axB.set_xticklabels([f"step {s}" for s in steps], fontsize=13, color=INK)
    axB.tick_params(axis="x", length=0, pad=7)
    axB.set_ylabel("oracle-v2's total, in points", fontsize=11.5)
    axB.set_ylim(0, (cap[-1] + conf[-1]) * 1.26)
    axB.set_xlim(-0.6, len(steps) - 0.4)

    # No "P010" prefix: the board frame's header already says it, and a figure
    # that repeats its page's title wastes the one line it has.
    fig.text(0.013, 1 - 0.33 / FH,
             "Only the invariant encoding clears the seed floor  —  and three "
             "quarters of what it buys is conformation-specific",
             fontsize=17, fontweight="bold", color=INK)
    fig.legend(handles=[
        Patch(facecolor=BLUE, alpha=0.13, label=f"± seed floor {floor:.2f}%  (inside = not an effect)"),
        Patch(facecolor=GREY, label="capacity + molecular geometry  (itself inside the floor)"),
        Patch(facecolor=ORANGE, label="conformation-specific, confirmed by the shuffle control"),
    ], loc="upper left", bbox_to_anchor=(0.012, 1 - 0.88 / FH), frameon=False,
        fontsize=11.5, ncol=3, columnspacing=2.0, handletextpad=0.7, handlelength=1.6)
    fig.text(0.013, 0.035,
             f"left: all five arms at step {step}, same baseline and seed, differing only in what h contains.   "
             "right: oracle-v2's total split by the shuffle control, at both checkpoints.   "
             "source  readout.md §17.1b and §14.6, parsed — never retyped.",
             fontsize=10, color=FAINT)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, facecolor=fig.get_facecolor())
    return out_png


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--readout", type=Path,
                   default=Path("artifacts/reports/P010/readout.md"))
    p.add_argument("--out", type=Path,
                   default=Path("artifacts/reports/P010/p010_arm_summary.png"))
    p.add_argument("--json", type=Path,
                   default=Path("artifacts/reports/P010/p010_arm_summary.json"))
    p.add_argument("--step", default="999")
    a = p.parse_args()
    if not a.readout.is_file():
        raise SystemExit(f"no {a.readout}; run from the workspace root")
    d = parse(a.readout)
    a.json.parent.mkdir(parents=True, exist_ok=True)
    a.json.write_text(json.dumps(d, indent=1, ensure_ascii=False))
    for k, short, *_ in ARMS:
        print(f"  {short:>14s}  499={d['arms'][k]['499']:+7.2f}  999={d['arms'][k]['999']:+7.2f}")
    print(f"  seed floor   499={d['floor']['499']}  999={d['floor']['999']}")
    print(f"  split 999    capacity={d['capacity']['999']} pp  "
          f"conformation={d['conformation']['999']} pp ({d['share']['999']}%)")
    out = draw(d, a.out, a.step)
    print(f"  wrote {out}\n  wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
