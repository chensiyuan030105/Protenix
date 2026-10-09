"""P010 §14.1 as a figure, parsed out of the readout rather than retyped.

The board's one-page summary needs the four-arm, three-band decomposition.  The
numbers already exist, published, in `artifacts/reports/P010/readout.md` §14.1 --
so the only way this figure can be wrong is a transcription error, and the way
to make that impossible is to not transcribe.  This parses that table and plots
it; if the table moves or its shape changes, the job fails instead of drawing a
stale picture.  Same contract as `kineidos.plot_heldout --verify` in P009.

What the figure has to carry
----------------------------
§14.4 is the point of the whole readout: low sigma is 61% of the sampling, and
*any* structured-but-wrong injection pays about +3.9% there -- the linear oracle
and shuffle differ by 0.06 points.  Only the correct conformation pays it off.
So the bars are grouped by sigma band with the sampling share on the axis, and
the seed floor is drawn as a band rather than left in a caption: an effect
inside it is not an effect.

Signs are kept as the readout writes them (negative = better, because it is a
relative change in a loss), and the axis is labelled to say so, rather than
flipping them to make taller bars mean better.

Run from the workspace root::

    sbatch --export=ALL,KINEIDOS_WORKSPACE=$PWD \\
      repos/research/kineidos-v3-diag/kineidos/slurm/p010_plot.sbatch
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

# Categorical slots 1-3 of the validated palette plus a muted slot for the arm
# that is a control rather than a treatment.  Checked with the dataviz
# validator at --pairs all on the light surface.
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8795a8"
INK, BODY, MUTE, FAINT, GRID = "#1a2332", "#3c4a5c", "#5c7390", "#8795a8", "#e7edf4"
SURFACE = "#fcfcfc"

#: Column order in §14.1, left to right after the two label columns.
ARMS = ("shuffle", "decoy", "oracle-v2", "linear oracle")
#: English only: the one matplotlib font here is DejaVu, which has sigma but
#: no CJK, and a missing glyph is dropped silently.  The Chinese commentary
#: lives in the Figma frame beside the image.
LABELS = {
    "shuffle": "shuffle  (capacity control)",
    "decoy": "decoy  (right molecule, wrong conformation)",
    "oracle-v2": "oracle-v2  (right conformation)",
    "linear oracle": "linear oracle  (a 2-D shadow after LN)",
}
COLOR = {"shuffle": GREY, "decoy": AQUA, "oracle-v2": ORANGE, "linear oracle": BLUE}
BANDS = ("高 σ", "中 σ", "低 σ", "全程（训练目标）")      # as the readout writes them
BAND_EN = {"高 σ": "high σ", "中 σ": "mid σ", "低 σ": "low σ",
           "全程（训练目标）": "overall\n(training objective)"}


def parse(readout: Path) -> dict:
    """§14.1's table, as {band: {arm: pct}}, plus the sampling shares and floor."""
    text = readout.read_text()
    start = text.find("### 14.1")
    if start < 0:
        raise SystemExit("§14.1 not found in the readout; the figure has no source")
    block = text[start:start + 4000]

    num = r"\*{0,2}([-+−][\d.]+)%\*{0,2}"
    rows, shares = {}, {}
    for band in BANDS:
        m = re.search(
            rf"^\|\s*\*{{0,2}}{re.escape(band)}\*{{0,2}}\s*\|\s*([\d.]+%)?\s*\|"
            rf"\s*{num}\s*\|\s*{num}\s*\|\s*{num}\s*\|\s*{num}\s*\|",
            block, re.M)
        if not m:
            raise SystemExit(f"§14.1: row {band!r} did not parse; table shape changed")
        if m.group(1):
            shares[band] = float(m.group(1).rstrip("%"))
        rows[band] = {a: float(m.group(2 + i).replace("−", "-"))
                      for i, a in enumerate(ARMS)}
    fm = re.search(r"种子底\s*([\d.]+)%", block)
    if not fm:
        raise SystemExit("§14.1: the seed floor line did not parse")
    return {"rows": rows, "shares": shares, "floor": float(fm.group(1)),
            "source": str(readout)}


def draw(d: dict, out_png: Path, step: str) -> Path:
    plt.rcParams.update({
        "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": BODY,
        "xtick.color": MUTE, "ytick.color": MUTE, "axes.titlecolor": INK,
        "font.family": "DejaVu Sans",
    })
    FW, FH = 18.4, 5.5
    fig = plt.figure(figsize=(FW, FH))
    fig.patch.set_facecolor(SURFACE)
    ax = fig.add_axes([0.052, 0.200, 0.938, 0.565])
    ax.set_facecolor("#ffffff")
    ax.grid(True, axis="y", color=GRID, lw=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    floor = d["floor"]
    ax.axhspan(-floor, floor, color=BLUE, alpha=0.13, lw=0, zorder=1)
    ax.axhline(0, color=FAINT, lw=1.1, zorder=2)

    x = np.arange(len(BANDS)); w = 0.2
    for k, arm in enumerate(ARMS):
        vals = [d["rows"][b][arm] for b in BANDS]
        ax.bar(x + (k - 1.5) * w, vals, w * 0.9, color=COLOR[arm], zorder=3,
               label=LABELS[arm])
        for xi, v in zip(x + (k - 1.5) * w, vals):
            ax.annotate(f"{v:+.2f}", xy=(xi, v), xytext=(0, 5 if v >= 0 else -15),
                        textcoords="offset points", ha="center", fontsize=9.5,
                        color=BODY, fontweight="bold" if arm == "oracle-v2" else "normal")

    ax.set_xticks(x)
    ax.set_xticklabels([BAND_EN[b] + (f"\n{d['shares'][b]:.1f}% of sampling"
                                      if b in d["shares"] else "")
                        for b in BANDS], fontsize=12, color=BODY, linespacing=1.6)
    ax.tick_params(axis="x", length=0, pad=8)
    ax.set_ylabel("change vs random-1gpu   ·   negative = better", fontsize=11.5, labelpad=8)
    ax.set_xlim(-0.55, len(BANDS) - 0.45)
    lo = min(min(r.values()) for r in d["rows"].values())
    hi = max(max(r.values()) for r in d["rows"].values())
    ax.set_ylim(lo - 0.18 * (hi - lo), hi + 0.22 * (hi - lo))

    fig.text(0.052, 1 - 0.30 / FH,
             f"P010   four arms x three σ bands, on the training objective  (step {step})",
             fontsize=16, fontweight="bold", color=INK)
    handles = [Patch(facecolor=COLOR[a], label=LABELS[a]) for a in ARMS]
    handles.append(Patch(facecolor=BLUE, alpha=0.13,
                         label=f"± seed floor {floor:.2f}%  (inside = indistinguishable)"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.050, 1 - 0.80 / FH),
               frameon=False, fontsize=11, ncol=5, columnspacing=1.3,
               handletextpad=0.6, handlelength=1.6)
    fig.text(0.052, 0.045,
             "Low σ is 61.2% of the sampling, and there every structured-but-wrong "
             "injection pays almost the same price (linear oracle +3.91%, shuffle +3.97% "
             "— 0.06 points apart). Only the right conformation pays it off (+0.55%).",
             fontsize=10.5, color=MUTE)
    fig.text(0.052, 0.012,
             "source  artifacts/reports/P010/readout.md §14.1 — parsed by "
             "kineidos.plot_sigma_arms, never retyped; if the table's shape changes the "
             "job fails instead of drawing a stale figure.",
             fontsize=9.5, color=FAINT)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, facecolor=fig.get_facecolor())
    return out_png


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--readout", type=Path,
                   default=Path("artifacts/reports/P010/readout.md"))
    p.add_argument("--out", type=Path,
                   default=Path("artifacts/reports/P010/p010_sigma_arms.png"))
    p.add_argument("--json", type=Path,
                   default=Path("artifacts/reports/P010/p010_sigma_arms.json"))
    p.add_argument("--step", default="499")
    a = p.parse_args()
    if not a.readout.is_file():
        raise SystemExit(f"no {a.readout}; run from the workspace root")
    d = parse(a.readout)
    a.json.parent.mkdir(parents=True, exist_ok=True)
    a.json.write_text(json.dumps(d, indent=1, ensure_ascii=False))
    for b in BANDS:
        print(f"  {b:>14s}  " + "  ".join(f"{k}={d['rows'][b][k]:+7.2f}" for k in ARMS))
    print(f"  seed floor {d['floor']}%")
    out = draw(d, a.out, a.step)
    print(f"  wrote {out}\n  wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
