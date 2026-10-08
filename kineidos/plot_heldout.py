#!/usr/bin/env python
"""P009's loss curves, drawn through the same loader the registered read uses.

The figure is not allowed to tell a different story from `readout.md`.  Two
measures enforce that:

  * The held-out numbers come from `kineidos.read_heldout` -- `find_run_dir`,
    `load_rows`, `ARMS`, `BINS` are imported, not re-implemented.  A second
    parser would be free to drift: `find_run_dir`'s timestamp anchor exists
    because a plain prefix match makes `p009_zero` swallow `p009_zero_seed2`,
    and a figure that fell into that would silently plot the noise-floor arm
    as the baseline.
  * `--verify artifacts/reports/P009/readout.md` re-reads the published tables
    and asserts the curve and the three bins agree to 1e-4.  If the runs have
    moved since the readout was written the job fails rather than drawing a
    figure that disagrees with the registered read.

Three panels, which is the argument in order:

  A  held-out total loss per evaluation round, four arms, with the train loss
     faint behind it.  `zero` and `zero-seed2` are the same configuration under
     a different seed, so the gap between them is what "no effect" looks like.
  B  `random - zero` against +/-|zero - zero-seed2|.  The treatment effect
     against the floor, round by round.
  C  the three stride bins at the read step, effect against floor, with paired
     standard errors.  The error bars crossing zero is the verdict: section
     6.2 reads bin 1, and bin 1 is indistinguishable.

The available information per bin (23 / 13 / 7%) is written as text, never as a
bar beside the loss differences: it is a different quantity in different units
and putting the two on one scale would imply they are comparable.

Chart text is English: the only fonts on this machine with the required
coverage are Latin-only, and a missing CJK glyph in matplotlib is dropped
silently.  The Chinese annotation lives in the Figma frame beside the image.

Run from the workspace root:

    PYTHONPATH=repos/research/kineidos-v3:repos/research/wp-v2 \
      <env>/bin/python -m kineidos.plot_heldout
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from kineidos.read_heldout import ARMS, BINS, bin_of, find_run_dir, load_rows  # noqa: E402

# Categorical slots 1-3 of the validated default palette.  Checked with the
# dataviz validator at `--pairs all`, light surface: all gates pass, worst CVD
# dE 9.2, worst normal-vision dE 24.0.  Aqua sits at 2.74:1 against the surface,
# below the 3:1 bar, so every arm carries a direct label at the right edge --
# that is the relief rule, not decoration.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, BODY, MUTE, FAINT, GRID = "#1a2332", "#3c4a5c", "#5c7390", "#8795a8", "#e7edf4"
SURFACE = "#fcfcfc"

# `zero` and `zero-seed2` share a hue because they are one configuration under
# two seeds; the dash carries the difference.  Colour follows the entity.
STYLE = {
    "random":     dict(color=ORANGE, ls="-",  label="random — h injected (treatment)"),
    "zero":       dict(color=BLUE,   ls="-",  label="zero — control, seed 42"),
    "zero-seed2": dict(color=BLUE,   ls="--", label="zero — same config, seed 43"),
    "none":       dict(color=AQUA,   ls="-",  label="none — cross-check, not in the verdict"),
}
ORDER = ("random", "zero", "zero-seed2", "none")
METRIC = "loss"


def heldout_curve(rows: dict[int, dict[int, dict]]) -> dict[int, float]:
    """{step: mean held-out loss} over whatever windows that round scored."""
    return {step: statistics.fmean(r[METRIC] for r in bucket.values())
            for step, bucket in rows.items()}


def train_curve(slurm_dir: Path, run_dir_name: str) -> list[tuple[int, float]]:
    """(step, train/loss.avg) out of the slurm log that owns this run directory.

    Matched on the full timestamped directory name, which appears in the
    "Using run name:" line.  Matching on the bare prefix would make
    `p009_zero` also select `p009_zero_seed2`'s log.
    """
    step_re = re.compile(r"Step (\d+) train metrics: \{")
    loss_re = re.compile(r"'train/loss\.avg':\s*(?:np\.float64\()?"
                         r"(-?\d+\.?\d*(?:[eE][-+]?\d+)?)")
    out: dict[int, float] = {}
    for path in sorted(slurm_dir.glob("p009-kineidos-*.err")):
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        if run_dir_name not in text:
            continue
        for line in text.splitlines():
            m = step_re.search(line)
            if not m:
                continue
            v = loss_re.search(line)
            if v:
                out[int(m.group(1))] = float(v.group(1))
    return sorted(out.items())


def paired(a: dict[int, dict], b: dict[int, dict], wids) -> tuple[float, float, int]:
    """mean(a - b), its standard error, n -- paired window by window."""
    d = [a[w][METRIC] - b[w][METRIC] for w in wids]
    n = len(d)
    if n < 2:
        return (d[0] if d else 0.0), 0.0, n
    return statistics.fmean(d), statistics.stdev(d) / (n ** 0.5), n


def bins_at(data: dict[str, dict[int, dict[int, dict]]], step: int) -> list[dict]:
    """Per-bin effect and floor at one round, on the windows every arm scored."""
    common = set.intersection(*(set(data[a][step]) for a in data))
    out = []
    for name, lo, hi, info in BINS:
        wids = sorted(w for w in common
                      if bin_of(int(data["zero"][step][w]["stride"])) == name)
        if not wids:
            continue
        zero_mean = statistics.fmean(data["zero"][step][w][METRIC] for w in wids)
        eff, eff_se, n = paired(data["random"][step], data["zero"][step], wids)
        flo, flo_se, _ = paired(data["zero"][step], data["zero-seed2"][step], wids)
        out.append(dict(
            bin=name, stride=f"{lo}-{hi}", n=n, info=info, zero=zero_mean,
            random=statistics.fmean(data["random"][step][w][METRIC] for w in wids),
            effect=eff, effect_se=eff_se, effect_pct=100 * eff / zero_mean,
            effect_se_pct=100 * eff_se / zero_mean,
            floor=abs(flo), floor_se=flo_se, floor_pct=100 * abs(flo) / zero_mean,
            floor_se_pct=100 * flo_se / zero_mean,
            t=abs(eff / eff_se) if eff_se else float("inf"),
        ))
    return out


def verify(readout: Path, curves: dict[str, dict[int, float]], bins: list[dict],
           step: int, tol: float = 1e-4) -> list[str]:
    """Assert the figure's numbers equal the published read's.  Returns problems."""
    text = readout.read_text()
    problems: list[str] = []

    cols = ["random", "zero", "zero-seed2", "none"]
    seen = 0
    for m in re.finditer(r"^\|\s*(\d+)\s*\|((?:\s*[\d.]+\s*\|){4})\s*$",
                         text, re.M):
        s = int(m.group(1))
        vals = [float(x) for x in m.group(2).strip().strip("|").split("|")]
        for arm, v in zip(cols, vals):
            got = curves.get(arm, {}).get(s)
            if got is None:
                problems.append(f"curve: {arm} has no step {s}, readout has {v}")
            elif abs(got - v) > 5e-5:          # the table is rounded to 4 places
                problems.append(f"curve: {arm}@{s} figure {got:.6f} vs readout {v}")
        seen += 1
    if seen == 0:
        problems.append("curve: no '各轮曲线' rows matched in the readout")

    by_name = {b["bin"]: b for b in bins}
    hits = 0
    for m in re.finditer(
            r"^\|\s*(bin\d)\s*\|[^|]*\|\s*(\d+)\s*\|[^|]*\|\s*([\d.]+)\s*\|"
            r"\s*([\d.]+)\s*\|\s*(-?[\d.]+)\s*\([^)]*\)\s*\|[^|]*\|[^|]*\|"
            r"\s*([\d.]+)\s*\(", text, re.M):
        name, n, zero, rnd, diff, floor = m.groups()
        b = by_name.get(name)
        if b is None:
            problems.append(f"bins: readout has {name}, figure does not")
            continue
        for label, got, want in (("n", b["n"], int(n)), ("zero", b["zero"], float(zero)),
                                 ("random", b["random"], float(rnd)),
                                 ("effect", b["effect"], float(diff)),
                                 ("floor", b["floor"], float(floor))):
            if abs(got - want) > (0 if label == "n" else 5e-5):
                problems.append(f"bins: {name}.{label} figure {got} vs readout {want}")
        hits += 1
    if hits != len(bins):
        problems.append(f"bins: matched {hits} readout rows, figure has {len(bins)}")
    return problems


#  Where each arm's end label sits, in points, so the four do not stack on one
#  another at the right edge.  `random` ends 1500 steps earlier than the rest,
#  so it needs no nudge.
LABEL_DY = {"random": 0, "zero": 11, "zero-seed2": -1, "none": -13}
XMIN = 400          # before this the train curves are still falling off the top


def draw(curves, trains, bins, step, stops, out_png):
    plt.rcParams.update({
        "font.size": 9.5, "axes.edgecolor": GRID, "axes.labelcolor": BODY,
        "xtick.color": MUTE, "ytick.color": MUTE, "axes.titlecolor": INK,
    })
    fig = plt.figure(figsize=(14.2, 11.6))
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(3, 1, height_ratios=[2.05, 1.0, 1.15], hspace=0.40,
                          left=0.058, right=0.985, top=0.862, bottom=0.185)
    axA, axB, axC = (fig.add_subplot(g) for g in gs)
    for ax in (axA, axB, axC):
        ax.set_facecolor("#ffffff")
        ax.grid(True, color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    # ---- A: held-out loss, four arms ---------------------------------------
    xmax = max(max(c) for c in curves.values())
    lo, hi = [], []
    for arm in ORDER:
        st = STYLE[arm]
        tr = trains.get(arm, [])
        if tr:
            ts = np.array([p[0] for p in tr]); tv = np.array([p[1] for p in tr])
            k = 4                       # 200-step mean; 50 steps is inside the noise
            sm = np.convolve(tv, np.ones(k) / k, mode="valid")
            ts = ts[k - 1:]
            axA.plot(ts, sm, color=st["color"], ls=st["ls"], lw=1.0,
                     alpha=0.30, zorder=2)
            vis = sm[ts >= XMIN]
            if vis.size:
                lo.append(vis.min()); hi.append(vis.max())
        s = sorted(curves[arm]); v = [curves[arm][x] for x in s]
        axA.plot(s, v, color=st["color"], ls=st["ls"], lw=2.0, marker="o", ms=5.5,
                 mfc="white", mew=1.7, zorder=5)
        axA.annotate(arm, xy=(s[-1], v[-1]), xytext=(9, LABEL_DY[arm]),
                     textcoords="offset points", color=st["color"], fontsize=9.5,
                     va="center", fontweight="bold")
        lo.append(min(v)); hi.append(max(v))

    pad = 0.055 * (max(hi) - min(lo))
    axA.set_ylim(min(lo) - pad, max(hi) + pad)
    axA.axvline(step, color=FAINT, lw=1.1, ls=(0, (4, 3)), zorder=1)
    axA.text(step - xmax * 0.008, 0.035, f"the read  (step {step})",
             transform=axA.get_xaxis_transform(), ha="right", fontsize=9,
             color=MUTE, style="italic")
    axA.set_ylabel("held-out total loss")
    axA.set_xlim(XMIN, xmax * 1.09)
    axA.set_title("P009   four arms: held-out loss per evaluation round",
                  loc="left", fontsize=13.5, fontweight="bold", pad=56)
    axA.text(0, 1.118,
             "256 paired windows, eval_seed 1234 in every arm, so all four score the "
             "same windows in the same order under the same noise.  "
             f"`random` stopped at step {stops['random']}; the other three reached "
             f"{stops['zero']}.",
             transform=axA.transAxes, fontsize=9, color=MUTE)
    handles = [Line2D([], [], color=STYLE[a]["color"], ls=STYLE[a]["ls"], lw=2.0,
                      marker="o", ms=5.5, mfc="white", mew=1.7, label=STYLE[a]["label"])
               for a in ORDER]
    handles.append(Line2D([], [], color=MUTE, lw=1.0, alpha=0.4,
                          label="train loss, 200-step mean"))
    axA.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.015),
               frameon=False, fontsize=9, ncol=5, handletextpad=0.6,
               columnspacing=1.5)

    # ---- B: effect against floor, round by round ---------------------------
    shared = sorted(set(curves["random"]) & set(curves["zero"]) & set(curves["zero-seed2"]))
    eff = np.array([curves["random"][s] - curves["zero"][s] for s in shared])
    flo = np.array([abs(curves["zero"][s] - curves["zero-seed2"][s]) for s in shared])
    axB.fill_between(shared, -flo, flo, color=BLUE, alpha=0.18, lw=0, zorder=1)
    axB.axhline(0, color=FAINT, lw=1.0, zorder=2)
    axB.plot(shared, eff, color=ORANGE, lw=2.0, marker="o", ms=5.5, mfc="white",
             mew=1.7, zorder=5)
    axB.axvline(step, color=FAINT, lw=1.1, ls=(0, (4, 3)), zorder=1)
    axB.annotate("random − zero", xy=(shared[-1], eff[-1]), xytext=(9, 0),
                 textcoords="offset points", color=ORANGE, fontsize=9.5,
                 va="center", fontweight="bold")
    axB.set_ylabel("difference in held-out loss")
    axB.set_xlim(XMIN, xmax * 1.09)
    axB.set_title("the effect, against the floor the same configuration produces "
                  "under a different seed", loc="left", fontsize=11.5,
                  fontweight="bold", pad=34)
    axB.legend(handles=[
        Line2D([], [], color=ORANGE, lw=2.0, marker="o", ms=5.5, mfc="white",
               mew=1.7, label="random − zero  (the treatment effect)"),
        Patch(facecolor=BLUE, alpha=0.18, label="± |zero − zero-seed2|  (the noise floor)"),
    ], loc="lower left", bbox_to_anchor=(0, 1.015), frameon=False, fontsize=9,
        ncol=2, columnspacing=2.2)

    # ---- C: the three stride bins at the read step -------------------------
    x = np.arange(len(bins)); w = 0.32
    ef = [abs(b["effect_pct"]) for b in bins]; efs = [b["effect_se_pct"] for b in bins]
    fl = [b["floor_pct"] for b in bins];      fls = [b["floor_se_pct"] for b in bins]
    axC.bar(x - w / 2, ef, w, color=ORANGE, yerr=efs, capsize=4,
            error_kw=dict(ecolor=BODY, lw=1.3), zorder=3,
            label="| random − zero |   (paired SE)")
    axC.bar(x + w / 2, fl, w, color=BLUE, yerr=fls, capsize=4,
            error_kw=dict(ecolor=BODY, lw=1.3), zorder=3,
            label="| zero − zero-seed2 |   the floor  (paired SE)")
    axC.axhline(0, color=FAINT, lw=1.0, zorder=2)
    for i, b in enumerate(bins):
        axC.annotate(f"{b['effect_pct']:+.2f}%\n|t| = {b['t']:.1f}",
                     xy=(i - w / 2, ef[i] + efs[i]), xytext=(0, 8),
                     textcoords="offset points", ha="center", fontsize=9,
                     color=BODY, fontweight="bold", linespacing=1.5)
        axC.annotate(f"{b['floor_pct']:.2f}%", xy=(i + w / 2, fl[i] + fls[i]),
                     xytext=(0, 8), textcoords="offset points", ha="center",
                     fontsize=9, color=BODY)
    top = max(e + s for e, s in zip(ef + fl, efs + fls))
    bot = min(0.0, min(e - s for e, s in zip(ef + fl, efs + fls)))
    axC.set_ylim(bot - 0.06 * (top - bot), top + 0.42 * (top - bot))
    axC.set_xticks(x)
    axC.set_xticklabels([f"{b['bin']}   stride {b['stride']}\n"
                         f"n = {b['n']} · available information ≈ {b['info']:.0%}"
                         for b in bins], fontsize=9.5, color=BODY, linespacing=1.6)
    axC.tick_params(axis="x", length=0, pad=10)
    axC.set_xlim(-0.6, len(bins) - 0.4)
    axC.set_ylabel("% of that bin's zero-arm loss")
    axC.set_title(f"at the read (step {step}): the effect is flat across the bins "
                  f"while the available information falls 23% → 13% → 7%",
                  loc="left", fontsize=11.5, fontweight="bold", pad=34)
    axC.legend(loc="lower left", bbox_to_anchor=(0, 1.015), frameon=False,
               fontsize=9, ncol=2, columnspacing=2.2)

    fig.text(0.058, 0.038,
             "source  runs/p009/p009_{random,zero,zero_seed2,none}_<ts>/heldout/step_*.rank*.jsonl   "
             "(loaded by kineidos.read_heldout, the same loader the registered read of §6.2 uses; "
             "this figure is asserted against readout.md before it is drawn)\n"
             "train loss  runs/slurm/p009-kineidos-*.err      "
             "panel C shows the two point estimates §6.2 divides.  Each error bar crosses the "
             "other quantity, and both cross zero (|t| < 2), which is why the ratio is reported "
             "but not interpreted —\n"
             "a floor that is itself unresolved from zero inflates any ratio it divides.",
             fontsize=8, color=FAINT, linespacing=1.9)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, facecolor=fig.get_facecolor())
    return out_png


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--runs", default="runs/p009")
    p.add_argument("--slurm", default="runs/slurm")
    p.add_argument("--step", type=int, default=None,
                   help="the read round; default is the latest present in every arm")
    p.add_argument("--verify", default="artifacts/reports/P009/readout.md")
    p.add_argument("--out", default="artifacts/reports/P009/p009_loss_curves.png")
    p.add_argument("--json", default="artifacts/reports/P009/p009_loss_curves.json")
    a = p.parse_args()

    base = Path(a.runs)
    if not base.is_dir():
        raise SystemExit(f"no run directory {base}; run from the workspace root")

    data, runs = {}, {}
    for arm, prefix in ARMS:
        d = find_run_dir(base, prefix)
        if d is None:
            print(f"  [skip] {arm}: no {prefix}_<timestamp>")
            continue
        rows = load_rows(d)
        if not rows:
            print(f"  [skip] {arm}: {d.name} has no heldout rounds")
            continue
        data[arm], runs[arm] = rows, d
        print(f"  {arm:11s} {d.name}  {len(rows)} rounds, last {max(rows)}")

    need = {"random", "zero", "zero-seed2"}
    if not need <= set(data):
        raise SystemExit(f"need {sorted(need)}, have {sorted(data)}")

    step = a.step or max(set.intersection(*(set(d) for d in data.values())))
    curves = {arm: heldout_curve(rows) for arm, rows in data.items()}
    trains = {arm: train_curve(Path(a.slurm), runs[arm].name) for arm in data}
    bins = bins_at(data, step)
    stops = {arm: max(c) for arm, c in curves.items()}
    print(f"\n  read step {step}; train points: "
          + ", ".join(f"{k} {len(v)}" for k, v in trains.items()))

    if a.verify:
        rp = Path(a.verify)
        if not rp.is_file():
            raise SystemExit(f"--verify {rp} not found; pass --verify '' to skip")
        problems = verify(rp, curves, bins, step)
        if problems:
            print(f"\n{len(problems)} disagreement(s) with {rp}:", file=sys.stderr)
            for q in problems[:20]:
                print("  " + q, file=sys.stderr)
            print("\nthe figure would contradict the registered read; not drawing.",
                  file=sys.stderr)
            return 2
        print(f"  verified against {rp}: curve and three bins agree")

    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(dict(
        step=step, runs={k: v.name for k, v in runs.items()},
        curves={k: {str(s): v for s, v in c.items()} for k, c in curves.items()},
        train={k: v for k, v in trains.items()}, bins=bins, stops=stops,
    ), indent=1, ensure_ascii=False))
    out = draw(curves, trains, bins, step, stops, Path(a.out))
    print(f"  wrote {out}\n  wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
