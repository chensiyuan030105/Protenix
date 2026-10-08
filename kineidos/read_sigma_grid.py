#!/usr/bin/env python
"""P010 section 6's readings, off the per-sigma jsonl.  Registered in advance.

The same discipline as kineidos/read_heldout.py: what counts as an effect is a
property of this file, written before the numbers are in, rather than of
whatever the numbers turn out to look like.  Section 6, item by item:

  * **Three bands, by c_skip** (section 1's table): high is sigma > 24.4
    (c_skip < 0.3, 14% of the training draw, `h` is the only extra
    information there is), mid is 5.3-24.4 (the skip term and the network's
    output have to agree on a reference frame), low is sigma < 5.3 (the output
    is mostly x_noisy copied through, so no history could matter).  The band is
    carried in each row by the scorer, not recomputed here.
  * **Paired, on (window_id, sigma, noise_idx).**  Every arm uses
    eval_seed=1234 and the same fixed window set, so the same key is the same
    window under the same noise at the same noise level.  That the keys agree
    is checked rather than assumed: if they do not, the per-band means compare
    different problems and everything below is void.
  * **The floor is |zero - zero-seed2| on that band**, not a standard error on
    a mean.  P004's reason stands: `none` -> `zero` is provably the identity at
    step 0 and still moved the held-out loss 0.31-1.70%, while `zero` ->
    `random` -- 73.45M parameters and a real h -- moved it 0.03%.
  * **The reading** (item 1): the high band above its floor with `random`
    lower means "h carries history"; all three bands the same sign and
    magnitude means "what we are reading is capacity", reported as such; the
    high band also below its floor means indistinguishable and the oracle gate
    decides.
  * **The oracle gate** (item 2): high-band paired MSE down more than 50% from
    `random-1gpu`, *and* `oracle-decoy`'s same number no larger than
    |zero-1gpu - random-1gpu|.  Both halves, because a drop that the decoy
    reproduces is the pipeline being busy rather than information arriving.

Also reports, because section 4 asks for it, the three dt bins inside each
band -- the effect should decay with dt, and one that does not is more likely
to be something else (P009 section 6.2 registered that in advance too).

`--identical A B` is section 4's and section 6 item 5b's acceptance instead of
a reading: it compares two scorings byte for byte and, if they differ, says
which key and by how much, which is the difference between "the tool is
nondeterministic" and "the fusion point moved".

Run as a slurm job, not on the login node -- AGENTS.md / P006 D19, and the
full set is ten arms x 11k rows:

    sbatch repos/research/kineidos-v3-diag/kineidos/slurm/p010_readout.sbatch
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Optional

# At module level, and all of them.  The grid's definition lives in
# score_sigma_grid and is imported rather than restated: a second copy would
# let the reader and the scorer disagree about which z a sigma stands for, and
# the z column is what the whole weighting argument rests on.  Importing them
# one at a time inside whichever function needed them next cost two jobs
# (2149439, 2149440) to two NameErrors.
from kineidos.score_sigma_grid import (P_MEAN, P_STD, SIGMA_DATA, band_of,
                                       c_skip_of)

BANDS = ("high", "mid", "low")
# stride bins, as P009 section 6.2 cut them: dt = stride * 0.1 ns.
DT_BINS = (("bin1", 1, 2), ("bin2", 3, 5), ("bin3", 6, 10))
METRICS = ("mse_aligned", "smooth_lddt", "loss_unweighted", "loss_edm_weighted")


def key_of(row: dict[str, Any]) -> tuple[int, float, int]:
    """What makes two rows the same measurement in two arms.

    sigma is rounded to 6 significant figures before it becomes part of a key.
    It is written to the file as a float and read back exactly, so this is not
    about round-tripping -- it is so that a grid recomputed from the same
    constants on another machine still joins.
    """
    return (int(row["window_id"]), float(f"{row['sigma']:.6g}"),
            int(row["noise_idx"]))


def load(path: Path) -> dict[tuple[int, float, int], dict[str, Any]]:
    out: dict[tuple[int, float, int], dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        k = key_of(row)
        if k in out:
            raise SystemExit(
                f"{path}: {k} appears twice. Ranks shard as "
                f"eval_windows[rank::world_size] and the grid is fixed, so a "
                f"key cannot collide unless the shard rule or the grid changed "
                f"mid-run."
            )
        out[k] = row
    return out


def load_arm(base: Path, arm: str, step: int) -> dict[tuple, dict[str, Any]]:
    """One arm's rows, ranks concatenated, with every window complete.

    The completeness check is not pedantry.  A `background` preemption kills
    the scorer mid-arm and leaves a jsonl holding whatever it had written --
    on 2026-10-08 two such files sat in runs/p010/sigma_grid next to two
    complete ones, at 3659 and 3720 rows against 11264.  A truncated arm reads
    as a complete arm over a smaller window set, which silently shrinks the
    paired set for *every* arm and moves every number in the table.

    The scorer writes row by row within a window, so a kill lands inside one:
    that window has fewer than len(sigmas) x n_noise rows, and the whole grid
    is present for every other window.  Refusing on an incomplete window
    catches the truncation at the only place it is visible.
    """
    paths = sorted(base.glob(f"{arm}_step{step}.rank*.jsonl"))
    if not paths:
        raise SystemExit(f"no {arm}_step{step}.rank*.jsonl under {base}")
    merged: dict[tuple, dict[str, Any]] = {}
    for path in paths:
        for k, row in load(path).items():
            if k in merged:
                raise SystemExit(f"{path}: {k} already came from another rank")
            merged[k] = row

    per_window: dict[int, int] = collections.Counter(
        k[0] for k in merged)
    sigmas = {k[1] for k in merged}
    noises = {k[2] for k in merged}
    full = len(sigmas) * len(noises)
    short = sorted(w for w, c in per_window.items() if c != full)
    if short:
        raise SystemExit(
            f"{arm} step {step}: window(s) {short[:3]} carry "
            f"{[per_window[w] for w in short[:3]]} rows, not the "
            f"{len(sigmas)} sigmas x {len(noises)} noise = {full} the rest do. "
            f"That is a scoring that was killed part way -- `background` has no "
            f"grace period and the file holds whatever had been written. Rerun "
            f"the arm (the sweep skips arms with a .done marker, so it will "
            f"redo only this one); do not read a truncated arm as a complete "
            f"one over fewer windows, which would shrink the paired set for "
            f"every arm."
        )
    return merged


def check_pairing(arms: dict[str, dict[tuple, dict[str, Any]]],
                  *, allow_ragged: bool = False) -> list[tuple]:
    """The shared keys, with "same key means same window" enforced."""
    sizes = {name: len({k[0] for k in rows}) for name, rows in arms.items()}
    if len(set(sizes.values())) > 1 and not allow_ragged:
        raise SystemExit(
            f"the arms scored different numbers of windows: "
            f"{dict(sorted(sizes.items(), key=lambda kv: kv[1]))}. Either an "
            f"arm is missing rounds, or kineidos.sigma_grid_windows differed "
            f"between them -- both are legitimate (a mid-training round may "
            f"score a prefix of the set) and both change what the table is "
            f"over, so say which with --allow-ragged rather than having it "
            f"decided by whichever arm was shortest."
        )
    common = sorted(set.intersection(*(set(a) for a in arms.values())))
    if not common:
        raise SystemExit("no (window, sigma, noise) key is present in every arm")
    ref_name = sorted(arms)[0]
    ref = arms[ref_name]
    for name, rows in arms.items():
        bad = [k for k in common
               if (rows[k]["sample_id"], rows[k]["stride"],
                   rows[k]["target_frame"])
               != (ref[k]["sample_id"], ref[k]["stride"], ref[k]["target_frame"])]
        if bad:
            raise SystemExit(
                f"{name}: key {bad[:3]} names a different window than in "
                f"{ref_name}. eval_seed must be 1234 and eval_windows must "
                f"match in every arm; otherwise the arms are not paired and no "
                f"difference below means anything."
            )
    for name, rows in arms.items():
        extra = len(rows) - len(common)
        if extra:
            print(f"  note: {name} has {extra} rows outside the shared set")
    return common


def paired(a: dict[tuple, dict], b: dict[tuple, dict], keys: list[tuple],
           metric: str) -> dict[str, float]:
    """mean(a) - mean(b) over `keys`, with the *paired* standard error.

    Paired, because the windows and the noise are shared: the standard error of
    the mean difference is the spread of the per-key differences, which is far
    smaller than either arm's own spread, and using the unpaired one would
    throw away the reason the seeds are fixed.
    """
    diffs = [a[k][metric] - b[k][metric] for k in keys]
    base = statistics.fmean(b[k][metric] for k in keys)
    mean = statistics.fmean(diffs)
    se = (statistics.stdev(diffs) / math.sqrt(len(diffs))
          if len(diffs) > 1 else float("nan"))
    return {
        "n": len(diffs), "mean": mean, "base": base,
        "rel": mean / base * 100.0 if base else float("nan"),
        "se": se, "t": abs(mean) / se if se and se == se and se > 0 else float("nan"),
    }


def band_keys(rows: dict[tuple, dict], keys: list[tuple], band: str,
              dt_bin: Optional[tuple[str, int, int]] = None) -> list[tuple]:
    out = [k for k in keys if rows[k]["sigma_band"] == band]
    if dt_bin is not None:
        _, lo, hi = dt_bin
        out = [k for k in out if lo <= rows[k]["stride"] <= hi]
    return out


def table(arms: dict[str, dict[tuple, dict]], keys: list[tuple],
          *, high: str, low: str, floor: Optional[tuple[str, str]],
          metric: str, split_dt: bool = False) -> list[str]:
    """One band-by-band table of `high - low`, against an optional floor."""
    ref = arms[low]
    head = (f"| 段 | {'Δt 箱 | ' if split_dt else ''}n | `{low}` | "
            f"`{high}−{low}` | 配对SE | \\|t\\| | ")
    head += "噪声底 | 底的SE | 比值 |" if floor else "|"
    lines = [head, "|---|" + ("---|" if split_dt else "")
             + "---:|---:|---:|---:|---:|"
             + ("---:|---:|---:|" if floor else "")]
    rows_out = []
    for band in BANDS:
        bins = list(DT_BINS) if split_dt else [None]
        for b in bins:
            ks = band_keys(ref, keys, band, b)
            if not ks:
                continue
            eff = paired(arms[high], arms[low], ks, metric)
            cell = (f"| {band} | " + (f"{b[0]} | " if split_dt else "")
                    + f"{eff['n']} | {eff['base']:.4f} | "
                    f"{eff['mean']:+.4f} ({eff['rel']:+.2f}%) | "
                    f"±{eff['se']:.4f} | {eff['t']:.1f} | ")
            if floor:
                fl = paired(arms[floor[0]], arms[floor[1]], ks, metric)
                ratio = (abs(eff["mean"]) / abs(fl["mean"])
                         if fl["mean"] else float("inf"))
                cell += (f"{abs(fl['mean']):.4f} ({abs(fl['rel']):.2f}%) | "
                         f"±{fl['se']:.4f} | {ratio:.2f} |")
                rows_out.append((band, b, eff, fl, ratio))
            else:
                cell += ""
                rows_out.append((band, b, eff, None, None))
            lines.append(cell)
    return lines, rows_out


def read_item1(arms: dict[str, dict[tuple, dict]], keys: list[tuple],
               names: dict[str, str]) -> list[str]:
    """Section 6 item 1, with the verdict spelled out rather than left to a
    reader comparing two columns."""
    out: list[str] = []
    for metric in ("loss_edm_weighted", "mse_aligned"):
        out.append(f"\n### 第 1 条 · `{metric}`\n")
        lines, rows = table(arms, keys, high=names["random"], low=names["zero"],
                            floor=(names["zero"], names["floor"]),
                            metric=metric)
        out += lines
        verdict = {}
        for band, _, eff, fl, ratio in rows:
            above = abs(eff["mean"]) > abs(fl["mean"])
            verdict[band] = (above, eff["mean"] < 0, eff, fl)
            if fl["t"] == fl["t"] and fl["t"] < 2.0:
                out.append(
                    f"\n> **{band} 段的噪声底没有从零分辨出来**（|t| = "
                    f"{fl['t']:.1f} < 2），所以该段的「比值」是在除一个与零无法"
                    f"区分的数，不可解读。")
        hi = verdict.get("high")
        lo = verdict.get("low")
        if hi and lo:
            if hi[0] and hi[1] and not lo[0]:
                out.append("\n**判定：`h` 带历史信息** —— 高 σ 段超底且 "
                           "`random` 更低，低 σ 段不超底（§6 第 1 条第一支）。")
            elif all(v[0] for v in verdict.values()) and len(
                    {v[1] for v in verdict.values()}) == 1:
                out.append("\n**判定：读到的是容量** —— 三段同号同量级，"
                           "如实写（§6 第 1 条第二支）。")
            elif not hi[0]:
                out.append("\n**判定：不可区分** —— 高 σ 段也不超底，"
                           "去向由 §6 第 2 条的 oracle 门决定（第三支）。")
            else:
                out.append("\n**判定：三支都不完全吻合**，逐条写出上表再议；"
                           "不要把它归到最近的一支。")
    return out


def per_sigma(arms: dict[str, dict[tuple, dict]], keys: list[tuple],
              *, high: str, low: str, floor: Optional[tuple[str, str]],
              metric: str) -> list[str]:
    """One row per grid point, which is what the three bands average over.

    The bands are section 6's registered cut and stay the verdict, but they can
    hide the shape inside themselves.  The high band holds only two grid points
    -- z = 1.5 and z = 2.0, sigma 45.7 and 96.8 -- while the training draw puts
    most of its high-band mass near z = 1.1-1.5, so the band mean over-weights
    the extreme tail by construction.  If a band effect turns out to live
    entirely at sigma 96.8, that is 2.3% of the draw and a noise level at which
    a 470-atom RNA is not a structure any more; the band table alone cannot say
    so.

    Also reported: `w`, each grid point's share of the training draw, as the
    Gaussian mass of the z-interval it stands for.  It is what the held-out
    loss weights by and the grid does not, which is the whole reason the two
    can disagree on the same checkpoint.
    """
    sigmas = sorted({k[1] for k in keys})
    sigma_data = SIGMA_DATA
    lines = [f"| σ (Å) | z | 段 | c_skip | 训练抽样占比 | n | `{low}` | "
             + rf"`{high}−{low}` | 配对SE | \|t\| | "
             + ("噪声底 | 比值 |" if floor else "|"),
             "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|"
             + ("---:|---:|" if floor else "")]
    # Each grid point stands for the z-interval halfway to its neighbours, so
    # the weights sum to one over the grid's span.  Not a claim that the grid
    # is a quadrature rule -- it is here so a reader can see which rows the
    # held-out average is actually made of.
    zs = [(math.log(sig / sigma_data) - P_MEAN) / P_STD for sig in sigmas]
    edges = [-math.inf] + [(a + b) / 2 for a, b in zip(zs, zs[1:])] + [math.inf]
    def phi(z: float) -> float:
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    for i, sig in enumerate(sigmas):
        ks = [k for k in keys if k[1] == sig]
        if not ks:
            continue
        eff = paired(arms[high], arms[low], ks, metric)
        w = phi(edges[i + 1]) - phi(edges[i])
        row = (f"| {sig:.2f} | {zs[i]:+.2f} | {band_of(sig)} | "
               f"{c_skip_of(sig):.3f} | {w * 100:.1f}% | {eff['n']} | "
               f"{eff['base']:.4f} | {eff['mean']:+.4f} ({eff['rel']:+.2f}%) | "
               f"±{eff['se']:.4f} | {eff['t']:.1f} | ")
        if floor:
            fl = paired(arms[floor[0]], arms[floor[1]], ks, metric)
            ratio = (abs(eff["mean"]) / abs(fl["mean"])
                     if fl["mean"] else float("inf"))
            row += f"{abs(fl['rel']):.2f}% | {ratio:.2f} |"
        lines.append(row)
    return lines


def read_item2(arms: dict[str, dict[tuple, dict]], keys: list[tuple],
               names: dict[str, str]) -> list[str]:
    """Section 6 item 2: the oracle gate, both halves."""
    need = ("oracle", "decoy", "random1", "zero1")
    if not all(names.get(k) and names[k] in arms for k in need):
        return ["\n### 第 2 条 · oracle 门\n",
                "oracle / decoy / random-1gpu / zero-1gpu 尚未全部就位，跳过。"]
    out = ["\n### 第 2 条 · oracle 门（`mse_aligned`，高 σ 段）\n"]
    lines, _ = table(arms, keys, high=names["oracle"], low=names["random1"],
                     floor=None, metric="mse_aligned")
    out += lines
    ks = band_keys(arms[names["random1"]], keys, "high")
    gate = paired(arms[names["oracle"]], arms[names["random1"]], ks,
                  "mse_aligned")
    decoy = paired(arms[names["decoy"]], arms[names["random1"]], ks,
                   "mse_aligned")
    pipe = paired(arms[names["zero1"]], arms[names["random1"]], ks,
                  "mse_aligned")
    drop = -gate["rel"]
    out.append(
        f"\n- oracle 对 `random-1gpu` 的高 σ 段配对 MSE 变化："
        f"**{gate['rel']:+.1f}%**（判据：下降 > 50%）"
        f"\n- decoy 的同一数字：**{decoy['rel']:+.1f}%**；"
        f"`|zero-1gpu − random-1gpu|` = **{abs(pipe['rel']):.1f}%**"
        f"（判据：decoy 不超过它）")
    passed_drop = drop > 50.0
    passed_decoy = abs(decoy["rel"]) <= abs(pipe["rel"])
    if passed_drop and passed_decoy:
        out.append("\n**门通过：通路能带信息。** 去 §6 第 3 条。")
    elif passed_drop and not passed_decoy:
        out.append("\n**oracle 与 decoy 同量级下降 ⟹ 管道效应，读数作废，"
                   "查管道**（§6 第 2 条第四行）。")
    elif not passed_drop:
        out.append("\n**门未通过：梯度到得了 `W_h` 却学不会读 —— 先验在抗。** "
                   "直接去 P011（§6 第 2 条第三行）。中 σ 段与 "
                   "`oracle-noaug` 的对照决定「共用旋转」是否列入必做。")
    return out


# `arm` is the file's own label -- the tree it came from, or the checkpoint's
# run directory -- so it differs by construction in exactly the comparisons
# that matter.  Section 4's acceptance compares `step0_none` against
# `step0_zero`, and section 6 item 5b compares `identity_...-diag` against
# `identity_...-diag-oracle`.  Byte comparison would report both as different
# on that field alone and say nothing about the numbers.
IDENTITY_IGNORE = ("arm",)


def flatten(row: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Every leaf of a row, so nothing is compared by accident or skipped."""
    out: dict[str, Any] = {}
    for key, value in row.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten(value, f"{name}."))
        else:
            out[name] = value
    return out


def identical(a: Path, b: Path) -> int:
    """Are these two scorings the same measurement, field by field.

    Exact equality on every leaf of every row, joined on (window_id, sigma,
    noise_idx) -- not a tolerance.  Section 4 and section 6 item 5b both ask
    for "逐位一致", and a tolerance here would hide the one failure these
    checks exist for: a change that moves the RNG stream by a single draw
    leaves every number plausible and every arm unpaired.

    Byte identity is reported as well, because it is the stronger statement and
    it is what the repeat-determinism check should see -- but it is not the
    verdict, since `arm` differs by construction in the cross-arm comparisons.
    """
    ta, tb = a.read_bytes(), b.read_bytes()
    byte_same = ta == tb
    print(f"  bytes: {'identical' if byte_same else f'{len(ta)} vs {len(tb)}'}")

    ra, rb = load(a), load(b)
    only = set(ra) ^ set(rb)
    if only:
        print(f"  [FAIL] {len(only)} (window, sigma, noise) keys are in one "
              f"file only, e.g. {sorted(only)[:3]}")
        return 1

    worst: dict[str, tuple[float, tuple]] = {}
    unequal: dict[str, tuple[Any, Any, tuple]] = {}
    for k in sorted(ra):
        fa, fb = flatten(ra[k]), flatten(rb[k])
        if set(fa) != set(fb):
            print(f"  [FAIL] {k} has different fields: "
                  f"{sorted(set(fa) ^ set(fb))}")
            return 1
        for field, va in fa.items():
            if field in IDENTITY_IGNORE:
                continue
            vb = fb[field]
            if va == vb:
                continue
            unequal.setdefault(field, (va, vb, k))
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                d = abs(va - vb)
                if d > worst.get(field, (0.0, None))[0]:
                    worst[field] = (d, k)

    if not unequal:
        print(f"  [PASS] {a.name} and {b.name} agree exactly on every field of "
              f"every one of {len(ra)} rows"
              + ("" if byte_same else f" (ignoring {list(IDENTITY_IGNORE)})"))
        return 0

    print(f"  [FAIL] {len(unequal)} fields differ over {len(ra)} rows")
    for field, (va, vb, k) in sorted(unequal.items()):
        if field in worst:
            d, kk = worst[field]
            print(f"    {field:24s} max |delta| = {d:.3e}  at {kk}")
        else:
            print(f"    {field:24s} {va!r} vs {vb!r}  at {k}")
    print("    a nonzero delta here is either the tool being "
          "nondeterministic or the code path having moved; the magnitudes say "
          "which -- a handful of 1e-7 scattered over rows is arithmetic, an "
          "offset on every row is not, and a difference in a non-numeric "
          "field means the two runs did not score the same windows.")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", default="runs/p010/sigma_grid")
    parser.add_argument("--step", type=int, default=2999)
    parser.add_argument("--out", default="artifacts/reports/P010/readout.md")
    parser.add_argument("--identical", nargs=2, metavar=("A", "B"),
                        help="two jsonl paths to compare field by field")
    parser.add_argument("--allow-ragged", action="store_true",
                        help="accept arms that scored different numbers of "
                             "windows, and read over the intersection")
    # Arm names as they appear in the file names.  Defaults are P009's four at
    # a checkpoint; the diagnostic arms are passed in once they exist.
    parser.add_argument("--random", default="p009_random")
    parser.add_argument("--zero", default="p009_zero")
    parser.add_argument("--floor", default="p009_zero_seed2")
    parser.add_argument("--none", default="p009_none")
    parser.add_argument("--oracle", default="")
    parser.add_argument("--decoy", default="")
    parser.add_argument("--random1", default="")
    parser.add_argument("--zero1", default="")
    args = parser.parse_args()

    if args.identical:
        return identical(Path(args.identical[0]), Path(args.identical[1]))

    base = Path(args.runs)
    names = {k: getattr(args, k) for k in
             ("random", "zero", "floor", "none", "oracle", "decoy",
              "random1", "zero1")}
    arms: dict[str, dict[tuple, dict]] = {}
    for role, arm in names.items():
        if not arm:
            continue
        if arm in arms:
            continue
        try:
            arms[arm] = load_arm(base, arm, args.step)
        except SystemExit as exc:
            print(f"  [skip] {role} = {arm}: {exc}")
            continue
        print(f"  {role:8s} {arm:22s} {len(arms[arm])} rows")

    required = [names["random"], names["zero"], names["floor"]]
    if not all(r in arms for r in required):
        print(f"\nthe reading needs {required}; have {sorted(arms)}. The floor "
              f"arm is not optional -- without it there is nothing to compare "
              f"an effect against.")
        return 2

    keys = check_pairing(arms, allow_ragged=args.allow_ragged)
    print(f"  paired on {len(keys)} (window, sigma, noise) keys")

    out = [f"# P010 逐 σ 读数（§6 的规则，step {args.step}）\n",
           "判据在 `kineidos.read_sigma_grid` 里，跑之前就写好了。三段按 c_skip 切"
           "（高 σ > 24.4 Å、中 5.3–24.4、低 < 5.3），各档在 "
           "`(window_id, sigma, noise_idx)` 上配对。\n",
           "| 角色 | 档 | 行数 |", "|---|---|---:|"]
    for role, arm in names.items():
        if arm and arm in arms:
            out.append(f"| {role} | `{arm}` | {len(arms[arm])} |")
    out += read_item1(arms, keys, names)
    out += read_item2(arms, keys, names)

    for metric in ("loss_edm_weighted", "mse_aligned"):
        out.append(f"\n### 多报 · 逐 σ（`{metric}`），三段平均的是这些行\n")
        out.append("「训练抽样占比」是该网格点代表的 z 区间在训练对数正态下的"
                   "质量 —— 留出 loss 按它加权，而本表等权，这是同一个 "
                   "checkpoint 上两者可能不一致的全部原因。\n")
        out += per_sigma(arms, keys, high=names["random"], low=names["zero"],
                         floor=(names["zero"], names["floor"]), metric=metric)

    out.append("\n### 多报 · 三段 × 三箱 Δt（`loss_edm_weighted`）\n")
    lines, _ = table(arms, keys, high=names["random"], low=names["zero"],
                     floor=(names["zero"], names["floor"]),
                     metric="loss_edm_weighted", split_dt=True)
    out += lines

    if names["none"] in arms:
        out.append("\n### 多报 · 管道效应 `zero − none` 按段\n")
        out.append("P009 §8 把它测成全程 +1.66%。若它也集中在高 σ 段，"
                   "那么第 2 条里「decoy 不超过 |zero − random|」这一条的"
                   "量级要按该段的值读，不是按全程的。\n")
        lines, _ = table(arms, keys, high=names["zero"], low=names["none"],
                         floor=None, metric="loss_edm_weighted")
        out += lines

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n")
    print(f"\nwrote {path}")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
