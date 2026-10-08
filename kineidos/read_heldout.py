#!/usr/bin/env python
"""P009 section 6.2's read, written before the numbers are in.

The rule is registered in advance and this is where it lives, so that "what
counts as an effect" is a property of the code rather than of whatever the
numbers turn out to look like.  Section 6.2, verbatim in behaviour:

  * Three bins: stride 1-2, 3-5, 6-10.  Available information roughly
    23% / 13% / 7%.
  * The noise floor of a bin is |zero - zero-seed2| on that bin.  It is not an
    error bar on a mean: P004 found that `none` -> `zero`, a transformation that
    is provably the identity at step 0, moved the held-out loss 0.31-1.70%,
    while `zero` -> `random` -- 73.45M parameters and a real h -- moved it
    0.03%.  Anything smaller than that floor is not measurable here.
  * The read is bin 1.  `random` counts as useful only if |random - zero|
    exceeds that bin's floor *and* random is the lower of the two.  Bins 2 and
    3 say whether the effect decays with dt, which is what should happen; an
    effect that does not decay is more likely something else.
  * No bin above its floor: report "indistinguishable", report no difference,
    and do not narrow the range further -- the next step is the oracle probe of
    section 8, item 1.
  * The ratio reported is loss difference over the bin's floor.  Not over the
    23%: the held-out loss measures denoising over 48 noise samples, only some
    of which sit where the history could help, so the loss difference is much
    smaller than the available information (P004 section 2.18).

`none` is reported but kept out of the verdict; it only cross-checks that
`zero`'s extra Linear is neutral.

The arms are paired window by window.  eval_seed is 1234 in every arm, so all
four score the same 256 windows in the same order under the same noise, and
window_id joins them.  That the window sets agree is checked, not assumed: if
they do not, the per-bin means are comparing different problems and everything
below is void.

Run from the workspace root:

    PYTHONPATH=repos/research/kineidos-v3:repos/research/wp-v2 \
      <env>/bin/python -m kineidos.read_heldout --out artifacts/reports/P009/readout.md
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path

# name -> the RUN_NAME prefix the sbatch gives it.  Order is the priority order
# of P009 section 5.
ARMS = (("random", "p009_random"), ("zero", "p009_zero"),
        ("zero-seed2", "p009_zero_seed2"), ("none", "p009_none"))
BINS = (("bin1", 1, 2, 0.23), ("bin2", 3, 5, 0.13), ("bin3", 6, 10, 0.07))
METRIC = "loss"


def refuse_oracle(run_dir: Path) -> None:
    """Refuse a run that was fed the answer.  P010 D2, item 8.

    P010's oracle arms hand the diffusion head a fixed random projection of the
    target frame as `h`.  That is label leakage by construction -- it is the
    point: the arm asks whether the pathway can carry information at all, and
    its numbers are meaningless as a measure of anything else.  Those arms live
    on a branch that is never merged (D2 item 9), which is the structural half
    of the protection; this is the other half, because a *run directory* can be
    copied anywhere and this file takes `--runs` as an argument.

    Two tests, because either alone can be defeated by a rename: the directory
    name, and `wp.mode` in the run's own env.lock.  A verdict is never produced
    from a run that trips either.
    """
    if "oracle" in run_dir.name.lower():
        raise SystemExit(
            f"refusing to read {run_dir}: its name says oracle. The oracle "
            f"arms are fed the target frame, so a held-out verdict built from "
            f"them would be a measurement of the leak. They belong in P010's "
            f"own readout, not in section 6.2's."
        )
    lock = run_dir / "env.lock"
    if not lock.is_file():
        return
    try:
        mode = json.loads(lock.read_text()).get("wp", {}).get("mode", "")
    except (OSError, json.JSONDecodeError):
        return
    if "oracle" in str(mode).lower():
        raise SystemExit(
            f"refusing to read {run_dir}: env.lock records wp.mode={mode!r}. "
            f"See the directory-name case above -- the mode is checked as well "
            f"because a rename defeats either test alone."
        )


def find_run_dir(base: Path, prefix: str) -> Path | None:
    """The newest run directory for one arm.

    Anchored on the timestamp init_basics appends, because a plain prefix match
    makes `p009_zero` swallow `p009_zero_seed2` -- which would silently read the
    noise-floor arm as the baseline and make the floor zero.
    """
    pattern = re.compile(rf"^{re.escape(prefix)}_\d{{8}}_\d{{6}}$")
    hits = sorted((d for d in base.iterdir()
                   if d.is_dir() and pattern.match(d.name)), key=lambda d: d.name)
    if not hits:
        return None
    refuse_oracle(hits[-1])
    return hits[-1]


def load_rows(run_dir: Path) -> dict[int, dict[int, dict]]:
    """{step: {window_id: row}} for one arm, ranks concatenated."""
    out: dict[int, dict[int, dict]] = {}
    heldout = run_dir / "heldout"
    if not heldout.is_dir():
        return out
    for path in sorted(heldout.glob("step_*.rank*.jsonl")):
        step = int(path.name.split("_")[1].split(".")[0])
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            bucket = out.setdefault(step, {})
            wid = int(row["window_id"])
            if wid in bucket:
                raise SystemExit(
                    f"{path}: window_id {wid} appears twice at step {step}. "
                    f"Ranks shard as eval_windows[rank::world_size], so ids "
                    f"cannot collide unless the shard rule or world size "
                    f"changed mid-run."
                )
            bucket[wid] = row
    return out


def bin_of(stride: int) -> str | None:
    for name, lo, hi, _ in BINS:
        if lo <= stride <= hi:
            return name
    return None


def permutation_rates(slurm_dir: Path, run_name: str) -> dict[str, float]:
    """The last reported permutation rates for one run, out of its slurm log.

    Section 6.1's sixth condition.  The rates ride in the periodic metrics line,
    so the log is the only place they exist; the run directory does not keep
    them.
    """
    rates: dict[str, float] = {}
    if not slurm_dir.is_dir():
        return rates
    for path in sorted(slurm_dir.glob("p009-kineidos-*")):
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        if f"run name: {run_name}" not in text:
            continue
        for line in text.splitlines():
            if "train metrics:" not in line:
                continue
            for key, value in re.findall(
                    r"'(train/[^']*perm[^']*)':\s*(?:np\.float64\()?"
                    r"(-?\d+\.?\d*(?:e[-+]?\d+)?)", line):
                rates[key] = float(value)
    return rates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", default="runs/p009")
    parser.add_argument("--slurm", default="runs/slurm")
    parser.add_argument("--step", type=int, default=None,
                        help="which evaluation round to judge on; default is "
                             "the latest round present in every arm")
    parser.add_argument("--metric", default=METRIC)
    parser.add_argument("--out", default="artifacts/reports/P009/readout.md")
    args = parser.parse_args()

    base = Path(args.runs)
    if not base.is_dir():
        raise SystemExit(f"no run directory {base}; run from the workspace root")

    runs: dict[str, Path] = {}
    data: dict[str, dict[int, dict[int, dict]]] = {}
    for arm, prefix in ARMS:
        run_dir = find_run_dir(base, prefix)
        if run_dir is None:
            print(f"  [skip] {arm}: no {prefix}_<timestamp> under {base}")
            continue
        rows = load_rows(run_dir)
        if not rows:
            print(f"  [skip] {arm}: {run_dir.name} has no heldout/ rounds yet")
            continue
        runs[arm] = run_dir
        data[arm] = rows
        print(f"  {arm:11s} {run_dir.name}  rounds: "
              f"{sorted(rows)[:3]}{'...' if len(rows) > 3 else ''} "
              f"({len(rows)} total, {len(rows[max(rows)])} windows at step "
              f"{max(rows)})")

    needed = {"random", "zero", "zero-seed2"}
    if not needed <= set(data):
        print(f"\nthe read needs {sorted(needed)}; have {sorted(data)}. "
              f"The floor arm (zero-seed2) is not optional -- without it there "
              f"is nothing to compare |random - zero| against.")
        return 2

    common = sorted(set.intersection(*(set(d) for d in data.values())))
    if not common:
        print("\nno evaluation round is present in every arm yet")
        return 2
    step = args.step if args.step is not None else common[-1]
    if step not in common:
        raise SystemExit(f"step {step} is not in every arm; shared: {common}")

    # The pairing, checked.  Same window_id must mean the same window.
    ref = data["zero"][step]
    for arm in data:
        other = data[arm][step]
        if set(other) != set(ref):
            raise SystemExit(
                f"{arm} scored a different window set at step {step}: "
                f"{len(other)} vs {len(ref)} ids. eval_seed must be 1234 in "
                f"every arm and EVAL_WINDOWS must match; otherwise the arms "
                f"are not paired and no difference below means anything."
            )
        bad = [w for w in ref if (other[w]["stride"], other[w]["sample_id"],
                                  other[w]["target_frame"])
               != (ref[w]["stride"], ref[w]["sample_id"], ref[w]["target_frame"])]
        if bad:
            raise SystemExit(
                f"{arm}: window_id {bad[:3]} names a different window than in "
                f"zero at step {step}. The held-out draw is not reproducible "
                f"across arms, so the comparison is void."
            )

    lines = [f"# P009 读数（§6.2 的规则，step {step}）", "",
             f"判据在 `kineidos.read_heldout` 里，跑之前就写好了。指标 "
             f"`{args.metric}`，{len(ref)} 个留出窗口，四档逐窗口配对"
             f"（`eval_seed=1234` 四档相同，`window_id` 对齐）。", ""]
    lines.append("| 档 | 运行目录 |")
    lines.append("|---|---|")
    for arm, _ in ARMS:
        if arm in runs:
            lines.append(f"| `{arm}` | `{runs[arm].name}` |")
    lines.append("")

    def bin_mean(arm: str, name: str) -> tuple[float, int]:
        vals = [r[args.metric] for w, r in data[arm][step].items()
                if bin_of(r["stride"]) == name]
        return (statistics.fmean(vals) if vals else float("nan")), len(vals)

    def paired(a: str, b: str, name: str) -> tuple[float, float, int]:
        """Mean and standard error of the per-window difference a - b.

        Reported alongside section 6.2's rule, not instead of it.  The rule
        compares |random - zero| against |zero - zero-seed2|, and both sides are
        single estimates: at step 499 the first was 0.0082 and the second 0.0051,
        a ratio of 1.6 that the rule reads as "above the floor" -- while the
        standard errors were 0.0061 and 0.0074, so neither side was resolved
        from zero and a floor that happened to come out small made a
        non-significant effect look like a finding.  The pairing is what makes
        this cheap: every arm scores the same windows under the same noise, so
        the per-window difference drops the between-trajectory variance that
        dominates the raw loss (std 0.254 against 0.054, a factor of 4.7).
        """
        ids = [w for w, r in data[b][step].items() if bin_of(r["stride"]) == name]
        d = [data[a][step][w][args.metric] - data[b][step][w][args.metric]
             for w in ids]
        if len(d) < 2:
            return (float("nan"), float("nan"), len(d))
        return statistics.fmean(d), statistics.stdev(d) / math.sqrt(len(d)), len(d)

    verdicts = []
    lines += ["## 三箱", "",
              "| 箱 | stride | n | 可用信息 | `zero` | `random` | "
              "`random−zero` | 配对SE | \\|t\\| | "
              "噪声底 \\|zero−seed2\\| | 底的SE | 比值 | 判定 |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for name, lo, hi, info in BINS:
        z, n = bin_mean("zero", name)
        r, _ = bin_mean("random", name)
        s2, _ = bin_mean("zero-seed2", name)
        effect, se_effect, _ = paired("random", "zero", name)
        floor_signed, se_floor, _ = paired("zero", "zero-seed2", name)
        floor = abs(floor_signed)
        ratio = abs(effect) / floor if floor > 0 else float("inf")
        above = abs(effect) > floor
        lower = effect < 0
        # Section 6.2's registered rule, unchanged.  The t statistics are
        # reported next to it, not folded into it.
        verdict = ("**random 更低且超噪声底**" if above and lower
                   else "超噪声底但 random 更高" if above
                   else "不可区分")
        t_effect = abs(effect / se_effect) if se_effect else float("nan")
        verdicts.append((name, above, lower, effect, floor, ratio, z,
                         se_effect, t_effect, se_floor))
        lines.append(
            f"| {name} | {lo}–{hi} | {n} | {info:.0%} | {z:.4f} | {r:.4f} | "
            f"{effect:+.4f} ({effect / z:+.2%}) | ±{se_effect:.4f} | "
            f"{t_effect:.1f} | {floor:.4f} ({floor / z:.2%}) | "
            f"±{se_floor:.4f} | {ratio:.2f} | {verdict} |")
    lines.append("")
    # Said once, in the report, because a ratio of two unresolved estimates is
    # the way this read goes wrong -- and it goes wrong from either side.  At
    # step 2499 bin 3's floor came out 0.0010 +/- 0.0066, |t| = 0.1, and the
    # ratio was 12.63; the registered rule read that as "above the floor" while
    # the denominator was indistinguishable from zero.  Checking only the
    # effect's |t|, as this did at first, is silent on exactly that case.
    weak_effect = [v[0] for v in verdicts if v[8] < 2.0]
    weak_floor = [v[0] for v in verdicts
                  if v[9] > 0 and abs(v[4] / v[9]) < 2.0]
    if weak_effect:
        lines += [
            f"> **{', '.join(weak_effect)} 的效应没有从零分辨出来**（|t| < 2）。", ""]
    if weak_floor:
        lines += [
            f"> **{', '.join(weak_floor)} 的噪声底没有从零分辨出来**（|t| < 2），"
            f"所以这些箱的「比值」是在除一个与零无法区分的数，不可解读——"
            f"底偶然抽小就会把任何效应放大成一个大比值。", ""]
    if weak_effect or weak_floor:
        lines += [
            "> §6.2 的「比值」是两个点估计相除，而两边各有自己的误差棒。"
            "判据按登记的规则给出，不改；以上是多报的诊断。", ""]

    if "none" in data:
        lines += ["## `none`（不进主结论，只复核 `zero` 的额外 Linear 是中性的）", "",
                  "| 箱 | `none` | `zero−none` | 该箱噪声底 |",
                  "|---|---:|---:|---:|"]
        for name, _, _, _ in BINS:
            nn, _ = bin_mean("none", name)
            z, _ = bin_mean("zero", name)
            s2, _ = bin_mean("zero-seed2", name)
            lines.append(f"| {name} | {nn:.4f} | {z - nn:+.4f} "
                         f"({(z - nn) / z:+.2%}) | {abs(z - s2):.4f} |")
        lines.append("")

    name, above, lower, effect, floor, ratio, z, se_effect, t_effect, se_floor = \
        verdicts[0]
    lines += ["## 判定", ""]
    floor_resolved = se_floor > 0 and abs(floor / se_floor) >= 2.0
    if above and lower and not floor_resolved:
        lines += [
            f"按 §6.2 的规则，箱 1 的差超过噪声底（{ratio:.2f} 倍）且方向正确，"
            f"**但那个底是 {floor:.4f} ± {se_floor:.4f}，|t| = "
            f"{abs(floor / se_floor) if se_floor else float('nan'):.1f}，"
            f"与零无法区分**。比值因此不可解读：这不是「效应超过了底」，"
            f"而是「底这一次抽小了」。需要的是把底测准——每档多个种子，"
            f"并把数据种子与初始化种子分开（§8 第 9 条）——而不是据此宣布结论。"]
    elif above and lower:
        decay = [v for v in verdicts[1:]]
        # Whether the bin-to-bin change is resolved, not just its sign: at step
        # 499 the effect looked like it grew with dt (+0.42% / +0.82% / +1.02%)
        # while bin1 - bin3 was -0.0112 +/- 0.0086, |t| = 1.3 -- no trend at all.
        spread = effect - verdicts[-1][3]
        se_spread = math.hypot(se_effect, verdicts[-1][7])
        t_spread = abs(spread / se_spread) if se_spread else float("nan")
        if t_spread < 2.0:
            shape = (f"**三箱之间没有分辨出差别**（箱1−箱3 = {spread:+.4f} "
                     f"± {se_spread:.4f}，|t| = {t_spread:.1f}），所以不要把"
                     f"箱间的升降当成 Δt 依赖")
        elif all(abs(v[3]) <= abs(effect) for v in decay):
            shape = "随 Δt 衰减，与预期一致"
        else:
            shape = "**不随 Δt 衰减——先怀疑读到了别的东西**（§6.2）"
        lines += [
            f"箱 1 上 `random` 比 `zero` 低 {abs(effect):.4f}"
            f"（{abs(effect) / z:.2%}），是该箱噪声底的 {ratio:.2f} 倍，"
            f"方向也对。按 §6.2 这算 **WP 注入有用**。", "",
            f"箱 2、3 的效应 {shape}。"]
    elif above:
        lines += [f"箱 1 上两档的差超过噪声底（{ratio:.2f} 倍），但 **`random` 更高**。"
                  f"按 §6.2 这不算「注入有用」。"]
    elif not any(v[1] for v in verdicts):
        lines += [
            "**三箱都不超过噪声底 → 不可区分。** 按 §6.2 不报差值。",
            "",
            "结论是 **Δt 不是唯一瓶颈**。下一步是 §8 第 1 条的 oracle 探针"
            "（把 `h` 换成由目标帧自己构造的特征，走同一条融合路径，"
            "再配一个「最近一帧线性嵌入」的地板），**不是**继续收窄到 "
            "`[0.1, 0.4]`——那只剩 4 个 stride，Δt 条件会退化成常数（§2.3）。",
            "",
            "读数时留 §8 第 2 条的余地：max_gain 是线性单帧预测器的指示值，"
            "不是上限；模型非线性、看 K=8 帧，真实可利用的信息应当更多。"]
    else:
        hit = [v[0] for v in verdicts if v[1]]
        lines += [f"箱 1 不可区分，但 {', '.join(hit)} 超过噪声底。"
                  f"§6.2 把主读数定在箱 1，所以这不算「注入有用」；"
                  f"长 Δt 上出现效应而短 Δt 上没有，与「历史带来信息」"
                  f"的方向相反，应当先怀疑读到了别的东西。"]
    lines.append("")

    lines += ["## 置换率（§6.1 第 6 条）", ""]
    any_rate = False
    for arm, _ in ARMS:
        if arm not in runs:
            continue
        rates = permutation_rates(Path(args.slurm), runs[arm].name)
        flags = {k: v for k, v in rates.items() if "is_permuted" in k}
        if flags:
            any_rate = True
            lines.append(f"- `{arm}`: " + ", ".join(
                f"`{k.split('/', 1)[1]}` = {v:.3f}" for k, v in sorted(flags.items())))
        else:
            lines.append(f"- `{arm}`: 日志里没有找到置换率")
    if any_rate:
        lines += ["",
                  "不为零则下一版关掉 `symmetric_permutation`（§8 第 4 条）："
                  "GAGU 两条链序列相同但构象不对称，而历史 `h` 是逐原子的，"
                  "置换一旦发生，模型按链 A 的历史预测、却被拿去和链 B 对比。"]
    lines.append("")

    lines += ["## 各轮曲线", "",
              "| step | " + " | ".join(f"`{a}`" for a, _ in ARMS if a in data)
              + " |",
              "|---|" + "---:|" * sum(1 for a, _ in ARMS if a in data)]
    for s in common:
        cells = []
        for arm, _ in ARMS:
            if arm not in data:
                continue
            vals = [r[args.metric] for r in data[arm][s].values()]
            cells.append(f"{statistics.fmean(vals):.4f}")
        lines.append(f"| {s} | " + " | ".join(cells) + " |")
    lines.append("")

    text = "\n".join(lines)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    print()
    print(text)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
