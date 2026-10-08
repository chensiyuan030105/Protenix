"""The read-out: three dt bins, the noise floor, per system, against the frozen
thresholds -- and the verdict sentence, written before the numbers existed.

The rules are P007's kickoff section "读数规则（事先登记，跑完不许改）" and
P009 section 6.2, and they live here rather than in a later judgement:

  * **Bins** are stride 1-2 / 3-5 / 6-10, i.e. 0.1-0.2 / 0.3-0.5 / 0.6-1.0 ns,
    carrying roughly 23% / 13% / 7% of the no-history error as available
    information.  The same three bins `kineidos.read_heldout` uses, so a window
    sits in the same bin in P009's denoising read and in this one.
  * **The floor** of a bin is |zero - zero-seed2| on that bin, paired window by
    window.  An arm difference smaller than it is reported as
    "indistinguishable" and no difference is quoted.  Without the `zero-seed2`
    arm there is no floor and no verdict -- that is stated, not worked around.
  * **The main read is bin 1**, and `random` counts as useful only if
    |random - zero| exceeds the floor *and* `random` is the lower.  Bins 2 and 3
    say whether the effect decays with dt; it should.
  * **skill = 1 - RMSD_model / RMSD_persistence**, on the same windows.  Not
    beating persistence means the history was not used, whatever the arm
    comparison says (P007 section 3.1).
  * **RMSF r is read per system** against that system's own ceiling in
    `thresholds.json`, never against a global number: MD's own 10 ns ceiling
    ranges from 0.685 to 0.979 over the 16 systems (P007 section 3.4).
  * `none` is reported and kept out of the verdict: it only cross-checks that
    `zero`'s extra Linear is neutral.

The paired standard error is printed beside the rule's comparison, never folded
into it.  P009's read had a case where a ratio of 1.6 over the floor looked like
a finding while neither side was resolved from zero; the pairing is what makes
the error cheap to have, since all arms score the same windows under the same
noise.

    PYTHONPATH=repos/research/kineidos-v3:repos/research/wp-v2 \\
      <env>/bin/python -m kineidos.bench.report \\
        --oneshot runs/p007/oneshot/dryrun \\
        --rollout runs/p007/rollout/dryrun \\
        --out artifacts/reports/P007/dryrun.md
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable, Optional

ARMS = ("random", "zero", "zero-seed2", "none", "pretrained")
BINS = (("bin1", 1, 2, 0.23), ("bin2", 3, 5, 0.13), ("bin3", 6, 10, 0.07))
FLOOR_PAIR = ("zero", "zero-seed2")
EFFECT_PAIR = ("random", "zero")


def bin_of(stride: int) -> Optional[str]:
    for name, lo, hi, _ in BINS:
        if lo <= stride <= hi:
            return name
    return None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def fmt(x: Any, digits: int = 3) -> str:
    if x is None:
        return "—"
    if isinstance(x, float) and not math.isfinite(x):
        return "—"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def mean_or_nan(values: Iterable[float]) -> float:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    return statistics.fmean(vals) if vals else float("nan")


def paired_difference(a: dict[int, float], b: dict[int, float]
                      ) -> tuple[float, float, int]:
    """Mean and standard error of a - b over the window ids both hold."""
    ids = sorted(set(a) & set(b))
    d = [a[i] - b[i] for i in ids
         if a[i] is not None and b[i] is not None
         and math.isfinite(a[i]) and math.isfinite(b[i])]
    if len(d) < 2:
        return (float("nan"), float("nan"), len(d))
    return statistics.fmean(d), statistics.stdev(d) / math.sqrt(len(d)), len(d)


def verdict_for(effect: float, floor: float, *, lower_is_better: bool = True
                ) -> str:
    """The registered rule, unchanged by anything the numbers do."""
    if not math.isfinite(effect) or not math.isfinite(floor):
        return "无噪声底，不判定"
    above = abs(effect) > floor
    better = effect < 0 if lower_is_better else effect > 0
    if above and better:
        return "**random 更好且超噪声底**"
    if above:
        return "超噪声底但 random 更差"
    return "不可区分"


# ------------------------------------------------------------------- one-shot


def oneshot_section(directory: Path, thresholds: dict[str, Any]) -> list[str]:
    baselines = {r["window_id"]: r for r in
                 read_jsonl(directory / "baselines.jsonl")} \
        if (directory / "baselines.jsonl").exists() else {}
    arms: dict[str, dict[int, dict[str, Any]]] = {}
    for arm in ARMS:
        path = directory / f"oneshot.{arm}.jsonl"
        if path.exists():
            arms[arm] = {r["window_id"]: r for r in read_jsonl(path)}
    if not arms:
        return [f"（`{directory}` 下没有 `oneshot.<arm>.jsonl`，跳过单步一节）", ""]

    lines = ["## 单步预测 RMSD（指标 1）", ""]
    meta_rows = []
    for arm in arms:
        meta_path = directory / f"meta.{arm}.json"
        step = ckpt = "—"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            model = meta.get("model", {})
            step = model.get("checkpoint_step", "—")
            ckpt = model.get("checkpoint", "—")
            n_step = model.get("N_step", "—")
            n_sample = model.get("N_sample", "—")
            commits = meta.get("worktrees", {})
            v3 = str(commits.get("kineidos-v3", {}).get("commit", "—"))[:12]
            wp = str(commits.get("wp-v2", {}).get("commit", "—"))[:12]
            meta_rows.append(f"| `{arm}` | {step} | {n_step} | {n_sample} | "
                             f"`{v3}` | `{wp}` | `{Path(str(ckpt)).name}` |")
    if meta_rows:
        lines += ["| 档 | checkpoint step | N_step | N_sample | "
                  "kineidos-v3 | wp-v2 | 文件 |",
                  "|---|---:|---:|---:|---|---|---|", *meta_rows, ""]

    # The pairing, checked rather than assumed.
    ids = set.intersection(*(set(a) for a in arms.values()))
    for arm, rows in arms.items():
        bad = [w for w in sorted(ids)
               if (rows[w]["sample_id"], rows[w]["target_frame"], rows[w]["stride"])
               != (next(iter(arms.values()))[w]["sample_id"],
                   next(iter(arms.values()))[w]["target_frame"],
                   next(iter(arms.values()))[w]["stride"])]
        if bad:
            lines += [f"**配对失败**：`{arm}` 的 window_id {bad[:3]} 指向与其它档"
                      f"不同的窗口。`eval_seed` 必须四档相同，否则下面任何差值都"
                      f"没有意义。", ""]
            return lines
    lines.append(f"{len(ids)} 个窗口在所有档上齐全，逐窗口配对。")
    lines.append("")

    for metric, label, digits in (("rmsd_mean", "RMSD 均值（5 样本）", 3),
                                  ("rmsd_best", "RMSD 最好样本", 3),
                                  ("ensemble_width", "系综宽度（样本两两）", 3),
                                  ("lddt_mean", "lDDT 均值", 4),
                                  ("valid_fraction", "有效帧比例", 3)):
        lines += [f"### {label}", "",
                  "| 箱 | stride | n | " + " | ".join(f"`{a}`" for a in arms)
                  + " | `random−zero` | 配对SE | \\|t\\| | 噪声底 | 比值 | 判定 |",
                  "|---|---|---:|" + "---:|" * (len(arms) + 2) + "---:|---:|---:|---|"]
        for name, lo, hi, _info in BINS:
            cells = []
            per_arm: dict[str, dict[int, float]] = {}
            for arm, rows in arms.items():
                vals = {w: rows[w][metric] for w in ids
                        if bin_of(rows[w]["stride"]) == name}
                per_arm[arm] = vals
                cells.append(fmt(mean_or_nan(vals.values()), digits))
            n = len(per_arm[next(iter(arms))])
            effect = se = float("nan")
            if set(EFFECT_PAIR) <= set(arms):
                effect, se, _ = paired_difference(per_arm[EFFECT_PAIR[0]],
                                                  per_arm[EFFECT_PAIR[1]])
            floor = float("nan")
            if set(FLOOR_PAIR) <= set(arms):
                floor_signed, _, _ = paired_difference(per_arm[FLOOR_PAIR[0]],
                                                       per_arm[FLOOR_PAIR[1]])
                floor = abs(floor_signed)
            ratio = (abs(effect) / floor if math.isfinite(floor) and floor > 0
                     else float("nan"))
            t = abs(effect / se) if se and math.isfinite(se) and se > 0 else float("nan")
            lower_is_better = metric not in ("lddt_mean", "valid_fraction",
                                             "ensemble_width")
            lines.append(
                f"| {name} | {lo}–{hi} | {n} | " + " | ".join(cells) +
                f" | {fmt(effect, digits + 1)} | {fmt(se, digits + 1)} | "
                f"{fmt(t, 1)} | {fmt(floor, digits + 1)} | {fmt(ratio, 2)} | "
                + (verdict_for(effect, floor, lower_is_better=lower_is_better)
                   if metric != "ensemble_width" else "（宽度无方向，只记录）")
                + " |")
        lines.append("")

    # skill against persistence, and the three model-free baselines.
    if baselines:
        lines += ["### 对三条无模型基线的 skill", "",
                  "skill = 1 − RMSD_model / RMSD_baseline，正数表示赢过该基线。"
                  "**赢不了 persistence 就不能说历史有用**（P007 §3.1）。", "",
                  "| 箱 | " + " | ".join(f"`{a}`" for a in arms)
                  + " | persistence | 静态 | replica | skill(pers) | skill(静态) |",
                  "|---|" + "---:|" * (len(arms) + 5)]
        for name, lo, hi, _ in BINS:
            sel = [w for w in sorted(ids)
                   if bin_of(arms[next(iter(arms))][w]["stride"]) == name]
            pers = mean_or_nan(baselines[w]["persistence_rmsd"] for w in sel
                               if w in baselines)
            stat = mean_or_nan(baselines[w]["static_rmsd"] for w in sel
                               if w in baselines)
            rep = mean_or_nan(baselines[w]["replica_rmsd_mean"] for w in sel
                              if w in baselines)
            cells = [fmt(mean_or_nan(arms[a][w]["rmsd_mean"] for w in sel))
                     for a in arms]
            best_arm = EFFECT_PAIR[0] if EFFECT_PAIR[0] in arms else next(iter(arms))
            model = mean_or_nan(arms[best_arm][w]["rmsd_mean"] for w in sel)
            lines.append(
                f"| {name} | " + " | ".join(cells) +
                f" | {fmt(pers)} | {fmt(stat)} | {fmt(rep)} | "
                f"{fmt(1 - model / pers if pers else float('nan'))} | "
                f"{fmt(1 - model / stat if stat else float('nan'))} |")
        lines += ["", f"skill 一列用 `{best_arm}` 档。", ""]

    # per system, bin 1 only: the main read, split the way P007 section 8 asks.
    lines += ["### 按体系分列（箱 1，主读数）", "",
              "| 体系 | n | " + " | ".join(f"`{a}` RMSD" for a in arms)
              + " | persistence | skill |", "|---|---:|" + "---:|" * (len(arms) + 2)]
    systems = sorted({arms[next(iter(arms))][w].get("system", "?") for w in ids})
    for system in systems:
        sel = [w for w in sorted(ids)
               if arms[next(iter(arms))][w].get("system") == system
               and bin_of(arms[next(iter(arms))][w]["stride"]) == "bin1"]
        if not sel:
            continue
        cells = [fmt(mean_or_nan(arms[a][w]["rmsd_mean"] for w in sel)) for a in arms]
        pers = mean_or_nan(baselines[w]["persistence_rmsd"] for w in sel
                           if w in baselines) if baselines else float("nan")
        best_arm = EFFECT_PAIR[0] if EFFECT_PAIR[0] in arms else next(iter(arms))
        model = mean_or_nan(arms[best_arm][w]["rmsd_mean"] for w in sel)
        lines.append(f"| {system} | {len(sel)} | " + " | ".join(cells) +
                     f" | {fmt(pers)} | "
                     f"{fmt(1 - model / pers if pers and math.isfinite(pers) else float('nan'))} |")
    lines.append("")

    # Chain swap and validity, which P007 section 8 item 2 asks for by name.
    lines += ["### 链交换率与有效帧比例", "",
              "| 档 | 链交换率（样本级） | 有效帧比例 | Bond MAE 均值 | "
              "O3′–P 最大偏差均值 | 碰撞 >0 的样本比例 |",
              "|---|---:|---:|---:|---:|---:|"]
    md_bond = thresholds["md_reference"]["all_64"]["bond_mae"]["mean"]
    for arm, rows in arms.items():
        swaps = mean_or_nan(rows[w]["swap_rate"] for w in ids)
        valid = mean_or_nan(rows[w]["valid_fraction"] for w in ids)
        bond = mean_or_nan(v for w in ids for v in rows[w]["bond_mae"])
        o3p = mean_or_nan(v for w in ids for v in rows[w]["o3p_maxdev"])
        clash = mean_or_nan(float(v > 0) for w in ids
                            for v in rows[w]["clash_vdw_count"])
        lines.append(f"| `{arm}` | {fmt(swaps)} | {fmt(valid)} | {fmt(bond, 4)} | "
                     f"{fmt(o3p, 4)} | {fmt(clash)} |")
    lines += ["", f"MD 参照：Bond MAE {md_bond:.4f} Å，O3′–P 最大偏差 "
              f"{thresholds['md_reference']['all_64']['o3p_maxdev']['mean']:.4f} Å，"
              f"碰撞 0。有效帧判据：O3′–P ≤ "
              f"{thresholds['validity']['o3p_maxdev_max_angstrom']:.4f} Å 且碰撞 ≤ "
              f"{thresholds['validity']['clash_vdw_count_max']} 对"
              f"（`thresholds.json`，{thresholds['frozen_on']} 冻结）。", ""]
    if not set(FLOOR_PAIR) <= set(arms):
        lines += [f"> **没有噪声底。** 本目录只有 {sorted(arms)}，缺 "
                  f"`{FLOOR_PAIR[1]}`，所以上面每一张表的「判定」一列都不成立，"
                  f"差值只能当作管线跑通的证据读，不能当作结论"
                  f"（P009 §6.2、P007 读数规则第一条）。", ""]
    return lines


# -------------------------------------------------------------------- rollout


def rollout_section(directory: Path, thresholds: dict[str, Any]) -> list[str]:
    arms: dict[str, list[dict[str, Any]]] = {}
    for arm in ARMS:
        path = directory / f"metrics.{arm}.jsonl"
        if path.exists():
            arms[arm] = read_jsonl(path)
    if not arms:
        return [f"（`{directory}` 下没有 `metrics.<arm>.jsonl`，跳过 rollout 一节）", ""]

    lines = ["## Rollout（指标 2–5 与 sanity 标志）", "",
             "每行一条 rollout。指标 2 与 4 的「valid」列只用有效帧，"
             "「all」列不过滤，两者并列（P007 §3.3）。", ""]
    lines += ["| 档 | 体系 | rep | 帧数 | 有效帧 | 首个无效帧 | Bond MAE | "
              "时滞偏差(valid) | 平台(模型/MD) | RMSF r | 幅度比 | "
              "sanity 首次触发 |",
              "|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|"]
    for arm, rows in arms.items():
        for row in sorted(rows, key=lambda r: (r["system"], r["replicate"])):
            lag = row.get("lag_valid") or {}
            rmsf = row.get("rmsf_valid") or row.get("rmsf_all") or {}
            lines.append(
                f"| `{arm}` | {row['system']} | {row['replicate']} | "
                f"{row['n_frames']} | {fmt(row['valid_fraction'])} | "
                f"{fmt(row.get('first_invalid_step'))} | "
                f"{fmt(row['bond_mae_all'], 4)} | "
                f"{fmt(lag.get('deviation_mean'))} | "
                f"{fmt(lag.get('plateau'))} / {fmt(lag.get('md_plateau'))} | "
                f"{fmt(rmsf.get('r_heavy'))} | {fmt(rmsf.get('amplitude_ratio'), 2)} | "
                f"{fmt(row['sanity'].get('hallucinated_transition_step'))} |")
    lines.append("")

    # Per system against that system's own ceiling.
    lines += ["### RMSF r 与幅度比，按体系对照 `thresholds.json`", "",
              "| 体系 | 档 | RMSF r（模型） | 10 ns 窗天花板 | 查表（须追平） | "
              "主轴地板 | 幅度比（模型） | MD replica 幅度比 | 读数 |",
              "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for arm, rows in arms.items():
        for row in sorted(rows, key=lambda r: (r["system"], r["replicate"])):
            system = row["system"]
            th = thresholds["per_system"].get(system)
            rmsf = row.get("rmsf_valid") or row.get("rmsf_all") or {}
            r = rmsf.get("r_heavy")
            if th is None:
                lines.append(f"| {system} | `{arm}` | {fmt(r)} | "
                             f"— | — | — | {fmt(rmsf.get('amplitude_ratio'), 2)} | "
                             f"— | 体系不在 thresholds.json 里 |")
                continue
            ceiling = th["rmsf_r"]["ceiling_10ns_window_random_heavy"]
            lookup = th["rmsf_r"]["must_match_lookup_heavy"]
            floor = th["rmsf_r"]["floor_principal_axis_heuristic"]
            ratio = rmsf.get("amplitude_ratio")
            md_ratio = th["rmsf_amplitude_ratio"]["r1_over_r4"]
            read = "—"
            if r is not None and math.isfinite(r):
                if r >= ceiling:
                    read = "达到 MD 的 10 ns 天花板"
                elif r >= lookup:
                    read = "超过查表基线"
                elif r >= floor:
                    read = "在地板与查表之间"
                else:
                    read = "**低于主轴地板**"
            lines.append(
                f"| {system} | `{arm}` | {fmt(r)} | {fmt(ceiling)} | {fmt(lookup)} "
                f"| {fmt(floor)} | {fmt(ratio, 2)} | {fmt(md_ratio, 2)} | {read} |")
    lines.append("")

    # Aggregate per arm per system with the across-replicate std P007 section 8
    # item 5 asks for by name.
    lines += ["### 每档每指标按体系分列（两条 rollout 的均值 ± 标准差）", "",
              "| 档 | 体系 | 有效帧 | RMSF r | 幅度比 | 时滞偏差 |",
              "|---|---|---|---|---|---|"]
    for arm, rows in arms.items():
        per: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            per.setdefault(row["system"], []).append(row)
        for system, group in sorted(per.items()):
            def spread(getter) -> str:
                vals = [getter(r) for r in group]
                vals = [v for v in vals if v is not None and math.isfinite(v)]
                if not vals:
                    return "—"
                if len(vals) == 1:
                    return f"{vals[0]:.3f} (n=1)"
                return f"{statistics.fmean(vals):.3f} ± {statistics.stdev(vals):.3f}"
            lines.append(
                f"| `{arm}` | {system} | {spread(lambda r: r['valid_fraction'])} | "
                f"{spread(lambda r: (r.get('rmsf_valid') or r.get('rmsf_all') or {}).get('r_heavy'))} | "
                f"{spread(lambda r: (r.get('rmsf_valid') or r.get('rmsf_all') or {}).get('amplitude_ratio'))} | "
                f"{spread(lambda r: (r.get('lag_valid') or {}).get('deviation_mean'))} |")
    lines.append("")

    # sanity, only where it is defined
    lines += ["### sanity 标志（只对从 I 出发的起点）", "",
              "| 档 | 体系 | rep | U7 翻出 | U18 翻出 | G4·G17 | G6·G15 | "
              "MD 基线（翻出/配对） | 首次连续 10 步 < 0.8 |",
              "|---|---|---:|---:|---:|---:|---:|---|---:|"]
    for arm, rows in arms.items():
        for row in sorted(rows, key=lambda r: (r["system"], r["replicate"])):
            s = row["sanity"]
            if not s.get("applies"):
                continue
            th = thresholds["per_system"].get(row["system"], {}).get("sanity_md", {})
            lines.append(
                f"| `{arm}` | {row['system']} | {row['replicate']} | "
                f"{fmt(s.get('u7_flipped_fraction'))} | "
                f"{fmt(s.get('u18_flipped_fraction'))} | "
                f"{fmt(s.get('g4g17_paired_fraction'))} | "
                f"{fmt(s.get('g6g15_paired_fraction'))} | "
                f"{fmt(th.get('u7_flipped'))} / {fmt(th.get('g4g17_paired'))} | "
                f"{fmt(s.get('hallucinated_transition_step'))} |")
    lines += ["", "χ(G6) / χ(G17) 的 syn 占比记录在 `metrics.<arm>.jsonl` 的 "
              "`sanity.syn_fraction_g4_g6_g15_g17` 里，不打分：构象 I 内部有"
              "syn/anti 子态（P007 §1.3）。", ""]
    return lines


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--oneshot", help="runs/p007/oneshot/<tag>")
    ap.add_argument("--rollout", help="runs/p007/rollout/<tag>")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="P007 读数")
    ap.add_argument("--preamble", help="a markdown file to put before the tables")
    args = ap.parse_args(argv)

    from kineidos.bench.oneshot import load_thresholds

    thresholds = load_thresholds()
    lines = [f"# {args.title}", "",
             f"判据写在 `kineidos.bench.report` 里，跑之前就定下来了；阈值与上下界"
             f"来自 `artifacts/reports/P007/thresholds.json`"
             f"（{thresholds['frozen_on']} 冻结，{thresholds['source']['n_trajectories']} "
             f"条 MD、{thresholds['source']['n_systems']} 个体系）。", ""]
    if args.preamble:
        lines += [Path(args.preamble).read_text().rstrip(), ""]
    if args.oneshot:
        lines += oneshot_section(Path(args.oneshot), thresholds)
    if args.rollout:
        lines += rollout_section(Path(args.rollout), thresholds)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
