#!/usr/bin/env python
"""P009 acceptance: what dt distribution the dataloader actually produces.

Section 6.1's second condition, in the form that can fail.  The quantity P009
rests on is the expected `max_gain` -- how much of the no-history error the
history could remove, averaged over the windows the model is trained on -- and
there are two ways to compute it.  One is to integrate the measured curve
analytically over [0.1, 1] ns, which is what `measure_decorrelation.mean_gain`
does; it returns 14.9% whatever the dataloader is doing, because the dataloader
is not one of its inputs.  The other is the one here: draw windows through the
real `GAGUWindowDataset.__getitem__`, histogram the strides they come out with,
and weight the measured curve by that histogram.  That number is 15.2% -- the
discrete value, higher than the continuous one because rounding to an integer
stride moves mass towards the short lags -- and it moves if anything in the
draw is wrong.  A dt_max left at 100, a clamp in `stride_for` firing, a seed
that does not reach the sampler: all of them change this and none of them change
mean_gain.

`build_window` is stubbed out, and nothing else is.  The draw is
`rng -> sample -> dt -> stride -> target`, and build_window is called after the
last of those, so stubbing it removes the coordinate work and leaves every
decision intact.  The samples themselves are loaded for real, because
`stride_for` reads `frame_interval_ns` and `n_frames` off them and standing in
for those two numbers would be standing in for the answer.

Also checks what section 6.1's fourth condition asks of the held-out set:
EVAL_WINDOWS=256 has to put at least 50 windows in each of the three dt bins,
and the thin bin is stride 10 at 2.2%.  Worth knowing before four arms spend
18 hours each.

Heavy enough for slurm (48 trajectories, ~5.4 GB); see kineidos/slurm/.

Run from the workspace root:

    PYTHONPATH=repos/research/kineidos-v3:repos/research/wp-v2 \
    LAYERNORM_TYPE=torch ATTN_IMPL=sdpa \
      <env>/bin/python -m kineidos.check_dt_range
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from kineidos.data import windows as windows_module
from kineidos.data.gagu import GAGUProtenixAdapter
from kineidos.data.windows import DT_MAX_NS, DT_MIN_NS, GAGUWindowDataset

# The bins P009 section 6.2 registered in advance, and the floor section 6.1's
# fourth condition puts under each of them.
BINS = (("bin1 stride 1-2 (0.1-0.2 ns)", 1, 2),
        ("bin2 stride 3-5 (0.3-0.5 ns)", 3, 5),
        ("bin3 stride 6-10 (0.6-1.0 ns)", 6, 10))
MIN_PER_BIN = 50

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def gain_curve(decorrelation: Path, system: str = "GAGU"):
    """max_gain(dt) by log interpolation of the measured curve.

    Log, not linear: the grid is log-spaced (strides 1, 2, 3, 4, 7, 11, ...), so
    strides 5, 6, 8, 9 and 10 are not measured points and linear interpolation
    across a factor-of-1.6 gap would sit noticeably low.  Same interpolation
    `mean_gain` uses, on the same rows -- only the weights differ.
    """
    blocks = json.loads(decorrelation.read_text())
    rows = next(b for b in blocks if b["system"] == system)["rows"]
    x = np.log10([r["dt_ns"] for r in rows])
    y = np.array([r["max_gain"] for r in rows])
    lo, hi = float(x[0]), float(x[-1])

    def gain_at(dt_ns: float) -> float:
        g = np.log10(dt_ns)
        if not lo <= g <= hi:
            raise SystemExit(
                f"dt={dt_ns} ns is outside the measured curve "
                f"[{10 ** lo:.3g}, {10 ** hi:.3g}] ns; rerun "
                f"kineidos.measure_decorrelation before trusting an "
                f"extrapolated gain"
            )
        return float(np.interp(g, x, y))

    return gain_at


def log_uniform_share(stride: int, interval_ns: float,
                      dt_min: float, dt_max: float) -> float:
    """The share of a LogUniform[dt_min, dt_max] draw that rounds to `stride`.

    `stride_for` rounds to nearest, so stride s claims
    [(s-1/2)*interval, (s+1/2)*interval] intersected with the range -- which is
    why stride 1 gets 17.6% rather than a tenth and stride 10 gets 2.2%.  This
    is the analytic expectation the measured histogram is compared against; it
    is not used to produce the headline number.
    """
    lo = max((stride - 0.5) * interval_ns, dt_min)
    hi = min((stride + 0.5) * interval_ns, dt_max)
    if hi <= lo:
        return 0.0
    return float(np.log(hi / lo) / np.log(dt_max / dt_min))


def draw(samples, *, length: int, seed: int, k: int) -> list[SimpleNamespace]:
    """`length` windows out of the real dataset, without the coordinate work."""
    real_build_window = windows_module.build_window

    def stub(sample, target_frame, *, stride, k, canonicalize=True, **kwargs):
        return SimpleNamespace(sample_id=sample.sample_id, target_frame=target_frame,
                               stride=stride,
                               delta_t_ns=stride * sample.frame_interval_ns)

    windows_module.build_window = stub
    try:
        # No dt_min_ns / dt_max_ns here on purpose: the trainer does not pass
        # them either (trainer.py's init_data), so the module constants are
        # what is under test.
        dataset = GAGUWindowDataset(samples, k=k, length=length, seed=seed,
                                    canonicalize=True)
        return [dataset[i] for i in range(len(dataset))]
    finally:
        windows_module.build_window = real_build_window


def histogram(drawn) -> dict[int, int]:
    counts: dict[int, int] = {}
    for w in drawn:
        counts[int(w.stride)] = counts.get(int(w.stride), 0) + 1
    return dict(sorted(counts.items()))


def report(label: str, drawn, gain_at, interval_ns: float) -> dict:
    counts = histogram(drawn)
    n = len(drawn)
    expected = 0.0
    print(f"\n  {label}: {n} windows")
    print(f"  {'stride':>6} {'dt (ns)':>8} {'count':>6} {'share':>7} "
          f"{'analytic':>9} {'max_gain':>9}")
    for stride, count in counts.items():
        dt = stride * interval_ns
        share = count / n
        gain = gain_at(dt)
        expected += share * gain
        print(f"  {stride:>6} {dt:>8.1f} {count:>6} {share:>6.1%} "
              f"{log_uniform_share(stride, interval_ns, DT_MIN_NS, DT_MAX_NS):>8.1%} "
              f"{gain:>8.1%}")
    print(f"  expected max_gain over the drawn windows: {expected:.1%}")

    bins = {}
    for name, lo, hi in BINS:
        in_bin = [c for s, c in counts.items() if lo <= s <= hi]
        weight = sum(
            log_uniform_share(s, interval_ns, DT_MIN_NS, DT_MAX_NS)
            * gain_at(s * interval_ns) for s in range(lo, hi + 1)
        ) / max(sum(log_uniform_share(s, interval_ns, DT_MIN_NS, DT_MAX_NS)
                    for s in range(lo, hi + 1)), 1e-12)
        bins[name] = {"count": sum(in_bin), "stride_range": [lo, hi],
                      "weighted_max_gain": weight}
        print(f"  {name}: {sum(in_bin)} windows, available information "
              f"{weight:.0%}")
    return {"n": n, "counts": {str(k): v for k, v in counts.items()},
            "expected_max_gain": expected, "bins": bins}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gagu-root", default="/mnt/xfs/home/mhg/Projects/ForSiyuan/"
                        "RNA-WorldParticle-Workspace/datasets/processed/"
                        "gagu_internal_loop_v0_1")
    parser.add_argument("--train-glob", default="*_r[123]")
    parser.add_argument("--held-out-glob", default="*_r4")
    parser.add_argument("--train-windows", type=int, default=10_000)
    parser.add_argument("--eval-windows", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42,
                        help="configs.seed, which seeds the training draw")
    parser.add_argument("--eval-seed", type=int, default=1234,
                        help="kineidos.eval_seed, which seeds the held-out draw")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--decorrelation",
                        default="artifacts/reports/P009/decorrelation.json")
    parser.add_argument("--out", default="artifacts/reports/P009/dt_histogram.json")
    parser.add_argument("--expect", type=float, default=0.152,
                        help="P009 section 2.4's discrete expectation")
    parser.add_argument("--tol", type=float, default=0.005)
    args = parser.parse_args()

    print("=== 1. the constants under test ===")
    check("DT_MIN_NS is the save interval", DT_MIN_NS == 0.1, f"{DT_MIN_NS}")
    check("DT_MAX_NS is 1.0 ns (P009 section 2.3)", DT_MAX_NS == 1.0, f"{DT_MAX_NS}")

    root = Path(args.gagu_root)
    train_dirs = sorted(root.glob(args.train_glob))
    held_dirs = sorted(root.glob(args.held_out_glob))
    print(f"\n=== 2. loading {len(train_dirs)} train + {len(held_dirs)} "
          f"held-out trajectories ===")
    if not train_dirs or not held_dirs:
        raise SystemExit(f"no trajectories under {root}")
    train = [GAGUProtenixAdapter(d).load() for d in train_dirs]
    held = [GAGUProtenixAdapter(d).load() for d in held_dirs]
    interval = train[0].frame_interval_ns
    check("every trajectory shares one save interval",
          all(s.frame_interval_ns == interval for s in train + held),
          f"{interval} ns")
    print(f"  {train[0].n_frames} frames each, {interval} ns apart")

    gain_at = gain_curve(Path(args.decorrelation))

    print("\n=== 3. the training draw ===")
    train_stats = report(f"train (seed={args.seed})",
                         draw(train, length=args.train_windows, seed=args.seed,
                              k=args.k), gain_at, interval)
    strides = [int(s) for s in train_stats["counts"]]
    check("every stride is an integer in 1..10",
          min(strides) >= 1 and max(strides) <= 10, f"{min(strides)}..{max(strides)}")
    got = train_stats["expected_max_gain"]
    check(f"expected max_gain is {args.expect:.1%} +/- {args.tol:.1%}",
          abs(got - args.expect) <= args.tol, f"{got:.2%}")

    print("\n=== 4. the held-out set, and the three bins ===")
    eval_stats = report(f"held-out (eval_seed={args.eval_seed})",
                        draw(held, length=args.eval_windows,
                             seed=args.eval_seed, k=args.k), gain_at, interval)
    for name, info in eval_stats["bins"].items():
        check(f"{name} has at least {MIN_PER_BIN} windows",
              info["count"] >= MIN_PER_BIN, f"{info['count']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "dt_min_ns": DT_MIN_NS, "dt_max_ns": DT_MAX_NS,
        "frame_interval_ns": interval,
        "k": args.k,
        "decorrelation": str(args.decorrelation),
        "train": {"seed": args.seed, "n_trajectories": len(train), **train_stats},
        "held_out": {"seed": args.eval_seed, "n_trajectories": len(held),
                     **eval_stats},
        "expected_max_gain_target": args.expect,
        "passed": not FAILS,
    }, indent=2) + "\n")
    print(f"\nwrote {out}")

    print(f"\n{len(FAILS)} failure(s)" + (": " + "; ".join(FAILS) if FAILS else ""))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
