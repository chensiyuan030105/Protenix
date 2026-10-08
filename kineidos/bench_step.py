#!/usr/bin/env python
"""Where a training step's time actually goes, so a budget is measured not guessed.

P009's four arms run at 6-7.5 s/step, which puts 10,000 steps at 17-21 hours and
makes every follow-up experiment a day long.  Three knobs could cut that, and
they are not equally free:

  * **the mini-rollout** -- 20 no-grad diffusion forwards every training step
    (`N_step_mini_rollout`, protenix.py's `if not self.skip_mini_rollout`).  With
    `skip_confidence` already true it has exactly one remaining effect: it
    permutes the label to match itself, which is the 30% chain permutation P009
    section 8.4 wants gone.  So turning it off buys speed *and* removes a
    confound -- but it changes the objective, so not on a running arm.
  * **the recycling depth** -- training draws `N_cycle ~ U{1..10}` from
    `RandomState(step)`, mean 5.5, each cycle a full 48-block Pairformer pass.
    Pinning it low is a protocol change and also removes a nuisance variable
    (P004 already pinned it for held-out scoring, for that reason).
  * **caching the frozen trunk** -- the trunk sees only `base_features`, which
    are identical for every window of a sample, so its output could be computed
    once per sample instead of every step.  What stops this being free is
    `pairformer.dropout = 0.25`: frozen parameters still get dropout in train
    mode, so the conditioning is stochastic and a cache would quietly mean
    "trunk dropout off".  That may well be the better protocol for an ablation
    measuring sub-percent effects -- STAR-MD precomputes its frozen OpenFold
    features for exactly this reason -- but it is a decision, not a free lunch.

Timed on one GPU with explicit `n_cycle`, so the depth is the variable rather
than a lottery, and with `torch.cuda.synchronize` around each phase.

Run from the workspace root, on a GPU (P010 D9):

    PYTHONPATH=repos/research/kineidos-v3-diag:repos/research/wp-v2 \
    LAYERNORM_TYPE=torch ATTN_IMPL=sdpa LD_LIBRARY_PATH=$ENV/lib \
      $ENV/bin/python -u -m kineidos.bench_step --mode random --out <path>

### What the first attempt got wrong (job 2149158, 2026-10-07)

It hit the one-hour limit having written six lines of output, the last of them
from DiffusionModule's constructor, and no json at all.  Three separate causes,
all fixed here, and the order matters because the first one hid the other two:

1. **Nothing said where it was.**  Between the last constructor print and the
   first timing line sit a checkpoint load, a trajectory load and a window
   build, and a reader could not tell which of them the hour went into.  Every
   phase now prints its own elapsed time, and every timed row is appended to
   the json as it is measured, so a run that is killed still reports what it
   had.  `--max-seconds` makes it stop on its own terms instead.
2. **The sweep had no budget.**  Eight configurations times (warmup + repeat)
   at `N_cycle=10` with a 48-sample diffusion batch and the rollout on is well
   over an hour on a contended node; the defaults here are cut to fit, and the
   budget check stops before starting a row it cannot finish.
3. **The configuration was written onto an already-parsed config.**  Three of
   those five assignments are read in a constructor and two at forward time, so
   "is this too late" had a different answer per line and no error either way
   (`configs.diffusion_batch_size = db` inside the last loop was read in
   Protenix.__init__ and so measured nothing -- all three rows were the same
   48-sample step under three different labels).  They now go through
   make_configs' `overrides`, which merges before parse, and the one value that
   genuinely is a runtime knob is set on the model where the model reads it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

T0 = time.perf_counter()


def say(message: str) -> None:
    """Progress with the clock attached, flushed.

    Both halves are load-bearing: without the timestamp a log cannot say which
    phase the wall time went into, and without the flush a job killed by the
    time limit loses the lines that would have said so.
    """
    elapsed = time.perf_counter() - T0
    print(f"[+{int(elapsed) // 60:02d}:{int(elapsed) % 60:02d}] {message}",
          flush=True)


def timed(fn, *, warmup: int, repeat: int) -> float:
    """Median seconds per call, cuda-synchronised."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        out.append(time.perf_counter() - t0)
    out.sort()
    return out[len(out) // 2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", default="random", choices=["none", "zero", "random"])
    parser.add_argument("--sample", default="gagu_100mM_K_agaguu_startI_r1")
    parser.add_argument("--cycles", default="1,4,10")
    parser.add_argument("--diffusion-batch", default="48,24")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    # The budget, not the queue's.  A row started with less than its own
    # estimated cost left is skipped, so the json that survives is complete
    # rows rather than a truncated one.
    parser.add_argument("--max-seconds", type=float, default=5400.0)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("needs a GPU: the point is the relative cost of the phases, and "
              "that does not transfer from CPU")
        return 2

    from protenix.config.extend_types import ListValue
    from protenix.utils.torch_utils import to_device

    from kineidos.data.gagu import GAGUProtenixAdapter
    from kineidos.data.windows import GAGUWindowDataset
    from kineidos.seeding import set_all_seeds
    from kineidos.train.batch import collate_window
    from kineidos.train.check_trainer import CHECKPOINT, make_configs
    from kineidos.train.trainer import KineidosTrainer

    import tempfile
    work = tempfile.mkdtemp(prefix="p010_bench_")
    set_all_seeds(0)

    # make_configs is built for a CPU smoke test; put the production numbers
    # back, or the timings describe the test and not the run.  Before parse,
    # not after -- see make_configs' docstring and the module note above.
    #
    # The triangle kernels come from the environment with the same default and
    # the same two names kineidos/train/main.py uses, so this script and the
    # training runs cannot silently disagree about which kernel was timed, and
    # a login-node invocation can drop to "torch" without editing the file.
    overrides = {
        "triangle_attention": os.environ.get("TRIANGLE_ATTENTION",
                                             "cuequivariance"),
        "triangle_multiplicative": os.environ.get("TRIANGLE_MULTIPLICATIVE",
                                                  "cuequivariance"),
        "diffusion_batch_size": 48,
        "model": {"N_cycle": 10},
        "sample_diffusion": {"N_step_mini_rollout": 20},
        # ListValue, not a bare list: the parser infers each key's dtype from
        # the default it is replacing, and a plain list where a ListValue is
        # expected loses that.
        "kineidos": {"train_samples": ListValue([args.sample])},
    }
    configs = make_configs(args.mode, work, max_steps=1, overrides=overrides)
    say(f"config parsed: triangle_attention={configs.triangle_attention} "
        f"diffusion_batch_size={configs.diffusion_batch_size} "
        f"N_cycle={configs.model.N_cycle}")

    trainer = KineidosTrainer(configs)
    model = trainer.raw_model
    say("trainer built, checkpoint loaded")

    root = __import__("pathlib").Path(configs.kineidos.gagu_root)
    sample = GAGUProtenixAdapter(root / args.sample).load()
    say(f"trajectory loaded: {args.sample}")
    window = GAGUWindowDataset([sample], k=8, length=1, seed=0)[0]
    batch = to_device(collate_window(window), trainer.device)
    say("window collated and on device")

    print(f"mode={args.mode}  {window.wp_position_nm.shape[1]} atoms  "
          f"stride={window.stride}  device={torch.cuda.get_device_name(0)}",
          flush=True)
    print(f"pairformer blocks={configs.model.pairformer.n_blocks}  "
          f"diffusion_batch_size={configs.diffusion_batch_size}  "
          f"N_step_mini_rollout={configs.sample_diffusion.N_step_mini_rollout}\n",
          flush=True)

    def forward(n_cycle: int):
        def go():
            trainer.optimizer.zero_grad(set_to_none=True)
            pred, label, _ = model(
                input_feature_dict=batch["input_feature_dict"],
                label_dict=batch["label_dict"],
                label_full_dict=batch["label_full_dict"],
                mode="train", current_step=0,
                symmetric_permutation=trainer.symmetric_permutation,
                n_cycle=n_cycle,
            )
            loss, _ = trainer.loss(feat_dict=batch["input_feature_dict"],
                                   pred_dict=pred, label_dict=label, mode="train")
            loss.backward()
        return go

    rows = []

    def record(row: dict) -> None:
        """Append to the json on every row, not once at the end.

        The previous version wrote the file after the last measurement, so the
        timeout threw away every row it had already paid for.
        """
        rows.append(row)
        if args.out:
            __import__("pathlib").Path(args.out).write_text(
                json.dumps(rows, indent=2) + "\n")

    def budget_left(estimate: float) -> bool:
        left = args.max_seconds - (time.perf_counter() - T0)
        if left < estimate:
            say(f"skipping: {left:.0f} s left, this row needs about "
                f"{estimate:.0f} s")
            return False
        return True

    cycles = [int(c) for c in args.cycles.split(",")]
    calls = args.warmup + args.repeat
    # Nothing is known about the step cost before the first row, so the first
    # estimate is deliberately generous and every later one uses what has been
    # measured, scaled by this row's cycle count.
    per_call = 30.0
    for skip in (False, True):
        model.skip_mini_rollout = skip
        for nc in cycles:
            if not budget_left(calls * per_call * max(nc, 1) / max(cycles[0], 1)):
                continue
            t = timed(forward(nc), warmup=args.warmup, repeat=args.repeat)
            per_call = t / max(nc, 1)
            record({"skip_mini_rollout": skip, "n_cycle": nc,
                    "diffusion_batch_size": 48, "sec_per_step": t})
            say(f"  skip_mini_rollout={str(skip):<5} N_cycle={nc:>2}  "
                f"{t:6.2f} s/step")

    print(flush=True)
    model.skip_mini_rollout = False
    for db in [int(x) for x in args.diffusion_batch.split(",")]:
        # On the model, not on the config.  Protenix.__init__ copies
        # configs.diffusion_batch_size into self.diffusion_batch_size
        # (protenix.py:159) and main_train_loop reads the attribute
        # (protenix.py:871), so assigning to the config here would leave every
        # row at 48 and label them 48 / 24 anyway.
        if not budget_left(calls * per_call * 10):
            continue
        model.diffusion_batch_size = db
        t = timed(forward(10), warmup=args.warmup, repeat=args.repeat)
        record({"skip_mini_rollout": False, "n_cycle": 10,
                "diffusion_batch_size": db, "sec_per_step": t})
        say(f"  diffusion_batch_size={db:>3} (N_cycle=10, rollout on)  "
            f"{t:6.2f} s/step")
    model.diffusion_batch_size = configs.diffusion_batch_size

    # The decomposition the three knobs are argued from.  Each is a difference
    # of measured totals rather than a separate timing, so the parts add up to
    # the whole by construction and no phase is double counted.
    def total(skip, nc, db=48):
        for r in rows:
            if (r["skip_mini_rollout"] == skip and r["n_cycle"] == nc
                    and r["diffusion_batch_size"] == db):
                return r["sec_per_step"]
        return float("nan")

    print("\n=== 分解 ===", flush=True)
    hi, lo = max(cycles), min(cycles)
    per_cycle = (total(True, hi) - total(True, lo)) / (hi - lo)
    print(f"  每一层 recycling（无 rollout，{lo}->{hi} 的斜率）: {per_cycle:.3f} s")
    print(f"  mini-rollout（N_cycle={hi} 时开关之差）      : "
          f"{total(False, hi) - total(True, hi):.3f} s")
    print(f"  其余（扩散头 fwd+bwd 等，N_cycle={lo} 无 rollout 减去 "
          f"{lo} 层 recycling）: {total(True, lo) - lo * per_cycle:.3f} s")
    production = (total(True, lo) + (5.5 - lo) * per_cycle
                  + (total(False, hi) - total(True, hi)))
    print(f"\n  生产口径（训练 N_cycle 平均 5.5、rollout 开）估计: "
          f"{production:.2f} s/step")
    print(f"  实测四档是 6.0-7.5 s/step（3 卡 DDP，含数据与同步开销）", flush=True)

    if args.out:
        # Rewritten once more with the derived numbers appended, so the file
        # carries the budget P010 section 8 is rewritten from and not only the
        # raw rows.
        __import__("pathlib").Path(args.out).write_text(json.dumps({
            "rows": rows,
            "derived": {
                "sec_per_recycling_layer": per_cycle,
                "sec_mini_rollout": total(False, hi) - total(True, hi),
                "sec_other_at_n_cycle_%d" % lo: total(True, lo) - lo * per_cycle,
                "sec_per_step_production_estimate": production,
            },
            "meta": {
                "mode": args.mode,
                "device": torch.cuda.get_device_name(0),
                "triangle_attention": configs.triangle_attention,
                "warmup": args.warmup, "repeat": args.repeat,
                "wall_seconds": time.perf_counter() - T0,
            },
        }, indent=2) + "\n")
        say(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
