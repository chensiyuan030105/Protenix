"""P009 acceptance: the per-window held-out rows actually land, and say enough.

Section 4's item 6 is the plan's only new implementation, and it has a property
that makes a late discovery expensive: it runs for the first time at the end of
the first evaluation round, which on a real arm is step 499 -- about an hour in,
four arms at once.  Anything wrong there is wrong four times before anyone
looks.  So one evaluation round is run here on CPU, with four windows instead of
256, and the file it writes is read back.

What is checked is not "a file appeared".  The file is the input to section
6.2's read, so the fields that read needs have to be present and sane: a stride
in 1..10 with a delta_t_ns that agrees with it, the loss terms the bins are
compared on, and a window_id that is unique across ranks.  The last of those is
the one that cannot be checked here -- this runs single-rank -- so the global
index is asserted against the sharding rule instead.

Also checks that the permutation rate reached the metric aggregator (section 4's
item 5), since the same evaluation round is the first place a reader would look
for it.

Run from the workspace root; needs the pretrained checkpoint and ~15 minutes on
CPU.  kineidos/slurm/p009_checks.sbatch does not include it (it is slower than
the other two put together); submit it on its own:

    PYTHONPATH=repos/research/kineidos-v3:repos/research/wp-v2 \
    LAYERNORM_TYPE=torch ATTN_IMPL=sdpa \
      <env>/bin/python -m kineidos.train.check_heldout_rows
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

# The held-out trajectory to score.  r4 of one sample: the real split's
# held-out axis, so the window construction is the one training will do.
HELD_OUT = "gagu_100mM_K_agaguu_startI_r4"
EVAL_WINDOWS = 4

# Section 6.2 compares bins on these.  A row missing one of them is a row the
# read cannot use.
REQUIRED = ("window_id", "sample_id", "target_frame", "stride", "delta_t_ns",
            "loss", "mse_loss", "smooth_lddt_loss")

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def main() -> int:
    from kineidos.data.windows import DT_MAX_NS, DT_MIN_NS
    from kineidos.seeding import set_all_seeds
    from kineidos.train.check_trainer import CHECKPOINT, make_configs
    from kineidos.train.trainer import KineidosTrainer

    if not os.path.exists(CHECKPOINT):
        print(f"missing checkpoint: {CHECKPOINT}")
        return 2

    work = tempfile.mkdtemp(prefix="p009_heldout_")
    try:
        set_all_seeds(0)
        # eval_interval=1 with max_steps=1: run() evaluates on the last step
        # regardless, but saying so explicitly is what makes this a test of the
        # eval path rather than of run()'s final-step special case.
        configs = make_configs("zero", work, max_steps=1)
        configs.eval_interval = 1
        configs.kineidos.held_out_samples = [HELD_OUT]
        configs.kineidos.eval_windows = EVAL_WINDOWS

        trainer = KineidosTrainer(configs)

        # Section 4 item 5: the permutation entries have to reach the
        # aggregator.  calc() empties it and log_interval is 1, so the only way
        # to see what was added is to watch the calls.
        added: list[str] = []
        real_add = trainer.train_metric_wrapper.add

        def spy_add(key, value, namespace="default"):
            added.append(key)
            return real_add(key, value, namespace=namespace)

        trainer.train_metric_wrapper.add = spy_add

        print("=== 1. one step and one evaluation round ===")
        trainer.run()
        check("the run finished", trainer.step >= 1, f"step {trainer.step}")

        perm_keys = sorted({k for k in added if "perm" in k})
        check("a permutation entry reached the aggregator", bool(perm_keys),
              ", ".join(perm_keys[:4]) or "none — log_dict was dropped again")
        check("one of them is an is_permuted flag",
              any("is_permuted" in k for k in perm_keys),
              ", ".join(k for k in perm_keys if "is_permuted" in k) or "none")

        print("\n=== 2. the file ===")
        out = Path(trainer.run_dir) / "heldout"
        files = sorted(out.glob("step_*.rank*.jsonl")) if out.is_dir() else []
        check("a jsonl per rank was written", bool(files),
              ", ".join(f.name for f in files) or f"nothing under {out}")
        if not files:
            return 1

        rows = [json.loads(line) for f in files for line in
                f.read_text().splitlines() if line.strip()]
        check(f"it holds {EVAL_WINDOWS} rows", len(rows) == EVAL_WINDOWS,
              f"{len(rows)}")

        print("\n=== 3. the fields section 6.2 reads ===")
        missing = sorted({k for k in REQUIRED for r in rows if k not in r})
        check("every required field is present", not missing,
              f"missing {missing}" if missing else ", ".join(REQUIRED))
        if missing:
            return 1

        strides = [r["stride"] for r in rows]
        check("stride is an integer in 1..10",
              all(isinstance(s, int) and 1 <= s <= 10 for s in strides),
              f"{sorted(set(strides))}")
        # The two have to agree, or a bin built on stride and a number reported
        # in ns would describe different windows.
        check("delta_t_ns == stride * 0.1 ns",
              all(abs(r["delta_t_ns"] - r["stride"] * 0.1) < 1e-9 for r in rows),
              f"{sorted({round(r['delta_t_ns'], 3) for r in rows})}")
        check(f"delta_t_ns within [{DT_MIN_NS}, {DT_MAX_NS}]",
              all(DT_MIN_NS - 1e-9 <= r["delta_t_ns"] <= DT_MAX_NS + 1e-9
                  for r in rows),
              f"{min(r['delta_t_ns'] for r in rows)}.."
              f"{max(r['delta_t_ns'] for r in rows)}")
        check("every loss is finite and positive",
              all(0.0 < r["loss"] < float("inf") for r in rows),
              f"{[round(r['loss'], 4) for r in rows]}")
        check("the losses differ between windows -- the row is per window",
              len({round(r["loss"], 6) for r in rows}) == len(rows),
              f"{len({round(r['loss'], 6) for r in rows})} distinct of {len(rows)}")

        # Single rank here, so the sharding rule reduces to 0 + i*1.  Asserting
        # the identity rather than the values is the point: it is what makes the
        # ids unique once there are three ranks.
        ids = sorted(r["window_id"] for r in rows)
        check("window_id is the global index (rank + i*world_size)",
              ids == list(range(len(rows))), f"{ids}")
        check("sample_id names the held-out trajectory",
              all(r["sample_id"] == HELD_OUT for r in rows),
              f"{sorted({r['sample_id'] for r in rows})}")

        print("\n  one row, as written:")
        print(f"    {json.dumps(rows[0], sort_keys=True)}")
    finally:
        shutil.rmtree(work, ignore_errors=True)

    print("\n" + "=" * 62)
    if FAILS:
        print(f"P009 逐窗口记录验收: {len(FAILS)} 项失败 — " + "; ".join(FAILS))
        return 1
    print("P009 逐窗口记录验收: 全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
