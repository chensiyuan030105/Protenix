#!/usr/bin/env python
"""P010's acceptance for everything that can be checked without a GPU.

Three of the four things this file tests have the same failure mode: they are
switches and defaults, and a wrong one does not raise -- it runs the wrong
experiment under the right name.  `kineidos/train/check_modes.py` and
`check_loss_switches.py` exist for that reason and this is the same kind of
file, extended per branch: the integration branch checks the per-sigma keys
and the round schedule, and each test branch adds its own layer to the bottom.

What is deliberately NOT here: anything that builds a model, loads a
checkpoint, reads a trajectory or touches a GPU.  Those go in
p010_sigma_grid.sbatch and p010_identity.sbatch, which ask for a GPU and get
one.  This file is config parsing and arithmetic, so it fits in the same CPU
job as the readout -- and it is still a slurm job, because the rule
(AGENTS.md, P006 D19) has no size qualifier and "it is only a parse" is how
things end up on the login node.

    sbatch --export=ALL,MODE=check \
      repos/research/kineidos-v3-diag/kineidos/slurm/p010_readout.sbatch
"""

from __future__ import annotations

import sys
from typing import Any

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


class FakeTrainer:
    """Just enough of a trainer for sigma_grid_due, which is pure arithmetic.

    A real trainer would need a GPU and a checkpoint to answer a question about
    integer division.  The method is bound off the class so that what is tested
    is the code the arms run, not a copy of its logic living here -- a test
    that reimplements the rule it is checking passes forever.
    """

    def __init__(self, step: int, *, eval_interval: int, max_steps: int,
                 sigma_grid_interval: int) -> None:
        from kineidos.train.trainer import KineidosTrainer

        self.step = step
        self.configs = _ns(
            eval_interval=eval_interval,
            max_steps=max_steps,
            kineidos=_ns(sigma_grid_interval=sigma_grid_interval),
        )
        self.sigma_grid_due = KineidosTrainer.sigma_grid_due.__get__(self)


class _ns:
    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)


def eval_rounds(max_steps: int, eval_interval: int) -> list[int]:
    """The steps `run()` evaluates on: (step + 1) % eval_interval == 0, plus
    the last step, which it evaluates whatever the interval says
    (runner/train.py:710)."""
    rounds = [s for s in range(max_steps) if (s + 1) % eval_interval == 0]
    if max_steps - 1 not in rounds:
        rounds.append(max_steps - 1)
    return rounds


def main() -> int:
    print("=== 1. the per-sigma keys exist, and default to today's behaviour ===")
    from kineidos.train.check_trainer import make_configs

    import tempfile
    cfg = make_configs("random", tempfile.mkdtemp(prefix="p010_check_")).kineidos
    for key, want, why in (
        ("sigma_grid", False,
         "on by default would make P009's protocol cost a second pass"),
        ("sigma_grid_noise", 4, "section 4 sizes the grid at 4 per sigma"),
        ("sigma_grid_interval", 0, "0 = every eval round = the old behaviour"),
        ("sigma_grid_windows", 0, "0 = all of them, which section 4 fixes"),
        ("sigma_grid_arm", "", "empty = derive it from the run"),
        ("sigma_grid_out", "", "empty = run_dir/sigma_grid"),
        ("score_checkpoint", "", "only the scorer reads it"),
    ):
        got = getattr(cfg, key, "<<missing>>")
        check(f"kineidos.{key} defaults to {want!r}", got == want,
              f"got {got!r}; {why}")

    print("\n=== 2. the sigma grid runs on the rounds it says it will ===")
    # 2000 steps, held-out every 250, grid every 500: the grid must land on
    # 499/999/1499/1999 and nowhere else.  Those four are what section 6 item 3
    # reads arm B's release against, so an off-by-one round here moves the
    # release out of the window being read.
    rounds = eval_rounds(2000, 250)
    check("eval rounds are where runner/train.py puts them",
          rounds == [249, 499, 749, 999, 1249, 1499, 1749, 1999], str(rounds))
    due = [s for s in rounds
           if FakeTrainer(s, eval_interval=250, max_steps=2000,
                          sigma_grid_interval=500).sigma_grid_due()]
    check("interval 500 over eval_interval 250 gives every other round",
          due == [499, 999, 1499, 1999], str(due))
    all_due = [s for s in rounds
               if FakeTrainer(s, eval_interval=250, max_steps=2000,
                              sigma_grid_interval=0).sigma_grid_due()]
    check("interval 0 gives every round, as before the key existed",
          all_due == rounds, f"{len(all_due)} of {len(rounds)}")
    # The last step is always included: run() evaluates there regardless, and
    # that round is the arm's final state -- the one every table quotes.
    last = FakeTrainer(1999, eval_interval=250, max_steps=2000,
                       sigma_grid_interval=1000).sigma_grid_due()
    check("the final round is always included", last,
          "run() evaluates the last step whatever the interval says, and that "
          "is the state the readout quotes")

    print("\n=== 3. the sigma grid is the grid section 4 registered ===")
    from kineidos.score_sigma_grid import (band_of, c_skip_of, edm_scale_of,
                                           sigma_grid)

    grid = sigma_grid()
    check("eleven sigmas", len(grid) == 11, str(len(grid)))
    want = [0.24, 0.51, 1.08, 2.28, 4.82, 5.33, 10.20, 21.60, 24.44, 45.72,
            96.79]
    check("and they are section 4's eleven",
          all(abs(a - b) < 0.01 for a, b in zip(grid, want)),
          ", ".join(f"{s:.2f}" for s in grid))
    bands = [band_of(s) for s in grid]
    check("two of them are in the high band (c_skip < 0.3)",
          bands.count("high") == 2,
          f"{bands.count('low')} low / {bands.count('mid')} mid / "
          f"{bands.count('high')} high")
    # The identity the two loss columns are derived through.  If this drifts,
    # mse_aligned and loss_unweighted are being divided by the wrong factor and
    # nothing in the output would say so.
    s = 10.0
    check("edm_scale is loss.py:1638's factor",
          abs(edm_scale_of(s) - (s**2 + 16.0**2) / (16.0 * s) ** 2) < 1e-12)
    check("c_skip is diffusion.py:602's",
          abs(c_skip_of(s) - 1.0 / (1.0 + (s / 16.0) ** 2)) < 1e-12)

    print("\n=== 4. read_heldout refuses a leaked arm ===")
    import json
    from pathlib import Path

    from kineidos.read_heldout import refuse_oracle

    tmp = Path(tempfile.mkdtemp(prefix="p010_check_oracle_"))
    named = tmp / "p009_oracle_20261008_120000"
    named.mkdir()
    relabelled = tmp / "p009_random_20261008_120000"
    relabelled.mkdir()
    (relabelled / "env.lock").write_text(json.dumps({"wp": {"mode": "oracle"}}))
    clean = tmp / "p009_random_20261008_130000"
    clean.mkdir()
    (clean / "env.lock").write_text(json.dumps({"wp": {"mode": "random"}}))
    for name, path in (("by directory name", named),
                       ("by env.lock's wp.mode", relabelled)):
        try:
            refuse_oracle(path)
            check(f"refused {name}", False, "it was accepted")
        except SystemExit as exc:
            check(f"refused {name}", "oracle" in str(exc).lower(),
                  str(exc)[:60])
    try:
        refuse_oracle(clean)
        check("and a real arm still passes", True)
    except SystemExit as exc:
        check("and a real arm still passes", False, str(exc)[:80])

    print("\n" + "=" * 62)
    print("验收:", "全部通过" if not FAILS else f"{len(FAILS)} 项失败 -> {FAILS}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
