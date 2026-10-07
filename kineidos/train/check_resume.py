"""Acceptance for resume, which is what makes a preemptible partition usable.

The only partition with working GPUs and free capacity is `background`:
PriorityTier=5, PreemptMode=REQUEUE, PreemptExemptTime=0.  A job there is
killed without grace whenever a tier-10 job wants the node.

Shipping this untested would be worse than not having it.  Without resume a
requeued job restarts from step 0 -- wasteful but alive.  With a broken resume
it crashes at startup and the run is simply gone.  So:

  * a checkpoint is written and latest.pt points at it;
  * a fresh trainer finds it, restores model, optimizer, scheduler and step;
  * the restored weights are bit-identical to the ones that were saved;
  * resuming across arms is refused rather than silently producing a run that
    is neither arm.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import torch

from kineidos.seeding import set_all_seeds
from kineidos.train.check_trainer import CHECKPOINT, check, fingerprint, make_configs
import kineidos.train.check_trainer as ct

FAILS = ct.FAILS


def main() -> int:
    from kineidos.train.trainer import KineidosTrainer

    work = tempfile.mkdtemp(prefix="p004_resume_")
    resume_dir = Path(work) / "resume"
    try:
        print("=== 1. a run writes latest.pt ===")
        set_all_seeds(0)
        cfg = make_configs("zero", work, max_steps=1)
        cfg.kineidos.resume_dir = str(resume_dir)
        cfg.checkpoint_interval = 1
        first = KineidosTrainer(cfg)
        first.run()
        link = resume_dir / "latest.pt"
        check("latest.pt exists", link.is_symlink() or link.exists(),
              str(link.resolve()) if link.exists() else "missing")
        check("the arm is recorded beside it",
              (resume_dir / "latest.arm").exists()
              and (resume_dir / "latest.arm").read_text().strip() == "zero",
              "resuming across arms has to be refusable")
        # The step the *checkpoint* records, which is not the step the finished
        # trainer sits at: run() saves while is_last_step is true and increments
        # self.step afterwards, so a one-step run leaves first.step == 1 and
        # writes 0.pt.  Comparing against first.step was this check's own
        # off-by-one.  The filename is the contract, so read it from there.
        saved_step = int(Path(link).resolve().stem)
        before = {n: fingerprint(p) for n, p in first.raw_model.named_parameters()}
        print(f"  checkpoint records step {saved_step}; trainer ended at "
              f"{first.step}")
        del first
        import gc
        gc.collect()

        print("\n=== 2. a fresh trainer resumes from it ===")
        set_all_seeds(0)
        cfg2 = make_configs("zero", work, max_steps=1)
        cfg2.kineidos.resume_dir = str(resume_dir)
        second = KineidosTrainer(cfg2)
        check("it continued rather than starting over",
              second.step == saved_step + 1 and second.step > 0,
              f"resumed at step {second.step}, checkpoint recorded {saved_step}")
        after = {n: fingerprint(p) for n, p in second.raw_model.named_parameters()}
        differing = [n for n in before if before[n] != after.get(n)]
        check("every weight came back bit-identical", not differing,
              f"{len(before)} tensors" if not differing
              else f"{len(differing)} differ: {differing[:3]}")
        check("the optimizer state came back too",
              bool(second.optimizer.state_dict()["state"]),
              "an empty state would mean Adam restarts its moments, which is a "
              "different trajectory from the one that was interrupted")
        del second
        gc.collect()

        print("\n=== 3. resuming into the wrong arm is refused ===")
        set_all_seeds(0)
        cfg3 = make_configs("random", work, max_steps=1)
        cfg3.kineidos.resume_dir = str(resume_dir)
        try:
            KineidosTrainer(cfg3)
            check("a `zero` checkpoint is refused by the `random` arm", False,
                  "it was accepted")
        except ValueError as exc:
            check("a `zero` checkpoint is refused by the `random` arm",
                  "neither arm" in str(exc), str(exc)[:72])
    finally:
        shutil.rmtree(work, ignore_errors=True)

    print("\n" + "=" * 62)
    print("验收:", "全部通过" if not FAILS else f"{len(FAILS)} 项失败 -> {FAILS}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())
