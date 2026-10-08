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

Run from the workspace root, on a GPU:

    PYTHONPATH=repos/research/kineidos-v3:repos/research/wp-v2 \
    LAYERNORM_TYPE=torch ATTN_IMPL=sdpa LD_LIBRARY_PATH=$ENV/lib \
      $ENV/bin/python -m kineidos.bench_step --mode random
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import torch


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
    parser.add_argument("--cycles", default="1,2,4,10")
    parser.add_argument("--diffusion-batch", default="48,24")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("needs a GPU: the point is the relative cost of the phases, and "
              "that does not transfer from CPU")
        return 2

    from protenix.utils.torch_utils import to_device

    from kineidos.data.gagu import GAGUProtenixAdapter
    from kineidos.data.windows import GAGUWindowDataset
    from kineidos.seeding import set_all_seeds
    from kineidos.train.batch import collate_window
    from kineidos.train.check_trainer import CHECKPOINT, make_configs
    from kineidos.train.trainer import KineidosTrainer

    import tempfile
    work = tempfile.mkdtemp(prefix="p009_bench_")
    set_all_seeds(0)

    configs = make_configs(args.mode, work, max_steps=1)
    # make_configs is built for a CPU smoke test; put the production numbers
    # back, or the timings describe the test and not the run.
    configs.model.N_cycle = 10
    configs.diffusion_batch_size = 48
    configs.sample_diffusion.N_step_mini_rollout = 20
    configs.triangle_attention = "cuequivariance"
    configs.triangle_multiplicative = "cuequivariance"
    configs.kineidos.train_samples = [args.sample]
    configs.load_checkpoint_path = CHECKPOINT

    trainer = KineidosTrainer(configs)
    model = trainer.raw_model
    root = __import__("pathlib").Path(configs.kineidos.gagu_root)
    sample = GAGUProtenixAdapter(root / args.sample).load()
    window = GAGUWindowDataset([sample], k=8, length=1, seed=0)[0]
    batch = to_device(collate_window(window), trainer.device)

    print(f"mode={args.mode}  {window.wp_position_nm.shape[1]} atoms  "
          f"stride={window.stride}  device={torch.cuda.get_device_name(0)}")
    print(f"pairformer blocks={configs.model.pairformer.n_blocks}  "
          f"diffusion_batch_size={configs.diffusion_batch_size}  "
          f"N_step_mini_rollout={configs.sample_diffusion.N_step_mini_rollout}\n")

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
    cycles = [int(c) for c in args.cycles.split(",")]
    for skip in (False, True):
        model.skip_mini_rollout = skip
        for nc in cycles:
            t = timed(forward(nc), warmup=args.warmup, repeat=args.repeat)
            rows.append({"skip_mini_rollout": skip, "n_cycle": nc,
                         "diffusion_batch_size": 48, "sec_per_step": t})
            print(f"  skip_mini_rollout={str(skip):<5} N_cycle={nc:>2}  "
                  f"{t:6.2f} s/step")

    print()
    model.skip_mini_rollout = False
    for db in [int(x) for x in args.diffusion_batch.split(",")]:
        configs.diffusion_batch_size = db
        t = timed(forward(10), warmup=args.warmup, repeat=args.repeat)
        rows.append({"skip_mini_rollout": False, "n_cycle": 10,
                     "diffusion_batch_size": db, "sec_per_step": t})
        print(f"  diffusion_batch_size={db:>3} (N_cycle=10, rollout on)  "
              f"{t:6.2f} s/step")
    configs.diffusion_batch_size = 48

    # The decomposition the three knobs are argued from.  Each is a difference
    # of measured totals rather than a separate timing, so the parts add up to
    # the whole by construction and no phase is double counted.
    def total(skip, nc, db=48):
        for r in rows:
            if (r["skip_mini_rollout"] == skip and r["n_cycle"] == nc
                    and r["diffusion_batch_size"] == db):
                return r["sec_per_step"]
        return float("nan")

    print("\n=== 分解 ===")
    hi, lo = max(cycles), min(cycles)
    per_cycle = (total(True, hi) - total(True, lo)) / (hi - lo)
    print(f"  每一层 recycling（无 rollout，{lo}->{hi} 的斜率）: {per_cycle:.3f} s")
    print(f"  mini-rollout（N_cycle={hi} 时开关之差）      : "
          f"{total(False, hi) - total(True, hi):.3f} s")
    print(f"  其余（扩散头 fwd+bwd 等，N_cycle={lo} 无 rollout 减去 "
          f"{lo} 层 recycling）: {total(True, lo) - lo * per_cycle:.3f} s")
    print(f"\n  生产口径（训练 N_cycle 平均 5.5、rollout 开）估计: "
          f"{total(True, lo) + (5.5 - lo) * per_cycle + (total(False, hi) - total(True, hi)):.2f} s/step")
    print(f"  实测四档是 6.0-7.5 s/step（3 卡 DDP，含数据与同步开销）")

    if args.out:
        __import__("pathlib").Path(args.out).write_text(json.dumps(rows, indent=2) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
