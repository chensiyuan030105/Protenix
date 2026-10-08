"""Acceptance for the P004 trainer: a few real steps, end to end.

Everything before this checked a piece in isolation.  This runs the loop the
ablation will run -- real GAGU windows, real pretrained checkpoint, real
optimizer -- for a handful of steps, and asks the questions whose wrong answers
are silent:

  * did the pretrained trunk actually load, with exactly the fusion missing;
  * after N steps, is every frozen parameter still bit-identical, and did the
    fusion's parameters move;
  * is the injection strength being recorded, so a run that is not injecting
    says so while it runs (plan section 5);
  * does a config whose arm and architecture disagree get refused.

Nothing is written outside base_dir, and checkpoint_interval is -1: the disk is
at 97% (AGENTS.md).
"""

from __future__ import annotations

import copy
import gc
import hashlib
import os
import shutil
import sys
import tempfile
from collections.abc import Mapping

import torch

CHECKPOINT = (
    "/mnt/xfs/home/mhg/Projects/ForSiyuan/RNA-WorldParticle-Workspace/runtimes/"
    "checkpoints/protenix_pretrained/protenix_base_default_v1.0.0.pt"
)

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def fingerprint(tensor: torch.Tensor) -> str:
    """A bit-exact digest of one tensor, holding one tensor's bytes at a time.

    Cloning every frozen parameter to compare later costs 165.23M floats --
    0.66 GB -- on a node with 31 GB shared between users.  Together with three
    live 368.60M models that was enough to get this script OOM-killed at the
    moment Adam allocated its two moment buffers for the random arm's 276.82M
    trainable parameters.  A digest answers the same question ("did this change
    at all") with a peak of one tensor.
    """
    return hashlib.blake2b(
        tensor.detach().cpu().contiguous().numpy().tobytes(), digest_size=16
    ).hexdigest()


def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, Mapping) and k in d and isinstance(d[k], Mapping):
            deep_update(d[k], v)
        else:
            d[k] = v
    return d


def make_configs(mode: str, base_dir: str, *, max_steps: int = 2,
                 wp_token_dim: int | None | str = "auto",
                 overrides: Mapping | None = None):
    """The smoke test's config, with `overrides` merged in before parse.

    `overrides` exists for callers that are not the smoke test -- P010's
    kineidos.bench_step wants the production numbers rather than the CPU
    defaults below.  It is merged into the dict *before* parse_configs for the
    same reason kineidos/train/main.py sets wp_token_dim there: several of
    these values are read in a constructor, so a value written onto an
    already-parsed config is read or ignored depending on which key it is, and
    nothing says which.  triangle_attention is read by Pairformer's
    constructor, diffusion_batch_size by Protenix.__init__ (protenix.py:159),
    while N_step_mini_rollout is read at forward time -- three different
    answers to "is it too late", from one line of assignment each.  Merging
    before parse makes the question not arise.
    """
    from configs.configs_base import configs as configs_base
    from configs.configs_data import data_configs
    from configs.configs_model_type import model_configs
    from protenix.config import parse_configs
    from kineidos.train.trainer import wp_token_dim_for

    name = "protenix_base_default_v1.0.0"
    # deepcopy, not {**configs_base}.  A shallow copy shares every nested dict
    # with the module-level default, so writing wp_token_dim for one arm here
    # changed it for every arm built later in the same process -- which is how
    # the `none` arm came to be refused for carrying the `zero` arm's 768.
    base = copy.deepcopy({**configs_base, **{"data": data_configs}})
    deep_update(base, copy.deepcopy(model_configs[name]))
    deep_update(base, {
        # No GPU on the login node; see check_loss_switches for why these two.
        "triangle_multiplicative": "torch",
        "triangle_attention": "torch",
        "diffusion_batch_size": 1,
        "model": {"N_cycle": 1},
        # 20 by default, and the mini-rollout runs every training step.  The
        # subject here is the loop's wiring, not the sampler, and on CPU the
        # difference is minutes per step.
        "sample_diffusion": {"N_step_mini_rollout": 2},
        "project": "p004", "run_name": "trainer_smoke", "base_dir": base_dir,
        "max_steps": max_steps, "log_interval": 1,
        "eval_interval": -1,          # evaluate() raises; see the trainer
        "checkpoint_interval": -1,    # the disk is at 97%
        "use_wandb": False,
        "load_checkpoint_path": CHECKPOINT,
        "load_params_only": True,
        "seed": 0,
        "wp": {"mode": mode},
        "kineidos": {"epoch_length": 64},
        # Tier 1 (plan 2.13): do not compute what is not supervised.
        "skip_confidence": True, "skip_distogram": True,
        "loss": {"weight": {"alpha_confidence": 0.0, "alpha_distogram": 0.0}},
    })
    dim = wp_token_dim_for(mode) if wp_token_dim == "auto" else wp_token_dim
    if dim is not None:
        base["model"]["diffusion_module"]["wp_token_dim"] = dim
    if overrides:
        deep_update(base, copy.deepcopy(dict(overrides)))
    return parse_configs(configs=base, arg_str=f"--model_name {name}",
                         fill_required_with_null=True)


def main() -> int:
    from kineidos.seeding import set_all_seeds
    from kineidos.train.trainer import KineidosTrainer

    if not os.path.exists(CHECKPOINT):
        print(f"missing checkpoint: {CHECKPOINT}")
        return 1

    work = tempfile.mkdtemp(prefix="p004_trainer_")
    try:
        print("=== 1. an arm whose architecture does not match is refused ===")
        try:
            # `zero` needs the fusion; building it without one would silently be
            # the `none` baseline wearing another name.
            KineidosTrainer(make_configs("zero", work, wp_token_dim=None))
            check("mismatched wp_token_dim is refused", False, "it was accepted")
        except ValueError as exc:
            check("mismatched wp_token_dim is refused",
                  "too late to reach the constructor" in str(exc), str(exc)[:64])

        print("\n=== 2. the `zero` arm trains for a few steps ===")
        set_all_seeds(0)
        trainer = KineidosTrainer(make_configs("zero", work, max_steps=2))

        frozen_before = {
            n: fingerprint(p)
            for n, p in trainer.raw_model.named_parameters() if not p.requires_grad
        }
        # Three tensors; cloning these is free and keeps the comparison direct.
        fusion_before = {
            n: p.detach().clone()
            for n, p in trainer.raw_model.named_parameters() if "wp_" in n
        }
        report = trainer.freeze_report
        check("no trainable parameter lies outside diffusion_module/wp_bridge",
              not report["trainable_outside_request"],
              f"leaked: {report['trainable_outside_request'][:3]}"
              if report["trainable_outside_request"]
              else f"all under {list(report['trainable_requested'])}")
        # In the `zero` arm the bridge builds no WorldParticle network, so it
        # holds no parameters -- which is why this asks the question of the
        # parameters and not of the report's module keys.
        bridge_params = list(trainer.raw_model.wp_bridge.parameters())
        check("the diffusion head is trainable",
              "diffusion_module" in report["trainable_by_module"],
              f"{report['trainable_by_module'].get('diffusion_module', 0) / 1e6:.2f}M")
        print(f"  wp_bridge holds {len(bridge_params)} parameters in the "
              f"{trainer.configs.wp.mode!r} arm")
        check("the fusion exists in this arm", bool(fusion_before),
              f"{len(fusion_before)} parameters")

        # train_metric_wrapper.calc() clears what it aggregated, and
        # log_interval is 1, so reading _metric_data after run() finds nothing.
        # Record the keys as they go in instead.
        added: set[str] = set()
        real_add = trainer.train_metric_wrapper.add

        def spy_add(key, value, namespace="default"):
            added.add(key)
            return real_add(key, value, namespace=namespace)

        trainer.train_metric_wrapper.add = spy_add

        trainer.run()
        check("it reached max_steps", trainer.step >= 2, f"step {trainer.step}")
        # run() calls evaluate() on the last step whatever eval_interval says,
        # so an evaluate that raises would discard every finished run.  This is
        # the regression test for that: run() returning at all is the check.
        check("the final step's evaluate did not abort the run", True,
              "evaluate prints why it is skipped instead of raising")

        moved_frozen = [
            n for n, p in trainer.raw_model.named_parameters()
            if n in frozen_before and fingerprint(p) != frozen_before[n]
        ]
        check("no frozen parameter moved in training", not moved_frozen,
              f"{len(frozen_before)} unchanged" if not moved_frozen
              else f"{len(moved_frozen)} moved: {moved_frozen[:3]}")
        # Which of the three *should* move in the `zero` arm is not "all of
        # them".  h is identically zero here, so:
        #   wp_fusion.weight  -- its c_l block has a real gradient (delta (x)
        #                        c_l) and must move; its h block's gradient is
        #                        delta (x) h = 0 and must stay exactly zero.
        #   wp_layernorm.*    -- the gradient reaching them is W_h^T delta = 0,
        #                        so neither has a reason to move.  `weight`
        #                        starts at ones and weight decay nudges it;
        #                        `bias` starts at zeros and does not move at
        #                        all.  Asserting "all three moved" failed on the
        #                        bias, correctly.
        moved_fusion = sorted(
            n.rsplit(".", 2)[-2] + "." + n.rsplit(".", 1)[-1]
            for n, p in trainer.raw_model.named_parameters()
            if n in fusion_before and not torch.equal(p, fusion_before[n])
        )
        check("the fusion's c_l half learned",
              any("wp_fusion" in n for n in moved_fusion), str(moved_fusion))
        encoder = trainer.raw_model.diffusion_module.atom_attention_encoder
        c_atom = encoder.c_atom
        h_half = encoder.wp_fusion.weight[:, c_atom:]
        check("and its h half is still exactly zero, since h was zero",
              bool((h_half == 0).all()),
              f"max |W_h| = {h_half.abs().max().item():.3e}")

        print("\n=== 3. the injection strength was recorded ===")
        from kineidos import observables

        stats = observables.read(trainer.raw_model)
        print("  " + observables.format_line(stats))
        check("observables are readable off the trained model",
              "injection_strength" in stats, str(sorted(stats)))
        check("and were logged under train/wp/ during the run",
              any(k.startswith("wp/") for k in added),
              str(sorted(k for k in added if k.startswith("wp/"))))
        check("including the one the `zero` arm depends on",
              "wp/w_cl_drift_from_identity" in added,
              "h-side observables are structurally zero in this arm, so the "
              "c_l block's drift is its only instrument")

        # Release the zero arm before building the next: three live models plus
        # their optimizer state is what the OOM was made of.
        del trainer, frozen_before, fusion_before, stats, added, real_add
        gc.collect()

        print("\n=== 4. `none` builds no fusion at all ===")
        set_all_seeds(0)
        # One step is enough here: this section asserts about the architecture,
        # not about anything moving (and at lr=0 nothing would).
        bare = KineidosTrainer(make_configs("none", work, max_steps=1))
        check("no wp_fusion parameter exists",
              not any("wp_fusion" in n for n, _ in bare.raw_model.named_parameters()))
        check("and observables say so rather than reporting zeros",
              observables.read(bare.raw_model) == {},
              "an empty reading is how `none` is distinguishable from a dead fusion")

        del bare
        gc.collect()

        print("\n=== 5. the `random` arm runs WorldParticle inside the model ===")
        # The only arm that exercises the whole path: open3d's continuous_conv,
        # the bridge as a submodule with parameters of its own, and
        # freeze_trunk having to leave those parameters trainable.  `zero` and
        # `none` cannot catch a mistake in any of that.
        # Two steps, not one, and the reason is worth knowing: af3_lr_scheduler
        # warms up, so the *first* optimizer step runs at lr = 0 and moves
        # nothing at all.  A one-step run therefore shows every parameter
        # unchanged, which is indistinguishable from a broken gradient path --
        # this check was written with max_steps=1 and reported exactly that.
        set_all_seeds(0)
        rnd = KineidosTrainer(make_configs("random", work, max_steps=2))
        wp_params = {n: fingerprint(p)
                     for n, p in rnd.raw_model.named_parameters()
                     if n.startswith("wp_bridge.")}
        # Only the fusion, not the bridge: "wp_" also matches wp_bridge.*
        fusion_rnd = {n: p.detach().clone()
                      for n, p in rnd.raw_model.named_parameters()
                      if "wp_fusion" in n or "wp_layernorm" in n}
        wp_numel = sum(p.numel() for n, p in rnd.raw_model.named_parameters()
                       if n.startswith("wp_bridge."))
        check("the bridge brought parameters of its own", bool(wp_params),
              f"{len(wp_params)} tensors, {wp_numel / 1e6:.2f}M")
        check("all of them are trainable",
              all(p.requires_grad for n, p in rnd.raw_model.named_parameters()
                  if n.startswith("wp_bridge.")),
              "WorldParticle is what P004 is asking about; freezing it would "
              "make `random` a slower `zero`")
        rnd_report = rnd.freeze_report
        check("and nothing else leaked into trainable",
              not rnd_report["trainable_outside_request"],
              f"leaked: {rnd_report['trainable_outside_request'][:3]}"
              if rnd_report["trainable_outside_request"] else "none")

        rnd.run()
        stats_rnd = observables.read(rnd.raw_model)
        print("  " + observables.format_line(stats_rnd))
        # This is the arm where h actually carries something, so the injection
        # strength is finally a real number rather than zero by construction.
        check("h reaches the fusion with a nonzero norm",
              stats_rnd.get("h_norm_after_layernorm", 0.0) > 0,
              f"|h| after LayerNorm = "
              f"{stats_rnd.get('h_norm_after_layernorm', float('nan')):.3e}")
        # The asymmetry of zero initialisation, which is the whole mechanism:
        #
        #   dL/dW_h = delta (x) h   -- nonzero, because W_h being zero does not
        #                              zero its own gradient.  This is why a
        #                              zero-initialised sum starts learning
        #                              where a multiplicative gate deadlocks.
        #   dL/dh   = W_h^T delta   -- exactly zero at step 0.
        #
        # So the thing to assert after one step is that W_h moved.  WorldParticle
        # itself receives no gradient until W_h has left zero, and if its weights
        # do move at step 0 the cause is weight decay, not signal: Adam divides
        # by sqrt(v_hat), which turns a gradient of wd*p into a step of order lr.
        # Asserting that WP moved would therefore pass for the wrong reason.
        fused_moved = [n for n, p in rnd.raw_model.named_parameters()
                       if "wp_fusion" in n and not torch.equal(p, fusion_rnd[n])]
        check("the fusion's W_h moved, which is what breaks the deadlock",
              bool(fused_moved), str(fused_moved))
        moved_wp = [n for n, p in rnd.raw_model.named_parameters()
                    if n in wp_params and fingerprint(p) != wp_params[n]]
        # Reported, not asserted, and the number is not what it looks like.
        # dL/dh = W_h^T delta is zero for both steps -- W_h is zero at step 0
        # and still zero entering step 1, because step 0 runs at lr=0 -- so none
        # of this motion is signal.  It is weight decay: the gradient is wd*p,
        # and Adam's division by sqrt(v_hat) turns that into a step of order
        # 0.05*lr even though wd is 1e-8.  The tensors that did NOT move are the
        # ones initialised to exactly zero, where wd*p is also zero.
        #
        # Worth watching rather than dismissing.  At lr~1e-3 that is ~5e-5 per
        # step, so a few thousand steps of waiting for W_h would shift these
        # weights by an amount comparable to their own scale.  If the injection
        # strength stays near zero, WorldParticle is not merely failing to learn
        # -- it is decaying while it waits.  That is a different failure from
        # v0.1's deadlock, and section 2.7's auxiliary supervision is what would
        # close the gap.
        print(f"  WorldParticle's own weights: {len(moved_wp)} of "
              f"{len(wp_params)} tensors differ after two steps -- weight decay, "
              f"not signal (dL/dh is zero for both steps). The {len(wp_params) - len(moved_wp)} "
              f"that held still are the zero-initialised ones.")
    finally:
        shutil.rmtree(work, ignore_errors=True)

    print("\n" + "=" * 62)
    print("验收:", "全部通过" if not FAILS else f"{len(FAILS)} 项失败 -> {FAILS}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())
