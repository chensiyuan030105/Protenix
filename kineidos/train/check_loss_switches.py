"""Acceptance for skip_confidence / skip_distogram / skip_mini_rollout.

The plan's tier-1 loss handling is "do not compute", not "weight by zero"
(plan 2.13).  What has to be true:

  * with the switches off, upstream behaviour is unchanged -- without this
    control, a switch that did nothing at all would pass everything below;
  * with them on, the four confidence keys and `distogram` are absent from
    pred_dict, so calculate_losses never builds those loss terms;
  * the diffusion losses are still built and finite, because 9 forbids
    touching them;
  * a configuration that weights a loss that will not exist is refused.

Runs the real model on a real GAGU window.  N_cycle and diffusion_batch_size
are turned down to 1 -- those are sample counts, not architecture, and the
control flow under test does not depend on them.
"""

from __future__ import annotations

import copy
import sys
from collections.abc import Mapping

import torch

from kineidos.seeding import set_all_seeds
from kineidos.train.batch import one_batch

FAILS: list[str] = []

CONFIDENCE_KEYS = ("plddt", "pae", "pde", "resolved")


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def build(**overrides):
    """The real model, with sample counts turned down."""
    from configs.configs_base import configs as configs_base
    from configs.configs_data import data_configs
    from configs.configs_inference import inference_configs
    from configs.configs_model_type import model_configs
    from protenix.config import parse_configs
    from protenix.model.protenix import Protenix

    name = "protenix_base_default_v1.0.0"
    # deepcopy, not a shallow {**configs_base}: a shallow copy shares every
    # nested dict with the module-level default, so writing into
    # base["model"]["diffusion_module"] below would change the default for any
    # model built later in this process.  That is how check_trainer's `none` arm
    # came to be refused for carrying the `zero` arm's wp_token_dim.
    base = copy.deepcopy(
        {**configs_base, **{"data": data_configs}, **inference_configs})

    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, Mapping) and k in d and isinstance(d[k], Mapping):
                deep_update(d[k], v)
            else:
                d[k] = v
        return d

    deep_update(base, copy.deepcopy(model_configs[name]))
    deep_update(base, {
        "diffusion_batch_size": 1,
        "model": {"N_cycle": 1},
        "sample_diffusion": {"N_step_mini_rollout": 2},
        # Login node, no GPU.  Both default to "cuequivariance", whose kernels
        # dlopen libcuda.so.1 and raise ImportError here.  "torch" is the same
        # computation in plain PyTorch on the same weights -- the counterpart of
        # LAYERNORM_TYPE=torch in AGENTS.md.  Training on a GPU node keeps the
        # defaults; nothing in this file's subject matter depends on which.
        "triangle_multiplicative": "torch",
        "triangle_attention": "torch",
    })
    deep_update(base, overrides)
    cfg = parse_configs(configs=base, arg_str=f"--model_name {name}",
                        fill_required_with_null=True)
    return Protenix(cfg), cfg


def forward_once(model, cfg, batch, step: int = 0):
    from protenix.utils.permutation.permutation import SymmetricPermutation

    perm = SymmetricPermutation(cfg, error_dir=None)
    model.train()
    pred, label, log = model(
        input_feature_dict=batch["input_feature_dict"],
        label_dict=batch["label_dict"],
        label_full_dict=batch["label_full_dict"],
        mode="train",
        current_step=step,
        symmetric_permutation=perm,
    )
    return pred, label, log, perm


# Asking "was this loss computed" by hooking the loss modules does not work:
# calculate_losses calls smooth_lddt_loss.dense_forward, bond_loss.sparse_forward
# and plddt_loss.forward_given_atom_lddt, and a forward hook sees none of those.
# Asking the returned metrics does not work either -- a loss outside the
# resolution window is computed, multiplied by 0.0 and never recorded
# (loss.py:1588-1604), which is what GAGU's resolution of -1.0 triggers.
#
# So intercept the dictionary itself.  calculate_losses builds `loss_fns` and
# hands it to aggregate_losses (loss.py:1799); its keys are exactly the terms
# that will be evaluated, before anything weights or drops them.
def losses_for(cfg, batch, pred, label):
    from protenix.model.loss import ProtenixLoss

    loss_obj = ProtenixLoss(cfg)
    seen: set[str] = set()
    original = loss_obj.aggregate_losses

    def spy(loss_fns, has_valid_resolution=None):
        seen.update(loss_fns)
        return original(loss_fns, has_valid_resolution)

    loss_obj.aggregate_losses = spy
    loss, loss_dict = loss_obj(
        feat_dict=batch["input_feature_dict"],
        pred_dict=pred,
        label_dict=label,
        mode="train",
    )
    return loss, loss_dict, seen


def main() -> int:
    print("=== 0. configurations that weight a loss that will not exist ===")
    cases = [
        ("skip_mini_rollout without skip_confidence",
         {"skip_mini_rollout": True}, "coordinate_mini"),
        ("skip_distogram with a nonzero alpha_distogram",
         {"skip_distogram": True}, "alpha_distogram"),
        ("skip_confidence with a nonzero alpha_confidence",
         {"skip_confidence": True}, "alpha_confidence"),
    ]
    for name, overrides, needle in cases:
        try:
            build(**overrides)
            check(name + " is refused", False, "it was accepted")
        except (ValueError, AssertionError) as exc:
            check(name + " is refused", needle in str(exc), str(exc)[:70])

    window, batch = one_batch()
    n_atom = batch["label_dict"]["coordinate"].shape[0]
    print(f"\n  GAGU window: {n_atom} atoms, sample {window.sample_id}")

    print("\n=== 1. switches off: upstream behaviour, as a control ===")
    set_all_seeds(0)
    model, cfg = build()
    base_batch = copy.deepcopy(batch)
    pred_a, label_a, log_a, _ = forward_once(model, cfg, base_batch)
    check("the confidence head ran", all(k in pred_a for k in CONFIDENCE_KEYS),
          str([k for k in CONFIDENCE_KEYS if k in pred_a]))
    check("the mini-rollout ran", "coordinate_mini" in pred_a)
    check("the distogram head ran", "distogram" in pred_a)
    _, _, built_a = losses_for(cfg, base_batch, pred_a, label_a)
    ran_a = sorted(built_a)
    check("their loss terms were built",
          {"plddt_loss", "pde_loss", "resolved_loss", "pae_loss",
           "distogram_loss"} <= set(ran_a), str(ran_a))

    # The question the plan left open: does permuting the label to match the
    # mini-rollout actually change anything on GAGU?  Atom permutation cannot
    # (atom_perm_list is identity by construction), but the two chains are
    # equivalent copies under one entity id, so chain permutation is live.
    permuted = [f"{k}={v}" for k, v in log_a.items() if "is_permuted" in k]
    moved = not torch.equal(label_a["coordinate"], batch["label_dict"]["coordinate"])
    print(f"  label permutation: {permuted or 'not logged'}; "
          f"label coordinates changed: {moved}")

    print("\n=== 2. switches on: the heads do not run ===")
    set_all_seeds(0)
    model_b, cfg_b = build(skip_confidence=True, skip_distogram=True,
                           loss={"weight": {"alpha_confidence": 0.0,
                                            "alpha_distogram": 0.0}})
    batch_b = copy.deepcopy(batch)
    pred_b, label_b, log_b, _ = forward_once(model_b, cfg_b, batch_b)
    check("no confidence keys in pred_dict",
          not any(k in pred_b for k in CONFIDENCE_KEYS),
          str([k for k in CONFIDENCE_KEYS if k in pred_b]) or "none")
    check("no distogram in pred_dict", "distogram" not in pred_b)
    check("the mini-rollout still ran, so the label is still permuted",
          "coordinate_mini" in pred_b,
          "skip_mini_rollout defaults off on purpose")

    loss_b, _, built_b = losses_for(cfg_b, batch_b, pred_b, label_b)
    ran_b = sorted(built_b)
    check("no confidence or distogram loss term is built at all",
          not ({"plddt_loss", "pde_loss", "resolved_loss", "pae_loss",
                "distogram_loss"} & set(ran_b)), str(ran_b))
    check("the diffusion loss terms still are",
          {"mse_loss", "smooth_lddt_loss"} <= set(ran_b), str(ran_b))
    check("the total loss is finite", torch.isfinite(loss_b).all(),
          f"loss = {loss_b.item():.4f}")

    print("\n" + "=" * 62)
    print("验收:", "全部通过" if not FAILS else f"{len(FAILS)} 项失败 -> {FAILS}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())
