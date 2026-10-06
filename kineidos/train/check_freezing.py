"""Acceptance for trunk freezing, on the real 368.48M model."""

from __future__ import annotations

import sys
from collections.abc import Mapping

import torch

from kineidos.seeding import set_all_seeds
from kineidos.train.freezing import (
    DISABLED_HEADS,
    TRAINABLE_MODULES,
    TRUNK_MODULES,
    format_report,
    freeze_trunk,
)

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def build(wp_token_dim=None):
    from configs.configs_base import configs as configs_base
    from configs.configs_data import data_configs
    from configs.configs_inference import inference_configs
    from configs.configs_model_type import model_configs
    from protenix.config import parse_configs
    from protenix.model.protenix import Protenix

    name = "protenix_base_default_v1.0.0"
    base = {**configs_base, **{"data": data_configs}, **inference_configs}

    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, Mapping) and k in d and isinstance(d[k], Mapping):
                deep_update(d[k], v)
            else:
                d[k] = v
        return d

    deep_update(base, model_configs[name])
    if wp_token_dim is not None:
        base["model"]["diffusion_module"]["wp_token_dim"] = wp_token_dim
    cfg = parse_configs(configs=base, arg_str=f"--model_name {name}",
                        fill_required_with_null=True)
    return Protenix(cfg)


def main() -> int:
    set_all_seeds(0)
    print("=== 1. the real model, fusion enabled ===")
    model = build(wp_token_dim=768)
    total = sum(p.numel() for p in model.parameters())
    print(f"  {total / 1e6:.2f}M parameters")
    check("the fusion reached the model through config",
          any("wp_fusion" in n for n, _ in model.named_parameters()),
          "diffusion_module.atom_attention_encoder.wp_fusion.weight")

    print("\n=== 2. freezing ===")
    report = freeze_trunk(model)
    print(format_report(report))
    # The point of the whole exercise: what is left trainable has to be exactly
    # what was asked for.  Checking "the named trunk modules got frozen" is the
    # weaker question and the one that passed while 0.46M of model-level trunk
    # parameters stayed trainable.
    check("exactly the requested modules remain trainable",
          set(report["trainable_by_module"]) == set(TRAINABLE_MODULES),
          str(sorted(report["trainable_by_module"])))
    # Ask it of the parameters, not of the report's keys.  `frozen` only lists
    # modules that hold frozen parameters, so constraint_embedder -- disabled in
    # config and therefore empty -- is absent from it for having nothing at all,
    # which is not the same as being left trainable.  The subset test could not
    # tell those two apart and failed on the harmless one.
    named = set(TRUNK_MODULES) | set(DISABLED_HEADS)
    still_trainable = sorted(
        name for name in named
        if any(p.requires_grad for p in getattr(model, name).parameters())
    )
    check("no named trunk module or disabled head has a trainable parameter",
          not still_trainable,
          f"{sorted(named)} all frozen" if not still_trainable
          else f"still trainable: {still_trainable}")
    check("the empty modules are named, so the log says what the trunk is",
          set(report["parameter_free"]) <= named,
          f"empty: {list(report['parameter_free']) or 'none'}")
    # The parameters that caused the original mistake: held on the model
    # itself rather than on any child module.
    direct = sum(p.numel() for p in model.parameters(recurse=False))
    model_level_trainable = sum(p.numel() for p in model.parameters(recurse=False)
                                if p.requires_grad)
    print(f"  model-level parameters: {direct / 1e6:.2f}M, "
          f"trainable {model_level_trainable / 1e6:.2f}M")
    check("model-level trunk parameters are frozen too",
          model_level_trainable == 0,
          "these feed s_trunk/z_trunk and were trainable before the fix")
    check("a substantial part of the model is frozen",
          0.0 < report["trainable_fraction"] < 1.0,
          f"{report['trainable_fraction']:.1%} trainable")

    print("\n=== 3. an optimizer step does not move a frozen parameter ===")
    # The first version of this section called get_adamw directly and passed.
    # The trainer does not call get_adamw: configs_base sets adam.use_adamw =
    # False, so get_optimizer takes its *other* branch and builds a plain
    # torch.optim.Adam whose single param group is `model.parameters()` --
    # unfiltered, frozen parameters included.  So the question "did a frozen
    # parameter enter the optimizer?" has different answers on the two branches
    # and is the wrong question either way.  The right one is whether a step can
    # move a frozen weight, and that is asked here of the branch the trainer
    # actually takes.
    from protenix.utils.training import get_optimizer

    cfg = model.configs
    opt = get_optimizer(cfg, model,
                        param_names=cfg.get("finetune_params_with_substring", [""]))
    in_opt = {id(p) for g in opt.param_groups for p in g["params"]}
    frozen_in_opt = [n for n, p in model.named_parameters()
                     if not p.requires_grad and id(p) in in_opt]
    # Reported rather than asserted: on this branch they are all in there, and
    # that is fine -- torch's Adam skips any parameter whose .grad is None, and
    # autograd never fills .grad for requires_grad=False.  Being in the group
    # costs a dict entry, not an update.
    print(f"  (frozen parameters sitting in the optimizer: {len(frozen_in_opt)} "
          f"-- inert, Adam skips grad=None)")

    # Grads only where autograd would put them, which is the realistic state.
    for p in model.parameters():
        p.grad = torch.randn_like(p) if p.requires_grad else None
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt.step()
    moved_frozen = [n for n, p in model.named_parameters()
                    if not p.requires_grad and not torch.equal(p, before[n])]
    moved_trainable = sum(1 for n, p in model.named_parameters()
                          if p.requires_grad and not torch.equal(p, before[n]))
    check("a step moves no frozen parameter", not moved_frozen,
          f"{len(moved_frozen)} moved: {moved_frozen[:3]}" if moved_frozen
          else f"{len(before) - moved_trainable} frozen tensors all unchanged")
    check("the same step does move the trainable ones", moved_trainable > 0,
          f"{moved_trainable} tensors moved")
    fusion_moved = [n for n, p in model.named_parameters()
                    if "wp_" in n and not torch.equal(p, before[n])]
    check("the fusion's parameters are among those that moved",
          len(fusion_moved) == sum(1 for n, _ in model.named_parameters()
                                   if "wp_" in n),
          str(fusion_moved))

    # The other branch, so that flipping use_adamw does not quietly change what
    # freezing means.  get_adamw filters on requires_grad, so there it is the
    # filter rather than grad=None that keeps the trunk still.
    from protenix.utils.training import get_adamw

    adamw = get_adamw(model=model, weight_decay=0.01, learning_rate=1e-4,
                      betas=(0.9, 0.95), device_type="cpu")
    in_adamw = {id(p) for g in adamw.param_groups for p in g["params"]}
    check("under use_adamw=True the frozen parameters are filtered out instead",
          not [n for n, p in model.named_parameters()
               if not p.requires_grad and id(p) in in_adamw],
          "both branches are safe, for different reasons")
    for p in model.parameters():
        p.grad = None

    print("\n=== 4. a stale module list is refused, not silently partial ===")
    try:
        freeze_trunk(model, expect_frozen=("pairformer_stack", "no_such_module"))
        check("a renamed trunk module raises", False, "it did not raise")
    except ValueError as exc:
        check("a renamed trunk module raises", "stale" in str(exc), str(exc)[:56])
    try:
        freeze_trunk(model, trainable=("no_such_module",))
        check("an unknown trainable module raises", False, "it did not raise")
    except ValueError as exc:
        check("an unknown trainable module raises", "cannot leave" in str(exc))

    print("\n" + "=" * 62)
    print("验收:", "全部通过" if not FAILS else f"{len(FAILS)} 项失败 -> {FAILS}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())
