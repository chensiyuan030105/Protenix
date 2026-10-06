"""Freeze the AF3 trunk, and say exactly what was frozen.

The plan requires the trunk to carry Protenix's pretrained weights and stay
frozen, so that any difference between the ablation's arms comes from the
fusion rather than from the trunk drifting.  Nothing upstream does this --
there is one `requires_grad` in the whole of runner/, and it belongs to a loss
placeholder -- so it is ours to get right.

Getting it wrong is silent in the direction that matters: if a module is left
trainable, the `none` arm stops being "Protenix with a frozen trunk" and the
baseline the whole comparison rests on has moved.  So this returns a report
rather than nothing, and the caller is expected to log it.

Where freezing has to happen: before the optimizer is built. get_adamw filters
on requires_grad (protenix/utils/training.py), so a parameter frozen first never
enters the optimizer and weight decay cannot touch it. Frozen after, and it is
already in a param group.
"""

from __future__ import annotations

import torch

# Named only as a staleness check, not as the thing that gets frozen -- see
# freeze_trunk's docstring for why listing them would be the wrong direction.
TRUNK_MODULES = (
    "input_embedder",
    "template_embedder",
    "msa_module",
    "constraint_embedder",
    "pairformer_stack",
)

# Not part of the trunk, but switched off in section 7's table -- distogram
# because the trunk that feeds it is frozen, confidence because MD trajectories
# carry no confidence labels.  Their losses being zero already stops them
# learning; freezing them as well means a checkpoint cannot come back with
# heads that drifted for reasons nobody logged.
DISABLED_HEADS = ("distogram_head", "confidence_head")


# Everything else is frozen.  The fusion lives inside diffusion_module.
TRAINABLE_MODULES = ("diffusion_module",)


def freeze_trunk(
    model: torch.nn.Module,
    *,
    trainable: tuple[str, ...] = TRAINABLE_MODULES,
    expect_frozen: tuple[str, ...] = TRUNK_MODULES + DISABLED_HEADS,
) -> dict[str, object]:
    """Freeze everything except `trainable`, and report what that came to.

    Note the direction.  An earlier version listed the trunk's modules and froze
    those, which is the wrong way round: Protenix also carries top-level
    parameters outside its named submodules -- linear_no_bias_sinit, zinit1,
    zinit2, relative_position_encoding, the recycling layernorms, about 0.46M in
    all -- and every one of them feeds s_trunk / z_trunk.  Listing by name left
    them trainable, so the trunk's outputs would have drifted and "frozen trunk"
    would have been false while nothing complained.

    Freezing by exclusion fails in the safe direction instead: a module upstream
    adds later is frozen by default, where listing would have left it training.

    `expect_frozen` is kept as a staleness alarm. It is not what does the
    freezing; it only asserts the modules we believe exist still do, so that a
    rename upstream surfaces here rather than as a quietly different baseline.

    Returns a report whose load-bearing entry is `trainable_by_module`: what is
    still trainable has to be exactly what was asked for.  `parameter_free`
    names the expected modules that exist but hold no parameters -- see the
    report key's comment below for why that is worth printing.
    """
    for name in expect_frozen:
        if getattr(model, name, None) is None:
            raise ValueError(
                f"{name!r} was expected on the model and is absent. The module "
                f"names in kineidos/train/freezing.py are stale relative to "
                f"protenix/model/protenix.py; check whether the trunk changed "
                f"shape before trusting any ablation run."
            )

    missing = [n for n in trainable if getattr(model, n, None) is None]
    if missing:
        raise ValueError(f"cannot leave {missing} trainable: not on the model")

    for p in model.parameters():
        p.requires_grad_(False)
    for name in trainable:
        for p in getattr(model, name).parameters():
            p.requires_grad_(True)

    by_module: dict[str, int] = {}
    frozen_by_module: dict[str, int] = {}
    for name, sub in model.named_children():
        t = sum(p.numel() for p in sub.parameters() if p.requires_grad)
        f = sum(p.numel() for p in sub.parameters() if not p.requires_grad)
        if t:
            by_module[name] = t
        if f:
            frozen_by_module[name] = f
    # Parameters held directly on the model rather than on a child module --
    # the ones that caused the original mistake.
    direct = sum(p.numel() for p in model.parameters(recurse=False))
    if direct:
        frozen_by_module["<model-level>"] = direct

    # Modules that are present but hold nothing. In the baseline config that is
    # constraint_embedder, whose four sub-embedders are all enable=False, so it
    # builds no submodules at all. (template_embedder is not empty despite
    # n_blocks=0 -- the projections outside the blocks still exist, which is why
    # this is measured rather than inferred from the config.) Freezing an empty
    # module is a no-op, which is harmless, but a reader of the log should not
    # have to open the config to learn which branches of the trunk are even
    # there. Reported, not raised: turning a branch off is the config's right,
    # we only refuse to be vague about it.
    parameter_free = tuple(
        name for name in expect_frozen
        if not any(True for _ in getattr(model, name).parameters())
    )

    total = sum(p.numel() for p in model.parameters())
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "frozen": frozen_by_module,
        "parameter_free": parameter_free,
        "frozen_parameters": total - total_trainable,
        "trainable_by_module": by_module,
        "trainable_parameters": total_trainable,
        "total_parameters": total,
        "trainable_fraction": total_trainable / total if total else 0.0,
    }


def format_report(report: dict[str, object]) -> str:
    """A few lines for the training log, naming what is trainable.

    Printed at startup every run. The cost of printing it is nothing; the cost
    of discovering three weeks later that the trunk was never frozen is a batch
    of conclusions.
    """
    lines = [
        f"frozen:    {report['frozen_parameters'] / 1e6:.2f}M across "
        f"{len(report['frozen'])} modules",
        f"trainable: {report['trainable_parameters'] / 1e6:.2f}M "
        f"({report['trainable_fraction']:.1%} of "
        f"{report['total_parameters'] / 1e6:.2f}M)",
    ]
    for name, n in sorted(report["trainable_by_module"].items(),
                          key=lambda kv: -kv[1]):
        lines.append(f"  trainable: {name:<22} {n / 1e6:8.2f}M")
    if report.get("parameter_free"):
        lines.append(
            "  empty (no parameters, so frozen vacuously): "
            + ", ".join(report["parameter_free"])
        )
    return "\n".join(lines)
