"""Load Protenix's checkpoint into a model that has the fusion added.

P003 verified the checkpoint loads into the unmodified baseline with
strict=True: 0 missing, 0 unexpected, 0 shape mismatches, 4174 tensors,
368.48M parameters.  Adding the fusion creates three parameters the checkpoint
cannot contain, so strict=True now raises.

The fix is not strict=False.  That tolerates *any* missing key, so "we
deliberately added three parameters" and "a module was renamed and half the
trunk silently did not load" produce the same log line.  Instead the load runs
non-strict and then asserts the missing set is *exactly* the known new keys --
equality, not containment, so one too many or one too few is an error.

This keeps P003's result at full strength rather than trading it away: the
claim changes from "nothing is missing" to "exactly these three are missing",
which is just as checkable.

The three need no checkpoint entry to be reproducible: wp_fusion is identity on
the c_l half and zero on the h half, wp_layernorm is ones and zeros.  Nothing
random, so the fusion's starting point does not depend on any seed.
"""

from __future__ import annotations

from typing import Iterable

import torch

# Under DiffusionModule; the trunk's own AtomAttentionEncoder in embedders.py is
# built with wp_token_dim=None and adds nothing.
FUSION_KEYS = frozenset({
    "diffusion_module.atom_attention_encoder.wp_fusion.weight",
    "diffusion_module.atom_attention_encoder.wp_layernorm.weight",
    "diffusion_module.atom_attention_encoder.wp_layernorm.bias",
})


def load_checkpoint(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    *,
    expect_new: Iterable[str] = (),
) -> dict[str, object]:
    """Load `state_dict`, allowing exactly `expect_new` to be absent from it.

    Args:
        model: the model to load into.
        state_dict: the checkpoint's "model" entry, already stripped of any
            "module." prefix left by DDP.
        expect_new: keys the model has and the checkpoint legitimately does not.
            Pass FUSION_KEYS when the fusion is enabled, nothing when it is not.

    Raises:
        RuntimeError: if the missing set differs from `expect_new` in either
            direction, or if anything is unexpected, or if a shared tensor has
            a different shape.  The message names the difference rather than a
            count, because a count tells you something is wrong and not what.
    """
    expect = set(expect_new)

    own = model.state_dict()
    shape_mismatch = {
        k: (tuple(own[k].shape), tuple(v.shape))
        for k, v in state_dict.items()
        if k in own and tuple(own[k].shape) != tuple(v.shape)
    }
    if shape_mismatch:
        raise RuntimeError(
            f"checkpoint shapes differ from the model for "
            f"{len(shape_mismatch)} tensors: {dict(list(shape_mismatch.items())[:5])}"
        )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    missing, unexpected = set(missing), set(unexpected)

    if missing != expect or unexpected:
        unaccounted = sorted(missing - expect)
        absent = sorted(expect - missing)
        raise RuntimeError(
            "checkpoint load did not match expectations.\n"
            f"  missing and not expected ({len(unaccounted)}): {unaccounted[:8]}\n"
            f"  expected to be missing but present ({len(absent)}): {absent}\n"
            f"  unexpected in checkpoint ({len(unexpected)}): {sorted(unexpected)[:8]}\n"
            "A key missing but not expected usually means a module was renamed, "
            "which strict=False would have hidden."
        )

    return {
        "tensors_loaded": len(state_dict),
        "parameters": sum(v.numel() for v in state_dict.values()),
        "left_at_initialisation": sorted(expect),
    }
