"""Plan section 5: what has to be visible while training, not afterwards.

v0.1 ran 5500 steps and produced a batch of conclusions; only a later look
inside the checkpoint showed the net injection strength was 1e-3 and the
WorldParticle branch had never participated in a prediction.  The point of this
module is that the same failure would be visible within a few hundred steps.

The quantity that matters is the *contribution* ratio, not the weight ratio.
A zero-initialised W_h can grow to a respectable norm and still contribute
nothing, because what reaches c_l' is W_h times LayerNorm(h), and the weight
norm says nothing about the scale of what it multiplies.  v0.1's gate looked
plausible by its own measure for exactly this reason.  So the encoder measures
the two halves' contributions during the forward pass
(AtomAttentionEncoder._fuse_wp_tokens) and this module reads them.
"""

from __future__ import annotations

from typing import Optional

import torch

# Below this, the WorldParticle half of c_l' is contributing less than a
# thousandth of what Protenix's own features do -- the regime v0.1 sat in for
# 5500 steps.  A warning rather than an error: early in training this is
# expected, since the branch starts at exactly zero by construction.
INJECTION_WARN_BELOW = 1e-3


def fusion_encoder(module: torch.nn.Module) -> Optional[torch.nn.Module]:
    """The AtomAttentionEncoder that carries the fusion, or None.

    Accepts either a whole Protenix model or the encoder itself, because
    callers have different things in hand: a training loop holds the model, a
    unit test holds the encoder.  An earlier version took only the model, so
    the test could not use this module at all and read the attribute directly
    -- which is how a key this module adds came to be missing from a line the
    test printed.

    Reached by attribute rather than by scanning for the class: the model has
    two AtomAttentionEncoders and only the diffusion module's carries the
    fusion, so a class search would also find the frozen trunk's.
    """
    if getattr(module, "wp_token_dim", None) is not None:
        return module
    enc = getattr(getattr(module, "diffusion_module", None),
                  "atom_attention_encoder", None)
    if enc is None or getattr(enc, "wp_token_dim", None) is None:
        return None
    return enc


def read(model: torch.nn.Module) -> dict[str, float]:
    """Diagnostics from the most recent fused forward pass.

    Empty if the fusion is absent or has not run yet -- which is itself worth
    logging, since in `none` mode it should be empty and in `random` mode it
    should not.
    """
    enc = fusion_encoder(model)
    if enc is None:
        return {}
    stats = dict(getattr(enc, "wp_diagnostics", {}) or {})
    if stats:
        # Recomputed from the weights rather than taken from the forward pass,
        # so a stale diagnostics dict cannot make the weights look current.
        with torch.no_grad():
            w = enc.wp_fusion.weight
            cl = w[:, : enc.c_atom].norm().item()
            h = w[:, enc.c_atom :].norm().item()
        stats["w_h_over_w_cl"] = h / cl if cl > 0 else float("nan")
    return stats


def warnings(stats: dict[str, float],
             threshold: float = INJECTION_WARN_BELOW) -> list[str]:
    """Human-readable alarms for a logger to print.

    Returned rather than printed so the caller decides the cadence; printing
    this every step would bury it.
    """
    out: list[str] = []
    s = stats.get("injection_strength")
    if s is not None and s == s and s < threshold:          # s == s rejects nan
        out.append(
            f"injection_strength {s:.3e} is below {threshold:.0e}: the "
            f"WorldParticle half of c_l' is contributing less than a thousandth "
            f"of Protenix's own features. Expected at the start, since W_h "
            f"begins at exactly zero; a problem if it persists. v0.1 sat here "
            f"for 5500 steps."
        )
    if stats.get("w_h_norm", 0.0) > 0 and not s:
        out.append(
            "W_h has left zero but the injection is still ~0, which means the "
            "weights grew without what they multiply mattering -- check "
            "h_norm_after_layernorm."
        )
    return out


def format_line(stats: dict[str, float]) -> str:
    """One compact line for a training log.

    A missing key prints as "?" rather than as nan.  The two mean different
    things -- "nobody recorded this" against "this was computed and came out
    undefined" -- and printing both as nan is the kind of conflation that costs
    an afternoon later.
    """
    if not stats:
        return "wp: no fusion active"

    def f(key: str) -> str:
        if key not in stats:
            return "?"
        v = stats[key]
        return f"{v:.3e}" if v == v else "nan"

    return (
        f"wp: inject={f('injection_strength')} "
        f"|W_h|/|W_cl|={f('w_h_over_w_cl')} "
        f"|h_contrib|={f('h_contribution_norm')} "
        f"|cl_contrib|={f('cl_contribution_norm')}"
    )
