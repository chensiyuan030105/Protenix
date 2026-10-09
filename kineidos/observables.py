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
        # Everything derived from the *weights* is recomputed here and
        # overwrites whatever the forward pass recorded, so a stale diagnostics
        # dict cannot make the weights look current.  The forward pass keeps
        # only what depends on the activations -- the two contribution norms,
        # the injection strength and |h| after LayerNorm -- which nothing here
        # can reconstruct.
        #
        # Splitting it any other way bit once already: w_cl_drift and the ratio
        # were recomputed while w_h_norm was left at the forward's value, and an
        # alarm that reads w_h_norm then silently saw a stale zero.
        with torch.no_grad():
            w = enc.wp_fusion.weight
            cl_block = w[:, : enc.c_atom]
            eye = torch.eye(enc.c_atom, device=w.device, dtype=w.dtype)
            cl = cl_block.norm().item()
            h = w[:, enc.c_atom :].norm().item()
            drift = (cl_block - eye).norm().item()
        stats["w_cl_norm"] = cl
        stats["w_h_norm"] = h
        stats["w_h_over_w_cl"] = h / cl if cl > 0 else float("nan")
        # The `zero` arm's only instrument.  There h is identically zero, so
        # dL/dW_h = delta (x) h is zero and W_h cannot move -- correctly, not as
        # a fault -- which makes every h-side number structurally zero.  What
        # that arm asks is whether the fusion pathway itself changes the model,
        # and the one part of it that can move is the c_l block: identity at the
        # start, with a real gradient of delta (x) c_l.
        stats["w_cl_drift_from_identity"] = drift
        # LayerNorm's own scale and shift, which is what the optimizer has
        # actually been using to switch the injection off: P009 watched gamma's
        # rms fall from 0.979 to 0.14 over 3150 steps, forty points, monotone,
        # no plateau.  Recorded here because that number was only ever
        # recovered by opening a checkpoint afterwards, and P010's gamma-freeze
        # arms read it per step -- arm B's question is whether gamma starts
        # falling again within 500 steps of being released (section 6, item 3),
        # which a per-checkpoint reading cannot answer.
        #
        # rms rather than the norm, so the number is comparable across widths
        # and starts at exactly 1.0 (gamma = ones at initialisation).
        with torch.no_grad():
            ln = getattr(enc, "wp_layernorm", None)
            if ln is not None and getattr(ln, "weight", None) is not None:
                stats["gamma_rms"] = ln.weight.float().pow(2).mean().sqrt().item()
            if ln is not None and getattr(ln, "bias", None) is not None:
                stats["beta_rms"] = ln.bias.float().pow(2).mean().sqrt().item()
    return stats


def warnings(stats: dict[str, float],
             threshold: float = INJECTION_WARN_BELOW,
             mode: str = "random") -> list[str]:
    """Human-readable alarms for a logger to print.

    Returned rather than printed so the caller decides the cadence; printing
    this every step would bury it.

    `mode` is the ablation arm, and it changes which alarm means anything.  In
    `zero` the tokens are identically zero by construction, so the injection
    strength is zero no matter what the weights do -- warning about it there is
    noise that trains people to ignore the warning that matters.  What `zero`
    can still say is whether W_h moved at all, since nothing but a gradient
    through the fusion can move it.
    """
    out: list[str] = []
    s = stats.get("injection_strength")
    w_h = stats.get("w_h_norm", 0.0)

    if mode == "zero":
        # h is identically zero here, so dL/dW_h = delta (x) h is zero and W_h
        # cannot move.  An alarm about W_h would therefore fire on every step of
        # every `zero` run and teach people to ignore alarms.  What this arm asks
        # is whether the fusion pathway itself changes the model, and the one
        # part of it that can move is the c_l block -- identity at the start,
        # with a real gradient of delta (x) c_l.
        drift = stats.get("w_cl_drift_from_identity")
        if drift is not None and drift == drift and drift == 0.0:
            out.append(
                "W_cl is still exactly the identity. In the `zero` arm that is "
                "the only thing that can move -- h is zero by construction, so "
                "W_h and the injection strength say nothing -- and no drift "
                "after training means no gradient reached the fusion at all."
            )
        if w_h != 0.0:
            out.append(
                f"W_h has left zero ({w_h:.3e}) in the `zero` arm, where its "
                f"gradient is identically zero. Weight decay can do this; "
                f"anything larger than that means h was not actually zero."
            )
        return out

    if s is not None and s == s and s < threshold:          # s == s rejects nan
        out.append(
            f"injection_strength {s:.3e} is below {threshold:.0e}: the "
            f"WorldParticle half of c_l' is contributing less than a thousandth "
            f"of Protenix's own features. Expected at the start, since W_h "
            f"begins at exactly zero; a problem if it persists. v0.1 sat here "
            f"for 5500 steps."
        )
    if w_h > 0 and not s:
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
        f"|cl_contrib|={f('cl_contribution_norm')} "
        f"W_cl-I={f('w_cl_drift_from_identity')}"
    )
