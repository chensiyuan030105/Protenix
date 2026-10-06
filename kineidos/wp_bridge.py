"""The only place that imports WorldParticle.

Everything that crosses between the two codebases goes through here: which arm
of the ablation is running, how the K history frames become one `h`, how
per-frame time reaches WorldParticle, and the checks that the window arriving
here is still the one the dataloader produced.

Keeping the import in one file is what makes "we changed WorldParticle" a
reviewable statement: a reader can see every assumption about it in one place
rather than hunting through the training loop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from kineidos.window_align import canonicalize_window

# Plan section 4.  `none` and `zero` do not run WorldParticle at all, which is
# what keeps the ablation's control arms independent of it.
MODES = ("none", "zero", "random", "pretrained")

# cconv_embedding_dim 384 x factor 2.  factor is 2 rather than 3 because
# uses_obstacle_features("molecular") is false -- we have no obstacle branch.
TOKEN_DIM = 768

# Plan section 2.10: 0.35 nm cutoff, which reaches hydrogen bonds (0.28-0.30)
# and stacking (0.34).  config.yaml's 0.012 gives a 0.054 nm cutoff and leaves
# every atom isolated.
PARTICLE_RADIUS_NM = 0.0778

# config.yaml, except particle_radius and other_feats_channels.
WP_CONFIG = dict(
    kernel_size=[4, 4, 4],
    cconv_embedding_dim=384,
    radius_scale=1.5,
    data="molecular",
    use_topology_conv=None,
    topology_search_radius=None,
    topology_blend_alpha=None,
    coordinate_mapping="ball_to_cube_volume_preserving",
    interpolation="linear",
    use_window=True,
    particle_position_rope_dim=48,
    slot_compression_ratio=4,
    slot_attn_iters=6,
    slot_attn_heads=16,
    num_decoder_layers=8,
    decoder_attn_heads=12,
    decoder_attn_ffn_hidden_dim=512,
    decoder_attn_dropout=0.1,
    output_hidden_dim=512,
    output_layers=5,
    timestep=0.01,
    gravity=(0.0, -9.81, 0.0),
    obstacle_feats_channels=3,
    verbose=False,
    velhead=True,
)
K_PARTITE = 2

CANONICAL_TOL_NM = 1e-4


class WorldParticleBridge(nn.Module):
    """Turns a history window into the per-particle tokens `h`, or into nothing.

    Args:
        mode: one of MODES.  `none` returns None, so the fusion in
            AtomAttentionEncoder is skipped entirely and the model is upstream
            Protenix.  `zero` returns zeros of the right shape, which exercises
            the fusion path while carrying no information -- that is the arm
            that separates "the pathway changes behaviour" from "WorldParticle's
            features are useful".  `random` and `pretrained` run WorldParticle.
        particle_radius_nm: see plan section 2.10.  Overridable because the
            ablation carries 0.0578 (0.26 nm cutoff) as an alternative.
        checkpoint: required by `pretrained`, which Stage 1 has not produced.
        time_channels: how many channels of per-frame time to give
            WorldParticle.  1 is the normalised position of the frame within
            the window; see _frame_time_feature.
    """

    def __init__(
        self,
        mode: str,
        *,
        particle_radius_nm: float = PARTICLE_RADIUS_NM,
        checkpoint: Optional[str | Path] = None,
        time_channels: int = 1,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.mode = mode
        self.token_dim = TOKEN_DIM
        self.particle_radius_nm = float(particle_radius_nm)
        self.time_channels = int(time_channels)
        self.wp = None

        if mode in ("random", "pretrained"):
            # Imported here, not at module scope: `none` and `zero` must not
            # depend on WorldParticle being importable at all, so that the
            # control arms keep running if something breaks on that side.
            from models.particle_network_cross_attn_feat import (
                ParticleNetworkCrossAttnLocalFeature,
            )

            if seed is not None:
                torch.manual_seed(seed)
            self.wp = ParticleNetworkCrossAttnLocalFeature(
                particle_radius=self.particle_radius_nm,
                other_feats_channels=self.time_channels,
                **WP_CONFIG,
            )

        if mode == "pretrained":
            if checkpoint is None:
                raise ValueError(
                    "mode='pretrained' needs a checkpoint, and Stage 1 has not "
                    "produced one yet (plan section 1). The first version runs "
                    "none / zero / random."
                )
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            state = state.get("model", state)
            missing, unexpected = self.wp.load_state_dict(state, strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    f"WorldParticle checkpoint does not match this model: "
                    f"{len(missing)} missing, {len(unexpected)} unexpected. "
                    f"first missing: {sorted(missing)[:4]}. Note the "
                    f"fluid->molecular rename changed state_dict keys "
                    f"(conv0_molecular, dense0_molecular), so upstream weights "
                    f"need remapping rather than strict=False."
                )

    # ---------------------------------------------------------------- checks

    def assert_window_canonical(self, window, tol_nm: float = CANONICAL_TOL_NM) -> float:
        """Confirm nothing transformed the window after the dataloader.

        Canonicalisation is idempotent, so re-aligning an already-canonical
        window gives the identity and moves nothing.  Drift means something
        moved it -- most plausibly an augmentation extended to the
        WorldParticle side, which would make `h` carry the window's pose again
        and undo what plan section 2.9.3 is for.

        Returns the drift so a caller can log it.  Cheap: one 3x3 SVD.

        Note what this does *not* test.  It cannot tell whether
        canonicalisation ran, and does not need to: GAGU's trajectories are
        pre-fitted, so skipping it shifts a window by ~5e-06 nm, under any
        sensible tolerance and harmless.  A transform applied afterwards moves
        atoms by nanometres.
        """
        if not getattr(window, "canonicalized", False):
            return float("nan")
        pos = window.wp_position_nm.detach().double().cpu().numpy()
        ref = window.ref_pos_nm.detach().double().cpu().numpy()
        again, _, _ = canonicalize_window(pos, None, ref)
        drift = float(np.abs(again - pos).max())
        if drift > tol_nm:
            raise ValueError(
                f"the window handed to WorldParticle is {drift:.3e} nm from "
                f"canonical pose (limit {tol_nm:.0e}); something transformed it "
                f"after the dataloader. Canonicalisation is idempotent, so a "
                f"canonical window re-aligns to itself. See plan section 2.9.4."
            )
        return drift

    # ------------------------------------------------------------- features

    def _frame_time_feature(self, window) -> torch.Tensor:
        """Per-frame time, as a channel WorldParticle can use.

        Normalised to the window's own span, giving -1 for the oldest frame and
        -1/K for the newest.  Two reasons for normalising rather than passing
        nanoseconds.

        Scale: raw times run to hundreds of ns while velocities are ~1e-3
        nm/ps, and the two share the convolution's input channels, so an
        unnormalised time channel would dominate by orders of magnitude.

        Non-redundancy: the *physical* timescale already reaches the model
        through AdaLN alongside the diffusion noise level (plan section 2.8),
        so this channel only has to say where in the window each frame sits.
        Pooling destroys order, which is the whole reason it exists.
        """
        t = window.wp_frame_time_ns.detach().float()
        span = t.abs().max().clamp(min=1e-12)
        return (t / span).unsqueeze(-1)          # [K, 1], in [-1, -1/K]

    # -------------------------------------------------------------- forward

    def forward(self, window) -> Optional[torch.Tensor]:
        """`h` for one window: [N, 768], or None in `none` mode.

        WorldParticle takes one frame at a time, so the K history frames are run
        separately and pooled.  Pooling is a mask-weighted mean: padded frames
        carry a repeat of the oldest real frame (kineidos/data/windows.py) and
        must not contribute, and a plain mean over K would quietly average them
        in near the start of a trajectory.
        """
        n_atom = window.wp_position_nm.shape[1]

        if self.mode == "none":
            return None
        if self.mode == "zero":
            # Shaped like the real thing and carrying nothing. The fusion runs,
            # so this arm answers whether the pathway alone changes behaviour.
            return window.wp_position_nm.new_zeros((n_atom, self.token_dim))

        self.assert_window_canonical(window)

        time_feat = self._frame_time_feature(window)
        mask = window.wp_frame_mask.detach().to(torch.bool)
        if not bool(mask.any()):
            raise ValueError("no valid history frame in this window")

        tokens = []
        for k in range(window.wp_position_nm.shape[0]):
            if not bool(mask[k]):
                continue
            other = time_feat[k].expand(n_atom, self.time_channels).contiguous()
            out = self.wp.compute_correction(
                window.wp_position_nm[k],
                window.wp_velocity_nm_per_ps[k],
                other,
                None,
                None,
                k_partite=K_PARTITE,
                return_tokens=True,
                skip_output_network=True,
            )
            tokens.append(out["tokens"])

        return torch.stack(tokens).mean(dim=0)
