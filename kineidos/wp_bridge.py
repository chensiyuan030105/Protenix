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
from kineidos.wp_features import OTHER_FEATS_CHANNELS, HistoryFeatureContract

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
    # P011 D1.  `particle_position_rope_dim` above is now inert -- the two 3D
    # RoPEs it sized are gone -- but it is left in because `wp-v3` still
    # accepts it and removing it from this dict would make the two configs
    # diverge for no reason.  These three are what replaced it.
    #
    # 2.0 nm for the attention bias: attention is all-to-all, so unlike the
    # convolution (bounded by the search radius) two atoms can be the whole
    # diameter of the molecule apart.  GAGU's 470 heavy atoms span ~2.5 nm, so
    # 16 centres at 0.13 nm spacing cover the molecule and the RBF's monotone
    # tail channel keeps anything beyond it ordered rather than saturated.
    distance_bias_cutoff=2.0,
    distance_bias_num_basis=16,
    # Per-branch LayerNorm on `other_feats` before the concat (D2 / STAR-MD
    # A12).  See models/super_particle_layers.py for why this branch and not
    # `ones` or the velocity slot.
    normalize_other_feats=True,
    # Per-branch LayerNorm on [conv0_out, dense0_out] before the
    # `particle_features` concat.  Same rule as above, one concat later, and
    # the checkup says it is the one that decides whether `h` carries
    # conformation at all: raw, the per-atom cosine between two conformations
    # goes from 0.9938 at conv0_molecular to 0.999991 at the concat, because
    # dense0 sees no positions and is pure identity.
    normalize_feature_branches=True,
)
K_PARTITE = 2

CANONICAL_TOL_NM = 1e-4


# The window tensors WorldParticle needs, as they travel in the feature dict.
# They go through the feature dict rather than as a Window object for one
# practical reason: the trainer calls to_device on the batch, which moves
# tensors inside dictionaries and would leave a dataclass's fields on the CPU.
WP_INPUT_KEYS = (
    "wp_position_nm",
    "wp_velocity_nm_per_ps",
    "wp_frame_time_ns",
    "wp_frame_mask",
    "wp_ref_pos_nm",
    "wp_canonicalized",
)


def wp_inputs_from_window(window) -> dict[str, torch.Tensor]:
    """The WP_INPUT_KEYS subset of a Window, all as tensors.

    `canonicalized` becomes a 0-d tensor rather than staying a Python bool so
    that it survives to_device and any collation unchanged in kind.

    `wp_velocity_nm_per_ps` is still collated and is no longer *consumed*:
    P011 D2 feeds zeros down WorldParticle's velocity slot.  It stays in the
    window and in this dict on purpose -- it is what makes "we stopped feeding
    velocities" a reversible decision and a one-line ablation rather than a
    data-pipeline change, and `forward` asserts against it so that the zeroing
    is visible at the point of use instead of being a missing key.
    """
    return {
        "wp_position_nm": window.wp_position_nm,
        "wp_velocity_nm_per_ps": window.wp_velocity_nm_per_ps,
        "wp_frame_time_ns": window.wp_frame_time_ns,
        "wp_frame_mask": window.wp_frame_mask,
        "wp_ref_pos_nm": window.ref_pos_nm,
        "wp_canonicalized": torch.tensor(bool(window.canonicalized)),
    }


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
        shuffle_h_seed: P011's `wp-inv-shuffle` arm.  Not None permutes `h`
            along the atom axis with a fixed permutation, which keeps the
            distribution of every channel and destroys the atom
            correspondence -- the negative control that separates "the history
            says something about *this* atom" from "the extra capacity
            helps".  P010 13.5 is why it exists: the `decoy` control was zero
            for a coordinate encoding and emphatically not zero for a distance
            encoding, so a control has to be designed against the encoding it
            is controlling for.
    """

    def __init__(
        self,
        mode: str,
        *,
        particle_radius_nm: float = PARTICLE_RADIUS_NM,
        checkpoint: Optional[str | Path] = None,
        seed: Optional[int] = None,
        shuffle_h_seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.mode = mode
        self.token_dim = TOKEN_DIM
        self.particle_radius_nm = float(particle_radius_nm)
        self.shuffle_h_seed = shuffle_h_seed
        self.features = None
        self.wp = None
        # Lazily built and then cached as a buffer, so it is written into the
        # checkpoint: a shuffle whose permutation changed between a run and its
        # resume would be a third arm nobody launched.
        self.register_buffer("shuffle_index", None, persistent=True)

        if mode in ("random", "pretrained"):
            # Imported here, not at module scope: `none` and `zero` must not
            # depend on WorldParticle being importable at all, so that the
            # control arms keep running if something breaks on that side.
            from models.particle_network_cross_attn_feat import (
                ParticleNetworkCrossAttnLocalFeature,
            )

            # **Seed first, then build -- both of them.**  The contract holds
            # the only learnable branch in the feature path (the atom-name
            # embedding), and the first version of this constructor built it
            # *above* the manual_seed, so those weights came from whatever
            # global RNG state the process happened to be in.  Two checkup runs
            # with identical arguments then reported conv0_molecular's per-atom
            # fraction as 45.9% and 67.1%, which looks exactly like a code
            # change and was in fact an unseeded embedding.  Caught by the
            # inconsistency on 2026-10-09, before any arm ran; it would have
            # made `wp-inv` and `wp-inv-seed2` differ by more than their
            # nominal seeds and put that difference into the seed floor the
            # whole read-out is measured against.
            if seed is not None:
                torch.manual_seed(seed)
            # P011 D2, plan section 5.1.  Built before WorldParticle, because
            # its channel count is WorldParticle's `other_feats_channels` and
            # the two must not be able to disagree.
            self.features = HistoryFeatureContract()
            self.wp = ParticleNetworkCrossAttnLocalFeature(
                particle_radius=self.particle_radius_nm,
                other_feats_channels=self.features.channels,
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

    def assert_window_canonical(self, feats, tol_nm: float = CANONICAL_TOL_NM) -> float:
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
        if not bool(feats["wp_canonicalized"]):
            return float("nan")
        pos = feats["wp_position_nm"].detach().double().cpu().numpy()
        ref = feats["wp_ref_pos_nm"].detach().double().cpu().numpy()
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

    def _frame_time_feature(self, feats) -> torch.Tensor:
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
        t = feats["wp_frame_time_ns"].detach().float()
        span = t.abs().max().clamp(min=1e-12)
        return (t / span).unsqueeze(-1)          # [K, 1], in [-1, -1/K]

    # -------------------------------------------------------------- forward

    def forward(self, feats) -> Optional[torch.Tensor]:
        """`h` for one window: [N, 768], or None in `none` mode.

        WorldParticle takes one frame at a time, so the K history frames are run
        separately and pooled.  Pooling is a mask-weighted mean: padded frames
        carry a repeat of the oldest real frame (kineidos/data/windows.py) and
        must not contribute, and a plain mean over K would quietly average them
        in near the start of a trajectory.
        """
        missing = [k for k in WP_INPUT_KEYS if k not in feats]
        if self.mode != "none" and missing:
            raise KeyError(
                f"the feature dict is missing {missing}; the collate has to put "
                f"the window's WorldParticle tensors there (see WP_INPUT_KEYS)"
            )
        position = feats["wp_position_nm"]
        n_atom = position.shape[1]

        if self.mode == "none":
            return None
        if self.mode == "zero":
            # Shaped like the real thing and carrying nothing. The fusion runs,
            # so this arm answers whether the pathway alone changes behaviour.
            return position.new_zeros((n_atom, self.token_dim))

        self.assert_window_canonical(feats)

        time_feat = self._frame_time_feature(feats)
        mask = feats["wp_frame_mask"].detach().to(torch.bool)
        if not bool(mask.any()):
            raise ValueError("no valid history frame in this window")

        # P011 D2.  Frame-independent, so it is built once and reused across
        # the K frames rather than K times -- and, more to the point, so that
        # "the identity does not depend on the frame" is a property of the code
        # and not of a convention somebody has to maintain.
        identity = self.features.identity(feats)
        if identity.shape[0] != n_atom:
            raise ValueError(
                f"the contract produced identity for {identity.shape[0]} "
                f"atoms but the history window has {n_atom}. These are the "
                f"same 470 heavy atoms in the same order -- `h` is "
                f"concatenated onto `c_l` position by position, so a mismatch "
                f"here is a silently wrong model, not a shape error."
            )

        # P011 D2: zeros, not the collated velocities.  Built from `position`
        # so dtype and device follow it.  Two reasons, and the second was
        # measured rather than argued: the content is thin (2e-4 nm/ps at a 100
        # ps save interval, P008 1.4), and the three components are a
        # *lab-frame vector*, so feeding them re-breaks the rotation invariance
        # D1 just built -- the symmetry probe puts that at 7.5e-04 against
        # 2.5e-07 at the stage where both are consumed, a factor of 3000
        # (artifacts/reports/P011/symmetry.md).
        zero_vel = torch.zeros_like(position[0])

        tokens = []
        for k in range(position.shape[0]):
            if not bool(mask[k]):
                continue
            other = self.features(time_feat[k], identity)
            out = self.wp.compute_correction(
                position[k],
                zero_vel,
                other,
                None,
                None,
                k_partite=K_PARTITE,
                return_tokens=True,
                skip_output_network=True,
            )
            tokens.append(out["tokens"])

        h = torch.stack(tokens).mean(dim=0)
        if self.shuffle_h_seed is not None:
            h = h[self._shuffle_index(n_atom, h.device)]
        return h

    # -------------------------------------------------------------- shuffle

    def _shuffle_index(self, n_atom: int, device) -> torch.Tensor:
        """A fixed permutation of the atom axis, built once and checkpointed.

        Distribution-preserving and correspondence-destroying: every channel's
        values over atoms are exactly the ones WorldParticle produced, and
        every one of them is attached to the wrong atom.  So this arm has the
        same capacity, the same magnitudes, the same LayerNorm statistics and
        the same gradient scale as `wp-inv`, and none of its per-atom
        information -- which is the only way to read "the history is specific
        to this conformation" off a difference between two arms (P010 14.5-14.6
        measured 74% of the oracle effect that way).

        Drawn from its own Generator, not the global RNG, for the reason the dt
        pathway restores the global state: the shuffle arm must differ from
        `wp-inv` in the permutation and in nothing else, and consuming global
        draws at construction time would also change the model's
        initialisation.
        """
        idx = self.shuffle_index
        if idx is None or idx.numel() != n_atom:
            if idx is not None:
                raise ValueError(
                    f"shuffle_index holds {idx.numel()} entries but this "
                    f"window has {n_atom} atoms. The permutation is "
                    f"checkpointed so that a resume cannot silently become a "
                    f"different arm; a size change means the atom set changed "
                    f"under it."
                )
            gen = torch.Generator().manual_seed(int(self.shuffle_h_seed))
            idx = torch.randperm(n_atom, generator=gen)
            self.shuffle_index = idx.to(device)
        return self.shuffle_index.to(device)
