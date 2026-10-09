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

import hashlib
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from kineidos.window_align import canonicalize_window

# Plan section 4.  `none` and `zero` do not run WorldParticle at all, which is
# what keeps the ablation's control arms independent of it.
#
# `oracle` is P010's probe and exists only on
# research/kineidos-v3-diag-oracle, which is never merged (D2 item 9).  It
# hands the diffusion head a fixed random projection of the *answer* -- the
# target frame, in the window's canonical pose -- and does not run
# WorldParticle at all.  By construction it is label leakage; the question it
# answers is whether the fusion pathway can carry information to the output,
# which is a property of the pathway and of the pretrained denoiser's prior,
# not of WorldParticle's features.  Its numbers are meaningless as a measure of
# anything else and must never reach a figure or a table outside P010's own
# readout.
MODES = ("none", "zero", "random", "pretrained", "oracle")

# The feature-dict key the oracle rides in.  Deliberately conspicuous and
# deliberately not inside input_feature_dict's normal namespace (D2 items 1-2):
# every mode but `oracle` raises if it finds this key, so a leak cannot travel
# by a collate being left switched on.
ORACLE_TARGET_KEY = "wp_oracle_target_nm"


# cconv_embedding_dim 384 x factor 2.  factor is 2 rather than 3 because
# uses_obstacle_features("molecular") is false -- we have no obstacle branch.
TOKEN_DIM = 768

# Plan section 2.10: 0.35 nm cutoff, which reaches hydrogen bonds (0.28-0.30)
# and stacking (0.34).  config.yaml's 0.012 gives a 0.054 nm cutoff and leaves
# every atom isolated.

# P010 section 8.6's replacement encoding.  Eight anchors and 48 frequencies
# multiply to TOKEN_DIM once sine and cosine are both taken; the assertion
# below is there because changing either without the other would silently
# produce a differently shaped h and the fusion's width check would blame the
# caller.
ORACLE_ENCODINGS = ("linear", "fourier_anchor")
ORACLE_K_ANCHOR = 8
ORACLE_M_FREQ = 48
# Wavelengths from the molecule's own size down to 0.25 nm.  Not finer: at a
# 0.05 nm floor the top frequency is 126 rad/nm, so a 0.01 nm displacement
# turns its phase by 1.26 rad and the channel wraps -- measured, a 0.1 A
# perturbation moved the LayerNormed encoding by 50%, which is a hash of the
# conformation rather than a coordinate the network can follow.  At 0.25 nm
# the response is linear in the displacement over 0.001-0.03 nm (x10 -> x9.93).
ORACLE_LAMBDA_MAX_NM = 4.0
ORACLE_LAMBDA_MIN_NM = 0.25
assert ORACLE_K_ANCHOR * ORACLE_M_FREQ * 2 == TOKEN_DIM


def oracle_omega(device=None, dtype=torch.float32) -> torch.Tensor:
    """The fixed frequency bank, in rad/nm.  Derived from constants rather than
    stored, so it needs no buffer and no checkpoint entry; `oracle_provenance`
    reports it so a run is still reproducible from its env.lock."""
    lam = torch.logspace(math.log10(ORACLE_LAMBDA_MAX_NM),
                         math.log10(ORACLE_LAMBDA_MIN_NM),
                         ORACLE_M_FREQ, device=device, dtype=dtype)
    return 2.0 * math.pi / lam


def oracle_anchor_index(n_atom: int, device=None) -> torch.Tensor:
    """Which atoms serve as distance references.  Spread along the chain and
    derived from n_atom alone: identical for every window, every arm and every
    run, so nothing has to be stored or matched at resume."""
    return torch.linspace(0, n_atom - 1, ORACLE_K_ANCHOR,
                          device=device).round().long()


def oracle_fourier_encode(target: torch.Tensor) -> torch.Tensor:
    """[N, 3] canonical-frame coordinates in nm -> [N, TOKEN_DIM], frame-free.

    Centred first.  Distances do not care about the offset, but `cdist` does:
    its default compute_mode expands |a-b|^2 as |a|^2 + |b|^2 - 2 a.b, and the
    canonical frame puts the molecule 5.17 nm from the origin, so those
    squared norms are ~27 while their difference is ~1.  Measured in float64
    before centring, the rotation+translation invariance held only to 3e-05;
    after centring and with the multiply-free mode, 5.7e-14.  float32 would
    have been far worse, and the symptom -- an oracle that is almost but not
    quite frame-free -- is one nobody would have gone looking for.
    """
    centred = target - target.mean(dim=0, keepdim=True)
    idx = oracle_anchor_index(target.shape[0], device=target.device)
    dist = torch.cdist(centred, centred[idx],
                       compute_mode="donot_use_mm_for_euclid_dist")
    phase = dist[..., None] * oracle_omega(target.device, target.dtype)
    return torch.cat([phase.sin(), phase.cos()],
                     dim=-1).reshape(target.shape[0], TOKEN_DIM)

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
        oracle_source: str = "",
        oracle_seed: int = 20261008,
        oracle_encoding: str = "linear",
    ) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.mode = mode
        self.token_dim = TOKEN_DIM
        self.particle_radius_nm = float(particle_radius_nm)
        self.time_channels = int(time_channels)
        self.wp = None
        self.oracle_seed: Optional[int] = None

        self.oracle_encoding = str(oracle_encoding)
        if self.oracle_encoding not in ORACLE_ENCODINGS:
            raise ValueError(
                f"oracle_encoding must be one of {ORACLE_ENCODINGS}, got "
                f"{self.oracle_encoding!r}"
            )

        if mode == "oracle":
            from kineidos.data import windows

            if oracle_source not in ("target", "decoy"):
                raise ValueError(
                    f"mode='oracle' needs oracle_source in ('target', "
                    f"'decoy'), got {oracle_source!r}. 'decoy' is the negative "
                    f"control and is not optional for a readable result "
                    f"(P010 D2 item 6)."
                )
            self.oracle_seed = int(oracle_seed)
            # Its own Generator, not the global RNG.  Drawing 2304 numbers from
            # the global stream would shift every later draw -- the diffusion
            # noise, the augmentation rotations -- so the oracle arm would stop
            # being paired with the baseline for a reason that has nothing to
            # do with the oracle.  Same hazard the fusion's own construction
            # order was written around (transformer.py's note on building
            # wp_layernorm last).
            gen = torch.Generator().manual_seed(self.oracle_seed)
            # 3 -> 768, unit-variance columns.  Full rank with probability one,
            # so the target is linearly recoverable from h and the only thing
            # standing between the oracle and a large effect is whether the
            # pathway and the prior let the information through.  Scaled by
            # 1/sqrt(3) so h's channel variance matches the input's rather than
            # being three times it; wp_layernorm would absorb the scale anyway,
            # but then the LayerNorm would be doing it rather than the design.
            projection = torch.randn(3, TOKEN_DIM, generator=gen) / math.sqrt(3)
            # A buffer, not a parameter: freeze_trunk makes every wp_bridge
            # *parameter* trainable, and a learnable projection would let the
            # arm reshape the answer instead of only transporting it.
            self.register_buffer("oracle_projection", projection)
            # Attaching the target to each window is the dataset's job, and the
            # dataset is built in init_data -- after after_model_built, which is
            # where this constructor runs, and before the dataloader forks.
            windows.set_oracle_source(oracle_source)

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

    def oracle_provenance(self) -> dict[str, object]:
        """What the oracle arm has to put in env.lock (D2 items 4 and 7).

        The projection is generated rather than stored, so the seed is what
        reproduces it -- and the digest is what proves the seed did.  Two runs
        that claim the same seed and hash different matrices is a question
        worth being able to ask.
        """
        if self.mode != "oracle":
            return {}
        from kineidos.data import windows

        p = self.oracle_projection.detach().cpu().contiguous()
        return {
            "oracle_source": windows.oracle_source(),
            "oracle_seed": self.oracle_seed,
            "oracle_encoding": self.oracle_encoding,
            "oracle_k_anchor": ORACLE_K_ANCHOR,
            "oracle_m_freq": ORACLE_M_FREQ,
            "oracle_lambda_nm": [ORACLE_LAMBDA_MAX_NM, ORACLE_LAMBDA_MIN_NM],
            "oracle_projection_shape": list(p.shape),
            "oracle_projection_sha256": hashlib.sha256(
                p.numpy().tobytes()).hexdigest(),
        }

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
        # D2 item 2, and first, before the early returns.  The oracle's key
        # existing in a non-oracle arm means the collate was left switched on,
        # which would make that arm's numbers unusable while looking entirely
        # normal -- the loss would simply be a little better.
        if self.mode != "oracle" and ORACLE_TARGET_KEY in feats:
            raise KeyError(
                f"{ORACLE_TARGET_KEY!r} is in the feature dict but wp.mode is "
                f"{self.mode!r}. That key is the target frame -- the answer. "
                f"Only the oracle probe may see it, and only on "
                f"research/kineidos-v3-diag-oracle. Something left "
                f"kineidos.data.windows.set_oracle_source switched on; find it "
                f"rather than removing this check."
            )

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

        if self.mode == "oracle":
            target = feats.get(ORACLE_TARGET_KEY)
            if target is None:
                raise KeyError(
                    f"mode='oracle' needs {ORACLE_TARGET_KEY!r} in the feature "
                    f"dict; kineidos/train/batch.py puts it there when the "
                    f"window carries it, and the window carries it when "
                    f"kineidos.data.windows.set_oracle_source is on. The "
                    f"constructor sets that, so an absent key means the bridge "
                    f"was built after the dataset."
                )
            # Same check as the real path: the oracle is expressed in the
            # window's canonical frame, so a window that something moved
            # afterwards would put the two in different frames.
            self.assert_window_canonical(feats)
            if tuple(target.shape) != (n_atom, 3):
                raise ValueError(
                    f"the oracle target is {tuple(target.shape)} but this "
                    f"window has {n_atom} atoms; these are the same atom set "
                    f"in the same order or the projection is nonsense"
                )
            if self.oracle_encoding == "fourier_anchor":
                # Same autocast discipline as the linear path below, and for
                # the same reason: the bridge runs under autocast(bfloat16)
                # while sample_diffusion_training consumes h with autocast
                # off, so h has to leave here as float32.
                with torch.autocast(device_type=position.device.type,
                                    enabled=False):
                    return oracle_fourier_encode(target.float())
            # h = x_target_canonical @ P, before wp_layernorm -- which is where
            # the real path hands over too, so the two differ in what h is and
            # in nothing else downstream.
            #
            # float32, with autocast explicitly off, and that is not a detail.
            # The bridge is called from Protenix.forward, which train_step
            # wraps in autocast(bfloat16); its output is consumed inside
            # sample_diffusion_training, which configs' skip_amp runs with
            # autocast DISABLED.  So wp_tokens has to be float32 -- the real
            # path's tokens come out of WorldParticle's own network as float32
            # and the fusion's weights are float32, which is why `random` has
            # run for 10,000 steps.  A matmul *under* autocast returns
            # bfloat16, so without this the oracle's h arrived as bf16 and
            # _fuse_wp_tokens' diagnostics block -- a raw `h @ w.T` with no
            # autocast to reconcile the two -- died with "expected mat1 and
            # mat2 to have the same dtype" three minutes into both oracle arms
            # (jobs 2149800 / 2149801).
            with torch.autocast(device_type=position.device.type,
                                enabled=False):
                return (target.float() @ self.oracle_projection.float())

        self.assert_window_canonical(feats)

        time_feat = self._frame_time_feature(feats)
        mask = feats["wp_frame_mask"].detach().to(torch.bool)
        if not bool(mask.any()):
            raise ValueError("no valid history frame in this window")

        tokens = []
        for k in range(position.shape[0]):
            if not bool(mask[k]):
                continue
            other = time_feat[k].expand(n_atom, self.time_channels).contiguous()
            out = self.wp.compute_correction(
                position[k],
                feats["wp_velocity_nm_per_ps"][k],
                other,
                None,
                None,
                k_partite=K_PARTITE,
                return_tokens=True,
                skip_output_network=True,
            )
            tokens.append(out["tokens"])

        return torch.stack(tokens).mean(dim=0)
