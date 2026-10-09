# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Union

import torch
import torch.nn as nn

from protenix.model.modules.embedders import FourierEmbedding, RelativePositionEncoding
from protenix.model.modules.primitives import LinearNoBias, Transition
from protenix.model.modules.transformer import (
    AtomAttentionDecoder,
    AtomAttentionEncoder,
    DiffusionTransformer,
)
from protenix.model.triangular.layers import LayerNorm
from protenix.model.utils import expand_at_dim, get_checkpoint_fn, permute_final_dims


class DiffusionConditioning(nn.Module):
    """
    Implements Algorithm 21 in AF3

    Args:
        sigma_data (float, optional): the standard deviation of the data. Defaults to 16.0.
        c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
        c_s (int, optional):  hidden dim [for single embedding]. Defaults to 384.
        c_s_inputs (int, optional): input embedding dim from InputEmbedder. Defaults to 449.
        c_noise_embedding (int, optional): noise embedding dim. Defaults to 256.
    """

    def __init__(
        self,
        sigma_data: float = 16.0,
        c_z: int = 128,
        c_s: int = 384,
        c_s_inputs: int = 449,
        c_noise_embedding: int = 256,
        # Kineidos P011 D4.  False keeps this module byte-identical to
        # upstream, parameters included.
        dt_conditioning: bool = False,
    ) -> None:
        super(DiffusionConditioning, self).__init__()
        self.sigma_data = sigma_data
        self.c_z = c_z
        self.c_s = c_s
        self.c_s_inputs = c_s_inputs
        # Line1-Line3:
        self.relpe = RelativePositionEncoding(c_z=c_z)
        self.layernorm_z = LayerNorm(2 * self.c_z, create_offset=False)
        self.linear_no_bias_z = LinearNoBias(
            in_features=2 * self.c_z, out_features=self.c_z, precision=torch.float32
        )
        # Line3-Line5:
        self.transition_z1 = Transition(c_in=self.c_z, n=2)
        self.transition_z2 = Transition(c_in=self.c_z, n=2)

        # Line6-Line7
        self.layernorm_s = LayerNorm(self.c_s + self.c_s_inputs, create_offset=False)
        self.linear_no_bias_s = LinearNoBias(
            in_features=self.c_s + self.c_s_inputs,
            out_features=self.c_s,
            precision=torch.float32,
        )
        # Line8-Line9
        self.fourier_embedding = FourierEmbedding(c=c_noise_embedding)
        self.layernorm_n = LayerNorm(c_noise_embedding, create_offset=False)
        self.linear_no_bias_n = LinearNoBias(
            in_features=c_noise_embedding,
            out_features=self.c_s,
            precision=torch.float32,
        )
        # Line10-Line12
        self.transition_s1 = Transition(c_in=self.c_s, n=2)
        self.transition_s2 = Transition(c_in=self.c_s, n=2)

        # ---------------- Kineidos P011 D4: the dt pathway ----------------
        # The model is asked to predict the structure dt later, and P009
        # established that it has never been told dt: `delta_t_ns` stops at the
        # collate's `basic` dict, no module under protenix/model/ reads it, and
        # `dt_fourier|dt_linear|dt_layernorm` had zero hits across all six
        # trees.  P004 2.8 designed this and it was never built.  The frame-time
        # channel the bridge does send is normalised by the window's own span,
        # which divides the stride out, so it cannot stand in.
        #
        # Deliberately the same three pieces, in the same order, as the noise
        # level immediately above -- FourierEmbedding -> LayerNorm ->
        # LinearNoBias, added to `single_s`.  STAR-MD conditions on
        # log10(dt) through AdaLN-Zero; this is the same idea at the one place
        # Protenix already has for "a scalar the whole diffusion head should
        # know".
        #
        # `initializer="zeros"` is what makes it inert at step 0, the same
        # AdaLN-Zero trick AdaptiveLayerNorm uses (primitives.py:117-120).
        # D16's weight_decay=0 matters here: at 1e-8 the optimiser would have
        # been shrinking a parameter whose gradient is zero, which is how P010
        # found the fusion's gate being destroyed by the optimiser rather than
        # by the data.
        self.dt_conditioning = bool(dt_conditioning)
        if self.dt_conditioning:
            # **The global RNG is restored afterwards, on purpose.**
            # nn.Linear.__init__ calls reset_parameters() before our
            # `initializer="zeros"` overwrites the weight, so building these
            # three modules consumes draws from the global generator and shifts
            # every later draw.  The dt ablation (`wp-inv` against
            # `wp-inv-nodt`) is only interpretable if the two arms differ in
            # the pathway and nothing else: without this, they would also get
            # different fusion and bridge initialisations, and since the
            # expected dt effect is *below* the seed floor (D4), that
            # difference would be most of what the ablation measured.
            # FourierEmbedding needs no such care -- it draws from its own
            # torch.Generator (embedders.py:229).  CPU state only, which is the
            # right one: modules are constructed on CPU and moved afterwards.
            _rng_state = torch.get_rng_state()
            # A different seed from the noise path's default 42, so that dt and
            # the noise level do not get the identical random projection.  They
            # take different inputs, so sharing would not be a bug, but two
            # pathways that are literally the same function of their argument
            # is a coincidence waiting to be mistaken for a finding.
            self.dt_fourier = FourierEmbedding(c=c_noise_embedding, seed=1011)
            self.dt_layernorm = LayerNorm(c_noise_embedding, create_offset=False)
            self.dt_linear = LinearNoBias(
                in_features=c_noise_embedding,
                out_features=self.c_s,
                precision=torch.float32,
                initializer="zeros",
            )
            torch.set_rng_state(_rng_state)

        print(f"Diffusion Module has {self.sigma_data}")

    def dt_embedding(self, delta_t_ns: torch.Tensor) -> torch.Tensor:
        """`log10(dt / 1 ns)` -> `[c_s]`, to be broadcast onto `single_s`.

        Returns a 1-d tensor so it broadcasts over both N_sample and N_token
        without the caller having to know how many leading batch axes there
        are.

        log10, not dt: P009 2.3 pinned the range to [0.1, 1] ns, over which a
        linear scale would hand the Fourier features a 10x spread while
        log10 hands them [-1, 0].  It is also STAR-MD's parameterisation
        (LogUniform[1e-2, 1e1] ns), so the two stay comparable.
        """
        dt = delta_t_ns
        if not torch.is_tensor(dt):
            dt = torch.tensor(float(dt))
        dt = dt.to(device=self.dt_linear.weight.device).float().reshape(-1)
        if dt.numel() != 1:
            raise ValueError(
                f"delta_t_ns must be one value per batch, got {dt.numel()}; "
                f"batch size is 1 by construction here (the loss calls "
                f"resolution.item())"
            )
        if not bool((dt > 0).all()):
            raise ValueError(
                f"delta_t_ns must be positive to take log10, got {float(dt)}"
            )
        log10_dt = torch.log10(dt).reshape(())
        return self.dt_linear(self.dt_layernorm(self.dt_fourier(log10_dt)))

    def prepare_cache(
        self,
        relp_feature: torch.Tensor,
        z_trunk: torch.Tensor,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        # Pair conditioning
        pair_z = torch.cat(
            tensors=[
                z_trunk,
                self.relpe(relp_feature),
            ],
            dim=-1,
        )  # [..., N_tokens, N_tokens, 2*c_z]
        pair_z = self.linear_no_bias_z(self.layernorm_z(pair_z))
        if inplace_safe:
            pair_z += self.transition_z1(pair_z)
            pair_z += self.transition_z2(pair_z)
        else:
            pair_z = pair_z + self.transition_z1(pair_z)
            pair_z = pair_z + self.transition_z2(pair_z)
        return pair_z

    def forward(
        self,
        t_hat_noise_level: torch.Tensor,
        relp_feature: torch.Tensor,
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        pair_z: torch.Tensor,
        inplace_safe: bool = False,
        use_conditioning: bool = True,
        delta_t_ns: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            t_hat_noise_level (torch.Tensor): the noise level
                [..., N_sample]
            delta_t_ns (torch.Tensor): Kineidos P011 D4, the physical time gap
                this prediction spans, in ns.  None skips the pathway, which
                is what makes the step-0 identity check possible on a single
                model instance rather than on two differently-seeded ones.
            asym_id (torch.Tensor): asym_id
            residue_index (torch.Tensor): residue_index
            entity_id (torch.Tensor): entity_id
            token_index (torch.Tensor): token_index
            sym_id (torch.Tensor): sym_id
            s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
                [..., N_tokens, c_s_inputs]
            s_trunk (torch.Tensor): single feature embedding from PairFormer (Alg17)
                [..., N_tokens, c_s]
            z_trunk (torch.Tensor): pair feature embedding from PairFormer (Alg17)
                [..., N_tokens, N_tokens, c_z]
            inplace_safe (bool): Whether it is safe to use inplace operations.
            use_conditioning (bool): Whether to drop the s/z embeddings.
        Returns:
            tuple[torch.Tensor, torch.Tensor]: embeddings s and z
                - s (torch.Tensor): [..., N_sample, N_tokens, c_s]
                - z (torch.Tensor): [..., N_tokens, N_tokens, c_z]
        """
        if pair_z is None:
            if not use_conditioning:
                if inplace_safe:
                    s_trunk *= 0
                    z_trunk *= 0
                else:
                    s_trunk = 0 * s_trunk
                    z_trunk = 0 * z_trunk
            pair_z = self.prepare_cache(relp_feature, z_trunk, inplace_safe)
        else:
            # Pair conditioning
            if inplace_safe:
                pair_z_clone = pair_z.clone()
                pair_z = pair_z_clone
        # Single conditioning
        single_s = torch.cat(
            tensors=[s_trunk, s_inputs], dim=-1
        )  # [..., N_tokens, c_s + c_s_inputs]
        single_s = self.linear_no_bias_s(self.layernorm_s(single_s))
        noise_n = self.fourier_embedding(
            t_hat_noise_level=torch.log(input=t_hat_noise_level / self.sigma_data) / 4
        ).to(
            single_s.dtype
        )  # [..., N_sample, c_in]
        single_s = single_s.unsqueeze(dim=-3) + self.linear_no_bias_n(
            self.layernorm_n(noise_n)
        ).unsqueeze(
            dim=-2
        )  # [..., N_sample, N_tokens, c_s]
        # P011 D4.  Added to the same `single_s`, after the noise level and
        # before the transitions, so dt reaches every block the noise level
        # reaches.  `dt_linear` is zero-initialised, so at step 0 this adds
        # exactly 0.0 and the sum is bit-identical to not adding it --
        # kineidos/check_dt_conditioning.py measures that, and measures that a
        # non-zero dt_linear does change the output, because an identity check
        # alone is satisfied by a pathway that is wired to nothing at all.
        if self.dt_conditioning and delta_t_ns is not None:
            single_s = single_s + self.dt_embedding(delta_t_ns).to(
                single_s.dtype
            )
        if inplace_safe:
            single_s += self.transition_s1(single_s)
            single_s += self.transition_s2(single_s)
        else:
            single_s = single_s + self.transition_s1(single_s)
            single_s = single_s + self.transition_s2(single_s)
        return single_s, pair_z


class DiffusionSchedule:
    """
    Diffusion schedule for training and inference.

    Args:
        sigma_data (float, optional): The standard deviation of the data. Defaults to 16.0.
        s_max (float, optional): The maximum noise level. Defaults to 160.0.
        s_min (float, optional): The minimum noise level. Defaults to 4e-4.
        p (float, optional): The exponent for the noise schedule. Defaults to 7.0.
        dt (float, optional): The time step size. Defaults to 1/200.
        p_mean (float, optional): The mean of the log-normal distribution for noise level sampling. Defaults to -1.2.
        p_std (float, optional): The standard deviation of the log-normal distribution for noise level sampling. Defaults to 1.5.
    """

    def __init__(
        self,
        sigma_data: float = 16.0,
        s_max: float = 160.0,
        s_min: float = 4e-4,
        p: float = 7.0,
        dt: float = 1 / 200,
        p_mean: float = -1.2,
        p_std: float = 1.5,
    ) -> None:
        self.sigma_data = sigma_data
        self.s_max = s_max
        self.s_min = s_min
        self.p = p
        self.dt = dt
        self.p_mean = p_mean
        self.p_std = p_std
        # self.T
        self.T = int(1 / dt) + 1  # 201

    def get_train_noise_schedule(self) -> torch.Tensor:
        return self.sigma_data * torch.exp(self.p_mean + self.p_std * torch.randn(1))

    def get_inference_noise_schedule(self) -> torch.Tensor:
        time_step_lists = torch.arange(start=0, end=1 + 1e-10, step=self.dt)
        inference_noise_schedule = (
            self.sigma_data
            * (
                self.s_max ** (1 / self.p)
                + time_step_lists
                * (self.s_min ** (1 / self.p) - self.s_max ** (1 / self.p))
            )
            ** self.p
        )
        return inference_noise_schedule


class DiffusionModule(nn.Module):
    """
    Implements Algorithm 20 in AF3

    Args:
        sigma_data (torch.float, optional): the standard deviation of the data. Defaults to 16.0.
        c_atom (int, optional): embedding dim for atom feature. Defaults to 128.
        c_atompair (int, optional): embedding dim for atompair feature. Defaults to 16.
        c_token (int, optional): feature channel of token (single a). Defaults to 768.
        c_s (int, optional):  hidden dim [for single embedding]. Defaults to 384.
        c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
        c_s_inputs (int, optional): hidden dim [for single input embedding]. Defaults to 449.
        atom_encoder (dict[str, int], optional): configs in AtomAttentionEncoder. Defaults to {"n_blocks": 3, "n_heads": 4}.
        transformer (dict[str, int], optional): configs in DiffusionTransformer. Defaults to {"n_blocks": 24, "n_heads": 16}.
        atom_decoder (dict[str, int], optional): configs in AtomAttentionDecoder. Defaults to {"n_blocks": 3, "n_heads": 4}.
        drop_path_rate (float, optional): drop path rate. Defaults to 0.0.
        blocks_per_ckpt: number of atom_encoder/transformer/atom_decoder blocks in each activation checkpoint
            Size of each chunk. A higher value corresponds to fewer
            checkpoints, and trades memory for speed. If None, no checkpointing is performed.
        use_fine_grained_checkpoint: whether use fine-gained checkpoint for finetuning stage 2
            only effective if blocks_per_ckpt is not None.
    """

    def __init__(
        self,
        sigma_data: float = 16.0,
        c_atom: int = 128,
        c_atompair: int = 16,
        c_token: int = 768,
        c_s: int = 384,
        c_z: int = 128,
        c_s_inputs: int = 449,
        atom_encoder: dict[str, int] = {"n_blocks": 3, "n_heads": 4},
        transformer: dict[str, int] = {
            "n_blocks": 24,
            "n_heads": 16,
            "drop_path_rate": 0,
        },
        atom_decoder: dict[str, int] = {"n_blocks": 3, "n_heads": 4},
        drop_path_rate: float = 0.0,
        blocks_per_ckpt: Optional[int] = None,
        use_fine_grained_checkpoint: bool = False,
        # Kineidos: width of WorldParticle's per-particle tokens, or None to
        # keep this module byte-identical to upstream.
        wp_token_dim: Optional[int] = None,
        # Kineidos P011 D4: condition the diffusion head on log10(dt).
        dt_conditioning: bool = False,
    ) -> None:
        super(DiffusionModule, self).__init__()
        self.sigma_data = sigma_data
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.c_token = c_token
        self.c_s_inputs = c_s_inputs
        self.c_s = c_s
        self.c_z = c_z

        # Grad checkpoint setting
        self.blocks_per_ckpt = blocks_per_ckpt
        self.use_fine_grained_checkpoint = use_fine_grained_checkpoint

        self.dt_conditioning = bool(dt_conditioning)
        self.diffusion_conditioning = DiffusionConditioning(
            sigma_data=self.sigma_data, c_z=c_z, c_s=c_s, c_s_inputs=c_s_inputs,
            dt_conditioning=self.dt_conditioning,
        )
        self.atom_attention_encoder = AtomAttentionEncoder(
            **atom_encoder,
            c_atom=c_atom,
            c_atompair=c_atompair,
            c_token=c_token,
            has_coords=True,
            c_s=c_s,
            c_z=c_z,
            blocks_per_ckpt=blocks_per_ckpt,
            # Kineidos: only this encoder fuses WorldParticle's tokens.  The
            # other instance of AtomAttentionEncoder lives in embedders.py
            # inside the frozen trunk and is left at wp_token_dim=None.
            wp_token_dim=wp_token_dim,
        )
        # Alg20: line4
        self.layernorm_s = LayerNorm(c_s, create_offset=False)
        self.linear_no_bias_s = LinearNoBias(
            in_features=c_s,
            out_features=c_token,
            precision=torch.float32,
            initializer="zeros",
        )
        self.diffusion_transformer = DiffusionTransformer(
            **transformer,
            c_a=c_token,
            c_s=c_s,
            c_z=c_z,
            blocks_per_ckpt=blocks_per_ckpt,
        )
        self.layernorm_a = LayerNorm(c_token, create_offset=False)
        self.atom_attention_decoder = AtomAttentionDecoder(
            **atom_decoder,
            c_token=c_token,
            c_atom=c_atom,
            c_atompair=c_atompair,
            blocks_per_ckpt=blocks_per_ckpt,
        )
        self.normalize = LayerNorm(c_z, create_offset=False, create_scale=False)

    def f_forward(
        self,
        r_noisy: torch.Tensor,
        t_hat_noise_level: torch.Tensor,
        input_feature_dict: dict[str, Union[torch.Tensor, int, float, dict]],
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        pair_z: torch.Tensor,
        p_lm: torch.Tensor,
        c_l: torch.Tensor,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
        use_conditioning: bool = True,
        enable_efficient_fusion: bool = False,
    ) -> torch.Tensor:
        """The raw network to be trained.
        As in EDM equation (7), this is F_theta(c_in * x, c_noise(sigma)).
        Here, c_noise(sigma) is computed in Conditioning module.

        Args:
            r_noisy (torch.Tensor): scaled x_noisy (i.e., c_in * x)
                [..., N_sample, N_atom, 3]
            t_hat_noise_level (torch.Tensor): the noise level, as well as the time step t
                [..., N_sample]
            input_feature_dict (dict[str, Union[torch.Tensor, int, float, dict]]): input feature
            s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
                [..., N_tokens, c_s_inputs]
            s_trunk (torch.Tensor): single feature embedding from PairFormer (Alg17)
                [..., N_tokens, c_s]
            z_trunk (torch.Tensor): pair feature embedding from PairFormer (Alg17)
                [..., N_tokens, N_tokens, c_z]
            pair_z (torch.Tensor): diffusion pair embedding
                [..., N_tokens, N_tokens, c_z]
            p_lm (torch.Tensor): MSA embedding
                [..., N_tokens, c_p_lm]
            c_l (torch.Tensor): ligand embedding
                [..., N_tokens, c_c_l]
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.
            use_conditioning (bool): Whether to drop the s/z embeddings in DiffusionConditioning.
            enable_efficient_fusion (bool): Whether to enable efficient fusion. Defaults to False.

        Returns:
            torch.Tensor: coordinates update
                [..., N_sample, N_atom, 3]
        """
        N_sample = r_noisy.size(-3)
        assert t_hat_noise_level.size(-1) == N_sample

        blocks_per_ckpt = self.blocks_per_ckpt
        if not torch.is_grad_enabled():
            blocks_per_ckpt = None
        # Conditioning, shared across difference samples
        # Diffusion_conditioning consumes 7-8G when token num is 768,
        # use checkpoint here if blocks_per_ckpt is not None.
        if blocks_per_ckpt:
            checkpoint_fn = get_checkpoint_fn()
            s_single, z_pair = checkpoint_fn(
                self.diffusion_conditioning,
                t_hat_noise_level,
                input_feature_dict["relp"],
                s_inputs,
                s_trunk,
                z_trunk,
                pair_z,
                inplace_safe,
                use_conditioning,
                # Positional, so this has to track DiffusionConditioning.forward's
                # order -- the same trap `wp_tokens` carries a note about a few
                # dozen lines below.  Omitting it here and not in the `else`
                # branch would give a run whose dt pathway was live only when
                # gradient checkpointing happened to be off, i.e. live in eval
                # and dead in training, which is both wrong and invisible.
                input_feature_dict.get("delta_t_ns", None),
            )
        else:
            s_single, z_pair = self.diffusion_conditioning(
                t_hat_noise_level,
                input_feature_dict["relp"],
                s_inputs=s_inputs,
                s_trunk=s_trunk,
                z_trunk=z_trunk,
                pair_z=pair_z,
                inplace_safe=inplace_safe,
                use_conditioning=use_conditioning,
                # P011 D4.  `.get`, not `[...]`: dt rides in the feature dict
                # like `wp_tokens` does, and a caller that does not put it
                # there (check_fusion, inference on a structure with no
                # trajectory) gets the pathway skipped rather than a KeyError.
                # KineidosTrainer refuses a config where the arm wants the
                # pathway and the batch has no dt, so "skipped" cannot happen
                # silently in a training run.
                delta_t_ns=input_feature_dict.get("delta_t_ns", None),
            )  # [..., N_sample, N_token, c_s], [..., N_token, N_token, c_z]

        # Expand embeddings to match N_sample
        s_trunk = expand_at_dim(s_trunk, dim=-3, n=1)  # [..., N_sample, N_token, c_s]
        z_pair = expand_at_dim(
            z_pair, dim=-4, n=1
        )  # [..., N_sample, N_token, N_token, c_z]
        # Fine-grained checkpoint for finetuning stage 2 (token num: 768) for avoiding OOM
        # Kineidos: h travels in input_feature_dict, the channel every other
        # per-atom input already uses, so no caller between here and the
        # dataloader needs a new argument.  Absent, or wp_token_dim None, and
        # the encoder ignores it.
        wp_tokens = input_feature_dict.get("wp_tokens", None)

        if blocks_per_ckpt and self.use_fine_grained_checkpoint:
            checkpoint_fn = get_checkpoint_fn()
            a_token, q_skip, c_skip, p_skip = checkpoint_fn(
                self.atom_attention_encoder,
                input_feature_dict["atom_to_token_idx"],
                input_feature_dict["ref_pos"],
                input_feature_dict["ref_charge"],
                input_feature_dict["ref_mask"],
                input_feature_dict["ref_atom_name_chars"],
                input_feature_dict["ref_element"],
                input_feature_dict["d_lm"],
                input_feature_dict["v_lm"],
                input_feature_dict["pad_info"],
                r_noisy,
                s_trunk,
                z_pair,
                p_lm,
                c_l,
                # Positional, so this has to track AtomAttentionEncoder.forward's
                # order.  Omitting it would shift inplace_safe into wp_tokens and
                # chunk_size into inplace_safe -- wrong even with fusion off, and
                # silent.
                wp_tokens,
                inplace_safe,
                chunk_size,
            )
        else:
            # Sequence-local Atom Attention and aggregation to coarse-grained tokens
            a_token, q_skip, c_skip, p_skip = self.atom_attention_encoder(
                input_feature_dict["atom_to_token_idx"],
                input_feature_dict["ref_pos"],
                input_feature_dict["ref_charge"],
                input_feature_dict["ref_mask"],
                input_feature_dict["ref_atom_name_chars"],
                input_feature_dict["ref_element"],
                input_feature_dict["d_lm"],
                input_feature_dict["v_lm"],
                input_feature_dict["pad_info"],
                r_l=r_noisy,
                s=s_trunk,
                z=z_pair,
                p_lm=p_lm,
                c_l=c_l,
                wp_tokens=wp_tokens,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
        # Upcast
        a_token = a_token.to(dtype=torch.float32)

        # Full self-attention on token level.
        if inplace_safe:
            a_token += self.linear_no_bias_s(
                self.layernorm_s(s_single)
            )  # [..., N_sample, N_token, c_token]
        else:
            a_token = a_token + self.linear_no_bias_s(
                self.layernorm_s(s_single)
            )  # [..., N_sample, N_token, c_token]
        if enable_efficient_fusion:
            z = self.normalize(z_pair.to(dtype=torch.float32))
            z = permute_final_dims(z, [2, 0, 1]).contiguous()
        else:
            z = z_pair.to(dtype=torch.float32)
        a_token = self.diffusion_transformer(
            a=a_token.to(dtype=torch.float32),  # Upcast all inputs
            s=s_single.to(dtype=torch.float32),
            z=z,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
            enable_efficient_fusion=enable_efficient_fusion,
        )

        a_token = self.layernorm_a(a_token)

        # Fine-grained checkpoint for finetuning stage 2 (token num: 768) for avoiding OOM
        if blocks_per_ckpt and self.use_fine_grained_checkpoint:
            checkpoint_fn = get_checkpoint_fn()
            r_update = checkpoint_fn(
                self.atom_attention_decoder,
                input_feature_dict["atom_to_token_idx"],
                a_token,
                q_skip,
                c_skip,
                p_skip,
                inplace_safe,
                chunk_size,
            )
        else:
            # Broadcast token activations to atoms and run Sequence-local Atom Attention
            r_update = self.atom_attention_decoder(
                atom_to_token_idx=input_feature_dict["atom_to_token_idx"],
                a=a_token,
                q_skip=q_skip,
                c_skip=c_skip,
                p_skip=p_skip,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )

        return r_update

    def forward(
        self,
        x_noisy: torch.Tensor,
        t_hat_noise_level: torch.Tensor,
        input_feature_dict: dict[str, Union[torch.Tensor, int, float, dict]],
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        pair_z: torch.Tensor,
        p_lm: torch.Tensor,
        c_l: torch.Tensor,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
        use_conditioning: bool = True,
        enable_efficient_fusion: bool = False,
    ) -> torch.Tensor:
        """One step denoise: x_noisy, noise_level -> x_denoised

        Args:
            x_noisy (torch.Tensor): the noisy version of the input atom coords
                [..., N_sample, N_atom,3]
            t_hat_noise_level (torch.Tensor): the noise level, as well as the time step t
                [..., N_sample]
            input_feature_dict (dict[str, Union[torch.Tensor, int, float, dict]]): input meta feature dict
            s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
                [..., N_tokens, c_s_inputs]
            s_trunk (torch.Tensor): single feature embedding from PairFormer (Alg17)
                [..., N_tokens, c_s]
            z_trunk (torch.Tensor): pair feature embedding from PairFormer (Alg17)
                [..., N_tokens, N_tokens, c_z]
            pair_z (torch.Tensor): diffusion pair embedding
                [..., N_tokens, N_tokens, c_z]
            p_lm (torch.Tensor): MSA embedding
                [..., N_tokens, c_p_lm]
            c_l (torch.Tensor): ligand embedding
                [..., N_tokens, c_c_l]
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.
            use_conditioning (bool): Whether to drop the s/z embeddings in DiffusionConditioning.
            enable_efficient_fusion (bool): Whether to enable efficient fusion. Defaults to False.

        Returns:
            torch.Tensor: the denoised coordinates of x
                [..., N_sample, N_atom,3]
        """
        # Scale positions to dimensionless vectors with approximately unit variance
        # As in EDM:
        #     r_noisy = (c_in * x_noisy)
        #     where c_in = 1 / sqrt(sigma_data^2 + sigma^2)
        r_noisy = (
            x_noisy
            / torch.sqrt(self.sigma_data**2 + t_hat_noise_level**2)[..., None, None]
        )

        # Compute the update given r_noisy (the scaled x_noisy)
        # As in EDM:
        #     r_update = F(r_noisy, c_noise(sigma))
        r_update = self.f_forward(
            r_noisy=r_noisy,
            t_hat_noise_level=t_hat_noise_level,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            pair_z=pair_z,
            p_lm=p_lm,
            c_l=c_l,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
            use_conditioning=use_conditioning,
            enable_efficient_fusion=enable_efficient_fusion,
        )

        # Rescale updates to positions and combine with input positions
        # As in EDM:
        #     D = c_skip * x_noisy + c_out * r_update
        #     c_skip = sigma_data^2 / (sigma_data^2 + sigma^2)
        #     c_out = (sigma_data * sigma) / sqrt(sigma_data^2 + sigma^2)
        #     s_ratio = sigma / sigma_data
        #     c_skip = 1 / (1 + s_ratio^2)
        #     c_out = sigma / sqrt(1 + s_ratio^2)

        s_ratio = (t_hat_noise_level / self.sigma_data)[..., None, None].to(
            r_update.dtype
        )
        x_denoised = (
            1 / (1 + s_ratio**2) * x_noisy
            + t_hat_noise_level[..., None, None]
            / torch.sqrt(1 + s_ratio**2)
            * r_update
        ).to(r_update.dtype)

        return x_denoised
