"""Load one arm's checkpoint and sample from it -- the inference path, not the
training objective.

This is the only module in `bench/` that imports torch and Protenix, and it is
the one place where the difference between P009's number and P007's number
lives.  P009 scores `sample_diffusion_training`, whose coordinate channel is a
noised copy of the answer (P004 section 2.18, fifth part).  Here the sampler
starts from pure noise and the history is the only thing that says where the
molecule should be, which is what "can it predict the next frame" means.

Four constraints from P007 section 3.0, all enforced here:

**The checkpoint load asks for equality, not `strict=False`.**  Through
`kineidos/checkpoint.py`, so a renamed module cannot hide behind a tolerated
gap (P004 section 2.11).  A checkpoint written by one of our own runs should
have *nothing* missing: it was saved from this architecture, and a gap means the
architecture moved under it.

**`eval()`, dropout off, `N_cycle` pinned to the configured 10.**  Upstream
draws the recycling depth from `RandomState(current_step)` in the training
branch; the inference branch uses `self.N_cycle` and is therefore already fixed,
but the number is logged because two logs are otherwise incomparable (P004
section 2.17).

**The sampling seed is a function of the window alone**, so the four arms draw
bit-identical noise on bit-identical windows and the arm comparison is paired.
`sampling_seed` is that function; nothing else may seed the sampler.

**The trunk is cached per sample, not per window.**  `get_pairformer_output`
reads sequence, MSA, template and `ref_pos` features, all of which are static
per GAGU trajectory -- `wp_tokens` is consumed inside the diffusion module and
never reaches the trunk (`protenix/model/modules/diffusion.py:423`).  So the
trunk result is identical for all 256 windows of one sample, and computing it 16
times instead of 256 is the same number for a sixteenth of the trunk cost.  The
cache cannot change any sampled coordinate because the seed is reset from
`sampling_seed` immediately before every `sample_diffusion` call; `inplace_safe`
is False throughout so that a cached tensor cannot be written through.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch

N_CYCLE = 10
"""The recycling depth every P004/P009 run was evaluated at."""


def required_env() -> dict[str, str]:
    """The four environment variables AGENTS.md calls load-bearing, as seen.

    Reported into every run's record rather than asserted: `LAYERNORM_TYPE` and
    `ATTN_IMPL` change which kernel runs, and a result whose log does not say
    which one cannot be compared with one that does.  `LAYERNORM_TYPE` is the
    one that fails loudly if wrong (no nvcc on any node here, so the fused path
    raises), the others fail quietly.
    """
    return {k: os.environ.get(k, "") for k in
            ("PYTHONPATH", "LAYERNORM_TYPE", "ATTN_IMPL", "LD_LIBRARY_PATH")}


def sampling_seed(window_id: int, *, base: int = 1234) -> int:
    """The sampler's seed for one window -- shared by all four arms.

    A function of the window index and nothing else: not of the arm, not of the
    checkpoint step, not of the order windows happen to be visited in.  That is
    what makes `random` minus `zero` a paired difference on each window rather
    than a difference of two independently noisy means.  `base` defaults to
    `kineidos.eval_seed`, the same number that drew the window set.
    """
    return int((base * 1_000_003 + int(window_id) * 7_919) % (2 ** 31 - 1))


def build_configs(*, wp_mode: str, n_step: int, n_sample: int,
                  extra: Sequence[str] = ()) -> Any:
    """Parse a config for inference, the same two-pass way training does.

    The second pass is not cosmetic: `DiffusionModule` is built as
    `DiffusionModule(**configs.model.diffusion_module)`, so `wp_token_dim` has
    to be inside that dict before `parse_configs` runs.  Setting it afterwards
    builds the baseline architecture under an ablation arm's name and says
    nothing while doing it (`kineidos/train/main.py`).
    """
    from configs.configs_base import configs as configs_base
    from configs.configs_data import data_configs
    from configs.configs_model_type import model_configs
    from protenix.config import parse_configs

    from kineidos.train.trainer import wp_token_dim_for

    configs_base["triangle_attention"] = os.environ.get(
        "TRIANGLE_ATTENTION", "cuequivariance")
    configs_base["triangle_multiplicative"] = os.environ.get(
        "TRIANGLE_MULTIPLICATIVE", "cuequivariance")

    args = ["--wp.mode", str(wp_mode),
            "--sample_diffusion.N_step", str(int(n_step)),
            "--sample_diffusion.N_sample", str(int(n_sample)),
            *extra]
    arg_str = " ".join(args)

    base = {**configs_base, **{"data": data_configs}}
    first = parse_configs({**configs_base, **{"data": data_configs}},
                          arg_str=arg_str, fill_required_with_null=True)

    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                deep_update(d[k], v)
            else:
                d[k] = v
        return d

    deep_update(base, model_configs[first.model_name])
    token_dim = wp_token_dim_for(wp_mode)
    if token_dim is not None:
        base["model"]["diffusion_module"]["wp_token_dim"] = token_dim
    return parse_configs(configs=base, arg_str=arg_str,
                         fill_required_with_null=True)


@dataclass
class LoadedArm:
    """One arm, ready to sample from."""

    model: Any
    configs: Any
    device: torch.device
    arm: str
    checkpoint: Path
    step: Optional[int]
    n_step: int
    n_sample: int
    info: dict[str, Any]
    trunk_cache: dict[str, tuple] = field(default_factory=dict)

    def describe(self) -> dict[str, Any]:
        return {"arm": self.arm, "checkpoint": str(self.checkpoint),
                "checkpoint_step": self.step, "N_step": self.n_step,
                "N_sample": self.n_sample, "N_cycle": N_CYCLE,
                "wp_token_dim": self.configs.model.diffusion_module.get(
                    "wp_token_dim", None),
                "particle_radius_nm": self.configs.wp.particle_radius_nm,
                "env": required_env(), **self.info}


def load_arm(checkpoint: str | Path, *, arm: str, device: str | torch.device,
             n_step: int, n_sample: int,
             particle_radius_nm: Optional[float] = None,
             wp_checkpoint: Optional[str] = None) -> LoadedArm:
    """Build the architecture for `arm` and load `checkpoint` into it.

    `arm` has to be given rather than read off the file: the arm decides whether
    the fusion exists at all (`wp_token_dim=None` for `none`), so a mismatch
    would be an architecture mismatch and not a labelling slip.  `latest.arm`
    beside a `latest.pt` records it; a numbered checkpoint under
    `runs/<run>/checkpoints/` is identified by its run directory.
    """
    from protenix.model.protenix import Protenix

    from kineidos.checkpoint import load_checkpoint
    from kineidos.wp_bridge import WorldParticleBridge

    checkpoint = Path(checkpoint)
    device = torch.device(device)
    extra = []
    if particle_radius_nm is not None:
        extra += ["--wp.particle_radius_nm", str(particle_radius_nm)]
    if wp_checkpoint:
        extra += ["--wp.checkpoint", str(wp_checkpoint)]
    configs = build_configs(wp_mode=arm, n_step=n_step, n_sample=n_sample,
                            extra=extra)
    if int(configs.model.N_cycle) != N_CYCLE:
        raise ValueError(
            f"model.N_cycle is {configs.model.N_cycle}, not {N_CYCLE}. The "
            f"held-out curves of P004 and P009 were all scored at {N_CYCLE} and "
            f"are not comparable across depths (P004 section 2.17)."
        )

    model = Protenix(configs).to(device)
    bridge = WorldParticleBridge(
        configs.wp.mode,
        particle_radius_nm=configs.wp.particle_radius_nm,
        checkpoint=configs.wp.checkpoint or None,
        seed=configs.wp.seed if configs.wp.seed >= 0 else None,
    )
    model.wp_bridge = bridge.to(device)

    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = raw["model"] if "model" in raw else raw
    if next(iter(state)).startswith("module."):
        state = {k[len("module."):]: v for k, v in state.items()}
    # Nothing may be missing: this file was written by this architecture.  If
    # anything is, the message from kineidos/checkpoint.py names the keys, and
    # the right response is to find out what moved -- not to widen the contract
    # (P004 section 2.11, and P007's "stop and ask" list).
    info = load_checkpoint(model, state, expect_new=())
    model.eval()

    step = raw.get("step") if isinstance(raw, dict) else None
    loaded = {"tensors_loaded": info["tensors_loaded"],
              "parameters_loaded": float(info["parameters"]) / 1e6,
              "left_at_initialisation": info["left_at_initialisation"],
              "has_wp_parameters": any(p.numel() for p in bridge.parameters())}
    return LoadedArm(model=model, configs=configs, device=device, arm=arm,
                     checkpoint=checkpoint,
                     step=int(step) if step is not None else None,
                     n_step=int(n_step), n_sample=int(n_sample), info=loaded)


# ------------------------------------------------------------------ sampling


def _prepare_features(loaded: LoadedArm, batch: dict[str, Any]) -> dict[str, Any]:
    """Protenix's own pre-forward feature work, which `forward` normally does.

    We do not call `forward(mode="inference")` because `main_inference_loop`
    runs the distogram and confidence heads, and these runs were trained with
    `skip_confidence`/`skip_distogram` -- those heads carry pretrained weights
    that nothing here fine-tuned and nothing here reads.  Everything up to and
    including the diffusion sampling is the same code path.
    """
    from protenix.model.protenix import update_input_feature_dict
    from protenix.utils.torch_utils import to_device

    feats = to_device(batch["input_feature_dict"], loaded.device)
    feats = loaded.model.relative_position_encoding.generate_relp(feats)
    return update_input_feature_dict(feats)


@torch.no_grad()
def trunk_for(loaded: LoadedArm, sample_id: str, batch: dict[str, Any]) -> tuple:
    """(features, s_inputs, s_trunk, z_trunk) for one GAGU trajectory, cached.

    Cached on `sample_id` because every window of one trajectory has the same
    static features; see the module docstring for why that is exact rather than
    an approximation.
    """
    hit = loaded.trunk_cache.get(sample_id)
    if hit is not None:
        return hit
    feats = _prepare_features(loaded, batch)
    s_inputs, s, z = loaded.model.get_pairformer_output(
        input_feature_dict=feats, N_cycle=N_CYCLE, inplace_safe=False,
        chunk_size=None, mc_dropout=False,
    )
    loaded.trunk_cache[sample_id] = (feats, s_inputs, s, z)
    return loaded.trunk_cache[sample_id]


@torch.no_grad()
def sample_frames(loaded: LoadedArm, window: Any, batch: dict[str, Any],
                  *, seed: int, n_sample: Optional[int] = None) -> np.ndarray:
    """Sample `n_sample` candidate target frames for one window.

    Returns [S, N, 3] in Angstroms, in the sampler's own arbitrary pose: the
    caller aligns before scoring (P004 section 2.19, fourth qualifier).

    The seed is set here, from the caller's `seed`, immediately before the
    sampler draws.  That is deliberate and load-bearing: the augmentation
    `random_transform` applies at every one of the ~N_step steps comes from
    numpy's global stream (`kineidos/seeding.py`), so seeding anywhere further
    away would let an unrelated numpy draw desynchronise two arms.
    """
    from kineidos.seeding import set_all_seeds
    from kineidos.wp_bridge import wp_inputs_from_window
    from protenix.utils.torch_utils import to_device

    feats, s_inputs, s, z = trunk_for(loaded, window.sample_id, batch)
    wp_feats = to_device(wp_inputs_from_window(window), loaded.device)
    feats = {**feats, **wp_feats}
    feats["wp_tokens"] = loaded.model.wp_bridge(feats)

    set_all_seeds(int(seed))
    schedule = loaded.model.inference_noise_scheduler(
        N_step=loaded.n_step, device=s_inputs.device, dtype=s_inputs.dtype)
    coords = loaded.model.sample_diffusion(
        denoise_net=loaded.model.diffusion_module,
        input_feature_dict=feats,
        s_inputs=s_inputs, s_trunk=s, z_trunk=z,
        pair_z=None, p_lm=None, c_l=None,
        N_sample=int(n_sample or loaded.n_sample),
        noise_schedule=schedule,
        inplace_safe=False,
        enable_efficient_fusion=False,
    )
    return coords.detach().float().cpu().numpy()


@torch.no_grad()
def lddt_complex(pred_angstrom: np.ndarray, true_angstrom: np.ndarray,
                 feats: dict[str, Any], device: str | torch.device = "cpu"
                 ) -> np.ndarray:
    """Protenix's own complex lDDT for [S, N, 3] predictions against one truth.

    Protenix's implementation rather than ours, because the number is meant to
    be the one Protenix reports: the 30 A inclusion radius for nucleotides, the
    same off-diagonal and coordinate masks, the same epsilon.  lDDT is a
    pairwise-distance quantity, so it needs no alignment -- and is also blind to
    a chain relabelling, which is why RMSD and not lDDT is the main read.
    """
    from protenix.metrics.lddt_metrics import LDDT
    from protenix.model.loss import compute_lddt_mask

    device = torch.device(device)
    pred = torch.as_tensor(np.asarray(pred_angstrom, dtype=np.float32), device=device)
    true = torch.as_tensor(np.asarray(true_angstrom, dtype=np.float32), device=device)
    if pred.ndim == 2:
        pred = pred[None]
    n = true.shape[-2]
    mask = torch.ones(n, device=device)
    distance = torch.cdist(true, true)
    is_nucleotide = feats["is_rna"].bool().to(device) | feats["is_dna"].bool().to(device)
    lddt_mask = compute_lddt_mask(
        true_distance=distance,
        distance_mask=mask[..., None] * mask[..., None, :],
        is_nucleotide=is_nucleotide,
    )
    return LDDT(eps=1e-10).forward(
        pred_coordinate=pred, true_coordinate=true, lddt_mask=lddt_mask,
    ).detach().float().cpu().numpy()
