#!/usr/bin/env python
"""Held-out scoring on a fixed sigma grid, per window and per noise sample.

P010 D1 and section 4.  The held-out loss averages over `diffusion_batch_size`
= 48 noise samples drawn from the training distribution, and `h` can only help
where the input does not already contain the answer:

    D = c_skip * x_noisy + c_out * F,      c_skip = 1 / (1 + (sigma/16)^2)

so at sigma > 24.4 (c_skip < 0.3) the network supplies nearly all of the output
and the history is the only extra information there is, while at sigma < 5.3
(c_skip > 0.9) the output is mostly the input copied through and no history
could matter.  Only 14% of the training draw lands in the first band.  A
difference that lives there and nowhere else is therefore divided by about
seven on its way into the mean -- which is the regime P009 read as
"indistinguishable" at -0.6% against a ~2% seed-to-seed floor.

This module reads the same number with sigma as a coordinate rather than as
something averaged away.  **Nothing else changes**: the window set, the seed
reset, the dropout-off path, the pinned recycling depth, the pairing across
arms and Protenix's own loss (whose Kabsch alignment is the one the training
objective uses) are all `KineidosTrainer.evaluate`'s, reused rather than
reimplemented.  The single substitution is `model.train_noise_sampler`.

### Why one forward per window and not one per (sigma, noise)

Section 4 sizes the job as 256 x 11 x 4 = 11k forwards per arm.  It is 11k
*denoiser* evaluations, but they do not need 11k trunk passes: the Pairformer
sees only the window's features, and sigma enters afterwards.  So the 44
(sigma, noise) pairs of one window ride in one forward as 44 diffusion samples
-- `sample_diffusion_training` draws the noise level once for all N_sample
(generator.py:355) and chunks only the denoise call -- and the trunk, which is
48 blocks times ten recycles, is paid once per window instead of 44 times.
Same arithmetic, same numbers, about 1/40 of the cost.

Per-sample values then come from calling Protenix's loss on one-sample slices
of its own output.  With N_sample = 1 the `.mean(dim=-1)` over the sample axis
is the identity, so each call returns that sample's loss exactly, computed by
the same code as the aggregate -- no second alignment, which is how a metric
ends up measuring something other than the objective (P009 section 8, item 8).

### The two loss columns

`loss_edm_weighted` is the cumulative loss verbatim, which is the held-out
metric's own quantity: `evaluate` calls the loss with mode="train", and that
branch multiplies the MSE and bond terms by the EDM per-sample scale
lambda(sigma) = (sigma^2 + sigma_data^2) / (sigma_data * sigma)^2
(loss.py:1638).  `loss_unweighted` is the same sum with that factor divided
back out of the terms that carry it.  Both are recorded because they answer
different questions -- what the optimizer saw, and how large the error
actually is -- and because dividing a logged number by a factor nobody wrote
down is how a table becomes unreproducible.

(P010 v0.2 section 1 said the held-out loss carries no lambda, citing
loss.py:1642.  That line is in the mode="eval" branch, which `evaluate` does
not take.  Registered as a plan correction; the "about seven" dilution above
is the 14% band occupancy and does not depend on it.)

### Output

`<out>/<arm>_step<N>.jsonl`, one line per (window, sigma, noise sample):

    {arm, step, window_id, sample_id, stride, delta_t_ns, sigma, noise_idx,
     mse_aligned, smooth_lddt, loss_unweighted, loss_edm_weighted, ...}

Those twelve keys are the interface to P007 section 3.7 and are fixed.  The
rest -- `c_skip`, `sigma_band`, `edm_scale`, `target_frame` and the raw loss
terms -- are additive, and are there so a reader can re-derive the four
reported numbers instead of trusting them.

Standalone, against a trained checkpoint (P010 D7: this runs on the P009
arms' step-3000 checkpoints without waiting for the arms to finish):

    PYTHONPATH=repos/research/kineidos-v3-diag:repos/research/wp-v2 \
    LAYERNORM_TYPE=torch ATTN_IMPL=sdpa LD_LIBRARY_PATH=$ENV/lib \
      $ENV/bin/python -u -m kineidos.score_sigma_grid \
        --wp.mode zero \
        --kineidos.score_checkpoint runs/p009/p009_zero_.../checkpoints/3000.pt \
        --kineidos.sigma_grid_arm p009_zero \
        --kineidos.sigma_grid_out runs/p010/sigma_grid \
        --kineidos.held_out_samples "<r4 names>"

Or inside training, by setting `kineidos.sigma_grid true`, which makes every
evaluation round write the per-sigma file beside the per-window one.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional

import torch

from protenix.utils.distributed import DIST_WRAPPER

from kineidos.seeding import set_all_seeds

# The training draw's own parameters, as constants to derive the grid from.
# Read off the config at runtime where possible; these are the fallback and the
# documentation of what the numbers below came from.
P_MEAN = -1.2
P_STD = 1.5
SIGMA_DATA = 16.0

# The quantiles of the training distribution, which is where the samples
# actually are, plus the two band edges, which is where the reading is cut.
GRID_Z = (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0)
# c_skip thresholds, not sigmas: the bands are defined by how much of the
# output the skip connection supplies, and the sigmas follow from sigma_data.
C_SKIP_LOW = 0.9      # above this, the output is essentially x_noisy copied
C_SKIP_HIGH = 0.3     # below this, the network supplies nearly everything

# Which loss terms calculate_losses multiplies by the EDM per-sample scale.
# Checked against the code rather than assumed: loss.py's mode="train" branch
# passes `per_sample_scale=diffusion_per_sample_scale` to mse_loss and
# bond_loss, and to nothing else -- smooth_lddt_loss has no such argument.
EDM_SCALED_TERMS = ("mse_loss", "bond_loss")


def sigma_from_z(z: float, *, p_mean: float = P_MEAN, p_std: float = P_STD,
                 sigma_data: float = SIGMA_DATA) -> float:
    """The noise level at quantile z of TrainingNoiseSampler's own draw."""
    return sigma_data * math.exp(p_mean + p_std * z)


def sigma_at_c_skip(c_skip: float, *, sigma_data: float = SIGMA_DATA) -> float:
    """Invert c_skip = 1 / (1 + (sigma/sigma_data)^2)."""
    return sigma_data * math.sqrt(1.0 / c_skip - 1.0)


def c_skip_of(sigma: float, *, sigma_data: float = SIGMA_DATA) -> float:
    return 1.0 / (1.0 + (sigma / sigma_data) ** 2)


def edm_scale_of(sigma: float, *, sigma_data: float = SIGMA_DATA) -> float:
    """lambda(sigma) as loss.py:1638 computes it, for one sample."""
    return (sigma**2 + sigma_data**2) / (sigma_data * sigma) ** 2


def band_of(sigma: float, *, sigma_data: float = SIGMA_DATA) -> str:
    """"low" / "mid" / "high", by c_skip.

    The two grid points that sit exactly on an edge (sigma = 16/3 and
    sigma = 16*sqrt(7/3)) fall in "mid" here, because a boundary point belongs
    with the band it bounds from below and the reading of interest is the high
    band.  `c_skip` is in every row, so a reader who wants the other convention
    does not have to rerun anything.
    """
    c = c_skip_of(sigma, sigma_data=sigma_data)
    if c > C_SKIP_LOW:
        return "low"
    if c < C_SKIP_HIGH:
        return "high"
    return "mid"


def sigma_grid(sigma_data: float = SIGMA_DATA, *, p_mean: float = P_MEAN,
               p_std: float = P_STD) -> list[float]:
    """Eleven noise levels: nine quantiles and the two band edges.

    Sorted, because the output is read band by band and a grid in draw order
    would interleave them.
    """
    grid = [sigma_from_z(z, p_mean=p_mean, p_std=p_std, sigma_data=sigma_data)
            for z in GRID_Z]
    grid += [sigma_at_c_skip(C_SKIP_LOW, sigma_data=sigma_data),
             sigma_at_c_skip(C_SKIP_HIGH, sigma_data=sigma_data)]
    return sorted(grid)


class FixedSigmaSampler:
    """A drop-in for `TrainingNoiseSampler` that hands back a prescribed vector.

    The substitution point, and the only one.  `sample_diffusion_training`
    calls `noise_sampler(size=..., device=...)` once for the whole diffusion
    batch (generator.py:355) and uses the result both as the noise scale and as
    the denoiser's conditioning, so replacing the object replaces the sigma of
    every sample and changes nothing else -- the augmentation, the standard
    normal draw, the chunking and the loss all stay where they were.

    Carries p_mean / p_std / sigma_data so that anything reading them off the
    sampler still finds them, and refuses a size it was not built for rather
    than broadcasting into one: a mismatch would mean `diffusion_batch_size`
    and the grid had drifted apart, which would silently rename the rows.
    """

    def __init__(self, sigmas: list[float], *, sigma_data: float = SIGMA_DATA,
                 p_mean: float = P_MEAN, p_std: float = P_STD) -> None:
        self.sigmas = list(sigmas)
        self.sigma_data = sigma_data
        self.p_mean = p_mean
        self.p_std = p_std

    def __call__(self, size: torch.Size,
                 device: torch.device = torch.device("cpu")) -> torch.Tensor:
        want = len(self.sigmas)
        if int(size[-1]) != want:
            raise ValueError(
                f"the sigma grid has {want} entries but the diffusion batch "
                f"asks for {int(size[-1])}. score_sigma_grid sets "
                f"model.diffusion_batch_size itself; if something else changed "
                f"it afterwards the rows would carry the wrong sigma."
            )
        out = torch.tensor(self.sigmas, device=device, dtype=torch.float32)
        return out.expand(*size)


def grid_plan(sigmas: list[float], n_noise: int) -> list[tuple[float, int]]:
    """The diffusion batch's layout: (sigma, noise_idx) per sample slot.

    Grouped by sigma rather than interleaved, so that a chunked denoise call
    (diffusion_chunk_size is 4 by default) keeps each sigma's samples together
    and a partial result is still whole sigmas.
    """
    return [(s, j) for s in sigmas for j in range(n_noise)]


@torch.no_grad()
def score(trainer: Any, *, step: int, arm: str, out_dir: Path,
          n_noise: int = 4, sigmas: Optional[list[float]] = None) -> Path:
    """Score the held-out set on the grid.  Returns the file it wrote.

    Deliberately a function of the trainer rather than a method on it: it is
    the same object `evaluate` runs on, so there is no second notion of "the
    held-out set" or "the model in eval mode" to drift from the first.
    """
    from protenix.utils.torch_utils import to_device

    from kineidos.train.batch import collate_window

    if not getattr(trainer, "eval_windows", None):
        raise ValueError(
            "no held-out windows: kineidos.held_out_samples is empty, so there "
            "is nothing to score. The grid is a held-out reading, not a "
            "training-set one."
        )

    sigma_data = float(trainer.configs.train_noise_sampler.sigma_data)
    if sigmas is None:
        sigmas = sigma_grid(
            sigma_data,
            p_mean=float(trainer.configs.train_noise_sampler.p_mean),
            p_std=float(trainer.configs.train_noise_sampler.p_std),
        )
    plan = grid_plan(sigmas, n_noise)

    model = trainer.raw_model
    was_training = model.training
    saved_sampler = model.train_noise_sampler
    saved_batch = model.diffusion_batch_size
    model.eval()
    # Same reset as `evaluate`, for the same reason: round k of one arm must
    # see bit-identical noise and augmentation to round k of another, or the
    # comparison is unpaired.  Reset here as well as there, so that calling
    # this after an evaluate in the same round does not inherit the RNG state
    # that evaluate left behind.
    set_all_seeds(trainer.configs.kineidos.eval_seed)
    model.train_noise_sampler = FixedSigmaSampler(
        [s for s, _ in plan], sigma_data=sigma_data,
        p_mean=float(trainer.configs.train_noise_sampler.p_mean),
        p_std=float(trainer.configs.train_noise_sampler.p_std))
    model.diffusion_batch_size = len(plan)

    weights = trainer.loss.loss_weight
    weight_mse = float(trainer.loss.mse_loss.weight_mse)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{arm}_step{step}.rank{DIST_WRAPPER.rank}.jsonl"
    shard = trainer.eval_windows[DIST_WRAPPER.rank::DIST_WRAPPER.world_size]
    written = 0
    try:
        with open(path, "w") as handle:
            for i, window in enumerate(shard):
                batch = to_device(collate_window(window), trainer.device)
                pred, label, _ = model(
                    input_feature_dict=batch["input_feature_dict"],
                    label_dict=batch["label_dict"],
                    label_full_dict=batch["label_full_dict"],
                    mode="train",
                    current_step=step,
                    symmetric_permutation=trainer.symmetric_permutation,
                    eval_training_objective=True,
                    n_cycle=trainer.configs.model.N_cycle,
                )
                drawn = pred["noise_level"]
                # The sampler's contract, checked against what came back: if
                # the layout and the output disagree, every sigma column is
                # mislabelled and no table built from this file means anything.
                for slot, (sigma, _) in enumerate(plan):
                    got = float(drawn[..., slot].reshape(-1)[0])
                    if abs(got - sigma) > 1e-3 * max(sigma, 1.0):
                        raise RuntimeError(
                            f"slot {slot} was conditioned on sigma={got:.6g} "
                            f"but the plan says {sigma:.6g}; the noise sampler "
                            f"substitution did not take effect"
                        )
                # The global index, as in write_heldout_rows: the shard is
                # eval_windows[rank::world_size], so rank r's i-th window is
                # r + i*world_size and numbering by i alone would give several
                # windows the same id.
                window_id = DIST_WRAPPER.rank + i * DIST_WRAPPER.world_size
                for slot, (sigma, noise_idx) in enumerate(plan):
                    single = dict(pred)
                    single["coordinate"] = pred["coordinate"][..., slot:slot + 1, :, :]
                    single["noise_level"] = drawn[..., slot:slot + 1]
                    _, terms = trainer.loss(
                        feat_dict=batch["input_feature_dict"],
                        pred_dict=single, label_dict=label, mode="train")
                    scale = edm_scale_of(sigma, sigma_data=sigma_data)
                    raw = {k: float(v) for k, v in terms.items()}
                    # The cumulative loss with lambda(sigma) taken back out of
                    # the two terms that carry it, rebuilt from the loss's own
                    # weights rather than from the 4.0 / 1/3 that happen to be
                    # in the config today.
                    unweighted = 0.0
                    for name, w in weights.items():
                        if name not in raw or not w:
                            continue
                        value = raw[name]
                        if name in EDM_SCALED_TERMS:
                            value /= scale
                        unweighted += w * value
                    row = {
                        "arm": arm,
                        "step": int(step),
                        "window_id": window_id,
                        "sample_id": window.sample_id,
                        "target_frame": int(window.target_frame),
                        "stride": int(window.stride),
                        "delta_t_ns": float(window.delta_t_ns),
                        "sigma": float(sigma),
                        "noise_idx": int(noise_idx),
                        # The aligned MSE itself, in A^2 per atom: the metric
                        # carries weight_mse and lambda, both of which are
                        # exact per-sample factors and divided back out here.
                        "mse_aligned": raw["mse_loss"] / (weight_mse * scale),
                        "smooth_lddt": raw["smooth_lddt_loss"],
                        "loss_unweighted": unweighted,
                        "loss_edm_weighted": raw["loss"],
                        "c_skip": c_skip_of(sigma, sigma_data=sigma_data),
                        "sigma_band": band_of(sigma, sigma_data=sigma_data),
                        "edm_scale": scale,
                        "terms": {k: v for k, v in raw.items()
                                  if not k.startswith("weighted_")},
                    }
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                    written += 1
                if (i + 1) % 16 == 0:
                    trainer.print(f"[sigma-grid] {arm} step {step}: "
                                  f"{i + 1}/{len(shard)} windows, "
                                  f"{written} rows")
    finally:
        model.train_noise_sampler = saved_sampler
        model.diffusion_batch_size = saved_batch
        if was_training:
            model.train()

    trainer.print(f"[sigma-grid] {arm} step {step}: {written} rows "
                  f"({len(shard)} windows x {len(sigmas)} sigma x {n_noise} "
                  f"noise) -> {path}")
    return path


# --------------------------------------------------------------- standalone

ARM_FROM_PREFIX = {
    "p009_random": "random",
    "p009_zero_seed2": "zero",
    "p009_zero": "zero",
    "p009_none": "none",
}


def arm_matches_state(state: dict[str, Any], mode: str) -> None:
    """Refuse a checkpoint whose architecture is not this arm's.

    The arm is not recorded inside a Protenix checkpoint -- `save_checkpoint`
    puts it in a `latest.arm` sidecar next to `latest.pt`, which a
    per-step file under checkpoints/ does not have.  So it is read off the
    weights instead, which is the stronger check anyway: scoring a `zero`
    checkpoint under the name `random` would load every shared key, leave the
    bridge at initialisation and produce a plausible table for an arm that
    never ran.
    """
    has_fusion = any("wp_fusion" in k for k in state)
    has_bridge = any(k.startswith("wp_bridge.") for k in state)
    want_fusion = mode != "none"
    want_bridge = mode in ("random", "pretrained")
    if has_fusion != want_fusion or has_bridge != want_bridge:
        raise SystemExit(
            f"this checkpoint has fusion={has_fusion} bridge={has_bridge}, "
            f"but wp.mode={mode!r} needs fusion={want_fusion} "
            f"bridge={want_bridge}. The arm is read off the weights because a "
            f"per-step checkpoint carries no arm marker; fix --wp.mode rather "
            f"than this check."
        )


def main() -> int:
    import logging

    from protenix.config import parse_sys_args

    from kineidos.checkpoint import load_checkpoint
    from kineidos.train.main import build_configs
    from kineidos.train.trainer import KineidosTrainer

    logging.basicConfig(
        format="%(asctime)s %(levelname)-7s %(message)s",
        level=logging.INFO, datefmt="%H:%M:%S")

    configs = build_configs(parse_sys_args())
    cfg = configs.kineidos
    target = cfg.score_checkpoint
    if not target:
        raise SystemExit(
            "set --kineidos.score_checkpoint to the trained checkpoint to "
            "score, or to the literal 'pretrained' for the step-0 reading. "
            "Defaulting it to the pretrained trunk would answer a different "
            "question and would look like a finished run."
        )

    # The trainer is built the way a training run builds it, pretrained trunk
    # and all, and the trained weights are loaded over the top afterwards.
    # Not through try_load_checkpoint: its pretrained branch requires the
    # missing key set to *equal* the fusion's, and a trained checkpoint is
    # missing nothing, so it would refuse. Not through resume() either, which
    # wants an optimizer state and a latest.arm sidecar this file has no reason
    # to carry. The cost is loading 3 GB twice, once.
    trainer = KineidosTrainer(configs)
    if target == "pretrained":
        # Section 4's first acceptance: `none` and `zero` must agree bit for
        # bit here.  The fusion is identity on the c_l half and zero on the h
        # half at initialisation (transformer.py's _init_wp_fusion) and `zero`
        # feeds it h = 0, so the two arms are provably the same function until
        # training moves the weights -- which makes any difference a fault in
        # this tool rather than a finding.
        step = 0
        trainer.print("scoring the model as constructed (step 0): pretrained "
                      "trunk, fusion at initialisation")
    else:
        checkpoint = torch.load(target, map_location=trainer.device,
                                weights_only=False)
        state = checkpoint["model"]
        if next(iter(state)).startswith("module."):
            state = {k[len("module."):]: v for k, v in state.items()}
        arm_matches_state(state, configs.wp.mode)
        load_checkpoint(trainer.raw_model, state, expect_new=())
        step = int(checkpoint["step"])
        trainer.print(f"scoring {target} at step {step}, "
                      f"arm {configs.wp.mode!r}")
    trainer.step = step

    arm = cfg.sigma_grid_arm or (
        f"step0_{configs.wp.mode}" if target == "pretrained"
        else Path(target).parents[1].name)
    out_dir = Path(cfg.sigma_grid_out or
                   (Path(trainer.run_dir) / "sigma_grid"))
    score(trainer, step=step, arm=arm, out_dir=out_dir,
          n_noise=int(cfg.sigma_grid_noise))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
