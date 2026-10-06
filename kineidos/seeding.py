"""One place to seed everything that actually decides a run's randomness.

torch.manual_seed is not enough here, and the reason is not the obvious one.
Protenix's parameter initialisation does go through numpy -- trunc_normal_init_
in triangular/layers.py draws via scipy's truncnorm.rvs -- but that turns out
not to matter: the checkpoint overwrites all 368.48M parameters with
strict loading, so whatever the initialiser produced is discarded.

What does matter runs *after* loading, every step:

    # protenix/utils/geometry.py, random_transform
    translation = np.random.uniform(-max_translation, max_translation, size=3)
    R = Rotation.random().as_matrix()        # scipy.spatial.transform -> numpy

That is the augmentation applied to the ground truth on every training step and
to the current estimate at every one of the ~200 sampling steps
(generator.py:196-203). It is the bulk of the stochasticity in both training and
inference, and no torch seed touches it.

torch still has to be seeded: WorldParticle's initialisers are all torch
(xavier_uniform_, uniform_, zeros_, randn), so the `random` arm's weights depend
on it, as does diffusion's noise sampling.

Our own window sampling is already explicit -- GAGUWindowDataset seeds
np.random.default_rng((seed, idx)) per index rather than drawing from a shared
stream -- and is unaffected by any of this.

Without both, a run records a seed that does not pin what the run did, which is
the plan's section 8 requirement failing while appearing to be met.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_all_seeds(seed: int, *, deterministic_torch: bool = False) -> dict[str, int]:
    """Seed every generator this stack draws from.  Returns what to record.

    Args:
        seed: the single number a run is identified by.  All generators take it
            directly rather than derived values, so "seed 7" means one thing.
        deterministic_torch: also ask cuDNN for deterministic kernels.  Off by
            default because it costs throughput and does not make a run
            bit-reproducible on its own -- different GPU counts or batch splits
            still diverge -- so it is a debugging tool, not a default.

    PYTHONHASHSEED is set but only takes effect in a fresh interpreter; it is
    recorded so a reader can tell whether it was in force.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return {
        "seed": seed,
        "python_random": seed,
        "numpy": seed,
        "torch": seed,
        "torch_cuda": seed if torch.cuda.is_available() else None,
        "pythonhashseed_env": os.environ.get("PYTHONHASHSEED"),
        "deterministic_torch": deterministic_torch,
    }
