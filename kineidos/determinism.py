"""Ask torch for reproducible kernels, and say what it cannot give.

Shared by both entry points, which is the whole point of the file.  It lived
in kineidos/score_sigma_grid.py and so reached only the scoring path:
kineidos/train/main.py never called it, so `KINEIDOS_DETERMINISTIC=1` did
nothing for a training run.  That is how job 2149625's three identity runs
came out with three different step-1 training losses -- 2.0212183, 2.0192595,
2.0210862 -- which reads exactly like "the branches changed the computation"
and was in fact the default nondeterministic kernels, on three runs of code
that differs in nothing that executes.  Exporting CUBLAS_WORKSPACE_CONFIG
alone does not enable deterministic algorithms; it only makes cuBLAS's
reductions reproducible *once they are asked for*.

What is measured so far:

  * **the forward is reproducible.**  Two scorings of one arm, same node,
    byte-identical over 2816 rows (2149525 / 2149526, compared by 2149583),
    and torch named no operation without a deterministic implementation.
  * **the backward is untested.**  It has more operations to replace, and
    `warn_only=True` *permits* the ones that cannot be replaced rather than
    raising -- so a training run under this switch is "as reproducible as
    torch can manage", not "reproducible".  Whether that is bit-exact is a
    measurement, and the warnings this prints are how it gets made.

Off unless KINEIDOS_DETERMINISTIC is set, so the normal path keeps the faster
kernels.  Determinism is for the acceptances, whose criterion *is* bit
identity; the arms read effects 400x above the nondeterminism envelope and buy
nothing from it.
"""

from __future__ import annotations

import os
import warnings

ENV_VAR = "KINEIDOS_DETERMINISTIC"
TRUTHY = ("1", "true", "True", "yes")


def wanted() -> bool:
    return os.environ.get(ENV_VAR, "") in TRUTHY


def enable() -> bool:
    """Turn on deterministic algorithms if asked.  Returns whether it did.

    `warn_only=True` rather than raising, and the reason is diagnostic: an
    operation with no deterministic implementation should be *named*, not hit
    as a crash on whichever one comes first.  Those names are the answer to
    "is the remaining spread the tool or the model" -- AF3 aggregates atoms
    into tokens with a segment sum, and a CUDA scatter-add is atomics, whose
    summation order is not reproducible.

    `filterwarnings("once")` because one of these fires per call: over a
    2000-step arm the same sentence would otherwise arrive tens of thousands
    of times and bury the log it is meant to inform.
    """
    import torch

    if not wanted():
        return False
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if not workspace:
        # Not fatal, but it has to be said: cuBLAS reads this when its first
        # handle is created, which is before any line of ours runs, so python
        # cannot set it late.  Without it the reductions stay nondeterministic
        # however the rest is configured.
        print(f"[determinism] CUBLAS_WORKSPACE_CONFIG is unset, so cuBLAS "
              f"reductions stay nondeterministic however the rest is "
              f"configured -- the sbatch must export it before python starts",
              flush=True)
    warnings.filterwarnings("once", message=".*deterministic.*")
    warnings.filterwarnings("once", message=".*nondeterministic.*")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    print(f"[determinism] on (warn_only); cuBLAS workspace="
          f"{workspace or '<unset>'}. Any 'does not have a deterministic "
          f"implementation' warning below names an operation that stays "
          f"nondeterministic, and is why a trained state may still not be "
          f"bit-reproducible.", flush=True)
    return True
