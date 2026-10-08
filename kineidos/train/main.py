"""Entry point for P004 training.

Mirrors runner/train.py's main() -- the same two-pass parse, so that command
line arguments keep the same precedence -- and adds one step between the
passes: the fusion's width has to reach DiffusionModule's constructor, and
DiffusionModule is built as `DiffusionModule(**configs.model.diffusion_module)`,
so wp_token_dim has to be in that dict before parse_configs runs.  Setting it
on an already-parsed config would be too late and would fail quietly, by
building the baseline architecture under an ablation arm's name.

Run from the workspace root (AGENTS.md):

    PYTHONPATH=repos/research/kineidos-v3:repos/research/wp-v2 \\
    LAYERNORM_TYPE=torch ATTN_IMPL=sdpa LD_LIBRARY_PATH=$ENV/lib \\
      $ENV/bin/python -m kineidos.train.main --wp.mode random ...
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping

import torch

from configs.configs_base import configs as configs_base
from configs.configs_data import data_configs
from configs.configs_model_type import model_configs
from protenix.config import parse_configs
from protenix.utils.distributed import DIST_WRAPPER

from kineidos.train.trainer import KineidosTrainer, wp_token_dim_for

# Undo protenix/data/pipeline/data_pipeline.py:36, which runs at import and sets
# torch's sharing strategy to "file_system".  Torch's own default on Linux is
# "file_descriptor", and the difference is who can destroy a batch in flight.
#
# Under "file_system" every tensor a dataloader worker hands to the main process
# is a *named* file under /dev/shm, opened by name on the far side.  Under
# "file_descriptor" the same file is unlinked the moment it is created and only
# its descriptor crosses the socket, so the mapping has no name for anything
# else to act on.
#
# This is not hypothetical.  On 2026-10-07 at 19:34:35 another job of ours
# landed on deep-chungus-1 and its prolog cleared /dev/shm; twenty seconds later
# both P009 arms on that node died inside `for batch in self.train_dl`, all six
# ranks at the same second, with
#
#   RuntimeError: unable to open shared memory object </torch_...> in
#   read-write mode: No such file or directory (2)
#
# raised from reductions.py's rebuild_storage_filename -- the "file_system" path.
# Neither resource was short: /dev/shm was 252 GB at 1% and the descriptor limit
# was 1048576.  The files had simply been deleted by someone else.  The two arms
# alone on their nodes were untouched, which is what makes this the explanation
# rather than a guess about memory.
#
# Upstream's choice is defensible for its own pipeline -- thousands of mmCIF
# workers can exhaust descriptors -- but we run three ranks of four workers on
# windows of 470 atoms, a few hundred descriptors against a million.
#
# It must be set before any worker is forked, hence here rather than in
# init_data, and after the protenix import that sets it the other way.
torch.multiprocessing.set_sharing_strategy("file_descriptor")


def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, Mapping) and k in d and isinstance(d[k], Mapping):
            deep_update(d[k], v)
        else:
            d[k] = v
    return d


def build_configs(arg_str: str):
    """The two-pass parse, with wp_token_dim placed between the passes.

    Separate from main() so that a tool which is not the training loop can
    reach the same configuration by the same route.  P010's
    kineidos.score_sigma_grid scores a checkpoint with `KineidosTrainer`, and
    building its config any other way would be a second answer to "what does
    this arm's architecture look like" -- exactly the gap the constructor check
    in KineidosTrainer.__init__ exists to catch.
    """
    from protenix.config import parse_sys_args  # noqa: F401  (documented route)

    # Same environment switches upstream's main() honours.  On a login node
    # both must be "torch": the cuequivariance kernels dlopen libcuda.so.1.
    configs_base["triangle_attention"] = os.environ.get(
        "TRIANGLE_ATTENTION", "cuequivariance")
    configs_base["triangle_multiplicative"] = os.environ.get(
        "TRIANGLE_MULTIPLICATIVE", "cuequivariance")

    first = parse_configs({**configs_base, **{"data": data_configs}},
                          arg_str=arg_str, fill_required_with_null=True)
    model_name = first.model_name
    wp_mode = first.wp.mode

    base = {**configs_base, **{"data": data_configs}}
    deep_update(base, model_configs[model_name])

    # The one addition to upstream's sequence.  Absent for `none`, so that arm
    # constructs exactly upstream's architecture rather than one with a fusion
    # that happens to be fed zeros -- that is what `zero` is for.
    token_dim = wp_token_dim_for(wp_mode)
    if token_dim is not None:
        base["model"]["diffusion_module"]["wp_token_dim"] = token_dim

    return parse_configs(configs=base, arg_str=arg_str,
                         fill_required_with_null=True)


def main() -> None:
    from protenix.config import parse_sys_args

    logging.basicConfig(
        format=("%(asctime)s,%(msecs)-3d %(levelname)-8s "
                "[%(filename)s:%(lineno)s %(funcName)s] %(message)s"),
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode="w",
    )
    configs = build_configs(parse_sys_args())
    model_name = configs.model_name
    wp_mode = configs.wp.mode
    token_dim = configs.model.diffusion_module.get("wp_token_dim", None)

    # Seeding is not done here.  AF3Trainer.init_env calls seed_everything
    # (protenix/utils/seed.py:22) immediately before the model is built, and it
    # already covers all four sources plan section 2.12 asks for -- random,
    # numpy, torch, torch.cuda -- from a rank-derived seed.  A set_all_seeds
    # call here would be overwritten by it and would read as if it were doing
    # something.
    # Fatal, not a warning.  Something importing protenix later could set it
    # back, and the symptom would be a crash hours in that looks like a cluster
    # problem rather than a configuration one.
    strategy = torch.multiprocessing.get_sharing_strategy()
    if strategy != "file_descriptor":
        raise SystemExit(
            f"refusing to start: torch sharing strategy is {strategy!r}, not "
            f"'file_descriptor'. Under 'file_system' the dataloader's tensors "
            f"are named files in /dev/shm and any job that clears it on this "
            f"node kills this run mid-batch; that is what happened to two arms "
            f"on 2026-10-07. Something re-set it after kineidos.train.main's "
            f"module level -- find it rather than removing this check."
        )
    logging.info(
        f"model={model_name} wp.mode={wp_mode} "
        f"wp_token_dim={token_dim} cycle={configs.model.N_cycle} "
        f"sharing={strategy} "
        f"rank={DIST_WRAPPER.rank}/{DIST_WRAPPER.world_size}"
    )
    KineidosTrainer(configs).run()


if __name__ == "__main__":
    main()
