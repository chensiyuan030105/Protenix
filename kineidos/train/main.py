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

from configs.configs_base import configs as configs_base
from configs.configs_data import data_configs
from configs.configs_model_type import model_configs
from protenix.config import parse_configs
from protenix.utils.distributed import DIST_WRAPPER

from kineidos.train.trainer import KineidosTrainer, wp_token_dim_for


def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, Mapping) and k in d and isinstance(d[k], Mapping):
            deep_update(d[k], v)
        else:
            d[k] = v
    return d


def main() -> None:
    from protenix.config import parse_sys_args

    logging.basicConfig(
        format=("%(asctime)s,%(msecs)-3d %(levelname)-8s "
                "[%(filename)s:%(lineno)s %(funcName)s] %(message)s"),
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode="w",
    )
    # Same environment switches upstream's main() honours.  On a login node
    # both must be "torch": the cuequivariance kernels dlopen libcuda.so.1.
    configs_base["triangle_attention"] = os.environ.get(
        "TRIANGLE_ATTENTION", "cuequivariance")
    configs_base["triangle_multiplicative"] = os.environ.get(
        "TRIANGLE_MULTIPLICATIVE", "cuequivariance")

    arg_str = parse_sys_args()
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

    configs = parse_configs(configs=base, arg_str=arg_str,
                            fill_required_with_null=True)

    # Seeding is not done here.  AF3Trainer.init_env calls seed_everything
    # (protenix/utils/seed.py:22) immediately before the model is built, and it
    # already covers all four sources plan section 2.12 asks for -- random,
    # numpy, torch, torch.cuda -- from a rank-derived seed.  A set_all_seeds
    # call here would be overwritten by it and would read as if it were doing
    # something.
    logging.info(
        f"model={model_name} wp.mode={wp_mode} "
        f"wp_token_dim={token_dim} cycle={configs.model.N_cycle} "
        f"rank={DIST_WRAPPER.rank}/{DIST_WRAPPER.world_size}"
    )
    KineidosTrainer(configs).run()


if __name__ == "__main__":
    main()
