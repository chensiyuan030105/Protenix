"""Environment acceptance for P004 -- also the environment's regression test.

Re-run this after anything touches the environment: a package installed into it,
a torch change, a venv rebuilt with --system-site-packages, a PYTHONPATH that
reaches into another environment.  Each check guards a failure mode that does
not announce itself:

  - open3d's compiled torch ops are ABI-bound to one torch minor version.  A
    mismatch either fails to load with an undefined c10 symbol or leaves the
    branch silently unexercised, so the build target is asserted equal to the
    installed torch rather than merely "importable".
  - More than one site-packages on sys.path means `import torch` resolves by
    path order again -- the shape of the editable-install incident AGENTS.md
    records.
  - continuous_conv's forward working does not mean the thing can train.  v0.1
    failed precisely because gradients never reached the WorldParticle branch,
    so the backward pass and non-zero gradients are checked, not the shape.

Exit code is 0 only if every check passes.

Usage (from the workspace root):
    PYTHONPATH=repos/research/kineidos-v3:repos/research/wp-v2 \
    LAYERNORM_TYPE=torch ATTN_IMPL=sdpa \
    LD_LIBRARY_PATH=/mnt/xfs/home/mhg/anaconda3/envs/kineidos-v2-slurm/lib \
      .../envs/kineidos-v2-slurm/bin/python -m kineidos.check_env

PROTENIX_CKPT overrides the checkpoint path.
"""
import os, sys, re
from collections.abc import Mapping
from pathlib import Path

FAILS = []
def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not cond: FAILS.append(name)

print("=== 1. 解释器与 torch/open3d ===")
import torch, open3d
from open3d import _build_config
print(f"  python {sys.version.split()[0]}  prefix {sys.prefix}")
check("torch == 2.13.0+cu126", torch.__version__ == "2.13.0+cu126", torch.__version__)
check("open3d == 0.20.0", open3d.__version__ == "0.20.0", open3d.__version__)
check("open3d 构建目标 == 实装 torch",
      _build_config["Pytorch_VERSION"] == torch.__version__,
      f'{_build_config["Pytorch_VERSION"]} vs {torch.__version__}')
check("site-packages 唯一（无 system-site-packages 叠加）",
      sum(1 for p in sys.path if p.endswith("site-packages")) == 1,
      str([p for p in sys.path if p.endswith("site-packages")]))

print("\n=== 2. 代码来源落在 repos/research/ ===")
import protenix
import models.super_particle_layers as spl
import TrajDataset
# Not a fixed worktree name: this read "kineidos-v2" until P009 moved the work
# to v3, at which point the check would have failed on a correct environment and
# the obvious repair would have been to edit the name again.  What has to hold is
# that protenix and kineidos come out of the *same* tree -- one of them resolving
# elsewhere is how the editable install of 2026-10-05 went unnoticed -- and that
# the tree is one of ours under repos/research/.
import kineidos
tree = Path(kineidos.__file__).resolve().parent.parent
check("protenix 与 kineidos 同一棵树", Path(protenix.__file__).resolve().is_relative_to(tree),
      f"{protenix.__file__} vs {tree}")
check("该树在 repos/research/ 下", "/repos/research/" in str(tree), str(tree))
check("WP models -> wp-v2", "/repos/research/wp-v2/" in spl.__file__, spl.__file__)

print("\n=== 3. 重命名后的名字 ===")
check("load_molecular_sample_by_data 存在", hasattr(TrajDataset, "load_molecular_sample_by_data"))
check("旧名 load_fluid_sample_by_data 已消失", not hasattr(TrajDataset, "load_fluid_sample_by_data"))
check('数据标识 "fluid" 保留', "fluid" in TrajDataset.OBSTACLE_DATA_TYPES)

print("\n=== 4. open3d 算子（含反向）===")
from open3d.ml.torch.python import ops
from open3d.ml.torch.ops import reduce_subarrays_sum
x = torch.rand(64, 3); q = torch.rand(16, 3)
t = ops.build_spatial_hash_table(x, 0.5, torch.LongTensor([0, 64]), 1/64)
r = ops.fixed_radius_search(x, q, 0.5, torch.LongTensor([0,64]), torch.LongTensor([0,16]),
                            t.hash_table_splits, t.hash_table_index, t.hash_table_cell_splits)
filters = torch.randn(4,4,4,4,8, requires_grad=True); feats = torch.randn(64,4, requires_grad=True)
out = ops.continuous_conv(filters=filters, out_positions=q, extents=torch.tensor([[1.0]]),
    offset=torch.zeros(3), inp_positions=x, inp_features=feats,
    inp_importance=torch.empty((0,)), neighbors_index=r.neighbors_index,
    neighbors_row_splits=r.neighbors_row_splits, neighbors_importance=torch.empty((0,)),
    align_corners=True, coordinate_mapping='ball_to_cube_radial',
    interpolation='linear', normalize=False)
out.sum().backward()
check("continuous_conv 前向", tuple(out.shape) == (16, 8), str(tuple(out.shape)))
check("continuous_conv 反向梯度非零", filters.grad.norm().item() > 0 and feats.grad.norm().item() > 0,
      f"{filters.grad.norm():.2f} / {feats.grad.norm():.2f}")
check("reduce_subarrays_sum", reduce_subarrays_sum(
      torch.ones_like(r.neighbors_index, dtype=torch.float32), r.neighbors_row_splits).shape[0] == 16)

print("\n=== 5. Protenix 建模 + checkpoint strict 加载 ===")
from configs.configs_base import configs as configs_base
from configs.configs_data import data_configs
from configs.configs_inference import inference_configs
from configs.configs_model_type import model_configs
from protenix.config import parse_configs
from protenix.model.protenix import Protenix
MODEL = "protenix_base_default_v1.0.0"
CKPT = os.environ.get(
    "PROTENIX_CKPT",
    "/mnt/xfs/home/mhg/Projects/ForSiyuan/RNA-WorldParticle-Workspace/runtimes/"
    "checkpoints/protenix_pretrained/protenix_base_default_v1.0.0.pt",
)
def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, Mapping) and k in d and isinstance(d[k], Mapping): deep_update(d[k], v)
        else: d[k] = v
    return d
base = {**configs_base, **{"data": data_configs}, **inference_configs}
deep_update(base, model_configs[MODEL])
cfg = parse_configs(configs=base, arg_str=f"--model_name {MODEL}", fill_required_with_null=True)
model = Protenix(cfg)
n = sum(p.numel() for p in model.parameters()) / 1e6
check("参数量 368.48M", abs(n - 368.48) < 0.01, f"{n:.2f}M")
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
sd = ck["model"]
if next(iter(sd)).startswith("module."): sd = {k[7:]: v for k, v in sd.items()}
missing, unexpected = model.load_state_dict(sd, strict=False)
check("missing == 0", len(missing) == 0, str(len(missing)))
check("unexpected == 0", len(unexpected) == 0, str(len(unexpected)))
model.load_state_dict(sd, strict=True)
check("strict=True 加载不抛异常", True)

print("\n" + ("=" * 60))
print("验收结果:", "全部通过" if not FAILS else f"{len(FAILS)} 项失败 -> {FAILS}")
sys.exit(0 if not FAILS else 1)
