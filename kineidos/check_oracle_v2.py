"""The "switched on, and it works" acceptance for oracle-v2, on the real bridge.

Not the prototype.  P010 section 6 item 5's lesson, which cost two jobs: an
identity check that runs a branch with its switch OFF says nothing about the
branch, and the oracle's dtype bug survived exactly that gap.  So this builds
`WorldParticleBridge(mode="oracle", oracle_encoding="fourier_anchor")` and
calls it the way `Protenix.forward` does -- inside autocast(bfloat16), with a
feature dict carrying the target key.

Six assertions, two of which are positive controls on the other four: the same
tests are run against `oracle_encoding="linear"` and must FAIL there, because
a test that has not caught the bug it was written for has not been tested.
"""
from __future__ import annotations

import math
import sys

import torch

from kineidos.wp_bridge import (ORACLE_TARGET_KEY, TOKEN_DIM,
                                WorldParticleBridge, oracle_anchor_index)

N = 470
torch.manual_seed(0)
# Measured GAGU geometry: 1.47 nm across, 5.17 nm from the canonical origin.
X = torch.randn(N, 3, dtype=torch.float32) * (1.465 / math.sqrt(3))
X = X + torch.tensor([3.0, -2.0, 3.7])
Q = torch.linalg.qr(torch.randn(3, 3))[0]
if torch.det(Q) < 0:
    Q[:, 0] *= -1
T = torch.tensor([7.0, -3.0, 11.0])
LN = torch.nn.LayerNorm(TOKEN_DIM)
fails: list[str] = []


def check(name: str, ok: bool, detail: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        fails.append(name)


def h_of(bridge, x):
    """Call the bridge the way Protenix.forward does: under autocast(bf16)."""
    # The full WP_INPUT_KEYS set: the bridge checks for all of them before
    # the mode branch, which is the right order -- an oracle arm whose collate
    # silently stopped producing the window would otherwise run happily.
    feats = {
        ORACLE_TARGET_KEY: x,
        # assert_window_canonical re-aligns the history to the reference; give
        # it the same coordinates so the transform is the identity.
        "wp_position_nm": x[None].expand(2, N, 3).contiguous(),
        "wp_velocity_nm_per_ps": torch.zeros(2, N, 3),
        "wp_frame_time_ns": torch.tensor([-1.0, -0.125]),
        "wp_frame_mask": torch.ones(2),
        "wp_ref_pos_nm": x,
        "wp_canonicalized": torch.tensor(True),
    }
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        return bridge(feats)


def run(encoding: str, *, expect_pass: bool) -> None:
    print(f"\n=== oracle_encoding={encoding!r} "
          f"（预期{'全过' if expect_pass else '在不变性与尺度两项上失败'}）===")
    b = WorldParticleBridge("oracle", oracle_source="target",
                            oracle_encoding=encoding)
    tag = "" if expect_pass else "（阳性对照）"

    try:
        h = h_of(b, X)
    except Exception as exc:                       # noqa: BLE001
        check(f"{tag}能调用", False, f"{type(exc).__name__}: {exc}")
        return
    check(f"{tag}形状", tuple(h.shape) == (N, TOKEN_DIM), f"{tuple(h.shape)}")
    check(f"{tag}autocast(bf16) 下仍是 float32", h.dtype == torch.float32,
          f"{h.dtype}")

    moved = h_of(b, X @ Q.T + T)
    inv = float((h - moved).abs().max())
    ok_inv = inv < 1e-3
    check(f"{tag}SE(3) 不变", ok_inv == expect_pass,
          f"最大差 {inv:.3e}" + ("" if expect_pass else "  ← 本该大，说明测法有效"))

    dscale = float((LN(h) - LN(h_of(b, X * 1.1))).abs().max())
    ok_scale = dscale > 1e-2
    check(f"{tag}LN 之后仍分辨得出整体缩放", ok_scale == expect_pass,
          f"最大差 {dscale:.3e}" + ("" if expect_pass else "  ← 本该小，这就是 §8 的病"))

    std = h.std(dim=-1, unbiased=False)
    flat = float(std.std() / std.mean())
    check(f"{tag}LN 归一化因子近乎常数", (flat < 0.05) == expect_pass,
          f"相对离散 {flat:.4f}")


run("fourier_anchor", expect_pass=True)
run("linear", expect_pass=False)

# 信息没丢：从锚距三角定位回坐标（只对 v2 有意义）
idx = oracle_anchor_index(N)
c = (X - X.mean(0)).double()
a = c[idx]
d = torch.cdist(c, a, compute_mode="donot_use_mm_for_euclid_dist")
A = 2.0 * (a[1:] - a[0])
rhs = (d[:, :1] ** 2 - d[:, 1:] ** 2) + (a[1:] ** 2).sum(-1) - (a[0] ** 2).sum()
rec = torch.linalg.lstsq(A, rhs.T).solution.T
rmsd = float((rec - c).pow(2).sum(-1).mean().sqrt())
print()
check("v2 的锚距可完全重建结构（信息没丢）", rmsd < 1e-5, f"RMSD {rmsd:.3e} nm")


# --------------------------------------------------------------------------
# readout.md 13.5: the shuffle control must keep everything about the injected
# signal except which atom each row describes.  Both halves are asserted,
# because a shuffle that also changed the distribution would be a different
# experiment and a shuffle that changed nothing would be no experiment.
# --------------------------------------------------------------------------
from kineidos.wp_bridge import oracle_shuffle_index  # noqa: E402

print("\n=== oracle_shuffle=True：边缘分布要保住，对应要摧毁 ===")
b_plain = WorldParticleBridge("oracle", oracle_source="target",
                              oracle_encoding="fourier_anchor")
b_shuf = WorldParticleBridge("oracle", oracle_source="target",
                             oracle_encoding="fourier_anchor",
                             oracle_shuffle=True)
h_p, h_s = h_of(b_plain, X), h_of(b_shuf, X)

check("shuffle 保住逐元素的边缘分布",
      bool(torch.allclose(h_p.flatten().sort().values,
                          h_s.flatten().sort().values, atol=1e-6)),
      "排序后逐元素相同")
check("shuffle 保住每个原子的范数集合",
      bool(torch.allclose(h_p.norm(dim=-1).sort().values,
                          h_s.norm(dim=-1).sort().values, atol=1e-5)),
      f"范数 rms {float(h_p.norm(dim=-1).mean()):.4f}（两者同）")
check("shuffle 摧毁了原子↔几何的对应",
      float((h_p - h_s).abs().max()) > 1e-2,
      f"逐位最大差 {float((h_p - h_s).abs().max()):.3e}")
idx = oracle_shuffle_index(N, 20261008)
check("shuffle 就是一个置换（可完全还原）",
      bool(torch.allclose(h_s[idx.argsort()], h_p, atol=1e-6)),
      "逆置换后与未 shuffle 的逐位相同")
check("shuffle 确实打乱（不是恒等）",
      int((idx != torch.arange(N)).sum()) > N * 0.9,
      f"{int((idx != torch.arange(N)).sum())}/{N} 个原子换了位置")

print("\n全部通过" if not fails else f"\n未通过：{fails}")
sys.exit(1 if fails else 0)
