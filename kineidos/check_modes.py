"""P004.4 acceptance: the four arms, the observables, and the bridge's guard."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from kineidos import observables
from kineidos.data.gagu import GAGUProtenixAdapter
from kineidos.data.windows import build_window
from kineidos.seeding import set_all_seeds
from kineidos.wp_bridge import MODES, TOKEN_DIM, WorldParticleBridge

SAMPLE = Path("/mnt/xfs/home/mhg/Projects/ForSiyuan/RNA-WorldParticle-Workspace/"
              "datasets/processed/gagu_internal_loop_v0_1/gagu_100mM_K_agaguu_startI_r1")
FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def main() -> int:
    set_all_seeds(0)

    print("=== 1. window ===")
    sample = GAGUProtenixAdapter(SAMPLE).load()
    w = build_window(sample, target_frame=5000, stride=100, k=8)
    n = w.wp_position_nm.shape[1]
    print(f"  {n} atoms, K={w.wp_position_nm.shape[0]}, dt={w.delta_t_ns} ns, "
          f"anchor RMSD {w.anchor_rmsd_to_ref_nm:.4f} nm")

    print("\n=== 2. the three arms the first version runs ===")
    hs = {}
    for mode in ("none", "zero", "random"):
        bridge = WorldParticleBridge(mode, seed=0)
        with torch.no_grad():
            h = bridge(w)
        hs[mode] = h
        shape = "None" if h is None else tuple(h.shape)
        print(f"  {mode:<10} h = {shape}")
    check("none yields no tokens, so the fusion is skipped", hs["none"] is None)
    check("zero yields the right shape", tuple(hs["zero"].shape) == (n, TOKEN_DIM))
    check("zero is exactly zero", bool((hs["zero"] == 0).all()))
    check("random yields the right shape", tuple(hs["random"].shape) == (n, TOKEN_DIM))
    # The arm has to carry information, or it is just a slower `zero`.
    check("random is not degenerate",
          float(hs["random"].std()) > 1e-6,
          f"std {float(hs['random'].std()):.4f}, "
          f"|h| {float(hs['random'].norm()):.1f}")

    print("\n=== 3. pretrained refuses to pretend ===")
    try:
        WorldParticleBridge("pretrained")
        check("pretrained without a checkpoint raises", False, "it did not raise")
    except ValueError as exc:
        check("pretrained without a checkpoint raises", "Stage 1" in str(exc),
              str(exc)[:56])
    try:
        WorldParticleBridge("nonsense")
        check("an unknown mode raises", False, "it did not raise")
    except ValueError:
        check("an unknown mode raises", True, f"modes are {MODES}")

    print("\n=== 4. pooling honours the validity mask ===")
    bridge = WorldParticleBridge("random", seed=0)
    early = build_window(sample, target_frame=300, stride=100, k=8)
    n_valid = int(early.wp_frame_mask.sum())
    print(f"  a window near the start: mask {early.wp_frame_mask.tolist()}")
    with torch.no_grad():
        h_early = bridge(early)
    check("a padded window still produces h", tuple(h_early.shape) == (n, TOKEN_DIM),
          f"{n_valid} of 8 frames valid")
    # Padding repeats the oldest real frame, so averaging it in would be
    # averaging a duplicate -- the mask is what stops that.  Compare against a
    # window whose frames are all valid and all distinct.
    with torch.no_grad():
        h_full = bridge(w)
    check("and differs from a fully valid window",
          not torch.allclose(h_early, h_full),
          f"relative difference "
          f"{float((h_early - h_full).norm() / h_full.norm()):.3e}")

    print("\n=== 5. the bridge refuses a window transformed after the dataloader ===")
    drift = bridge.assert_window_canonical(w)
    check("a clean window passes", drift < 1e-4, f"drift {drift:.3e} nm")
    g = np.random.default_rng(3)
    q, r = np.linalg.qr(g.normal(size=(3, 3)))
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    meddled = type(w)(**w.__dict__)
    meddled.wp_position_nm = torch.from_numpy(
        w.wp_position_nm.double().numpy() @ q.T).float()
    try:
        bridge(meddled)
        check("a rotated window is refused", False, "it did not raise")
    except ValueError as exc:
        check("a rotated window is refused", "canonical pose" in str(exc),
              str(exc)[:58])

    print("\n=== 6. per-frame time is bounded and ordered ===")
    t = bridge._frame_time_feature(w)
    print(f"  {[round(float(x), 3) for x in t.squeeze(-1)]}")
    check("shape is [K, channels]", tuple(t.shape) == (8, 1))
    check("bounded in [-1, 0)", bool((t <= 0).all() and (t >= -1).all()))
    check("oldest frame is -1", abs(float(t[0]) + 1.0) < 1e-6)
    # Order is the only thing this channel has to carry; pooling destroys it
    # otherwise, and the physical scale goes to AdaLN separately.
    check("strictly increasing towards the newest frame",
          bool((t.squeeze(-1).diff() > 0).all()))

    print("\n=== 7. the observables (plan section 5) ===")
    from protenix.model.modules.transformer import AtomAttentionEncoder

    enc = AtomAttentionEncoder(
        has_coords=True, c_token=768, c_atom=128, c_atompair=16, c_s=384,
        c_z=128, n_blocks=1, n_heads=4, n_queries=32, n_keys=128,
        wp_token_dim=TOKEN_DIM)
    check("nothing is reported before a fused forward",
          not getattr(enc, "wp_diagnostics", {}))

    c_l = torch.randn(1, 2, n, 128)
    with torch.no_grad():
        enc._fuse_wp_tokens(c_l, hs["random"])
    # Read through the module's own accessor rather than the attribute, so the
    # test exercises what a training loop would call.
    stats = observables.read(enc)
    check("read() works from the encoder, not only from a whole model",
          "w_h_over_w_cl" in stats, f"{len(stats)} fields")
    for k in ("injection_strength", "h_contribution_norm", "cl_contribution_norm",
              "w_cl_norm", "w_h_norm", "h_norm_after_layernorm"):
        print(f"  {k:<26} {stats.get(k, float('nan')):.6e}")
    check("injection strength is exactly zero at initialisation",
          stats["injection_strength"] == 0.0,
          "by construction: W_h starts at zero")
    check("the c_l half contributes", stats["cl_contribution_norm"] > 0)
    check("h is non-trivial after LayerNorm", stats["h_norm_after_layernorm"] > 0)

    # The alarm must fire in exactly the situation v0.1 was in.
    warns = observables.warnings(stats)
    check("a zero injection raises the alarm", len(warns) >= 1,
          warns[0][:58] if warns else "no warning")
    with torch.no_grad():
        enc.wp_fusion.weight[:, 128:].normal_(0.0, 0.05)
        enc._fuse_wp_tokens(c_l, hs["random"])
    moved = observables.read(enc)
    check("a missing field prints as ? rather than nan",
          "?" in observables.format_line({"injection_strength": 1.0}),
          observables.format_line({"injection_strength": 1.0}))
    print(f"  after perturbing W_h: injection "
          f"{moved['injection_strength']:.4e}")
    check("a real injection clears the alarm",
          moved["injection_strength"] > 1e-3 and not observables.warnings(moved))
    print(f"  {observables.format_line(moved)}")

    print("\n" + "=" * 62)
    print("P004.4 验收:", "全部通过" if not FAILS else f"{len(FAILS)} 项失败 -> {FAILS}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())
