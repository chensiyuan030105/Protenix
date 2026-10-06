"""P004.2 acceptance for the dataloader.  Exit 0 only if every check passes."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from kineidos.data.gagu import GAGUProtenixAdapter
from kineidos.data.windows import (
    DT_MAX_NS,
    DT_MIN_NS,
    GAGUWindowDataset,
    build_window,
    sample_delta_t_ns,
    stride_for,
)

SAMPLE = Path("/mnt/xfs/home/mhg/Projects/ForSiyuan/RNA-WorldParticle-Workspace/"
              "datasets/processed/gagu_internal_loop_v0_1/gagu_100mM_K_agaguu_startI_r1")

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def main() -> int:
    print("=== 1. sample ===")
    sample = GAGUProtenixAdapter(SAMPLE).load()
    print(f"  {sample.sample_id}: {sample.n_atoms} atoms, {sample.n_tokens} tokens, "
          f"{sample.n_frames} frames, {sample.duration_ns:.0f} ns")
    check("470 heavy atoms", sample.n_atoms == 470, str(sample.n_atoms))
    nn = float(np.linalg.norm(
        sample.position_nm[0][:, None, :] - sample.position_nm[0][None, :, :], axis=-1
    )[np.triu_indices(sample.n_atoms, k=1)].min())
    check("positions are nanometres", 0.05 <= nn <= 0.3,
          f"nearest pair {nn:.4f} nm")
    check("Angstrom view is 10x", np.allclose(
        sample.position_angstrom[0], sample.position_nm[0] * 10.0))

    print("\n=== 2. dt sampling is LogUniform[0.1, 100] ns ===")
    rng = np.random.default_rng(0)
    dts = np.array([sample_delta_t_ns(rng) for _ in range(200_000)])
    print(f"  range [{dts.min():.4f}, {dts.max():.2f}] ns, "
          f"median {np.median(dts):.3f}")
    check("within bounds", DT_MIN_NS <= dts.min() and dts.max() <= DT_MAX_NS)
    # LogUniform means log(dt) is uniform; its median is the geometric mean of
    # the bounds, and each decade holds an equal share.  Testing the decades
    # catches a linear-uniform draw, which would pile 90% into the top one.
    geo = float(np.sqrt(DT_MIN_NS * DT_MAX_NS))
    check("median at the geometric mean", abs(np.median(dts) - geo) / geo < 0.02,
          f"{np.median(dts):.3f} vs {geo:.3f}")
    decades = [((dts >= 10.0 ** e) & (dts < 10.0 ** (e + 1))).mean()
               for e in (-1, 0, 1)]
    print(f"  decade shares: {[round(float(x), 3) for x in decades]}")
    check("decades equally populated", all(abs(x - 1 / 3) < 0.01 for x in decades))

    print("\n=== 3. stride clamping ===")
    check("dt below the save interval still gives stride >= 1",
          stride_for(0.001, 0.1, 10000, 8) == 1)
    check("dt at 100 ns gives stride 1000",
          stride_for(100.0, 0.1, 10000, 8) == 1000)
    check("stride capped so the window fits",
          stride_for(10_000.0, 0.1, 10000, 8) == (10000 - 1) // 8,
          str(stride_for(10_000.0, 0.1, 10000, 8)))

    print("\n=== 4. one window ===")
    w = build_window(sample, target_frame=5000, stride=100, k=8)
    print(f"  target {w.target_frame}, stride {w.stride}, dt {w.delta_t_ns} ns")
    print(f"  history {w.history_frames.tolist()}")
    print(f"  wp_position {tuple(w.wp_position_nm.shape)}  "
          f"frame_time {w.wp_frame_time_ns.tolist()}")
    check("history is K frames", w.wp_position_nm.shape == (8, 470, 3))
    check("history strictly before the target",
          bool((w.history_frames < w.target_frame).all()))
    check("history ordered oldest first",
          bool((np.diff(w.history_frames) > 0).all()))
    check("dt matches stride", w.delta_t_ns == 100 * 0.1)
    check("relative time is -K*dt .. -dt", torch.allclose(
        w.wp_frame_time_ns,
        torch.tensor([-80.0, -70.0, -60.0, -50.0, -40.0, -30.0, -20.0, -10.0])))
    check("all frames valid away from the start", bool(w.wp_frame_mask.all()))
    check("canonicalisation ran", np.isfinite(w.anchor_rmsd_to_ref_nm),
          f"anchor RMSD {w.anchor_rmsd_to_ref_nm:.4f} nm")

    print("\n=== 5. padding and mask near the start ===")
    w2 = build_window(sample, target_frame=300, stride=100, k=8)
    print(f"  history {w2.history_frames.tolist()}")
    print(f"  mask    {w2.wp_frame_mask.tolist()}")
    n_valid = int(w2.wp_frame_mask.sum())
    check("only the reachable frames are valid", n_valid == 3, str(n_valid))
    check("padded entries repeat the oldest real frame",
          bool((w2.history_frames[: 8 - n_valid] == w2.history_frames[8 - n_valid]).all()))
    check("padding is a real configuration, not zeros",
          bool(w2.wp_position_nm[0].abs().sum() > 0))

    print("\n=== 6. canonicalisation makes the window pose-independent ===")
    # The property P004.1 established, checked on the dataloader's own output
    # because that is where it has to hold.
    #
    # Only the non-reference frames are rotated, and that is the correct model
    # rather than a convenience.  ref_pos is a fixed, sample-level artefact --
    # the PDB, which Protenix uses too -- while the pose of each trajectory
    # frame is what varies, since the molecule tumbles and a window drawn from
    # late frames arrives in whatever orientation the dynamics produced.
    # Rotating the reference as well would leave nothing fixed to canonicalise
    # against: the algebra degenerates to the "align to the window's own oldest
    # frame" case, V' = R V R^T, and the result still carries R^T
    # (kineidos/window_align.py).
    g = np.random.default_rng(7)
    q, r = np.linalg.qr(g.normal(size=(3, 3)))
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    pos_rot = sample.position_nm.copy()
    vel_rot = sample.velocity_nm_per_ps.copy()
    pos_rot[1:] = pos_rot[1:] @ q.T
    vel_rot[1:] = vel_rot[1:] @ q.T
    rotated = type(sample)(**{**sample.__dict__,
                              "position_nm": pos_rot,
                              "velocity_nm_per_ps": vel_rot})
    w_rot = build_window(rotated, target_frame=5000, stride=100, k=8)
    drift = float((w.wp_position_nm - w_rot.wp_position_nm).abs().max())
    check("reposing the window leaves the canonicalised output unchanged",
          drift < 1e-4, f"max |diff| {drift:.3e} nm")
    # And the velocities, which take the rotation and not the translation.
    vdrift = float((w.wp_velocity_nm_per_ps - w_rot.wp_velocity_nm_per_ps).abs().max())
    check("velocities canonicalise with the positions", vdrift < 1e-4,
          f"max |diff| {vdrift:.3e} nm/ps")

    print("\n=== 7. dataset is deterministic per index ===")
    ds = GAGUWindowDataset([sample], k=8, length=64, seed=0)
    a, b = ds[11], ds[11]
    check("same index gives the same window",
          a.target_frame == b.target_frame and a.stride == b.stride
          and torch.equal(a.wp_position_nm, b.wp_position_nm))
    strides = sorted({ds[i].stride for i in range(32)})
    print(f"  strides seen over 32 draws: {strides}")
    check("strides vary over draws", len(strides) > 5, str(len(strides)))
    check("every draw leaves at least one real history frame",
          all(bool(ds[i].wp_frame_mask.any()) for i in range(32)))

    print("\n" + "=" * 62)
    print("P004.2 验收:", "全部通过" if not FAILS else f"{len(FAILS)} 项失败 -> {FAILS}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())
