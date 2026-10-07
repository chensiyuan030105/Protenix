"""How much can the history possibly tell us about the target frame?

P004 picked `Δt ~ LogUniform[0.1, 100] ns` (plan section 2.8) by what GAGU can
supply: 0.01 ns is below the 100 ps save interval, and 10 ns would leave a
decade of the 1 μs trajectories unused.  Both true, and both about data
availability -- neither is about how long the molecule stays correlated with
itself.  This module measures that, because if the target has forgotten the
history then no amount of conditioning can help and the loss-minimising move is
to shut the `h` pathway off.  See plan P009.

Two numbers per trajectory:

  persistence(Δt)   RMSD between frames Δt apart, after optimal superposition.
                    This is the error of "copy the most recent frame".
  s                 RMSD of a frame to the trajectory mean.  This is the error
                    of "ignore the history and emit a generic conformer".

Under stationarity and a Gaussian approximation these two pin down the
correlation that a predictor could exploit,

    persistence(Δt)² = 2·s²·(1 − ρ(Δt))            ⟹  ρ(Δt)

and the best predictor that shrinks one history frame toward the mean,
`x̂ = mean + ρ·(x(t−Δt) − mean)`, has error `s·√(1−ρ²)`.  Reported as
`max_gain = 1 − √(1−ρ²)`: the fraction of the no-history error that the
history could remove.

`max_gain` is indicative, not a ceiling.  It assumes a *linear* predictor on
*one* frame, while the model is nonlinear and sees K=8.  It is reported because
the alternative -- comparing two hand-picked predictors -- hides how little
room there is: at Δt = 100 ns on GAGU, persistence is worse than emitting the
mean, yet the honest statement is not "history hurts" (it never does) but
"there is 0.1% to win and our measurement noise is larger than that".

Cross-replica RMSD is reported alongside as the empirical decorrelation
plateau.  Where persistence stays below it even at the largest Δt, the
trajectory has not finished mixing and the plateau is the stricter reference.

Run from the workspace root:

    PYTHONPATH=repos/research/kineidos-v2:repos/research/wp-v2 \
    LAYERNORM_TYPE=torch ATTN_IMPL=sdpa \
      <env>/bin/python -m kineidos.measure_decorrelation --out artifacts/reports/P009
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

GAGU_ROOT = ("/mnt/xfs/home/mhg/Projects/ForSiyuan/RNA-WorldParticle-Workspace"
             "/datasets/processed/gagu_internal_loop_v0_1")
ATLAS_ROOT = "datasets/raw/atlas_2022_06_13/protein"

# One decade either side of where GAGU's curve turns over, so the lag grid is
# not chosen to flatter any particular answer.
N_LAGS = 16
N_ORIGINS = 150
N_PAIRS = 400


def kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """RMSD of each [N,3] in `a` to its counterpart in `b`, after superposition.

    Batched over the leading axis.  The reflection fix matters: without it a
    mirror image can score as a good fit, which on a chiral backbone is not a
    small error but a different molecule.
    """
    a = a - a.mean(-2, keepdims=True)
    b = b - b.mean(-2, keepdims=True)
    h = np.einsum("mni,mnj->mij", a, b)
    u, _, vt = np.linalg.svd(h)
    det = np.sign(np.linalg.det(np.einsum("mij,mjk->mik",
                                         vt.transpose(0, 2, 1),
                                         u.transpose(0, 2, 1))))
    d = np.zeros_like(h)
    d[:, 0, 0] = d[:, 1, 1] = 1.0
    d[:, 2, 2] = det
    rot = np.einsum("mij,mjk,mkl->mil", vt.transpose(0, 2, 1), d,
                    u.transpose(0, 2, 1))
    return np.sqrt(((np.einsum("mij,mnj->mni", rot, a) - b) ** 2)
                   .sum(-1).mean(-1))


def max_gain(persistence: float, spread: float) -> tuple[float, float]:
    """(rho, max_gain) from the two measured RMSDs.  See the module docstring."""
    rho = 1.0 - persistence ** 2 / (2.0 * spread ** 2)
    return rho, 1.0 - float(np.sqrt(max(0.0, 1.0 - rho ** 2)))


def curve(frames: list[np.ndarray], interval_ns: float, cross: float,
          rng: np.random.Generator) -> dict:
    """persistence / rho / max_gain over a log-spaced lag grid."""
    n = min(x.shape[0] for x in frames)
    max_lag = n // 8          # K=8 windows cannot reach further anyway
    lags = np.unique(np.round(
        np.logspace(0, np.log10(max_lag), N_LAGS)).astype(int))

    spread = float(np.mean([
        kabsch_rmsd(x[rng.choice(x.shape[0], size=N_PAIRS, replace=False)],
                    np.repeat(x.mean(0, keepdims=True), N_PAIRS, 0)).mean()
        for x in frames]))

    rows = []
    for lag in lags:
        vals = []
        for x in frames:
            avail = x.shape[0] - lag
            if avail < 20:
                continue
            idx = rng.choice(avail, size=min(N_ORIGINS, avail), replace=False)
            vals.append(kabsch_rmsd(x[idx], x[idx + lag]))
        if not vals:
            continue
        v = np.concatenate(vals)
        rho, gain = max_gain(float(v.mean()), spread)
        rows.append({"dt_ns": float(lag * interval_ns), "stride": int(lag),
                     "persistence": float(v.mean()), "std": float(v.std()),
                     "rho": rho, "max_gain": gain})
    return {"spread_to_mean": spread, "cross_replica": cross, "rows": rows}


def mean_gain(rows: list[dict], lo_ns: float, hi_ns: float) -> float:
    """max_gain averaged log-uniformly over [lo, hi] -- the sampling we use."""
    x = np.log10([r["dt_ns"] for r in rows])
    y = np.array([r["max_gain"] for r in rows])
    g = np.linspace(np.log10(lo_ns), np.log10(hi_ns), 600)
    return float(np.interp(g, x, y).mean())


def heavy_atom_index(pdb: str) -> np.ndarray:
    """Indices of non-hydrogen atoms, read from the PDB element column.

    GAGU's npz carries all 712 atoms while the model sees the 470 heavy ones.
    Mixing the two shifts every RMSD -- hydrogens fluctuate most -- so the
    subset is taken here rather than assumed.
    """
    idx, i = [], 0
    with open(pdb) as handle:
        for line in handle:
            if line.startswith(("ATOM", "HETATM")):
                element = line[76:78].strip() or line[12:16].strip()[0]
                if element != "H":
                    idx.append(i)
                i += 1
    return np.array(idx)


def load_gagu(n_traj: int, rng: np.random.Generator) -> dict:
    held_out = sorted(glob.glob(f"{GAGU_ROOT}/*_r4"))[:n_traj]
    if not held_out:
        raise FileNotFoundError(f"no *_r4 samples under {GAGU_ROOT}")
    frames, cross = [], []
    for directory in held_out:
        heavy = heavy_atom_index(glob.glob(f"{directory}/*.pdb")[0])
        with np.load(glob.glob(f"{directory}/*.npz")[0],
                     allow_pickle=False) as npz:
            x = np.asarray(npz["position"], dtype=np.float64)[:, heavy] * 10.0
        frames.append(x)
        # Same sample, same salt, same starting conformer, different seed.
        base = os.path.basename(directory)[:-3]
        for replica in ("r1", "r2", "r3"):
            other = f"{GAGU_ROOT}/{base}_{replica}"
            found = glob.glob(f"{other}/*.npz")
            if not found:
                continue
            with np.load(found[0], allow_pickle=False) as npz:
                y = np.asarray(npz["position"], dtype=np.float64)[:, heavy] * 10.0
            m = min(x.shape[0], y.shape[0])
            j = rng.choice(m, size=N_PAIRS, replace=False)
            cross.append(kabsch_rmsd(x[j], y[j]))
    out = curve(frames, 0.1, float(np.concatenate(cross).mean()), rng)
    out.update(system="GAGU", n_atoms=int(frames[0].shape[1]),
               n_frames=int(frames[0].shape[0]), interval_ns=0.1,
               atoms="heavy", trajectories=[os.path.basename(d) for d in held_out])
    return out


def load_atlas(pids: list[str], rng: np.random.Generator) -> list[dict]:
    import mdtraj as md
    results = []
    for pid in pids:
        top = md.load(f"{ATLAS_ROOT}/{pid}/{pid}.pdb")
        ca = top.topology.select("name CA")
        def xyz(replica: str) -> np.ndarray:
            t = md.load(f"{ATLAS_ROOT}/{pid}/{pid}_prod_{replica}_fit.xtc",
                        top=top, atom_indices=ca)
            return t.xyz.astype(np.float64) * 10.0
        x, y = xyz("R1"), xyz("R2")
        m = min(x.shape[0], y.shape[0])
        j = rng.choice(m, size=N_PAIRS, replace=False)
        out = curve([x], 0.01, float(kabsch_rmsd(x[j], y[j]).mean()), rng)
        out.update(system=pid, n_atoms=int(len(ca)), n_frames=int(x.shape[0]),
                   interval_ns=0.01, atoms="CA", trajectories=[f"{pid}_prod_R1"])
        results.append(out)
    return results


def render(block: dict, ranges: list[tuple[float, float]]) -> str:
    lines = [
        f"### {block['system']}  "
        f"({block['n_atoms']} {block['atoms']} atoms, {block['n_frames']} frames "
        f"@ {block['interval_ns']} ns = {block['n_frames']*block['interval_ns']:.0f} ns)",
        "",
        f"no-history error `s` = {block['spread_to_mean']:.2f} A | "
        f"cross-replica plateau = {block['cross_replica']:.2f} A",
        "",
        "| dt (ns) | stride | persistence (A) | rho | max_gain |",
        "|---:|---:|---:|---:|---:|",
    ]
    for r in block["rows"]:
        lines.append(f"| {r['dt_ns']:.2f} | {r['stride']} | "
                     f"{r['persistence']:.2f} +- {r['std']:.2f} | "
                     f"{r['rho']:.3f} | {r['max_gain']*100:.1f}% |")
    lines += ["", "log-uniform averages:", ""]
    for lo, hi in ranges:
        lines.append(f"- `[{lo:g}, {hi:g}]` ns "
                     f"({np.log10(hi/lo):.1f} decades): "
                     f"**{mean_gain(block['rows'], lo, hi)*100:.1f}%**")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="artifacts/reports/P009")
    parser.add_argument("--gagu-trajectories", type=int, default=4)
    parser.add_argument("--atlas", nargs="*", default=["1a62_A", "1ab1_A"],
                        help="ATLAS chains for the cross-system reference; "
                             "empty to skip (mdtraj and the 680 GB tree)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    blocks = [load_gagu(args.gagu_trajectories, rng)]
    gagu_ranges = [(0.1, 100.0), (0.1, 10.0), (0.1, 2.5), (0.1, 1.0), (0.1, 0.4)]
    text = ["# Decorrelation and the room left for the history",
            "",
            "Generated by `kineidos.measure_decorrelation`; see its docstring "
            "for what `rho` and `max_gain` mean and what they assume.",
            "",
            render(blocks[0], gagu_ranges)]

    if args.atlas:
        for block in load_atlas(list(args.atlas), rng):
            blocks.append(block)
            text.append(render(block, [(0.01, 10.24), (0.01, 1.0)]))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "decorrelation.json").write_text(json.dumps(blocks, indent=1))
    (out / "decorrelation.md").write_text("\n".join(text))
    print("\n".join(text))
    print(f"\nwrote {out/'decorrelation.md'} and {out/'decorrelation.json'}")


if __name__ == "__main__":
    main()
