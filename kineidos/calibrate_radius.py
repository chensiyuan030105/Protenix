"""Calibrate WorldParticle's particle_radius for molecular scale.

config.yaml ships 0.012, tuned for the official fluid data.  That value cannot
be inherited:

    filter_extent = radius_scale * 6 * particle_radius = 1.5 * 6 * 0.012 = 0.108 nm

and ParticleRadiusResearch searches at 0.5 * filter_extent
(continuous_conv.py:313), so the actual cutoff is 0.054 nm -- well under the
0.131 nm mean bond length the plan measures in section 2.5.  Every atom would be
isolated and the convolution would see nothing.

The radius is not a free knob with a "more neighbours is better" answer, because
`extents` does two jobs in ops.continuous_conv.  It sets the search cutoff, and
it sets the scale of the ball_to_cube coordinate mapping -- offsets within the
cutoff are mapped into a kernel_size (4x4x4) grid.  So a wider cutoff buys
coverage and pays with resolution: the kernel cell spans

    cell = 2 * cutoff / kernel_size = cutoff / 2

Two neighbours closer together than one cell land in the same cell and the
convolution cannot tell them apart.  With a 0.131 nm bond length, resolving
bonded from non-bonded neighbours wants cell < 0.131, i.e. cutoff < 0.26 nm,
while seeing base stacking wants several times that.  The tension is real and
this script measures both sides of it rather than picking a neighbour count.

Criteria, stated before the numbers:

1. **Connectivity.** No isolated atoms, and every chain internally connected:
   components <= n_chains.  "No isolated atoms" alone is not sufficient -- the
   plan's section 2.5 found a topology scheme that left only 2 isolated atoms
   while having lost every backbone phosphodiester bond, i.e. 22 disconnected
   nucleotides, so components is the indicator that catches both.

   Note the direction of the inequality.  Section 2.5's criterion was
   components *equal* to the chain count, but that was for a bond topology,
   where bonds do not cross chains.  This is a distance graph, and GAGU is a
   base-paired duplex: once the cutoff reaches hydrogen-bond range the two
   strands are legitimately joined, so components collapsing to 1 is the
   physically correct outcome and not a failure.  Requiring equality here would
   reject exactly the cutoffs that see the base pairing.
2. **Kernel resolution.** cell / bond_length, reported so the coverage versus
   resolution trade is visible rather than implied.  Treat it as a soft
   indicator: `interpolation` is "linear", so the kernel is trilinearly
   interpolated between grid cells and two neighbours sharing a cell are
   blended rather than made indistinguishable.  Resolution degrades smoothly
   instead of hitting a wall.
3. **Information reaching h.** Across-particle spread and effective rank of `h`.
   A representation that is nearly identical for every particle cannot condition
   anything, however many neighbours fed it.  Measured at random init, where it
   reflects how much input geometry the architecture propagates.
4. **Cost.** Neighbour pairs grow as cutoff^3 and every pair is a conv term.

Usage (from the workspace root):
    PYTHONPATH=repos/research/kineidos-v3:repos/research/wp-v2 \
    LAYERNORM_TYPE=torch ATTN_IMPL=sdpa \
    LD_LIBRARY_PATH=/mnt/xfs/home/mhg/anaconda3/envs/kineidos-v2-slurm/lib \
      .../envs/kineidos-v2-slurm/bin/python -m kineidos.calibrate_radius
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from kineidos.probe_h_symmetry import (
    K_PARTITE,
    WP_CONFIG,
    build_model,
    flatten,
    load_gagu_heavy,
    run_once,
)

KERNEL = 4  # WP_CONFIG["kernel_size"] is [4, 4, 4]
RADIUS_SCALE = WP_CONFIG["radius_scale"]
BOND_NM = 0.1309  # plan section 2.5, measured on GAGU: mean heavy-atom bond length


def cutoff_to_particle_radius(cutoff_nm: float) -> float:
    """Invert cutoff = 0.5 * radius_scale * 6 * particle_radius.

    The sweep is parameterised by cutoff because that is the physical quantity;
    particle_radius is the knob that happens to express it, and the factor 4.5
    between them (with radius_scale 1.5) is easy to drop.
    """
    return cutoff_nm / (0.5 * RADIUS_SCALE * 6.0)


def pair_distance_shells(pos: np.ndarray, n_bins: int = 120, r_max: float = 1.2) -> dict:
    """Radial distribution of heavy-atom pairs, to locate the physical shells.

    The shells are what the cutoff has to be chosen against, so they are measured
    from the data rather than taken from chemical intuition.
    """
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    iu = np.triu_indices(len(pos), k=1)
    dist = d[iu]
    hist, edges = np.histogram(dist, bins=n_bins, range=(0.0, r_max))
    centres = 0.5 * (edges[:-1] + edges[1:])
    nz = hist > 0
    first = float(centres[nz][0]) if nz.any() else float("nan")
    return {
        "nearest_pair_nm": float(dist.min()),
        "first_occupied_bin_nm": first,
        "median_pair_nm": float(np.median(dist)),
        "max_pair_nm": float(dist.max()),
        "hist": hist.tolist(),
        "bin_centres": [round(c, 4) for c in centres],
    }


def neighbour_stats(pos: np.ndarray, cutoff: float, n_chains: int) -> dict:
    """Geometry of the neighbour graph at one cutoff, chain-aware."""
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    adj = d <= cutoff
    counts = adj.sum(axis=1)

    rows, cols = np.nonzero(adj)
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=adj.shape)
    n_comp, _labels = connected_components(graph, directed=False)

    return {
        "cutoff_nm": cutoff,
        "particle_radius": cutoff_to_particle_radius(cutoff),
        "mean_neighbours": float(counts.mean()),
        "median_neighbours": float(np.median(counts)),
        "max_neighbours": int(counts.max()),
        "isolated": int((counts == 0).sum()),
        "components": int(n_comp),
        # <= rather than ==: cross-chain edges are expected once the cutoff
        # reaches non-bonded contact range, and for a base-paired duplex they
        # are the point.  See criterion 1 in the module docstring.
        "connectivity_ok": bool(n_comp <= n_chains and int((counts == 0).sum()) == 0),
        "pairs": int(adj.sum()),
        "kernel_cell_nm": cutoff / 2.0,
        "cell_over_bond": (cutoff / 2.0) / BOND_NM,
    }


def h_information(h: torch.Tensor) -> dict:
    """How much `h` distinguishes particles from one another.

    `across_particle_std` is the spread of each channel across particles, summed
    in quadrature: zero means every particle got the same vector.
    `effective_rank` is the participation ratio of the centred matrix's singular
    values, exp(-sum p log p) with p = s^2 / sum s^2 -- a smooth count of how
    many directions the representation actually uses, which a hard rank
    threshold would obscure.
    """
    a = h.double()
    centred = a - a.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(centred)
    p = (s ** 2) / (s ** 2).sum()
    p = p[p > 0]
    eff_rank = float(torch.exp(-(p * p.log()).sum()))
    return {
        "across_particle_std": float(centred.std(dim=0).norm()),
        "mean_abs": float(a.abs().mean()),
        "effective_rank": eff_rank,
        "dim": int(a.shape[1]),
    }


def chain_count(sample_dir: Path) -> int:
    """Chains from the PDB's TER records.

    MDTraj writes blank chain IDs, so TER is the only marker -- the same
    reasoning as the legacy GAGU adapter.
    """
    sample_id = sample_dir.name
    n_ter = 0
    with (sample_dir / f"{sample_id}.pdb").open() as fh:
        for line in fh:
            if line[:6].strip().upper() == "TER":
                n_ter += 1
    return max(n_ter, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=Path,
                    default=Path("/mnt/xfs/home/mhg/Projects/ForSiyuan/"
                                 "RNA-WorldParticle-Workspace/datasets/processed/"
                                 "gagu_internal_loop_v0_1/gagu_100mM_K_agaguu_startI_r1"))
    ap.add_argument("--cutoffs", type=float, nargs="+",
                    default=[0.054, 0.15, 0.20, 0.26, 0.35, 0.45, 0.55, 0.70, 0.90])
    ap.add_argument("--frames", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args()

    print("=== 1. geometry of the system ===")
    pos, vel, _ref = load_gagu_heavy(args.sample, n_frames=args.frames)
    n_chains = chain_count(args.sample)
    print(f"  chains (from TER): {n_chains}")
    shells = pair_distance_shells(pos[0])
    print(f"  nearest heavy-atom pair: {shells['nearest_pair_nm']:.4f} nm")
    print(f"  median pair distance:    {shells['median_pair_nm']:.4f} nm")
    print(f"  largest pair distance:   {shells['max_pair_nm']:.4f} nm")
    print(f"  bond length used as the resolution yardstick: {BOND_NM} nm "
          f"(plan section 2.5)")

    print("\n=== 2. the official value, for reference ===")
    official_cutoff = 0.5 * RADIUS_SCALE * 6 * 0.012
    print(f"  particle_radius 0.012 -> filter_extent "
          f"{RADIUS_SCALE * 6 * 0.012:.4f} nm -> cutoff {official_cutoff:.4f} nm")
    print(f"  that is {official_cutoff / BOND_NM:.2f} x the bond length, "
          f"so no bonded neighbour is inside it")

    print("\n=== 3. sweep ===")
    print(f"  {'cutoff':>7} {'radius':>8} {'nb_mean':>8} {'nb_max':>7} {'isol':>5} "
          f"{'comp':>5} {'pairs':>7} {'cell/bond':>10} {'h_std':>9} {'h_rank':>7}")

    model = build_model(cutoff_to_particle_radius(args.cutoffs[0]), seed=args.seed)
    rows = []
    for cutoff in args.cutoffs:
        radius = cutoff_to_particle_radius(cutoff)
        model.apply_radius_override(particle_radius=radius)

        # Trust nothing: confirm the override actually reached the buffer the op
        # reads, rather than only the Python attribute.
        want = np.float32(RADIUS_SCALE * 6 * radius)
        got = float(model.local_feature_extractor.filter_extent_tensor)
        assert abs(got - float(want)) < 1e-6, (
            f"apply_radius_override did not update filter_extent_tensor: "
            f"{got} != {want}")

        geo = neighbour_stats(pos[0], cutoff, n_chains)
        out, cap = run_once(model, pos[0], vel[0])
        info = h_information(out["tokens"])
        wp_nb = float(model.local_feature_extractor.conv0_molecular._avg_neighbors)

        row = {**geo, **{f"h_{k}": v for k, v in info.items()},
               "wp_reported_mean_neighbours": wp_nb}
        rows.append(row)
        print(f"  {cutoff:>7.3f} {radius:>8.4f} {geo['mean_neighbours']:>8.1f} "
              f"{geo['max_neighbours']:>7d} {geo['isolated']:>5d} "
              f"{geo['components']:>5d} {geo['pairs']:>7d} "
              f"{geo['cell_over_bond']:>10.2f} "
              f"{info['across_particle_std']:>9.2f} "
              f"{info['effective_rank']:>7.1f}")

    print("\n=== 4. against the criteria ===")
    ok = [r for r in rows if r["connectivity_ok"]]
    print(f"  connectivity (no isolated atoms, components <= {n_chains}): "
          f"{[round(r['cutoff_nm'], 3) for r in ok] or 'none'}")
    resolving = [r for r in rows if r["cell_over_bond"] < 1.0]
    print(f"  kernel cell below one bond length (soft): "
          f"{[round(r['cutoff_nm'], 3) for r in resolving] or 'none'}")
    if rows:
        best_rank = max(rows, key=lambda r: r["h_effective_rank"])
        print(f"  highest h effective rank: cutoff {best_rank['cutoff_nm']:.3f} nm "
              f"(rank {best_rank['h_effective_rank']:.1f} of {best_rank['h_dim']})")
    print("\n  No single number settles this: coverage, kernel resolution and the")
    print("  spread of h pull in different directions, and none of them measures")
    print("  downstream performance.  The chosen value and the reasoning are in")
    print("  the plan, section 2.10; this sweep is the evidence behind it.")

    record = {"sample": str(args.sample), "n_chains": n_chains,
              "bond_length_nm": BOND_NM, "shells": shells,
              "official_cutoff_nm": official_cutoff, "sweep": rows}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(record, indent=2) + "\n")
        print(f"\nwrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
