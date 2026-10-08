"""Metric 1: single-step prediction RMSD on P009's own held-out windows.

Give the model 8 history frames and a dt, let it sample the 9th frame, and
measure how far the sample is from the frame MD actually produced.  This is the
only number in the project that is on the same footing as the persistence,
static and cross-replica baselines, because it is the same task: predict the
target from its past, with no part of the answer in the input (P004 section
2.18.1).

**The window set is P009's, not a new one.**  Same 16 held-out trajectories in
the same order, `k=8`, `length=256`, `seed=1234`, `canonicalize=True`, and dt in
[0.1, 1] ns -- so `window_id` here and `window_id` in
`runs/p009/<arm>/heldout/step_<N>.rank<r>.jsonl` are the same window, and the
denoising loss and the prediction error can be read side by side per window.
`dt_max_ns` is passed explicitly even though the module default is already 1.0:
the default is what P009's first change moved, and a benchmark that silently
inherits it would be scoring a different problem if it ever moved again.

**Two stages, because the baselines are not the model's.**  `--stage baselines`
needs no GPU and no checkpoint: persistence is a frame of the held-out
trajectory, the static baseline is its mean structure, and the cross-replica
baseline is r1-r3 at the same time index.  All three are identical for all four
arms, so they are computed once on CPU and the GPU stage only samples.  Running
`--stage both` does both in one process, which is what the dry run does.

Output, one line per window per arm, in `runs/p007/oneshot/<tag>/`:

    windows.jsonl            the window set itself: id, sample, target, stride, dt
    baselines.jsonl          persistence / static / replica RMSD and lDDT
    oneshot.<arm>.jsonl      the model's samples
    meta.json                checkpoint, step, commits, N_step, N_sample, env

    PYTHONPATH=repos/research/kineidos-v3:repos/research/wp-v2 \\
    LAYERNORM_TYPE=torch ATTN_IMPL=sdpa LD_LIBRARY_PATH=$ENV/lib:$LD_LIBRARY_PATH \\
      $ENV/bin/python -m kineidos.bench.oneshot --arm random \\
        --checkpoint runs/p004/p004_random_20261007_000106/checkpoints/3999.pt \\
        --out runs/p007/oneshot/dryrun --n-windows 32 --n-samples 2 --n-step 20
"""

from __future__ import annotations

import argparse
import json
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np

from kineidos.bench import metrics as M

GAGU_ROOT = Path(
    "/mnt/xfs/home/mhg/Projects/ForSiyuan/RNA-WorldParticle-Workspace/datasets/"
    "processed/gagu_internal_loop_v0_1")
HELD_OUT_GLOB = "*_r4"
EVAL_SEED = 1234
EVAL_WINDOWS = 256
WINDOW_K = 8
DT_MIN_NS = 0.1
DT_MAX_NS = 1.0


def held_out_samples(root: Path = GAGU_ROOT) -> list[str]:
    """The 16 held-out trajectory names, in the order P009's runs saw them.

    `kineidos/slurm/p004_train.sbatch` builds the list with
    `ls -d $GAGU_ROOT/*_r4 | xargs -n1 basename | paste -sd,`, so the order is
    the shell's sort.  The order matters: `GAGUWindowDataset.__getitem__` picks
    a sample with `rng.integers(len(samples))`, so a permuted list gives
    different windows under the same seed and the jsonl files stop joining.
    """
    names = sorted(p.name for p in root.glob(HELD_OUT_GLOB) if p.is_dir())
    if len(names) != 16:
        raise SystemExit(f"{len(names)} held-out trajectories under {root}, expected 16")
    return names


def load_window_set(root: Path = GAGU_ROOT, *, n_windows: int = EVAL_WINDOWS,
                    seed: int = EVAL_SEED, k: int = WINDOW_K):
    """(samples, dataset) for the fixed held-out window set.

    Windows are drawn one at a time from the returned dataset rather than
    materialised in a list: `__getitem__` is seeded by its index, so window i is
    window i however it is reached, and 256 windows each holding a clone of the
    feature dict is memory spent for nothing.
    """
    from kineidos.data.gagu import GAGUProtenixAdapter
    from kineidos.data.windows import GAGUWindowDataset

    names = held_out_samples(root)
    samples = [GAGUProtenixAdapter(root / n).load() for n in names]
    dataset = GAGUWindowDataset(
        samples, k=k, length=int(n_windows), seed=int(seed), canonicalize=True,
        dt_min_ns=DT_MIN_NS, dt_max_ns=DT_MAX_NS,
    )
    return samples, dataset


def topologies_for(samples) -> dict[str, M.Topology]:
    """One `Topology` per held-out trajectory, from its own frame 0.

    Per trajectory and not once for all of them: the bond table is built from
    coordinates, and although all four contexts happen to have 470 heavy atoms
    and 526 bonds, they are not the same 526 bonds.
    """
    from kineidos.data.gagu import _sample_paths

    out = {}
    for sample in samples:
        _, _, pdb = _sample_paths(Path(sample.sample_dir))
        frame0 = np.asarray(sample.position_angstrom[0], dtype=np.float64)
        topo = M.build_topology(pdb, frame0)
        if topo.n_atoms != sample.n_atoms:
            raise AssertionError(
                f"{sample.sample_id}: the topology has {topo.n_atoms} heavy "
                f"atoms and the loaded sample {sample.n_atoms}; the hydrogen "
                f"filter differs between kineidos/data/gagu.py and "
                f"bench/metrics.py and every per-atom index is then wrong"
            )
        out[sample.sample_id] = topo
    return out


def window_rows(dataset) -> Iterator[tuple[int, Any]]:
    for i in range(len(dataset)):
        yield i, dataset[i]


# ------------------------------------------------------------------ baselines


def replica_frames(root: Path, sample_id: str, frames: np.ndarray,
                   heavy: np.ndarray) -> dict[str, np.ndarray]:
    """The r1-r3 counterparts of `sample_id` at the given frame indices.

    One npz per replica is read and thrown away again: only the requested
    frames are kept, so the peak cost is one trajectory's position array
    (56 MB) rather than three times sixteen of them.
    """
    out = {}
    for rep in (1, 2, 3):
        other = sample_id[:-1] + str(rep)
        path = root / other / f"{other}.npz"
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as z:
            pos = np.asarray(z["position"][:, heavy], dtype=np.float64) * 10.0
        out[other] = pos[frames].copy()
        del pos
    return out


def compute_baselines(samples, dataset, topos, out_dir: Path) -> None:
    """persistence / static / cross-replica RMSD for every window.

    Model-free and therefore arm-free: written once and read by every arm's
    report.  P007 section 4 gives the numbers these should land on -- 0.98,
    1.37 and 2.06 A as cross-system medians -- so a gross disagreement here is
    a bug in the window set, not a finding.
    """
    by_id = {s.sample_id: s for s in samples}
    static: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []

    # Group the windows by trajectory so each trajectory's replicas are read once.
    per_sample: dict[str, list[tuple[int, Any]]] = {}
    for i, window in window_rows(dataset):
        per_sample.setdefault(window.sample_id, []).append((i, window))

    for sample_id, items in sorted(per_sample.items()):
        sample = by_id[sample_id]
        topo = topos[sample_id]
        if sample_id not in static:
            static[sample_id] = M.mean_structure(sample.position_angstrom)
        frames = np.array([w.target_frame for _, w in items], dtype=np.int64)
        others = replica_frames(GAGU_ROOT, sample_id, frames,
                               np.asarray(sample.heavy_atom_indices))
        for pos_in_group, (window_id, window) in enumerate(items):
            target = np.asarray(sample.position_angstrom[window.target_frame],
                                dtype=np.float64)
            last_history = int(window.history_frames[-1])
            persistence = M.rmsd_chain_aware(
                np.asarray(sample.position_angstrom[last_history], dtype=np.float64),
                target, topo)
            stat = M.rmsd_chain_aware(static[sample_id], target, topo)
            rep = {name: M.rmsd_chain_aware(arr[pos_in_group], target, topo)
                   for name, arr in others.items()}
            rows.append({
                "window_id": window_id,
                "sample_id": sample_id,
                "target_frame": int(window.target_frame),
                "stride": int(window.stride),
                "delta_t_ns": float(window.delta_t_ns),
                "last_history_frame": last_history,
                "persistence_rmsd": persistence.rmsd,
                "persistence_swapped": bool(persistence.swapped),
                "static_rmsd": stat.rmsd,
                "replica_rmsd": {k: v.rmsd for k, v in rep.items()},
                "replica_rmsd_mean": float(np.mean([v.rmsd for v in rep.values()]))
                                     if rep else None,
            })
    rows.sort(key=lambda r: r["window_id"])
    with open(out_dir / "baselines.jsonl", "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"[baselines] {len(rows)} windows -> {out_dir / 'baselines.jsonl'}")
    persistence = np.array([r["persistence_rmsd"] for r in rows])
    print(f"[baselines] persistence median {np.median(persistence):.3f} A, "
          f"static {np.median([r['static_rmsd'] for r in rows]):.3f} A, "
          f"replica {np.median([r['replica_rmsd_mean'] for r in rows if r['replica_rmsd_mean']]):.3f} A")


# ---------------------------------------------------------------- model stage


def run_model(samples, dataset, topos, out_dir: Path, *, arm: str,
              checkpoint: Path, n_samples: int, n_step: int,
              device: str) -> dict[str, Any]:
    from kineidos.bench import model as BM
    from kineidos.train.batch import collate_window

    loaded = BM.load_arm(checkpoint, arm=arm, device=device,
                         n_step=n_step, n_sample=n_samples)
    by_id = {s.sample_id: s for s in samples}
    path = out_dir / f"oneshot.{arm}.jsonl"
    started = time.time()
    thresholds = load_thresholds()
    n_written = 0
    with open(path, "w") as handle:
        for window_id, window in window_rows(dataset):
            sample = by_id[window.sample_id]
            topo = topos[window.sample_id]
            batch = collate_window(window)
            seed = BM.sampling_seed(window_id, base=EVAL_SEED)
            coords = BM.sample_frames(loaded, window, batch, seed=seed)
            coords = np.asarray(coords, dtype=np.float64)
            if coords.ndim == 4:        # [batch, S, N, 3] -> [S, N, 3]
                coords = coords[0]
            target = np.asarray(sample.position_angstrom[window.target_frame],
                                dtype=np.float64)

            rmsd, swapped = M.rmsd_chain_aware_batch(coords, target, topo)
            lddt = BM.lddt_complex(coords, target, batch["input_feature_dict"],
                                   device=loaded.device)
            obs = M.bond_observables(coords, topo)
            system = system_key(window.sample_id)
            valid = M.valid_frames(
                obs,
                o3p_maxdev_max=thresholds["validity"]["o3p_maxdev_max_angstrom"],
                clash_vdw_count_max=thresholds["validity"]["clash_vdw_count_max"])
            # Ensemble width: the mean pairwise RMSD *between* samples, which is
            # the spread of the conditional distribution the model is sampling.
            # The std of the five RMSD-to-truth values is a different thing (how
            # unevenly good the samples are) and both are kept.
            if len(coords) > 1:
                pairs = [(a, b) for a in range(len(coords))
                         for b in range(a + 1, len(coords))]
                width = float(np.mean(M.kabsch_rmsd_pairs(
                    coords[[a for a, _ in pairs]], coords[[b for _, b in pairs]])))
            else:
                width = float("nan")
            row = {
                "window_id": window_id,
                "arm": arm,
                "sample_id": window.sample_id,
                "system": system,
                "target_frame": int(window.target_frame),
                "stride": int(window.stride),
                "delta_t_ns": float(window.delta_t_ns),
                "seed": seed,
                "rmsd": [float(x) for x in rmsd],
                "rmsd_mean": float(np.mean(rmsd)),
                "rmsd_best": float(np.min(rmsd)),
                "rmsd_std": float(np.std(rmsd)),
                "ensemble_width": width,
                "chain_swapped": [bool(x) for x in swapped],
                "swap_rate": float(np.mean(swapped)),
                "lddt": [float(x) for x in np.atleast_1d(lddt)],
                "lddt_mean": float(np.mean(lddt)),
                "lddt_best": float(np.max(lddt)),
                "bond_mae": [float(x) for x in obs["bond_mae"]],
                "o3p_maxdev": [float(x) for x in obs["o3p_maxdev"]],
                "clash_vdw_count": [int(x) for x in obs["clash_vdw_count"]],
                "valid": [bool(x) for x in valid],
                "valid_fraction": float(np.mean(valid)),
                "anchor_rmsd_to_ref_nm": float(window.anchor_rmsd_to_ref_nm),
                "inter_frame_rotation_deg": float(window.inter_frame_rotation_deg),
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            n_written += 1
            if n_written <= 3 or n_written % 25 == 0:
                print(f"[{arm}] window {window_id:4d} dt={window.delta_t_ns:.1f} ns "
                      f"rmsd {row['rmsd_mean']:.3f} (best {row['rmsd_best']:.3f}) "
                      f"lddt {row['lddt_mean']:.4f} valid {row['valid_fraction']:.2f} "
                      f"swap {row['swap_rate']:.2f} "
                      f"[{(time.time() - started) / n_written:.1f} s/window]")
    print(f"[{arm}] {n_written} windows -> {path} "
          f"in {time.time() - started:.0f} s")
    return loaded.describe()


# ---------------------------------------------------------------------- misc


def system_key(sample_id: str) -> str:
    """`gagu_100mM_K_agaguu_startI_r4` -> `100mM_agaguu_I`, the key used by
    `thresholds.json`."""
    parts = sample_id.split("_")
    return f"{parts[1]}_{parts[3]}_{parts[4][len('start'):]}"


@lru_cache(maxsize=4)
def _thresholds_cached(path: Optional[str]) -> str:
    return json.dumps(_load_thresholds_uncached(Path(path) if path else None))


def load_thresholds(path: Optional[Path] = None) -> dict[str, Any]:
    """Cached wrapper, so a caller may ask repeatedly without re-reading."""
    return json.loads(_thresholds_cached(str(path) if path else None))


def _load_thresholds_uncached(path: Optional[Path] = None) -> dict[str, Any]:
    """The frozen thresholds.  Fatal if absent: scoring validity without them
    would mean inventing a threshold at the call site, which is the thing P007
    section 4 exists to prevent."""
    from kineidos.env_lock import find_workspace_root

    # Found from the workspace root, which AGENTS.md requires every run to start
    # from -- not from __file__, which resolves inside the worktree and would
    # reach a different artifacts/ (or none) depending on which worktree the
    # import came from.
    path = Path(path) if path else \
        find_workspace_root() / "artifacts/reports/P007/thresholds.json"
    if not path.exists():
        raise SystemExit(
            f"no thresholds at {path}. Run P007.0's "
            f"artifacts/reports/P007/calibrate/freeze_thresholds.py first; "
            f"the validity threshold has to predate any model's score."
        )
    return json.loads(path.read_text())


def write_env_lock(out_dir: Path) -> None:
    """One env.lock per run directory, as AGENTS.md requires.

    The directory name does not fix its contents: what is installed, which
    commit each of the two trees was on, and which environment variables were in
    force all have to be on disk, or a log months later points at a state that
    no longer exists.  `meta.<arm>.json` carries the commits too; this is the
    file the workspace convention names, in the format every other run uses.

    Written per run and not per arm: several arms share one output directory and
    they run in the same environment.  A second arm overwrites the first's copy
    with an equivalent one.
    """
    from kineidos.env_lock import collect, find_workspace_root

    try:
        record = collect(find_workspace_root())
    except SystemExit as exc:
        print(f"[env.lock] skipped: {exc}")
        return
    (out_dir / "env.lock").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n")
    for label, tree in record.get("worktrees", {}).items():
        flag = " [DIRTY]" if tree.get("dirty") else ""
        print(f"[env.lock] {label:12s} {str(tree.get('commit'))[:12]} "
              f"({tree.get('branch')}){flag}")


def worktree_commits() -> dict[str, Any]:
    from kineidos.env_lock import collect, find_workspace_root

    try:
        return collect(find_workspace_root()).get("worktrees", {})
    except Exception as exc:                                  # pragma: no cover
        return {"error": repr(exc)}


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", choices=("none", "zero", "random", "pretrained"))
    ap.add_argument("--checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", choices=("baselines", "model", "both"),
                    default="both")
    ap.add_argument("--n-windows", type=int, default=EVAL_WINDOWS)
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--n-step", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gagu-root", default=str(GAGU_ROOT))
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_env_lock(out_dir)
    root = Path(args.gagu_root)

    print(f"[setup] window set: {args.n_windows} windows, seed {EVAL_SEED}, "
          f"k={WINDOW_K}, dt in [{DT_MIN_NS}, {DT_MAX_NS}] ns")
    samples, dataset = load_window_set(root, n_windows=args.n_windows)
    topos = topologies_for(samples)

    # Written through a temporary file and renamed into place.  Several arms run
    # against the same output directory at once and all of them write the same
    # window set; three processes writing one path with "w" can interleave at
    # different offsets and leave a file that is neither of them.  os.replace is
    # one rename syscall, so the published file is always one process's whole
    # output.
    tmp = out_dir / f"windows.jsonl.{os.getpid()}"
    with open(tmp, "w") as handle:
        for window_id, window in window_rows(dataset):
            handle.write(json.dumps({
                "window_id": window_id, "sample_id": window.sample_id,
                "system": system_key(window.sample_id),
                "target_frame": int(window.target_frame),
                "history_frames": [int(f) for f in window.history_frames],
                "stride": int(window.stride),
                "delta_t_ns": float(window.delta_t_ns),
                "n_valid_history": int(window.wp_frame_mask.sum()),
            }, sort_keys=True) + "\n")
    os.replace(tmp, out_dir / "windows.jsonl")

    meta: dict[str, Any] = {
        "window_set": {"n_windows": args.n_windows, "eval_seed": EVAL_SEED,
                       "k": WINDOW_K, "dt_min_ns": DT_MIN_NS,
                       "dt_max_ns": DT_MAX_NS,
                       "held_out_samples": held_out_samples(root)},
        "worktrees": worktree_commits(),
        "thresholds": {"o3p_maxdev_max_angstrom":
                       load_thresholds()["validity"]["o3p_maxdev_max_angstrom"],
                       "clash_vdw_count_max":
                       load_thresholds()["validity"]["clash_vdw_count_max"],
                       "frozen_on": load_thresholds()["frozen_on"]},
        "env": {k: os.environ.get(k, "") for k in
                ("PYTHONPATH", "LAYERNORM_TYPE", "ATTN_IMPL",
                 "PYTORCH_CUDA_ALLOC_CONF", "SLURM_JOB_ID")},
    }

    if args.stage in ("baselines", "both"):
        compute_baselines(samples, dataset, topos, out_dir)

    if args.stage in ("model", "both"):
        if not (args.arm and args.checkpoint):
            raise SystemExit("--stage model needs --arm and --checkpoint")
        meta["model"] = run_model(
            samples, dataset, topos, out_dir, arm=args.arm,
            checkpoint=Path(args.checkpoint), n_samples=args.n_samples,
            n_step=args.n_step, device=args.device)

    name = f"meta.{args.arm}.json" if args.stage != "baselines" else "meta.json"
    (out_dir / name).write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    print(f"[done] {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
