"""Autoregressive rollout: 8 history frames in, 100 steps of 0.1 ns out.

The protocol is P007 section 3.2 and nothing here chooses any part of it:

    start      the first 8 frames of each held-out r4 trajectory (0.8 ns),
               both starting conformations
    dt         0.1 ns = stride 1.  P009 section 2.2 item 3: the inference
               operating point is ~0.25 tau_half, which on GAGU is 0.1 ns
               (rho = 0.69).  1 ns is rho = 0.30 and 2.2% of the training
               windows -- evaluating there would test the model where it has
               seen least
    length     100 steps = 10 ns.  The RMSF ceiling of a 10 ns window is 0.90
               (median over 16 systems); 30 ns buys 0.92 for three times the
               cost, so 10 ns is the operating point and 30 ns a later subset
    replicates 2 per start, differing only in the sampling seed

Each step: build the window (canonicalised, with the inter-frame rotation
guard), compute `h`, sample one frame, **Kabsch-align it to the previous frame**
with both chain labellings tried, take the backward-difference velocity, push it
into the history and drop the oldest frame.

The alignment is not optional.  Sampling starts from pure noise and there is no
orientation to read out of the conditioning, so the output pose is arbitrary
(P004 section 2.19, fourth qualifier); an unaligned frame pushed into the window
makes `build_window`'s guard fire on the first step, which is the guard working
rather than failing.  The relabelling is applied to the frame that is kept, so
atom identity stays consistent along the rollout -- without it, an arm with no
history (`none`, `zero`) would swap strands between consecutive steps and the
RMSF profile would be the average of two labellings.

**The velocity channel changes definition at the seam, and that is recorded
rather than fixed.**  The seed window carries MD's own velocities, which
`metadata.json` declares and measurement confirms to be central differences of
the saved coordinates, (x[t+1] - x[t-1]) / 2dt.  A rollout has no future frame,
so a generated frame's velocity is the backward difference (x[t] - x[t-1]) / dt.
Measured on `gagu_100mM_K_agaguu_startI_r4` over 2000 frames: the backward
difference has **1.743x** the RMS magnitude of the stored central difference and
correlates with it at **r = 0.574**.  Both follow exactly from
central = (back(t) + back(t+1)) / 2 with consecutive backward differences
anticorrelated at rho = -0.342: the ratio is 1/sqrt((1+rho)/2) and the
correlation is sqrt((1+rho)/2).  So the velocity the model is fed after the
first eight steps is a noisier, systematically larger estimate than the one it
trained on -- which is the concrete form of P009 section 2.5's remark that at a
100 ps saving interval the velocity channel is nearly empty.

Two stages:

    --stage generate   GPU.  Samples, writes one npz per rollout plus a
                       per-step jsonl of validity and geometry diagnostics.
    --stage score      CPU.  Reads the npz and produces metrics 2-5 and the
                       sanity flag against the same-start MD window.

Splitting them means a preempted generate costs no scoring, a changed metric
costs no sampling, and the scoring runs where mdtraj's SASA is cheap.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from kineidos.bench import metrics as M
from kineidos.bench.oneshot import (GAGU_ROOT, held_out_samples, load_thresholds,
                                    system_key, topologies_for, worktree_commits,
                                    write_env_lock)

K = 8
STRIDE = 1
N_STEPS = 100
LAGS = tuple(range(1, 31))
"""Lags 1-30 frames = 0.1-3 ns, the range P007 section 3.2 asks for on a
100-frame rollout."""
MIN_FRAMES_FOR_LAG = 12
"""Below this many frames a lag curve says nothing; above it, whichever lags
fit are used and reported."""
SANITY_WINDOW = 10
SANITY_THRESHOLD = 0.8


class RolloutBuffer:
    """A GAGUSample-shaped view of a rolling K+1 frame history.

    `build_window` is reused exactly as the dataloader uses it -- the
    canonicalisation, the padding rule and the inter-frame rotation guard are
    the same code on the same path -- so this holds the attributes that
    function reads, and nothing else.  Written out rather than delegated with
    `__getattr__`: `position_angstrom` is a property computed from
    `position_nm`, so a delegating shim would quietly hand back the *parent's*
    coordinates for the label while giving WorldParticle the buffer's.
    """

    def __init__(self, base: Any, k: int = K):
        self.base = base
        self.sample_id = base.sample_id
        self.sample_dir = base.sample_dir
        self.frame_interval_ns = float(base.frame_interval_ns)
        self.base_features = base.base_features
        self.coordinate_mask = base.coordinate_mask
        self.atom_array = base.atom_array
        self.token_array = base.token_array
        self._ref_pos_nm = np.asarray(base.ref_pos_nm(), dtype=np.float64)
        n = base.n_atoms
        # K history frames plus one slot for the target.  build_window reads
        # position_angstrom[target_frame] for the label; a rollout has no label,
        # so the slot holds a copy of the newest history frame and the label is
        # never read (sampling takes only input_feature_dict).
        self.position_nm = np.zeros((k + 1, n, 3), dtype=np.float64)
        self.velocity_nm_per_ps = np.zeros((k + 1, n, 3), dtype=np.float64)
        self.k = int(k)

    # -- the parts of GAGUSample's interface build_window uses
    @property
    def position_angstrom(self) -> np.ndarray:
        return self.position_nm * 10.0

    @property
    def velocity_angstrom_per_ps(self) -> np.ndarray:
        return self.velocity_nm_per_ps * 10.0

    @property
    def n_frames(self) -> int:
        return int(self.position_nm.shape[0])

    @property
    def n_atoms(self) -> int:
        return int(self.position_nm.shape[1])

    def ref_pos_nm(self) -> np.ndarray:
        return self._ref_pos_nm

    # -- the buffer itself
    def seed_from(self, positions_nm: np.ndarray, velocities_nm: np.ndarray) -> None:
        if positions_nm.shape[0] != self.k:
            raise ValueError(f"need {self.k} seed frames, got {positions_nm.shape[0]}")
        self.position_nm[:self.k] = positions_nm
        self.velocity_nm_per_ps[:self.k] = velocities_nm
        self.position_nm[self.k] = positions_nm[-1]
        self.velocity_nm_per_ps[self.k] = velocities_nm[-1]

    def push(self, frame_nm: np.ndarray) -> None:
        """Append one frame, dropping the oldest.

        Its velocity is the backward difference against the frame it follows --
        the only estimate available without a future frame.  See the module
        docstring for how far that is from the central difference training saw.
        """
        dt_ps = self.frame_interval_ns * 1000.0
        previous = self.position_nm[self.k - 1]
        velocity = (frame_nm - previous) / dt_ps
        # Shifting a slice onto an overlapping one: numpy detects the overlap
        # and buffers (verified on 2.4.6 as well as documented since 1.13), so
        # this is a correct shift and not a smear.  Do not "optimise" it into an
        # explicit loop that writes forwards.
        self.position_nm[:self.k - 1] = self.position_nm[1:self.k]
        self.velocity_nm_per_ps[:self.k - 1] = self.velocity_nm_per_ps[1:self.k]
        self.position_nm[self.k - 1] = frame_nm
        self.velocity_nm_per_ps[self.k - 1] = velocity
        self.position_nm[self.k] = frame_nm
        self.velocity_nm_per_ps[self.k] = velocity


def rollout_seed(sample_id: str, replicate: int, *, base: int = 1234) -> int:
    """The sampling seed for one rollout -- shared by all four arms.

    A function of the trajectory and the replicate index only, so arm A's
    replicate 0 of a trajectory sees the same noise as arm B's (P007 section
    3.0, item 3).
    """
    h = 0
    for ch in sample_id:
        h = (h * 131 + ord(ch)) % (2 ** 31 - 1)
    return int((base * 1_000_003 + h * 97 + int(replicate) * 7_919) % (2 ** 31 - 1))


def generate(samples, topos, out_dir: Path, *, arm: str, checkpoint: Path,
             n_steps: int, n_step_sampler: int, device: str,
             replicates: int, only: Optional[list[str]] = None) -> dict[str, Any]:
    from kineidos.bench import model as BM
    from kineidos.data.windows import build_window
    from kineidos.train.batch import collate_window

    loaded = BM.load_arm(checkpoint, arm=arm, device=device,
                         n_step=n_step_sampler, n_sample=1)
    thresholds = load_thresholds()["validity"]
    by_id = {s.sample_id: s for s in samples}
    chosen = [s for s in samples if not only or s.sample_id in only]
    print(f"[rollout] {len(chosen)} starts x {replicates} replicates x "
          f"{n_steps} steps, N_step={n_step_sampler}")

    index: list[dict[str, Any]] = []
    for sample in chosen:
        topo = topos[sample.sample_id]
        seed_pos = np.asarray(sample.position_nm[:K], dtype=np.float64)
        seed_vel = np.asarray(sample.velocity_nm_per_ps[:K], dtype=np.float64)
        for replicate in range(replicates):
            seed = rollout_seed(sample.sample_id, replicate)
            buffer = RolloutBuffer(sample, k=K)
            buffer.seed_from(seed_pos, seed_vel)
            frames: list[np.ndarray] = []
            rows: list[dict[str, Any]] = []
            started = time.time()
            stopped_at = None
            stop_reason = None
            for step in range(1, n_steps + 1):
                try:
                    window = build_window(buffer, target_frame=K, stride=STRIDE,
                                          k=K, canonicalize=True)
                except ValueError as exc:
                    stopped_at, stop_reason = step, str(exc)
                    print(f"  [stop] {sample.sample_id} r{replicate} step {step}: "
                          f"{exc}")
                    break
                batch = collate_window(window)
                coords = np.asarray(
                    BM.sample_frames(loaded, window, batch,
                                     seed=seed + step, n_sample=1),
                    dtype=np.float64)
                while coords.ndim > 3:
                    coords = coords[0]
                raw = coords[0]
                previous = buffer.position_angstrom[buffer.k - 1]
                placed, swapped = M.align_frame_to(raw, previous, topo)
                obs = M.bond_observables(placed[None], topo)
                valid = bool(M.valid_frames(
                    obs,
                    o3p_maxdev_max=thresholds["o3p_maxdev_max_angstrom"],
                    clash_vdw_count_max=thresholds["clash_vdw_count_max"])[0])
                rows.append({
                    "step": step,
                    "arm": arm,
                    "sample_id": sample.sample_id,
                    "replicate": replicate,
                    "chain_swapped": bool(swapped),
                    "valid": valid,
                    "bond_mae": float(obs["bond_mae"][0]),
                    "o3p_maxdev": float(obs["o3p_maxdev"][0]),
                    "clash_vdw_count": int(obs["clash_vdw_count"][0]),
                    "rmsd_to_previous": M.kabsch_rmsd(placed, previous),
                    "inter_frame_rotation_deg": float(window.inter_frame_rotation_deg),
                    "anchor_rmsd_to_ref_nm": float(window.anchor_rmsd_to_ref_nm),
                })
                frames.append(placed)
                buffer.push(placed / 10.0)
            if not frames:
                raise SystemExit(
                    f"{sample.sample_id} r{replicate} produced no frames: "
                    f"{stop_reason}. If the inter-frame rotation guard fires on "
                    f"step 1 even after alignment, the alignment or the chain "
                    f"relabelling is wrong -- P007's kickoff says stop and ask."
                )
            stem = f"rollout.{arm}.{sample.sample_id}.rep{replicate}"
            np.savez_compressed(
                out_dir / f"{stem}.npz",
                coords_angstrom=np.stack(frames).astype(np.float32),
                seed=np.array(seed), start_frames=np.arange(K),
                md_first_target_frame=np.array(K),
                n_steps=np.array(len(frames)),
            )
            with open(out_dir / f"{stem}.steps.jsonl", "w") as handle:
                for row in rows:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
            entry = {
                "arm": arm, "sample_id": sample.sample_id,
                "system": system_key(sample.sample_id), "replicate": replicate,
                "seed": seed, "n_frames": len(frames),
                "stopped_at": stopped_at, "stop_reason": stop_reason,
                "npz": f"{stem}.npz", "steps": f"{stem}.steps.jsonl",
                "valid_fraction": float(np.mean([r["valid"] for r in rows])),
                "swap_rate": float(np.mean([r["chain_swapped"] for r in rows])),
                "seconds": time.time() - started,
            }
            index.append(entry)
            print(f"  {sample.sample_id} r{replicate}: {len(frames)} frames, "
                  f"valid {entry['valid_fraction']:.2f}, swap "
                  f"{entry['swap_rate']:.2f}, {entry['seconds']:.0f} s")
    (out_dir / f"index.{arm}.json").write_text(
        json.dumps(index, indent=1, sort_keys=True) + "\n")
    return {"model": loaded.describe(), "rollouts": index}


# ------------------------------------------------------------------- scoring


def score_one(coords: np.ndarray, md_window: np.ndarray, topo: M.Topology,
              *, thresholds: dict[str, Any], started_from: str,
              with_sasa: bool = True) -> dict[str, Any]:
    """Metrics 2-5 and the sanity flag for one generated trajectory.

    `md_window` is the same-start, same-length stretch of the trajectory the
    rollout began from: metric 2 is the difference between two lag curves and
    metric 4 is a correlation against that window's RMSF, so the reference has
    to be the window and not the full microsecond (the full trajectory's RMSF is
    a different, easier target -- its ceiling is 0.92 against the window's 0.90).

    Metrics 2 and 4 are computed twice: on the valid frames only, which is what
    P007 section 3.3 asks for, and on everything, so that a filtered number can
    never be mistaken for the whole picture.
    """
    obs = M.bond_observables(coords, topo)
    valid = M.valid_frames(
        obs, o3p_maxdev_max=thresholds["o3p_maxdev_max_angstrom"],
        clash_vdw_count_max=thresholds["clash_vdw_count_max"])

    out: dict[str, Any] = {
        "n_frames": int(len(coords)),
        "valid_fraction": float(valid.mean()),
        "first_invalid_step": (int(np.flatnonzero(~valid)[0]) + 1
                               if (~valid).any() else None),
        "bond_mae_all": float(obs["bond_mae"].mean()),
        "bond_mae_valid": (float(obs["bond_mae"][valid].mean())
                           if valid.any() else None),
        "o3p_maxdev_all": float(obs["o3p_maxdev"].mean()),
        "o3p_maxdev_max": float(obs["o3p_maxdev"].max()),
        "clash_vdw_count_mean": float(obs["clash_vdw_count"].mean()),
        "clash_vdw_count_max": int(obs["clash_vdw_count"].max()),
    }

    md_lag = M.rmsd_vs_lag(md_window, LAGS)
    out["md_lag"] = {"lags": md_lag["lags"].tolist(),
                     "mean": md_lag["mean"].tolist()}
    for label, subset in (("all", np.ones(len(coords), bool)), ("valid", valid)):
        # Enough frames for a curve at all, not enough for every lag: the
        # lags that do not fit are dropped by rmsd_vs_lag and the ones used are
        # recorded, so a short rollout (the dry run's 20 steps) still exercises
        # this and still says what it covered.
        if subset.sum() < MIN_FRAMES_FOR_LAG:
            out[f"lag_{label}"] = None
            continue
        gen = M.rmsd_vs_lag(coords[subset], LAGS)
        # Compare lag for lag.  Filtering removes frames, so a filtered
        # trajectory's "lag 1" is not always 0.1 ns apart -- recorded, because a
        # large deviation after heavy filtering may be the filtering.
        common = [i for i, l in enumerate(gen["lags"]) if l in set(md_lag["lags"])]
        gi = np.array(common, dtype=int)
        mi = np.array([list(md_lag["lags"]).index(gen["lags"][i]) for i in common],
                      dtype=int)
        deviation = gen["mean"][gi] - md_lag["mean"][mi]
        out[f"lag_{label}"] = {
            "lags": gen["lags"][gi].tolist(),
            "mean": gen["mean"][gi].tolist(),
            "deviation_mean": float(np.nanmean(deviation)),
            "deviation_abs_mean": float(np.nanmean(np.abs(deviation))),
            "plateau": float(np.nanmean(gen["mean"][gi][-5:])),
            "md_plateau": float(np.nanmean(md_lag["mean"][mi][-5:])),
            "n_frames_used": int(subset.sum()),
        }

    md_rmsf = M.rmsf(M.align_to(md_window, md_window[0]))
    for label, subset in (("all", np.ones(len(coords), bool)), ("valid", valid)):
        if subset.sum() < 10:
            out[f"rmsf_{label}"] = None
            continue
        aligned = M.align_to(coords[subset], coords[subset][0])
        gen_rmsf = M.rmsf(aligned)
        out[f"rmsf_{label}"] = {
            "r_heavy": M.rmsf_pearson(gen_rmsf, md_rmsf),
            "r_c1p": M.rmsf_pearson(gen_rmsf[topo.c1p_idx], md_rmsf[topo.c1p_idx]),
            "amplitude_ratio": float(gen_rmsf.mean() / md_rmsf.mean()),
            "mean_rmsf_model": float(gen_rmsf.mean()),
            "mean_rmsf_md": float(md_rmsf.mean()),
            "n_frames_used": int(subset.sum()),
        }

    loop = M.loop_observables(coords, topo, sasa_stride=1, with_sasa=with_sasa)
    paired = loop["paired"].astype(float)
    sanity: dict[str, Any] = {
        "applies": started_from == "I",
        "g4g17_paired_fraction": float(paired[:, 0].mean()),
        "g6g15_paired_fraction": float(paired[:, 1].mean()),
        "syn_fraction_g4_g6_g15_g17": M.syn_fraction(loop["chi"]).tolist(),
        "first_sustained_drop": {},
    }
    series = {"g4g17_paired": paired[:, 0], "g6g15_paired": paired[:, 1]}
    if with_sasa:
        flipped = loop["flipped"].astype(float)
        sanity["u7_flipped_fraction"] = float(flipped[:, 0].mean())
        sanity["u18_flipped_fraction"] = float(flipped[:, 1].mean())
        sanity["control_c10_c21_sasa_mean"] = loop["sasa"][:, 2:].mean(0).tolist()
        series["u7_flipped"] = flipped[:, 0]
        series["u18_flipped"] = flipped[:, 1]
    for name, values in series.items():
        hit = M.first_sustained_drop(values, threshold=SANITY_THRESHOLD,
                                     window=SANITY_WINDOW)
        sanity["first_sustained_drop"][name] = None if hit is None else hit + 1
    hits = [v for v in sanity["first_sustained_drop"].values() if v is not None]
    sanity["hallucinated_transition_step"] = min(hits) if hits else None
    out["sanity"] = sanity
    return out


def score(samples, topos, out_dir: Path, *, arm: str) -> None:
    index = json.loads((out_dir / f"index.{arm}.json").read_text())
    by_id = {s.sample_id: s for s in samples}
    thresholds = load_thresholds()["validity"]
    rows = []
    for entry in index:
        sample = by_id[entry["sample_id"]]
        topo = topos[entry["sample_id"]]
        with np.load(out_dir / entry["npz"]) as z:
            coords = np.asarray(z["coords_angstrom"], dtype=np.float64)
            first = int(z["md_first_target_frame"])
        md_window = np.asarray(
            sample.position_angstrom[first:first + len(coords)], dtype=np.float64)
        started_from = system_key(sample.sample_id).split("_")[-1]
        row = {**{k: entry[k] for k in ("arm", "sample_id", "system",
                                        "replicate", "seed", "stopped_at")},
               **score_one(coords, md_window, topo, thresholds=thresholds,
                           started_from=started_from)}
        rows.append(row)
        lag = row.get("lag_valid") or row.get("lag_all") or {}
        rmsf = row.get("rmsf_valid") or row.get("rmsf_all") or {}
        print(f"  {row['sample_id']} r{row['replicate']}: valid "
              f"{row['valid_fraction']:.2f}, bond MAE {row['bond_mae_all']:.4f}, "
              f"lag dev {lag.get('deviation_mean', float('nan')):+.3f}, "
              f"RMSF r {rmsf.get('r_heavy', float('nan')):.3f}, ratio "
              f"{rmsf.get('amplitude_ratio', float('nan')):.2f}, sanity step "
              f"{row['sanity']['hallucinated_transition_step']}")
    path = out_dir / f"metrics.{arm}.jsonl"
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"[score] {len(rows)} rollouts -> {path}")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", required=True,
                    choices=("none", "zero", "random", "pretrained"))
    ap.add_argument("--checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", choices=("generate", "score", "both"),
                    default="generate")
    ap.add_argument("--steps", type=int, default=N_STEPS)
    ap.add_argument("--n-step", type=int, default=200,
                    help="the sampler's denoising steps, not the rollout length")
    ap.add_argument("--replicates", type=int, default=2)
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to these sample ids (P007.4's 100 mM subset)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gagu-root", default=str(GAGU_ROOT))
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_env_lock(out_dir)
    root = Path(args.gagu_root)

    from kineidos.data.gagu import GAGUProtenixAdapter

    names = held_out_samples(root)
    if args.only:
        names = [n for n in names if n in set(args.only)]
        if not names:
            raise SystemExit(f"--only matched none of the 16 held-out names")
    samples = [GAGUProtenixAdapter(root / n).load() for n in names]
    topos = topologies_for(samples)

    meta: dict[str, Any] = {
        "protocol": {"k": K, "stride": STRIDE, "dt_ns": 0.1,
                     "steps": args.steps, "replicates": args.replicates,
                     "sampler_n_step": args.n_step,
                     "lags": list(LAGS),
                     "sanity_window": SANITY_WINDOW,
                     "sanity_threshold": SANITY_THRESHOLD},
        "starts": names,
        "worktrees": worktree_commits(),
        "env": {k: os.environ.get(k, "") for k in
                ("PYTHONPATH", "LAYERNORM_TYPE", "ATTN_IMPL",
                 "PYTORCH_CUDA_ALLOC_CONF", "SLURM_JOB_ID")},
    }

    if args.stage in ("generate", "both"):
        if not args.checkpoint:
            raise SystemExit("--stage generate needs --checkpoint")
        meta.update(generate(samples, topos, out_dir, arm=args.arm,
                             checkpoint=Path(args.checkpoint), n_steps=args.steps,
                             n_step_sampler=args.n_step, device=args.device,
                             replicates=args.replicates, only=names))
    if args.stage in ("score", "both"):
        score(samples, topos, out_dir, arm=args.arm)

    (out_dir / f"meta.rollout.{args.arm}.json").write_text(
        json.dumps(meta, indent=1, sort_keys=True) + "\n")
    print(f"[done] {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
