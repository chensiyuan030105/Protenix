"""P004.1: measure how WorldParticle's per-particle tokens `h` transform.

The fusion point concatenates `h` onto Protenix's `c_l` and lets the diffusion
head condition on it.  For that to mean anything, `h` has to carry geometric
information in a form the head can use, which requires knowing which of three
cases holds:

    h(Rx) = h(x)                  h is a rotation invariant (scalars only)
    h(Rx) recoverable from h(x)   h is equivariant under some representation
    neither                       h is tied to the lab frame -- a blocking item

The plan (section 2.9) treats "neither" as blocking rather than cosmetic: a model
conditioned on a lab-frame-bound signal learns a rule that does not survive
rotation, and rollout drifts.  `h` is 768 mixed channels, so the answer is
measured here rather than argued from the architecture -- some channels may be
invariant while others are not.

Translation is tested alongside rotation.  MD frames carry no canonical origin,
and 3D RoPE is applied to absolute positions, so translation sensitivity would
be just as damaging and is just as easy to check.

Usage (from the workspace root):
    PYTHONPATH=repos/research/kineidos-v2:repos/research/wp-v2 \
    LAYERNORM_TYPE=torch ATTN_IMPL=sdpa \
    LD_LIBRARY_PATH=/mnt/xfs/home/mhg/anaconda3/envs/kineidos-v2-slurm/lib \
      .../envs/kineidos-v2-slurm/bin/python -m kineidos.probe_h_symmetry
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from kineidos.window_align import canonicalize_window

GAGU_DEFAULT = Path(
    "/mnt/xfs/home/mhg/Projects/ForSiyuan/RNA-WorldParticle-Workspace/datasets/"
    "processed/gagu_internal_loop_v0_1/gagu_100mM_K_agaguu_startI_r1"
)

# config.yaml, except particle_radius -- see pick_particle_radius().
WP_CONFIG = dict(
    kernel_size=[4, 4, 4],
    cconv_embedding_dim=384,
    radius_scale=1.5,
    data="molecular",
    use_topology_conv=None,
    topology_search_radius=None,
    topology_blend_alpha=None,
    coordinate_mapping="ball_to_cube_volume_preserving",
    interpolation="linear",
    use_window=True,
    particle_position_rope_dim=48,
    slot_compression_ratio=4,
    slot_attn_iters=6,
    slot_attn_heads=16,
    num_decoder_layers=8,
    decoder_attn_heads=12,
    decoder_attn_ffn_hidden_dim=512,
    decoder_attn_dropout=0.1,
    output_hidden_dim=512,
    output_layers=5,
    timestep=0.01,
    gravity=(0.0, -9.81, 0.0),
    other_feats_channels=0,
    obstacle_feats_channels=3,
    verbose=False,
    velhead=True,
)
K_PARTITE = 2


def load_gagu_heavy(sample_dir: Path, n_frames: int = 1, window_from: int = 0
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Frame 0 heavy-atom positions and velocities, in the PDB's atom order.

    Protenix removes hydrogens (Filter.remove_hydrogens, several call sites in
    data/core/parser.py), so its ref_pos is heavy atoms only.  `h` is
    concatenated onto c_l position by position, so WorldParticle has to see the
    same set in the same order -- this is not a tuning choice.
    """
    sample_id = sample_dir.name
    npz_path = sample_dir / f"{sample_id}.npz"
    pdb_path = sample_dir / f"{sample_id}.pdb"

    elements: list[str] = []
    xyz: list[tuple[float, float, float]] = []
    with pdb_path.open() as fh:
        for line in fh:
            if line[:6].strip().upper() not in {"ATOM", "HETATM"}:
                continue
            el = line[76:78].strip().upper()
            if not el:  # fall back to the atom name when the column is blank
                el = next((c for c in line[12:16].strip() if c.isalpha()), "")
                el = el.upper()
            elements.append(el)
            xyz.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))

    pdb_xyz = np.asarray(xyz, dtype=np.float64)  # Angstrom
    # Frames spread across the trajectory, not consecutive: consecutive frames
    # are nearly identical, and near-duplicate rows would leave the linear fit
    # ill-conditioned for the same reason too few rows would.
    with np.load(npz_path) as z:
        total = z["position"].shape[0]
        idx = np.linspace(window_from, total - 1, n_frames, dtype=int)
        pos_nm = np.asarray(z["position"][idx], dtype=np.float64)  # [F, A, 3] nm
        vel_nm = np.asarray(z["velocity"][idx], dtype=np.float64)
        # The reference conformer is always trajectory frame 0, which is what the
        # PDB holds and what Protenix uses as ref_pos.  Loading it separately
        # from the window matters: with window_from=0 the window's anchor *is*
        # the reference, the unrotated alignment degenerates to the identity,
        # and the test stops exercising the case training will actually see.
        ref_nm = np.asarray(z["position"][0], dtype=np.float64)
    print(f"  window frames: {list(idx)} of {total}; reference = frame 0")

    assert len(elements) == pos_nm.shape[1], (
        f"PDB has {len(elements)} atoms, npz has {pos_nm.shape[1]}"
    )

    # The atom-order contract, checked by coordinates rather than by shape: a
    # permuted array of the same length would pass a shape check and train a
    # quietly wrong model.
    # Compared against ref_nm, which is trajectory frame 0 -- not against the
    # window's first frame, which is frame `window_from` and has no reason to
    # match the PDB.
    align = float(np.abs(pdb_xyz - ref_nm * 10.0).max())
    assert align < 1e-3, f"PDB is not npz frame 0: max |diff| = {align:.3e} A"
    print(f"  atom order contract: max |pdb - npz[0]| = {align:.6e} A")

    heavy = np.array([e != "H" for e in elements], dtype=bool)
    print(f"  atoms: {len(elements)} all -> {int(heavy.sum())} heavy "
          f"({int((~heavy).sum())} hydrogens removed)")
    return pos_nm[:, heavy], vel_nm[:, heavy], ref_nm[heavy]


def pick_particle_radius(pos: np.ndarray, target_neighbors: int = 24) -> float:
    """A particle_radius giving a non-degenerate neighbourhood.

    config.yaml ships 0.012, which makes filter_extent = radius_scale * 6 * r =
    0.108 nm -- below the 0.131 nm mean bond length measured in the plan's
    section 2.5.  Officially-tuned for their fluid data, every atom here would
    be isolated, and a symmetry measured on an empty neighbour graph says
    nothing about the real one.  So the radius is chosen from this geometry,
    and reported, instead of inherited.
    """
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    kth = np.sort(d, axis=1)[:, min(target_neighbors, d.shape[0] - 1)]
    cutoff = float(np.median(kth))
    # ParticleRadiusResearch searches at `0.5 * extents` (continuous_conv.py:313),
    # so the cutoff is half the filter extent, not the extent.  Omitting this
    # factor cost a 2x radius and left 5.2 neighbours where 24 were intended.
    return 2.0 * cutoff / (WP_CONFIG["radius_scale"] * 6.0)


def random_rotation(seed: int) -> np.ndarray:
    """A uniformly random proper rotation (QR of a Gaussian, det forced to +1)."""
    g = np.random.default_rng(seed)
    q, r = np.linalg.qr(g.normal(size=(3, 3)))
    q *= np.sign(np.diag(r))          # fix QR's sign ambiguity
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1                 # reflection -> rotation
    return q


def build_model(particle_radius: float, seed: int = 0):
    from models.particle_network_cross_attn_feat import (
        ParticleNetworkCrossAttnLocalFeature,
    )

    torch.manual_seed(seed)
    model = ParticleNetworkCrossAttnLocalFeature(
        particle_radius=particle_radius, **WP_CONFIG
    )
    model.eval()  # decoder_attn_dropout is 0.1; without eval() the two passes
    return model  # would differ for reasons unrelated to geometry


# The pipeline, stage by stage.  Finding which stage first loses the symmetry
# is the point: the plan calls "neither invariant nor equivariant" a blocking
# item and requires the cause be located before building on top of it.
STAGES = (
    ("1_local_feature_extractor", "local_feature_extractor", "particle"),
    ("2_slot_attention", "super_particle_cross_attn_feat", "super_particle"),
    ("3_super_particle_decoder", "super_particle_decoder", "particle"),
    ("4_output_network", "output_network", "particle"),
)


def run_once(model, pos: np.ndarray, vel: np.ndarray) -> tuple[dict, dict]:
    """One forward pass; returns compute_correction's output and every stage's
    activation, taken from the modules themselves rather than reconstructed."""
    captured: dict[str, torch.Tensor] = {}

    handles = []
    for label, attr, _kind in STAGES:
        module = getattr(model, attr, None) or getattr(
            model.local_feature_extractor, attr, None)
        if module is None:
            continue

        def hook(_m, _i, output, _label=label):
            # local_feature_extractor returns a tuple; slot attention returns
            # (features, positions).  Keep the first tensor in either case.
            t = output[0] if isinstance(output, (tuple, list)) else output
            captured[_label] = t.detach().clone()

        handles.append(module.register_forward_hook(hook))

    try:
        with torch.no_grad():
            out = model.compute_correction(
                torch.tensor(pos, dtype=torch.float32),
                torch.tensor(vel, dtype=torch.float32),
                None, None, None,
                k_partite=K_PARTITE,
                return_tokens=True,
                skip_output_network=False,
            )
    finally:
        for h in handles:
            h.remove()
    return out, captured


def flatten(t: torch.Tensor) -> torch.Tensor:
    """[1, M, C] or [M, C] -> [M, C]."""
    return t.squeeze(0) if t.dim() == 3 and t.shape[0] == 1 else t


def prepare(pos: np.ndarray, vel: np.ndarray, ref: np.ndarray, *,
            rot: np.ndarray | None = None, shift: np.ndarray | None = None,
            align: bool) -> tuple[np.ndarray, np.ndarray, dict]:
    """Apply a rigid motion to the window, then canonicalise if asked.

    The motion goes on first: the question being measured is what the model sees
    when the *input data* arrives in an arbitrary pose, so canonicalisation has
    to run on the moved window exactly as it would at training time.

    `ref` is deliberately never moved.  It is the fixed external reference that
    makes the pose canonical -- rotating it along with the data would reproduce
    the "align to the window's own oldest frame" construction, which leaves the
    global orientation in place (see kineidos/window_align.py).
    """
    p, v = pos, vel
    if rot is not None:
        p, v = p @ rot.T, v @ rot.T
    if shift is not None:
        p = p + shift
    if not align:
        return p, v, {}
    p, v, info = canonicalize_window(p, v, ref)
    return p, v, info


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative difference, scaled by the reference so it reads as a fraction."""
    denom = a.norm().item()
    return (a - b).norm().item() / denom if denom > 0 else float("nan")


def linear_map_residuals(h: torch.Tensor, h_t: torch.Tensor,
                         train_frac: float = 0.75) -> dict[str, float]:
    """Fit A with h @ A ~= h_t and report the residual on held-out rows.

    The in-sample residual alone proves nothing here.  A is 768x768, so with a
    single 470-particle frame the system has far more unknowns than equations
    and lstsq reaches ~1e-13 whether or not any equivariance exists -- an
    earlier version of this probe reported exactly that and concluded
    "equivariant" from an artefact of an underdetermined fit.

    If h really transforms under a linear representation of R, A is a property
    of R and not of the sample, so it must predict rows it never saw.  Rows are
    therefore stacked across frames until there are comfortably more than 768 of
    them, and the residual that counts is the held-out one.
    """
    a, b = h.double(), h_t.double()
    n_rows, n_feat = a.shape
    n_train = int(n_rows * train_frac)
    if n_train <= n_feat:
        raise ValueError(
            f"underdetermined: {n_train} training rows for {n_feat} features. "
            "Stack more frames (--frames) so the fit is identifiable."
        )
    sol = torch.linalg.lstsq(a[:n_train], b[:n_train]).solution
    in_res = (a[:n_train] @ sol - b[:n_train]).norm().item() / b[:n_train].norm().item()
    out_res = (a[n_train:] @ sol - b[n_train:]).norm().item() / b[n_train:].norm().item()
    return {"train_rows": n_train, "feature_dim": n_feat,
            "in_sample": in_res, "held_out": out_res}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=Path, default=GAGU_DEFAULT)
    ap.add_argument("--frames", type=int, default=4,
                    help="frames stacked for the linear fit; 470 heavy atoms "
                         "each, and the fit needs well over 768 rows to be "
                         "identifiable at all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--window-from", type=int, default=0,
                    help="first trajectory frame of the window; the reference "
                         "conformer stays frame 0 regardless, so a non-zero "
                         "value is what makes the anchor differ from the "
                         "reference as it will in training")
    ap.add_argument("--align", action="store_true",
                    help="canonicalise each window before WorldParticle sees it "
                         "(kineidos/window_align.py); this is the remedy P004.1 "
                         "settled on, and with it h must come out invariant")
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args()

    print("=== 1. input ===")
    pos, vel, ref = load_gagu_heavy(args.sample, n_frames=args.frames,
                                    window_from=args.window_from)
    n_frames, n, _ = pos.shape
    radius = pick_particle_radius(pos[0])
    extent = radius * WP_CONFIG["radius_scale"] * 6
    print(f"  particle_radius: {radius:.6f} nm  (config.yaml ships 0.012)")
    print(f"  filter_extent {extent:.4f} nm -> search cutoff {extent / 2:.4f} nm")

    print("\n=== 2. model ===")
    model = build_model(radius, seed=args.seed)
    print(f"  {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params, eval mode")

    R = random_rotation(args.seed + 1)
    print(f"  R: det = {np.linalg.det(R):+.6f}, "
          f"orthogonality err = {np.abs(R @ R.T - np.eye(3)).max():.2e}")
    t = np.array([1.7, -0.9, 2.3])

    print("\n=== 3. forward passes ===")
    print(f"  canonicalisation: {'ON' if args.align else 'OFF'}")

    p_plain, v_plain, info = prepare(pos, vel, ref, align=args.align)
    p_rot, v_rot, info_rot = prepare(pos, vel, ref, rot=R, align=args.align)
    p_tr, v_tr, _ = prepare(pos, vel, ref, shift=t, align=args.align)
    if args.align:
        print(f"  anchor RMSD to ref: {info['anchor_rmsd_to_ref']:.4f} nm "
              f"(rotated window: {info_rot['anchor_rmsd_to_ref']:.4f} nm)")
        drift = float(np.abs(p_plain - p_rot).max())
        print(f"  canonicalised windows agree: max |aligned(x) - aligned(Rx)| "
              f"= {drift:.3e} nm")

    plain, rotated, translated = [], [], []
    for f in range(n_frames):
        plain.append(run_once(model, p_plain[f], v_plain[f]))
        rotated.append(run_once(model, p_rot[f], v_rot[f]))
        if f == 0:  # translation needs no stacking: no fit is involved
            translated.append(run_once(model, p_tr[f], v_tr[f]))
        print(f"  frame {f + 1}/{n_frames} done")

    out0, cap0 = plain[0]
    h = out0["tokens"]
    print("\n=== 4. h identity and shape ===")
    print(f"  returned keys: {sorted(out0)}")
    print(f"  h shape: {tuple(h.shape)}")
    same = torch.equal(h, flatten(cap0["3_super_particle_decoder"]))
    print(f"  h is the super_particle_decoder output, bit-for-bit: {same}")
    avg_nb = float(model.local_feature_extractor.conv0_molecular._avg_neighbors)
    print(f"  mean neighbours per particle: {avg_nb:.1f}")

    checks = {
        "h_shape_is_N_by_768": tuple(h.shape) == (n, 768),
        "h_is_decoder_output": bool(same),
        "neighbour_graph_non_degenerate": avg_nb > 2.0,
    }

    print("\n=== 5. where the rotation symmetry goes ===")
    print(f"  {'stage':<28} {'rows':>6} {'invariance':>12} {'lin in-sample':>14} "
          f"{'lin held-out':>13}")
    stage_results = {}
    for label, _attr, kind in STAGES:
        if label not in cap0:
            continue
        a = torch.cat([flatten(c[label]) for _o, c in plain]).double()
        b = torch.cat([flatten(c[label]) for _o, c in rotated]).double()
        inv = rel(a, b)
        entry = {"rows": a.shape[0], "dim": a.shape[1], "invariance_residual": inv}
        try:
            lin = linear_map_residuals(a, b)
            entry["linear"] = lin
            cells = f"{lin['in_sample']:>14.3e} {lin['held_out']:>13.3e}"
        except ValueError as exc:
            entry["linear_skipped"] = str(exc)
            cells = f"{'underdetermined':>28}"
        print(f"  {label:<28} {a.shape[0]:>6} {inv:>12.3e} {cells}")
        stage_results[label] = entry

    print("\n=== 6. end to end ===")
    inv_tr = rel(flatten(cap0["3_super_particle_decoder"]),
                 flatten(translated[0][1]["3_super_particle_decoder"]))
    print(f"  h translation invariance  ||h(x+t)-h(x)|| / ||h(x)|| = {inv_tr:.4e}")

    pc = out0["pos_correction"]
    pc_rot = rotated[0][0]["pos_correction"]
    equiv_pred = rel(pc @ torch.tensor(R.T, dtype=pc.dtype), pc_rot)
    print(f"  pred equivariance ||pred(Rx)-R.pred(x)|| / ||R.pred(x)|| = {equiv_pred:.4e}")
    print("  (measured at random init: an architecture that is structurally "
          "equivariant\n   is equivariant at any weights, so this is a property "
          "of the design,\n   not of training)")

    TOL = 1e-4
    h_stage = stage_results.get("3_super_particle_decoder", {})
    h_inv = h_stage.get("invariance_residual", float("nan"))
    h_lin = h_stage.get("linear", {}).get("held_out")
    if h_inv < TOL:
        verdict = "INVARIANT"
    elif h_lin is not None and h_lin < TOL:
        verdict = "EQUIVARIANT (linear representation of R)"
    elif h_lin is None:
        verdict = "UNDETERMINED (linear fit underdetermined -- stack more frames)"
    else:
        verdict = "NEITHER -- blocking, see plan section 2.9"

    print(f"\n  h rotation verdict: {verdict}")
    print(f"  h translation:      {'INVARIANT' if inv_tr < TOL else 'NOT invariant'}")
    first_broken = next(
        (lbl for lbl, _a, _k in STAGES
         if lbl in stage_results and stage_results[lbl]["invariance_residual"] >= TOL),
        None)
    print(f"  symmetry first lost at: {first_broken}")

    print("\n=== 7. checks ===")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")

    record = {
        "sample": str(args.sample), "seed": args.seed, "frames": n_frames,
        "window_from": args.window_from, "canonicalized": bool(args.align),
        "anchor_rmsd_to_ref": info.get("anchor_rmsd_to_ref"),
        "n_particles": n, "particle_radius": radius,
        "filter_extent_nm": extent, "search_cutoff_nm": extent / 2,
        "mean_neighbours": avg_nb, "h_shape": list(h.shape),
        "checks": checks, "stages": stage_results,
        "h_rotation_verdict": verdict,
        "h_translation_invariance_residual": inv_tr,
        "h_translation_invariant": inv_tr < TOL,
        "prediction_equivariance_residual": equiv_pred,
        "symmetry_first_lost_at": first_broken, "tolerance": TOL,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(record, indent=2) + "\n")
        print(f"\nwrote {args.report}")

    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
