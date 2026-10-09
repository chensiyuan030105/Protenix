"""P011 section 5.3: does `h` carry per-atom information at all?

This is the probe P009 10.7 was written from, committed for the first time --
that appendix was produced by a throwaway script and the numbers in it have
never been reproducible from any branch.  It answers one question with hooks
rather than with argument: **at which layer does the per-atom part of
WorldParticle's activations disappear**, and the decomposition it uses is the
exact one P009 settled on after getting it wrong once:

    ||T||^2 = N * ||mean_atoms(T)||^2 + ||T - mean_atoms(T)||^2

so "per-atom fraction" is the second term over the total.  An earlier version
used `std(dim=0).mean() / mean(dim=0).abs().mean()`, which takes two separate
means and is not a fraction of anything.

What P009 measured, on the old 5-channel contract (eval mode, random init):

    conv0_molecular      4.347%        <- the neighbour graph does carry geometry
    dense0_molecular     0.000%  (1.57e-07)
    decoder q_proj       0.000%  (1.59e-05)
    wp_tokens            0.000%
    h = LN(wp_tokens)    0.000%

and the cause: of the five channels, two were cross-atom constants 400-5000x
larger than the only per-atom signal (three velocity components at 2e-4).
`dense0_molecular` is a Linear(5 -> 384); the two constants decided it.

The gates (plan section 5.3, registered before running):

    dense0 per-atom fraction   > 10%
    wp_tokens per-atom fraction > 50%
    same input twice, per-atom cos > 0.999        <- P009's gate (1)
    reconstruction vs online heldout_wp/*, 4 dp   <- P009's gate (2)

**Both gates on `h` exist because P009 got this quantity wrong four times in a
row, every time by reasoning about it instead of measuring it.** Gate (1) is
there because `SuperParticleDecoder` carries decoder_attn_dropout=0.1: without
`.eval()` the per-atom part -- a tiny fraction of the output -- is swamped by
dropout noise, and "the same input twice" came out at cos ~= 0.  Gate (2) is
there because a reconstruction that nobody checked against the training loop's
own logged numbers is just a second implementation, and the first four attempts
were all second implementations that disagreed with the first.

Gate (2) needs an online log, which does not exist until an arm has run an
evaluation round, so it is deferred rather than skipped: `--heldout-wp` takes
the arm's first round and compares.  Everything else runs before any arm.

Usage (from the workspace root):
    PYTHONPATH=repos/research/kineidos-v4:repos/research/wp-v3 \
    LAYERNORM_TYPE=torch ATTN_IMPL=sdpa LD_LIBRARY_PATH=$ENV/lib \
      $ENV/bin/python -m kineidos.probe_wp_content --windows 256
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from kineidos import determinism
from kineidos.train.batch import GAGU_ROOT, collate_window
from kineidos.wp_bridge import TOKEN_DIM, WorldParticleBridge

# Where the hooks go.  Each one is a place P009 found the signal either
# surviving or already gone, so the list is the shape of that finding rather
# than a sample of convenient layers.
HOOKS = (
    ("conv0_molecular", "local_feature_extractor.conv0_molecular"),
    ("dense0_molecular", "local_feature_extractor.dense0_molecular"),
    # `particle_features` = cat([conv0_out, dense0_out]), and it is 768 wide --
    # `cconv_embedding_dim * 2`, the same TOKEN_DIM the fusion expects.  So it
    # is a candidate `h` that needs no change on the Protenix side, and
    # measuring it here is how the choice between it and the decoder's output
    # gets made on numbers instead of on preference.  D1 freezes the
    # `skip_output_network` path, so this is a reading, not a change.
    ("particle_features", "local_feature_extractor"),
    ("decoder_q_proj",
     "super_particle_decoder.layers.0.multihead_cross_attn.q_proj"),
)


def resolve(root: torch.nn.Module, dotted: str) -> torch.nn.Module:
    obj = root
    for part in dotted.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def per_atom_fraction(t: torch.Tensor) -> float:
    """The fraction of squared magnitude that is not the cross-atom mean.

    `t` must be `[N, C]` or `[1, N, C]`, atoms on the leading axis after the
    optional squeeze.  **Raises on anything else**, and that is why: the first
    version silently accepted `[F, N, 1]`, failed the `shape[0] == 1` squeeze,
    and averaged over the *frame* axis -- so a number reported as "per-atom
    fraction of |d_anchor|" was in fact its across-frame fraction.  A wrong
    axis gives a plausible number, which is the kind of error that only an
    assertion catches.
    """
    x = t.detach().float()
    if x.dim() == 3 and x.shape[0] == 1:
        x = x[0]
    if x.dim() != 2:
        raise ValueError(
            f"per_atom_fraction needs [N, C] (atoms leading), got "
            f"{tuple(t.shape)}. Reduce the frame axis at the call site; "
            f"averaging over it here is what produced a mislabelled number."
        )
    mean = x.mean(dim=0, keepdim=True)
    total = x.pow(2).sum()
    const = mean.pow(2).sum() * x.shape[0]
    if float(total) == 0.0:
        return float("nan")
    return float(((total - const) / total).clamp(min=0.0))


def per_atom_part(t: torch.Tensor) -> torch.Tensor:
    x = t.detach().float()
    if x.dim() == 3 and x.shape[0] == 1:
        x = x[0]
    return (x - x.mean(dim=0, keepdim=True)).reshape(-1)


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    na, nb = a.norm(), b.norm()
    if float(na) == 0.0 or float(nb) == 0.0:
        return float("nan")
    return float((a @ b) / (na * nb))


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    d = a.detach().float().norm()
    if float(d) == 0.0:
        return float("nan")
    return float((a.detach().float() - b.detach().float()).norm() / d)


def random_rotation(seed: int) -> np.ndarray:
    g = np.random.default_rng(seed)
    q, r = np.linalg.qr(g.normal(size=(3, 3)))
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


class Capture:
    """Forward hooks, and `h` built the way the fusion builds it."""

    def __init__(self, bridge: WorldParticleBridge) -> None:
        self.bridge = bridge
        self.acts: dict[str, torch.Tensor] = {}
        self.handles = []
        for label, dotted in HOOKS:
            module = resolve(bridge.wp, dotted)

            def hook(_m, _i, output, _label=label):
                t = output[0] if isinstance(output, (tuple, list)) else output
                # The last frame of the window wins.  The pooling is a mean
                # over frames, so any single frame is representative of where
                # the signal is, and keeping all K would multiply the memory
                # for no extra answer.
                self.acts[_label] = t.detach().clone()

            self.handles.append(module.register_forward_hook(hook))

    def close(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []


def layernorm_768() -> torch.nn.Module:
    """The fusion's `wp_layernorm`, at initialisation.

    The real one lives in AtomAttentionEncoder and is built with Protenix's own
    LayerNorm; at initialisation its weight is ones and its offset is absent
    (`create_offset=False`), so this reproduces it exactly without building a
    whole model.  Using the same class rather than nn.LayerNorm matters for
    gate (2): the comparison is against numbers the real module produced.
    """
    from protenix.model.triangular.layers import LayerNorm

    ln = LayerNorm(TOKEN_DIM)
    ln.eval()
    return ln


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=256,
                    help="how many held-out windows; 256 is the arms' "
                         "kineidos.eval_windows, and the set is drawn with the "
                         "same seed so these are literally the same windows")
    ap.add_argument("--eval-seed", type=int, default=1234)
    ap.add_argument("--window-k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42,
                    help="WorldParticle's initialisation seed, as wp.seed")
    ap.add_argument("--gagu-root", type=Path, default=GAGU_ROOT)
    ap.add_argument("--held-out", type=str, default="",
                    help="comma-separated sample names; default is every *_r4 "
                         "directory, which is what the arms hold out")
    ap.add_argument("--heldout-wp", type=Path, default=None,
                    help="gate (2): a training run's logged heldout_wp/* "
                         "values, to reconcile this reconstruction against. "
                         "Needs an arm to have finished one evaluation round, "
                         "so it is passed on a second invocation.")
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args()

    determinism.enable()

    from kineidos.data.gagu import GAGUProtenixAdapter
    from kineidos.data.windows import GAGUWindowDataset, build_window

    print("=== 1. the held-out set ===")
    if args.held_out:
        names = [n for n in args.held_out.split(",") if n]
    else:
        names = sorted(p.name for p in args.gagu_root.glob("*_r4"))
    print(f"  {len(names)} samples: {', '.join(names[:4])}"
          f"{' ...' if len(names) > 4 else ''}")
    samples = [GAGUProtenixAdapter(args.gagu_root / n).load() for n in names]
    eval_ds = GAGUWindowDataset(samples, k=args.window_k, length=args.windows,
                                seed=args.eval_seed, canonicalize=True)
    print(f"  {args.windows} windows, seed {args.eval_seed} -- the same draw "
          f"the arms score")

    print("\n=== 2. the bridge, at random initialisation ===")
    bridge = WorldParticleBridge("random", seed=args.seed)
    bridge.eval()          # decoder_attn_dropout is 0.1; see gate (1)
    n_params = sum(p.numel() for p in bridge.parameters())
    print(f"  other_feats_channels = {bridge.features.channels} "
          f"(was 1 under the old contract)")
    print(f"  {n_params / 1e6:.2f}M parameters, eval mode")
    ln = layernorm_768()

    cap = Capture(bridge)
    fractions: dict[str, list[float]] = {k: [] for k, _ in HOOKS}
    fractions["wp_tokens"] = []
    fractions["h"] = []
    h_norms, h_first = [], None
    branch_rms: dict[str, float] = {}

    print(f"\n=== 3. {args.windows} windows ===")
    with torch.no_grad():
        for i in range(args.windows):
            batch = collate_window(eval_ds[i])
            feats = batch["input_feature_dict"]
            wp_tokens = bridge(feats)
            h = ln(wp_tokens)
            for label, _ in HOOKS:
                fractions[label].append(per_atom_fraction(cap.acts[label]))
            fractions["wp_tokens"].append(per_atom_fraction(wp_tokens))
            fractions["h"].append(per_atom_fraction(h))
            h_norms.append(float(h.norm()))
            if i == 0:
                h_first = h.clone()
                rot_ref = {k: cap.acts[k].clone() for k, _ in HOOKS}
                branch_rms = {**bridge.features.branch_rms,
                              **bridge.wp.local_feature_extractor._branch_rms}
                first_feats = feats
            if (i + 1) % 32 == 0:
                print(f"  {i + 1}/{args.windows}")

    print("\n=== 4. per-branch rms before the concat (section 4, item 4) ===")
    print("  P009 10.7 measured 1.0 against 2e-4 here, i.e. 5000:1, and that "
          "ratio\n  is why dense0's per-atom output was 1.57e-07 of its "
          "magnitude.")
    for k in sorted(branch_rms):
        print(f"  {k:<22} {branch_rms[k]:>10.4f}")

    print("\n=== 5. per-atom fraction by layer ===")
    print(f"  {'layer':<20} {'mean':>10} {'min':>10} {'max':>10}   "
          f"P009 (old contract)")
    p009 = {"conv0_molecular": "4.347%", "dense0_molecular": "0.000%",
            "decoder_q_proj": "0.000%", "wp_tokens": "0.000%", "h": "0.000%",
            "particle_features": "(not measured)"}
    summary = {}
    for label in [k for k, _ in HOOKS] + ["wp_tokens", "h"]:
        v = np.array(fractions[label], dtype=float)
        summary[label] = {"mean": float(v.mean()), "min": float(v.min()),
                          "max": float(v.max())}
        print(f"  {label:<20} {v.mean():>9.3%} {v.min():>9.3%} "
              f"{v.max():>9.3%}   {p009.get(label, '')}")

    # ---------------------------------------------------------- gate (1)
    print("\n=== 6. gate (1): the same input twice ===")
    print("  P009 section 8 item 8: the first step of any measurement of `h`. "
          "Without\n  .eval() the decoder's dropout gave cos ~= 0 on identical "
          "input.")
    with torch.no_grad():
        h_again = ln(bridge(first_feats))
    gate1_cos = cos(per_atom_part(h_first), per_atom_part(h_again))
    gate1_max = float((h_first - h_again).abs().max())
    print(f"  per-atom cos = {gate1_cos:.6f}   max|dh| = {gate1_max:.3e}")

    # --------------------------------------------- rotation, with contract
    print("\n=== 7. rotation invariance, with the real contract ===")
    print("  The section 4 probe runs WorldParticle with other_feats=None and "
          "a zeroed\n  velocity slot, so every atom's input feature vector is "
          "*identical* and the\n  only per-atom variation comes through the "
          "neighbour graph.  That is a\n  degenerate regime: P009 put the "
          "per-atom part at 1.57e-07 of dense0's\n  magnitude, which is the "
          "same order as the perturbation a float32 rotation\n  introduces, so "
          "the relative residual there is a ratio of two small numbers.\n  "
          "This is the same measurement in the regime the arms actually run.")
    R = random_rotation(args.seed + 1)
    rot = torch.tensor(R, dtype=torch.float32)
    with torch.no_grad():
        rot_feats = dict(first_feats)
        rot_feats["wp_position_nm"] = first_feats["wp_position_nm"] @ rot.T
        # The rotated copy is, truthfully, no longer in canonical pose, and
        # `assert_window_canonical` would reject it -- correctly, since its job
        # is to catch a transform applied after the dataloader.  Saying so is
        # the honest way past it; suppressing the check would not be.
        rot_feats["wp_canonicalized"] = torch.tensor(False)
        h_rot = ln(bridge(rot_feats))
    rot_resid = rel(h_first, h_rot)
    rot_cos = cos(per_atom_part(h_first), per_atom_part(h_rot))
    rot_by_layer = {k: rel(v, cap.acts[k]) for k, v in rot_ref.items()}
    print(f"  ||h(Rx) - h(x)|| / ||h(x)|| = {rot_resid:.4e}")
    print(f"  per-atom cos                = {rot_cos:.6f}")
    for k, v in rot_by_layer.items():
        print(f"  {k:<22} {v:.4e}")

    # ------------------------------------------------- content, reported
    print("\n=== 8. what `h` encodes (reported, no gate) ===")
    print("  P009 measured 0.92-0.93 for the same molecule in different "
          "conformations\n  and 0.977 for stride 1 against stride 10, and "
          "concluded that the per-atom\n  part encoded *which atom this is* "
          "and was almost blind to dt.  Those\n  numbers were taken on a "
          "signal whose magnitude was 0.000% of the output,\n  so they say what "
          "the residual looked like, not what the model could use.")
    sample = samples[0]
    content = {}
    with torch.no_grad():
        pairs = {
            "same_molecule_two_conformations": (
                build_window(sample, 1000, stride=10, k=args.window_k),
                build_window(sample, 5000, stride=10, k=args.window_k)),
            "stride_1_vs_10": (
                build_window(sample, 5000, stride=1, k=args.window_k),
                build_window(sample, 5000, stride=10, k=args.window_k)),
        }
        # Measured at **every** hooked layer, not only at `h`.  That is the
        # measurement that decides what to do when `h` turns out to be
        # near-constant: if an early layer separates two conformations and `h`
        # does not, the information exists and something downstream is
        # destroying it; if no layer separates them, the features themselves
        # carry no conformation and the answer is somewhere else entirely
        # (plan section 12's fallback -- token-level pairwise distances on the
        # fusion side).  Reading it only at `h`, as the first version did,
        # cannot tell those two apart, and they have opposite next steps.
        layers = [k for k, _ in HOOKS] + ["wp_tokens", "h"]
        for label, (wa, wb) in pairs.items():
            acts = {}
            for tag, w in (("a", wa), ("b", wb)):
                wpt = bridge(collate_window(w)["input_feature_dict"])
                acts[tag] = {k: cap.acts[k].clone() for k, _ in HOOKS}
                # wp_tokens is the bridge's output, `h` is that after the
                # fusion's LayerNorm.  Kept separate rather than deriving one
                # from the other: P010 section 8 found `wp_layernorm` removing
                # a third of the oracle's content, so the two are not the same
                # tensor up to a scale and must not be reported as if they were.
                acts[tag]["wp_tokens"] = wpt.clone()
                acts[tag]["h"] = ln(wpt)
            content[label] = {}
            print(f"  {label}")
            for lyr in layers:
                a, b = acts["a"][lyr], acts["b"][lyr]
                content[label][lyr] = {
                    "per_atom_cos": cos(per_atom_part(a), per_atom_part(b)),
                    "constant_cos": cos(
                        a.detach().float().reshape(-1, a.shape[-1]).mean(dim=0),
                        b.detach().float().reshape(-1, b.shape[-1]).mean(dim=0)),
                }
                print(f"    {lyr:<22} per-atom cos "
                      f"{content[label][lyr]['per_atom_cos']:+.6f}   "
                      f"constant cos "
                      f"{content[label][lyr]['constant_cos']:+.6f}")

    # ------------------------------------------- what the history contains
    # The structural fact this measures: after D2 there is **exactly one** path
    # from conformation into `particle_features`, namely the distance-dependent
    # filter inside conv0_molecular.  `other_feats` is frame-independent except
    # for one per-frame scalar (the frame index, identical across atoms), the
    # velocity slot is zero, and `dense0_molecular` takes no positions.  So the
    # eight history frames differ *only* in their geometry, and the bridge then
    # mean-pools them.
    #
    # Which raises the question this section answers: how much do the eight
    # frames differ at all, and does pooling them leave anything that one frame
    # would not have given?  If `h(pooled)` is indistinguishable from `h(newest
    # frame only)`, the window is costing 8x the forward passes for no
    # information -- and no training arm can read a history effect that is not
    # in `h`.
    print("\n=== 8b. the eight history frames, and what pooling does ===")
    from kineidos.data.windows import build_window as _bw

    win = _bw(samples[0], 5000, stride=10, k=args.window_k)
    wfeats = collate_window(win)["input_feature_dict"]
    pos = wfeats["wp_position_nm"]
    mask = wfeats["wp_frame_mask"].to(torch.bool)
    time_feat = bridge._frame_time_feature(wfeats)
    frame_acts, frame_tokens = [], []
    with torch.no_grad():
        ident = bridge.features.identity(wfeats)
        zero_vel = torch.zeros_like(pos[0])
        for k in range(pos.shape[0]):
            if not bool(mask[k]):
                continue
            out = bridge.wp.compute_correction(
                pos[k], zero_vel, bridge.features(time_feat[k], ident),
                None, None, k_partite=2, return_tokens=True,
                skip_output_network=True)
            frame_tokens.append(out["tokens"])
            frame_acts.append({k2: cap.acts[k2].clone() for k2, _ in HOOKS})
    nf = len(frame_tokens)
    print(f"  window: target {win.target_frame}, stride {win.stride}, "
          f"dt {win.delta_t_ns:.3f} ns, {nf} real frames of {pos.shape[0]}")
    print(f"  {'layer':<20} {'oldest vs newest':>18} {'adjacent frames':>18}")
    frames = {}
    for lyr in [k for k, _ in HOOKS]:
        a = per_atom_part(frame_acts[0][lyr])
        b = per_atom_part(frame_acts[-1][lyr])
        adj = [cos(per_atom_part(frame_acts[i][lyr]),
                   per_atom_part(frame_acts[i + 1][lyr]))
               for i in range(nf - 1)]
        frames[lyr] = {"oldest_vs_newest_cos": cos(a, b),
                       "adjacent_cos_mean": float(np.mean(adj))}
        print(f"  {lyr:<20} {frames[lyr]['oldest_vs_newest_cos']:>18.6f} "
              f"{frames[lyr]['adjacent_cos_mean']:>18.6f}")
    pooled = torch.stack(frame_tokens).mean(dim=0)
    newest = frame_tokens[-1]
    frames["pooled_vs_newest_cos"] = cos(per_atom_part(pooled),
                                         per_atom_part(newest))
    frames["pooled_vs_newest_rel"] = rel(pooled, newest)
    print(f"  pooled over {nf} frames vs the newest frame alone: "
          f"per-atom cos {frames['pooled_vs_newest_cos']:.6f}, "
          f"rel {frames['pooled_vs_newest_rel']:.3e}")
    # The spread across frames, against the spread across atoms, both on the
    # per-atom part.  This is the ratio that says whether the history axis
    # carries anything comparable to the identity axis.
    stack = torch.stack([per_atom_part(t) for t in frame_tokens])
    across_frames = float(stack.std(dim=0).pow(2).mean().sqrt())
    across_atoms = float(stack.mean(dim=0).pow(2).mean().sqrt())
    frames["across_frames_rms"] = across_frames
    frames["across_atoms_rms"] = across_atoms
    frames["frame_to_atom_ratio"] = across_frames / max(across_atoms, 1e-30)
    print(f"  per-atom part: rms across frames {across_frames:.4e}, "
          f"rms across atoms {across_atoms:.4e}, ratio "
          f"{frames['frame_to_atom_ratio']:.4e}")

    # --------------------------------- what differs between frames, physically
    # Section 8b says the frames differ by 1.7% at conv0.  This says what that
    # 1.7% *is*, in nanometres, and then tests the one candidate the project's
    # own literature note already measured and ranked
    # (repos/literature/topics/history-featurization.md section 5.1, job
    # 2149808): the per-atom displacement.
    #
    # `d_step` and `d_anchor` are measured there as **vectors** -- 99.96% /
    # 99.91% per-atom, rms 9.8e-02 / 1.5e-01 nm, i.e. naturally on the same
    # scale as the frame-time channel and only 7x below `ones`.  But a
    # displacement vector rotates with the window, so feeding its three
    # components would re-break D1 exactly as the velocity does (the symmetry
    # probe puts that at 3598x).  Their **magnitudes** are scalars, hence
    # invariant, and that is what is measured here: one channel each, which is
    # what section 4 item 1 of that note says every paper has and our contract
    # does not.
    print("\n=== 8c. what differs between frames, in nanometres ===")
    pos_f = pos[mask].double()
    nfm = pos_f.shape[0]
    adj_rmsd = [float((pos_f[i + 1] - pos_f[i]).pow(2).sum(-1).mean().sqrt())
                for i in range(nfm - 1)]
    span_rmsd = float((pos_f[-1] - pos_f[0]).pow(2).sum(-1).mean().sqrt())
    print(f"  adjacent-frame RMSD  {np.mean(adj_rmsd):.4f} nm "
          f"(min {min(adj_rmsd):.4f}, max {max(adj_rmsd):.4f})")
    print(f"  oldest-to-newest     {span_rmsd:.4f} nm over "
          f"{win.delta_t_ns * (nfm - 1):.2f} ns")
    # d_anchor: displacement from the window's oldest (anchor) frame.
    # d_step:   displacement from the previous frame.
    d_anchor = (pos_f - pos_f[0:1]).norm(dim=-1)          # [F, N] scalars
    d_step = torch.cat([torch.zeros(1, pos_f.shape[1], dtype=pos_f.dtype),
                        (pos_f[1:] - pos_f[:-1]).norm(dim=-1)])
    geom = {"adjacent_rmsd_nm": float(np.mean(adj_rmsd)),
            "span_rmsd_nm": span_rmsd, "span_ns": win.delta_t_ns * (nfm - 1)}
    print(f"\n  candidate motion features (vector vs its own magnitude):")
    # **Vector against magnitude, and this is the measurement that decides
    # whether the encoder has to be equivariant rather than merely invariant.**
    # The displacement is a type-1 quantity: under a rotation it maps to R.d.
    # Its magnitude is a scalar, hence invariant, which is why the invariance
    # requirement pushed towards |d| -- but taking the norm discards the
    # direction, and the direction is where the per-atom content is.  Both are
    # measured per frame, with atoms on the leading axis, and averaged over
    # frames; passing the frame axis into per_atom_fraction is the error the
    # assertion above now refuses.
    d_anchor_vec = pos_f - pos_f[0:1]                      # [F, N, 3]
    d_step_vec = torch.cat([torch.zeros_like(pos_f[0:1]),
                            pos_f[1:] - pos_f[:-1]])       # [F, N, 3]
    for nm, vec, mag in (("d_anchor", d_anchor_vec, d_anchor),
                         ("d_step", d_step_vec, d_step)):
        # Frame 0 is identically zero for d_anchor and for d_step, so it is
        # excluded: its per-atom fraction is undefined, not 0.
        pa_vec = float(np.mean([per_atom_fraction(vec[k].float())
                                for k in range(1, nfm)]))
        pa_mag = float(np.mean([per_atom_fraction(mag[k].unsqueeze(-1).float())
                                for k in range(1, nfm)]))
        c = cos(per_atom_part(mag[1].unsqueeze(-1).float()),
                per_atom_part(mag[-1].unsqueeze(-1).float()))
        geom[nm] = {"rms": float(mag.pow(2).mean().sqrt()),
                    "per_atom_fraction_vector": pa_vec,
                    "per_atom_fraction_magnitude": pa_mag,
                    "direction_share_lost_to_norm": pa_vec - pa_mag,
                    "cos_first_last_frame_magnitude": c}
        print(f"  {nm:<14} rms {geom[nm]['rms']:.4f} nm   "
              f"per-atom: vector {pa_vec:>7.2%}  magnitude {pa_mag:>7.2%}   "
              f"|.| cos(f1,f-1) {c:+.4f}")
    # The decisive one: does the candidate separate two *conformations* where
    # the current contract does not?  Compared on the same pair of windows
    # section 8 uses, so the numbers sit beside each other.
    w2 = _bw(samples[0], 1000, stride=10, k=args.window_k)
    p2 = collate_window(w2)["input_feature_dict"]["wp_position_nm"]
    p2 = p2[collate_window(w2)["input_feature_dict"]["wp_frame_mask"].to(torch.bool)].double()
    d2 = (p2 - p2[0:1]).norm(dim=-1)
    c_conf = cos(per_atom_part(d_anchor[-1].unsqueeze(-1).float()),
                 per_atom_part(d2[-1].unsqueeze(-1).float()))
    geom["d_anchor_cos_two_conformations"] = c_conf
    print(f"\n  |d_anchor| between the two conformations of section 8: "
          f"per-atom cos {c_conf:+.6f}")
    print(f"  the current contract reaches {content['same_molecule_two_conformations']['h']['per_atom_cos']:+.6f} at `h` "
          f"and {content['same_molecule_two_conformations']['conv0_molecular']['per_atom_cos']:+.6f} at conv0")

    # ---------------------------------------------------------- gate (2)
    print("\n=== 9. gate (2): reconstruction against the online log ===")
    recon = {"h_norm_after_layernorm": float(np.mean(h_norms)),
             "h_norm_after_layernorm_window0": float(h_norms[0])}
    gate2 = None
    if args.heldout_wp is not None and args.heldout_wp.exists():
        online = json.loads(args.heldout_wp.read_text())
        got = float(online.get("h_norm_after_layernorm", float("nan")))
        want = recon["h_norm_after_layernorm_window0"]
        gate2 = abs(got - want) < 5e-4 * max(abs(want), 1.0)
        print(f"  online {got:.4f} vs reconstructed {want:.4f} -> "
              f"{'AGREE' if gate2 else 'DISAGREE'}")
    else:
        print(f"  deferred: needs an arm's first evaluation round. Reconstructed "
              f"|h| = {recon['h_norm_after_layernorm_window0']:.4f} (window 0), "
              f"{recon['h_norm_after_layernorm']:.4f} (mean over "
              f"{args.windows}); re-run with --heldout-wp once an arm has "
              f"logged heldout_wp/h_norm_after_layernorm.")

    cap.close()

    # ----------------------------------------------------------- verdict
    gates = {
        "dense0 per-atom fraction > 10%":
            summary["dense0_molecular"]["mean"] > 0.10,
        "wp_tokens per-atom fraction > 50%":
            summary["wp_tokens"]["mean"] > 0.50,
        "gate (1) same input twice, per-atom cos > 0.999": gate1_cos > 0.999,
    }
    if gate2 is not None:
        gates["gate (2) reconstruction agrees with the online log"] = gate2

    print("\n=== 10. gates ===")
    ok = True
    for name, passed in gates.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok &= bool(passed)
    print(f"\n=== section 5.3: {'PASS' if ok else 'FAIL'} ===")
    if not ok:
        print("  The first two not passing means `h` is still mostly a "
              "cross-atom\n  constant and no training arm should start -- plan "
              "section 5.3 sends that\n  back to the contract in 5.1 to find "
              "the next constant source, it does not\n  lower the bar.")

    record = {
        "windows": args.windows, "eval_seed": args.eval_seed,
        "wp_seed": args.seed, "other_feats_channels": bridge.features.channels,
        "parameters": n_params, "branch_rms": branch_rms,
        "per_atom_fraction": summary,
        "gate1_same_input_cos": gate1_cos, "gate1_max_abs_diff": gate1_max,
        "rotation_residual_with_contract": rot_resid,
        "rotation_per_atom_cos": rot_cos,
        "rotation_residual_by_layer": rot_by_layer,
        "content": content, "frames": frames, "geometry": geom,
        "reconstruction": recon,
        "gates": gates, "pass": bool(ok),
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(record, indent=2) + "\n")
        print(f"\nwrote {args.report}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
