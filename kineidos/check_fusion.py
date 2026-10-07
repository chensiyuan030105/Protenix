"""P004.3 acceptance: the fusion point must be invisible at initialisation.

The hard requirement is that a model carrying the fusion produces bit-identical
output to the unmodified Protenix before training moves any weight.  That is
what makes the `none` and `zero` arms of the ablation meaningful: any difference
later is the fusion doing something, not the fusion existing.

Checked against a real control built in the same process -- two
AtomAttentionEncoders, one with wp_token_dim=None and one with 768, fed the same
tensors under the same seed -- rather than against remembered numbers.  A
regression here is a change in behaviour, and only a side-by-side run can say
so.

Usage (from the workspace root):
    PYTHONPATH=repos/research/kineidos-v3:repos/research/wp-v2 \
    LAYERNORM_TYPE=torch ATTN_IMPL=sdpa \
    LD_LIBRARY_PATH=/mnt/xfs/home/mhg/anaconda3/envs/kineidos-v2-slurm/lib \
      .../envs/kineidos-v2-slurm/bin/python -m kineidos.check_fusion
"""

from __future__ import annotations

import sys

import numpy as np
import torch

from protenix.model.modules.transformer import AtomAttentionEncoder

WP_DIM = 768          # cconv_embedding_dim 384 x factor 2 (no obstacle branch)
C_ATOM = 128
FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def make(wp_token_dim, seed: int = 0):
    """An encoder, seeded as far as seeding goes here.

    Both generators are seeded because Protenix's initialisers do not all use
    torch: trunc_normal_init_ (triangular/layers.py) draws through scipy's
    truncnorm.rvs, which reads numpy's global RNG, so torch.manual_seed alone
    leaves parameter initialisation unreproducible.  Worth knowing beyond this
    test -- a run recorded with only a torch seed does not pin its own weights.

    Even so, the step-0 comparison does not rely on two constructions agreeing;
    see align_shared().
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    enc = AtomAttentionEncoder(
        has_coords=True, c_token=768, c_atom=C_ATOM, c_atompair=16,
        c_s=384, c_z=128, n_blocks=1, n_heads=4, n_queries=32, n_keys=128,
        wp_token_dim=wp_token_dim,
    )
    enc.eval()
    return enc


def align_shared(src, dst) -> None:
    """Copy every parameter and buffer `dst` shares with `src`.

    The step-0 check compares two models that must differ *only* by the fusion.
    Relying on two constructions to coincide would make it a test of RNG
    determinism instead -- and that test would fail, because Protenix's
    trunc_normal_init_ draws through scipy's truncnorm.rvs, which reads numpy's
    global RNG rather than torch's.  Copying states the intent directly and
    holds whatever the initialisers do.
    """
    with torch.no_grad():
        src_p = dict(src.named_parameters())
        for n, v in dst.named_parameters():
            if n in src_p:
                v.copy_(src_p[n])
        src_b = dict(src.named_buffers())
        for n, v in dst.named_buffers():
            if n in src_b:
                v.copy_(src_b[n])


def real_batch(n_sample: int = 2, seed: int = 1):
    """One real GAGU window through Protenix's own feature pipeline.

    Synthetic tensors were tried first and are not worth the trouble: d_lm,
    v_lm and pad_info have shapes that depend on the n_queries/n_keys
    windowing, so hand-built ones test the fusion against inputs the model
    never sees.  Real features also exercise the diffusion module's two call
    branches, including the checkpointed one that passes arguments positionally.
    """
    from pathlib import Path
    from kineidos.data.gagu import GAGUProtenixAdapter
    from kineidos.data.windows import build_window

    sample_dir = Path(
        "/mnt/xfs/home/mhg/Projects/ForSiyuan/RNA-WorldParticle-Workspace/datasets/"
        "processed/gagu_internal_loop_v0_1/gagu_100mM_K_agaguu_startI_r1")
    sample = GAGUProtenixAdapter(sample_dir).load()
    w = build_window(sample, target_frame=5000, stride=100, k=8)
    n_atom = w.wp_position_nm.shape[1]

    # The minimal collate P004.2 deliberately left out: batch, then let
    # Protenix compute d_lm / v_lm / pad_info (Algorithm 5 lines 1-3) and the
    # relative-position encoding.  These want the batch dimension, which is why
    # they belong here and not in build_window.
    from protenix.model.protenix import update_input_feature_dict
    from protenix.model.modules.embedders import RelativePositionEncoding

    feats = {}
    for k, v in w.features.items():
        if torch.is_tensor(v):
            feats[k] = v.clone() if k == "atom_to_token_idx" else v.unsqueeze(0).clone()
        else:
            feats[k] = v
    feats = update_input_feature_dict(feats)
    feats = RelativePositionEncoding(c_z=128).generate_relp(feats)

    g = torch.Generator().manual_seed(seed)
    return sample, w, feats, torch.randn(n_atom, WP_DIM, generator=g)


def main() -> int:
    print("=== 1. the fusion is off by default ===")
    plain = make(None)
    names = {n for n, _ in plain.named_parameters()}
    check("wp_token_dim=None creates no parameters",
          not any(n.startswith("wp_") for n in names),
          f"{len(names)} parameters, none named wp_*")

    fused = make(WP_DIM)
    fnames = {n for n, _ in fused.named_parameters()}
    added = sorted(fnames - names)
    print(f"  fusion adds: {added}")
    check("fusion adds exactly the expected parameters",
          set(added) == {"wp_layernorm.weight", "wp_layernorm.bias",
                         "wp_fusion.weight"},
          str(added))
    # Names alone are not enough.  Constructing the fusion draws from the
    # global RNG, so if those modules are built before the shared ones, every
    # shared weight differs while the name sets still match -- which is exactly
    # how the first version of this passed while the step-0 comparison failed.
    align_shared(plain, fused)
    shared_plain = dict(plain.named_parameters())
    shared_fused = {n: v for n, v in fused.named_parameters() if n in names}
    same_values = set(shared_fused) == names and all(
        torch.equal(shared_plain[n], shared_fused[n]) for n in names)
    check("shared parameters are identical in value, not just in name",
          same_values, f"{len(names)} tensors copied from the control")

    print("\n=== 2. block initialisation ===")
    w = fused.wp_fusion.weight                       # [c_atom, c_atom + wp_dim]
    check("fusion weight shape", tuple(w.shape) == (C_ATOM, C_ATOM + WP_DIM),
          str(tuple(w.shape)))
    check("c_l half is the identity",
          torch.equal(w[:, :C_ATOM], torch.eye(C_ATOM)),
          f"max |W_cl - I| = {(w[:, :C_ATOM] - torch.eye(C_ATOM)).abs().max():.3e}")
    check("h half is exactly zero", torch.equal(w[:, C_ATOM:],
                                                torch.zeros(C_ATOM, WP_DIM)),
          f"max |W_h| = {w[:, C_ATOM:].abs().max():.3e}")
    ln = fused.wp_layernorm
    check("LayerNorm starts at gamma=1, beta=0",
          torch.equal(ln.weight, torch.ones(WP_DIM))
          and torch.equal(ln.bias, torch.zeros(WP_DIM)))

    print("\n=== 3. step 0 is bit-identical to unmodified Protenix ===")
    sample, w, feats, h = real_batch()
    n_atom = w.wp_position_nm.shape[1]
    print(f"  real GAGU window: {n_atom} atoms, "
          f"{feats['restype'].shape[-2]} tokens")

    gg = torch.Generator().manual_seed(2)
    kw = dict(
        atom_to_token_idx=feats["atom_to_token_idx"],
        ref_pos=feats["ref_pos"], ref_charge=feats["ref_charge"],
        ref_mask=feats["ref_mask"],
        ref_atom_name_chars=feats["ref_atom_name_chars"],
        ref_element=feats["ref_element"],
        d_lm=feats["d_lm"], v_lm=feats["v_lm"], pad_info=feats["pad_info"],
        r_l=torch.randn(1, 2, n_atom, 3, generator=gg),
        s=torch.randn(1, feats["restype"].shape[-2], 384, generator=gg),
        z=torch.randn(1, feats["restype"].shape[-2],
                      feats["restype"].shape[-2], 128, generator=gg),
    )
    with torch.no_grad():
        ref_out = plain(**kw)
        fused_out = fused(**kw, wp_tokens=h)
        fused_no_h = fused(**kw)
    labels = ("a_token", "q_l", "c_l", "p_lm")
    for i, lab in enumerate(labels):
        same = torch.equal(ref_out[i], fused_out[i])
        d = (ref_out[i] - fused_out[i]).abs().max().item()
        check(f"{lab} identical with h injected", same, f"max |diff| = {d:.3e}")
    for i, lab in enumerate(labels):
        check(f"{lab} identical with h absent",
              torch.equal(ref_out[i], fused_no_h[i]))

    print("\n=== 4. and it is not identical once the weights move ===")
    # A zero-initialised branch that stayed zero under any weights would mean
    # the path is dead, which is how v0.1 failed. Perturbing W_h must change the
    # output, or the fusion is decoration.
    with torch.no_grad():
        fused.wp_fusion.weight[:, C_ATOM:].normal_(0.0, 0.01)
    with torch.no_grad():
        moved = fused(**kw, wp_tokens=h)
    d = (ref_out[2] - moved[2]).abs().max().item()
    check("perturbing W_h changes c_l", d > 1e-6, f"max |diff| = {d:.3e}")

    print("\n=== 5. gradient reaches both halves and h itself ===")
    fused2 = make(WP_DIM)
    align_shared(plain, fused2)
    h2 = h.clone().requires_grad_(True)
    out = fused2(**kw, wp_tokens=h2)
    out[0].sum().backward()
    gw = fused2.wp_fusion.weight.grad
    check("W_h receives gradient at step 0",
          gw is not None and gw[:, C_ATOM:].abs().max().item() > 0,
          f"max |dL/dW_h| = {gw[:, C_ATOM:].abs().max():.3e}")
    # h's own gradient is zero at step 0 and must be: W_h is zero, so dL/dh =
    # W_h^T . dL/dc_l' = 0.  That is the intended order -- W_h moves first, and
    # h starts receiving gradient on the next step.  A non-zero value here would
    # mean the zero-init had not taken.
    check("dL/dh is zero at step 0, as zero-init requires",
          h2.grad is not None and h2.grad.abs().max().item() == 0.0,
          f"max |dL/dh| = {h2.grad.abs().max():.3e}")

    print("\n=== 6. shape mismatches are refused, not broadcast ===")
    for bad, why in ((torch.randn(n_atom // 2, WP_DIM), "wrong atom count"),
                     (torch.randn(n_atom, 256), "wrong token width")):
        try:
            fused(**kw, wp_tokens=bad)
            check(f"refuses {why}", False, "it did not raise")
        except ValueError as exc:
            check(f"refuses {why}", True, str(exc)[:58])

    print("\n" + "=" * 62)
    print("P004.3 验收:", "全部通过" if not FAILS else f"{len(FAILS)} 项失败 -> {FAILS}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())
