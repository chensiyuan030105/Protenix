"""P011 section 6 acceptance: the dt pathway must be invisible at step 0.

Same requirement, and the same reason, as kineidos/check_fusion.py: a pathway
added to a pretrained model has to start as the identity, so that any later
difference is the pathway *doing* something rather than the pathway *existing*.

But an identity check on its own is worthless here, and that is the point D8
makes: "关掉等于基线" is silent about "打开能用".  A pathway wired to nothing at
all -- `delta_t_ns` never reaching the module, a `.get` returning None forever,
an argument landing in the wrong positional slot -- passes an identity check
perfectly.  P010 section 16 has three instances of exactly that shape, each
caught by a *different* check than the one that looked relevant.

So there are four checks, in two pairs:

  1. dt present vs dt absent, same model, bit-identical.      (identity)
  2. two different dt values, bit-identical.                  (identity)
  3. with dt_linear perturbed, check 1 must now FAIL.         (the control)
  4. with dt_linear perturbed, check 2 must now FAIL.         (the control)

3 and 4 are the ones that would catch a pathway that is not connected: if
writing a non-zero dt_linear does not change the output, then nothing the
pathway computes reaches `single_s`, whatever the identity said.

Both run on one model instance rather than on two differently-seeded ones.  That
matters: building the dt modules consumes draws from the global RNG (nn.Linear's
reset_parameters runs before `initializer="zeros"` overwrites the weight), so a
dt-on and a dt-off construction cannot be compared parameter-for-parameter
unless the comparison is done this way.  DiffusionConditioning restores the RNG
state around that construction for the arms' sake, but this check does not rely
on it -- it does not need to.

Usage (from the workspace root):
    PYTHONPATH=repos/research/kineidos-v4:repos/research/wp-v3 \
    LAYERNORM_TYPE=torch ATTN_IMPL=sdpa LD_LIBRARY_PATH=$ENV/lib \
      $ENV/bin/python -m kineidos.check_dt_conditioning
"""

from __future__ import annotations

import numpy as np
import torch

from protenix.model.modules.diffusion import DiffusionConditioning

C_S, C_Z, C_S_INPUTS = 384, 128, 449
N_TOKEN, N_SAMPLE = 24, 2
FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          + (f"  -- {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def make(dt_conditioning: bool, seed: int = 0) -> DiffusionConditioning:
    # numpy as well as torch: Protenix's trunc_normal_init_ draws through
    # scipy's truncnorm.rvs, which reads numpy's global RNG, so a torch seed
    # alone does not pin these parameters (check_fusion.py's `make` says the
    # same and it is worth repeating here -- a run recorded with only a torch
    # seed does not pin its own weights).
    torch.manual_seed(seed)
    np.random.seed(seed)
    mod = DiffusionConditioning(
        c_z=C_Z, c_s=C_S, c_s_inputs=C_S_INPUTS,
        dt_conditioning=dt_conditioning,
    )
    mod.eval()
    return mod


def inputs(seed: int = 1):
    g = torch.Generator().manual_seed(seed)

    def r(*shape):
        return torch.randn(*shape, generator=g)

    return dict(
        t_hat_noise_level=torch.rand(N_SAMPLE, generator=g) * 10 + 0.1,
        relp_feature=None,
        s_inputs=r(N_TOKEN, C_S_INPUTS),
        s_trunk=r(N_TOKEN, C_S),
        z_trunk=r(N_TOKEN, N_TOKEN, C_Z),
        # A pair_z given explicitly, so `prepare_cache` -- and therefore
        # RelativePositionEncoding, which would need a whole feature dict -- is
        # not on the path.  The dt term is added to `single_s`, which this
        # branch reaches in full.
        pair_z=r(N_TOKEN, N_TOKEN, C_Z),
    )


def main() -> int:
    print("=== 1. dt_conditioning=False is byte-identical to upstream ===")
    off = make(False)
    on = make(True)
    off_names = {n for n, _ in off.named_parameters()}
    on_names = {n for n, _ in on.named_parameters()}
    added = sorted(on_names - off_names)
    check("dt_conditioning=False adds no parameter",
          not (off_names - on_names) and added,
          f"added when on: {added}")
    n_added = sum(p.numel() for n, p in on.named_parameters() if n in added)
    print(f"  the pathway is {n_added} parameters "
          f"({n_added / sum(p.numel() for p in on.parameters()):.3%} of the module)")

    print("\n=== 2. dt_linear starts at exactly zero ===")
    w = on.dt_linear.weight
    check("dt_linear.weight == 0 exactly", bool((w == 0).all()),
          f"max|w| = {float(w.detach().abs().max()):.3e}")
    check("dt_fourier uses its own seed, not the noise path's 42",
          on.dt_fourier.seed != on.fourier_embedding.seed,
          f"dt {on.dt_fourier.seed} vs noise {on.fourier_embedding.seed}")

    print("\n=== 3. step-0 identity, on one model instance ===")
    kw = inputs()
    with torch.no_grad():
        base, _ = on(**kw, delta_t_ns=None)
        with_dt, _ = on(**kw, delta_t_ns=torch.tensor(0.1))
        other_dt, _ = on(**kw, delta_t_ns=torch.tensor(1.0))
    check("dt present == dt absent, bit for bit",
          torch.equal(base, with_dt),
          f"max|d| = {float((base - with_dt).abs().max()):.3e}")
    check("dt=0.1 == dt=1.0, bit for bit",
          torch.equal(with_dt, other_dt),
          f"max|d| = {float((with_dt - other_dt).abs().max()):.3e}")

    print("\n=== 4. the control: with dt_linear non-zero, 3 must fail ===")
    print("  Without this, a pathway that is wired to nothing would pass "
          "section 3\n  perfectly -- which is the whole of D8's point, and "
          "three of P010's\n  section-16 failures.")
    with torch.no_grad():
        on.dt_linear.weight.normal_(0.0, 0.02)
        pert_none, _ = on(**kw, delta_t_ns=None)
        pert_01, _ = on(**kw, delta_t_ns=torch.tensor(0.1))
        pert_10, _ = on(**kw, delta_t_ns=torch.tensor(1.0))
    d_present = float((pert_none - pert_01).abs().max())
    d_value = float((pert_01 - pert_10).abs().max())
    check("perturbed: dt present now differs from dt absent",
          d_present > 1e-6, f"max|d| = {d_present:.3e}")
    check("perturbed: dt=0.1 now differs from dt=1.0",
          d_value > 1e-6, f"max|d| = {d_value:.3e}")
    check("perturbed: dt absent still equals the unperturbed run "
          "(nothing leaks when dt is None)",
          torch.equal(pert_none, base),
          f"max|d| = {float((pert_none - base).abs().max()):.3e}")

    print("\n=== 5. log10 is applied, and a non-positive dt is refused ===")
    on2 = make(True)
    with torch.no_grad():
        # A *random* weight, not `fill_(1/256)`.  The uniform fill made this
        # section pass for the wrong reason: dt_layernorm has
        # create_offset=False, so its output has zero mean over the 256
        # channels, and a uniform weight sums it to ~0.  The first assertion
        # below then read max|d| = 1.5e-08 -- float noise -- and would have
        # passed just as well with the pathway disconnected.
        on2.dt_linear.weight.normal_(0.0, 0.05, generator=torch.Generator().manual_seed(7))
        e01 = on2.dt_embedding(torch.tensor(0.1))
        e10 = on2.dt_embedding(torch.tensor(1.0))
    # log10(0.1) = -1 and log10(1.0) = 0, so the two embeddings must differ;
    # and the one at dt = 1 ns is the Fourier features of exactly 0.
    d = float((e01 - e10).abs().max())
    check("dt_embedding(0.1) differs from dt_embedding(1.0) by more than noise",
          d > 1e-3, f"max|d| = {d:.3e}")
    zero_in = torch.cos(2 * torch.pi * on2.dt_fourier.b)
    ref = on2.dt_linear(on2.dt_layernorm(zero_in))
    check("dt_embedding(1.0) is the embedding of log10 = 0",
          torch.allclose(e10, ref, atol=1e-6),
          f"max|d| = {float((e10 - ref).abs().max()):.3e}")
    for bad in (0.0, -1.0):
        try:
            on2.dt_embedding(torch.tensor(bad))
            check(f"delta_t_ns={bad} is refused", False, "no error raised")
        except ValueError:
            check(f"delta_t_ns={bad} is refused", True)

    print(f"\n=== section 6: {'PASS' if not FAILS else 'FAIL'} ===")
    if FAILS:
        for f in FAILS:
            print(f"  failed: {f}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())
