#!/usr/bin/env python
"""P010's acceptance for everything that can be checked without a GPU.

Three of the four things this file tests have the same failure mode: they are
switches and defaults, and a wrong one does not raise -- it runs the wrong
experiment under the right name.  `kineidos/train/check_modes.py` and
`check_loss_switches.py` exist for that reason and this is the same kind of
file, extended per branch: the integration branch checks the per-sigma keys
and the round schedule, and each test branch adds its own layer to the bottom.

What is deliberately NOT here: anything that builds a model, loads a
checkpoint, reads a trajectory or touches a GPU.  Those go in
p010_sigma_grid.sbatch and p010_identity.sbatch, which ask for a GPU and get
one.  This file is config parsing and arithmetic, so it fits in the same CPU
job as the readout -- and it is still a slurm job, because the rule
(AGENTS.md, P006 D19) has no size qualifier and "it is only a parse" is how
things end up on the login node.

    sbatch --export=ALL,MODE=check \
      repos/research/kineidos-v3-diag/kineidos/slurm/p010_readout.sbatch
"""

from __future__ import annotations

import sys
from typing import Any

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


class FakeTrainer:
    """Just enough of a trainer for sigma_grid_due, which is pure arithmetic.

    A real trainer would need a GPU and a checkpoint to answer a question about
    integer division.  The method is bound off the class so that what is tested
    is the code the arms run, not a copy of its logic living here -- a test
    that reimplements the rule it is checking passes forever.
    """

    def __init__(self, step: int, *, eval_interval: int, max_steps: int,
                 sigma_grid_interval: int) -> None:
        from kineidos.train.trainer import KineidosTrainer

        self.step = step
        self.configs = _ns(
            eval_interval=eval_interval,
            max_steps=max_steps,
            kineidos=_ns(sigma_grid_interval=sigma_grid_interval),
        )
        self.sigma_grid_due = KineidosTrainer.sigma_grid_due.__get__(self)


class _ns:
    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)


def eval_rounds(max_steps: int, eval_interval: int) -> list[int]:
    """The steps `run()` evaluates on: (step + 1) % eval_interval == 0, plus
    the last step, which it evaluates whatever the interval says
    (runner/train.py:710)."""
    rounds = [s for s in range(max_steps) if (s + 1) % eval_interval == 0]
    if max_steps - 1 not in rounds:
        rounds.append(max_steps - 1)
    return rounds


def check_no_empty_args() -> None:
    """No `--key ""` in any P010 submission script.

    An empty value cannot survive protenix's command line.  parse_sys_args
    builds the argument string as `f"{k} {v} "` and parse_configs splits it on
    whitespace, so an empty v leaves its key with nothing after it and shifts
    every later pair by one.  The symptom is argparse's "expected one
    argument", ten seconds in.

    A one-second static check because the cost is not the ten seconds.  On
    2026-10-08 job 2149587 died this way *after* the 33-minute acceptance it
    depended on had passed, and took four queued arms with it as
    DependencyNeverSatisfied -- so the price of one empty string was the whole
    chain's latency plus a resubmission.  The right place to catch an argument
    that can never work is before anything is submitted.

    The value to pass instead is no flag at all: every config key has a
    default, and for the keys where "" is the meaningful value -- resume_dir,
    sigma_grid_out, sigma_grid_arm -- "" *is* the default.
    """
    import re
    from pathlib import Path

    slurm = Path(__file__).resolve().parent / "slurm"
    bad: list[str] = []
    scanned = 0
    # `--key ""` and `--key ''`, with the quotes adjacent: a quoted empty
    # string is the only way to write this that bash will pass through as an
    # empty argv entry.
    pattern = re.compile(r"(--[A-Za-z0-9_.-]+)\s+([\"']{2})(?=\s|\\|$)")
    for path in sorted(slurm.glob("p010_*")):
        text = path.read_text()
        scanned += 1
        for line in text.splitlines():
            stripped = line.strip()
            # Usage comments carry example command lines; they are
            # documentation, not what runs.
            if stripped.startswith("#"):
                continue
            for key, _ in pattern.findall(line):
                bad.append(f"{path.name}: {key} \"\"")
    check(f"{scanned} P010 scripts pass no empty values", not bad,
          "; ".join(bad) if bad else
          "an empty value shifts every later --key/value pair by one")


def check_optimizer_leaves_idle_params_alone(*, cfg_root: Any) -> None:
    """A parameter with no data gradient must keep its value.  D16.

    This tests the *property*, not the setting, and that is the whole point:
    `weight_decay = 0` is today's fix, but `eps`, the optimizer class, or a
    future `use_adamw` would each change the answer, and a test that asserted
    "weight_decay is 0" would pass while the artefact came back by another
    route.

    What it is defending against (P009 section 8 item 8).  With plain Adam,
    weight decay is coupled -- added to the gradient and then normalised by
    Adam.  With `wd` and `eps` both 1e-8, a parameter whose data gradient is
    ~0 gets

        update = lr * (wd*theta)/(wd*theta + eps) = lr * theta/(theta + 1)

    and `wd` cancels out, so for theta << 1 it is exponential decay at rate =
    lr regardless of how small wd is.  Measured with these hyperparameters:
    theta goes 1 -> 0.604 -> 0.068 -> 0.000299 over 500 / 2000 / 5000 steps.

    The victim class is narrow -- non-zero initial value *and* a data gradient
    below ~1e-8*theta -- and in this codebase it is exactly
    wp_layernorm.weight, initialised to 1.  Which is the gate of the fusion.
    The experiment this project keeps running is "add a gated side branch and
    see whether the model uses it", so an optimizer that closes such a gate
    whatever the data does is aimed squarely at the quantity being read.
    P009 nearly concluded "the optimizer is refusing the history" from it.
    """
    import torch

    adam = cfg_root.adam
    lr = float(cfg_root.lr)
    steps = 2000
    # The real optimizer, built the way protenix builds it, on one scalar
    # standing in for wp_layernorm.weight: value 1, gradient exactly 0.
    theta = torch.tensor([1.0], requires_grad=True)
    Opt = torch.optim.AdamW if adam.use_adamw else torch.optim.Adam
    opt = Opt([theta], lr=lr, betas=(float(adam.beta1), float(adam.beta2)),
              weight_decay=float(adam.weight_decay))
    for _ in range(steps):
        opt.zero_grad()
        theta.grad = torch.zeros_like(theta)
        opt.step()
    moved = abs(theta.item() - 1.0)
    check(f"a gamma-like parameter survives {steps} zero-gradient steps",
          moved < 1e-6,
          f"it is now {theta.item():.6f} (moved {moved:.2e}); "
          f"optimizer={Opt.__name__} lr={lr} wd={adam.weight_decay} -- the "
          f"fusion's gate would be closed by arithmetic rather than by data, "
          f"and that value is what sections 6.3 reads")

    # And the complement: a parameter *with* a real gradient must still move,
    # so the fix cannot have been "turn the optimizer off".
    theta2 = torch.tensor([1.0], requires_grad=True)
    opt2 = Opt([theta2], lr=lr, betas=(float(adam.beta1), float(adam.beta2)),
               weight_decay=float(adam.weight_decay))
    for _ in range(100):
        opt2.zero_grad()
        theta2.grad = torch.full_like(theta2, 1e-4)
        opt2.step()
    check("while a parameter with a real gradient still moves",
          abs(theta2.item() - 1.0) > 1e-3,
          f"it is now {theta2.item():.6f} after 100 steps at grad=1e-4")


def check_readout_on_fixture() -> None:
    """Drive kineidos.read_sigma_grid over four tiny arms.

    Written because two readout jobs died on NameError -- a name that lives in
    score_sigma_grid being used in read_sigma_grid -- after loading 45,000 rows
    each.  Nothing about those failures needed real data to find; they needed
    the code path to be executed at all.  Four arms of two windows exercise
    every table the real readout prints, in about a second.

    A fixture rather than the real files, so this says nothing about the
    numbers and cannot start passing or failing because an arm was rerun.
    """
    import json
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    from kineidos.score_sigma_grid import (band_of, c_skip_of, edm_scale_of,
                                           sigma_grid)

    tmp = Path(tempfile.mkdtemp(prefix="p010_check_readout_"))
    grid = sigma_grid()
    arms = {"p009_random": 1.02, "p009_zero": 1.00,
            "p009_zero_seed2": 1.01, "p009_none": 0.99}
    # Two windows, one in dt bin 1 and one in bin 3, so the dt-split table has
    # something in more than one row.
    windows = [(0, 1, 0.1), (1, 8, 0.8)]
    for arm, scale in arms.items():
        path = tmp / f"{arm}_step2999.rank0.jsonl"
        with open(path, "w") as handle:
            for wid, stride, dt in windows:
                for sig in grid:
                    for noise in range(2):
                        base = (1.0 + sig / 16.0) * (1.0 + 0.1 * noise)
                        handle.write(json.dumps({
                            "arm": arm, "step": 2999, "window_id": wid,
                            "sample_id": "gagu_fixture_r4",
                            "target_frame": 1000 + wid, "stride": stride,
                            "delta_t_ns": dt, "sigma": sig, "noise_idx": noise,
                            "mse_aligned": base * scale,
                            "smooth_lddt": 0.3 * base * scale,
                            "loss_unweighted": 2.0 * base * scale,
                            "loss_edm_weighted": 3.0 * base * scale,
                            "c_skip": c_skip_of(sig),
                            "sigma_band": band_of(sig),
                            "edm_scale": edm_scale_of(sig),
                            "terms": {"mse_loss": base * scale,
                                      "smooth_lddt_loss": 0.3 * base * scale,
                                      "bond_loss": 0.0,
                                      "loss": 3.0 * base * scale},
                        }, sort_keys=True) + "\n")
    out = tmp / "readout.md"
    proc = subprocess.run(
        [sys.executable, "-m", "kineidos.read_sigma_grid",
         "--runs", str(tmp), "--step", "2999", "--out", str(out)],
        capture_output=True, text=True)
    check("the readout runs to completion", proc.returncode == 0,
          (proc.stderr.strip().splitlines() or ["ok"])[-1][:120])
    text = out.read_text() if out.is_file() else ""
    for want in ("第 1 条", "逐 σ", "三箱 Δt", "zero − none"):
        check(f"and its report contains {want!r}", want in text,
              f"{len(text)} bytes written")
    # The fixture makes random uniformly 2% worse than zero at every sigma, so
    # the registered verdict must be the capacity branch -- same sign, same
    # magnitude in all three bands.  That checks the verdict logic, not the
    # data.
    check("a uniform effect reads as the capacity branch",
          "读到的是容量" in text,
          "the fixture is random = 1.02 x zero at every sigma, which is "
          "section 6 item 1's second branch by construction")

    # And the truncation guard, since that is the other thing a readout must
    # never do silently.
    short = tmp / "p009_none_step2999.rank0.jsonl"
    lines = short.read_text().splitlines()
    short.write_text("\n".join(lines[:-3]) + "\n")
    proc = subprocess.run(
        [sys.executable, "-m", "kineidos.read_sigma_grid",
         "--runs", str(tmp), "--step", "2999", "--out", str(out)],
        capture_output=True, text=True)
    check("a truncated arm is refused",
          proc.returncode != 0 and "killed part way" in (proc.stdout + proc.stderr),
          (proc.stdout + proc.stderr).strip().splitlines()[-1][:100]
          if (proc.stdout + proc.stderr).strip() else "no output")


def main() -> int:
    print("=== 1. the per-sigma keys exist, and default to today's behaviour ===")
    from kineidos.train.check_trainer import make_configs

    import tempfile
    cfg = make_configs("random", tempfile.mkdtemp(prefix="p010_check_")).kineidos
    for key, want, why in (
        ("sigma_grid", False,
         "on by default would make P009's protocol cost a second pass"),
        ("sigma_grid_noise", 4, "section 4 sizes the grid at 4 per sigma"),
        ("sigma_grid_interval", 0, "0 = every eval round = the old behaviour"),
        ("sigma_grid_windows", 0, "0 = all of them, which section 4 fixes"),
        ("sigma_grid_arm", "", "empty = derive it from the run"),
        ("sigma_grid_out", "", "empty = run_dir/sigma_grid"),
        ("score_checkpoint", "", "only the scorer reads it"),
    ):
        got = getattr(cfg, key, "<<missing>>")
        check(f"kineidos.{key} defaults to {want!r}", got == want,
              f"got {got!r}; {why}")

    print("\n=== 2. the sigma grid runs on the rounds it says it will ===")
    # 2000 steps, held-out every 250, grid every 500: the grid must land on
    # 499/999/1499/1999 and nowhere else.  Those four are what section 6 item 3
    # reads arm B's release against, so an off-by-one round here moves the
    # release out of the window being read.
    rounds = eval_rounds(2000, 250)
    check("eval rounds are where runner/train.py puts them",
          rounds == [249, 499, 749, 999, 1249, 1499, 1749, 1999], str(rounds))
    due = [s for s in rounds
           if FakeTrainer(s, eval_interval=250, max_steps=2000,
                          sigma_grid_interval=500).sigma_grid_due()]
    check("interval 500 over eval_interval 250 gives every other round",
          due == [499, 999, 1499, 1999], str(due))
    all_due = [s for s in rounds
               if FakeTrainer(s, eval_interval=250, max_steps=2000,
                              sigma_grid_interval=0).sigma_grid_due()]
    check("interval 0 gives every round, as before the key existed",
          all_due == rounds, f"{len(all_due)} of {len(rounds)}")
    # The last step is always included: run() evaluates there regardless, and
    # that round is the arm's final state -- the one every table quotes.
    last = FakeTrainer(1999, eval_interval=250, max_steps=2000,
                       sigma_grid_interval=1000).sigma_grid_due()
    check("the final round is always included", last,
          "run() evaluates the last step whatever the interval says, and that "
          "is the state the readout quotes")

    print("\n=== 3. the sigma grid is the grid section 4 registered ===")
    from kineidos.score_sigma_grid import (band_of, c_skip_of, edm_scale_of,
                                           sigma_grid)

    grid = sigma_grid()
    check("eleven sigmas", len(grid) == 11, str(len(grid)))
    want = [0.24, 0.51, 1.08, 2.28, 4.82, 5.33, 10.20, 21.60, 24.44, 45.72,
            96.79]
    check("and they are section 4's eleven",
          all(abs(a - b) < 0.01 for a, b in zip(grid, want)),
          ", ".join(f"{s:.2f}" for s in grid))
    bands = [band_of(s) for s in grid]
    check("two of them are in the high band (c_skip < 0.3)",
          bands.count("high") == 2,
          f"{bands.count('low')} low / {bands.count('mid')} mid / "
          f"{bands.count('high')} high")
    # The identity the two loss columns are derived through.  If this drifts,
    # mse_aligned and loss_unweighted are being divided by the wrong factor and
    # nothing in the output would say so.
    s = 10.0
    check("edm_scale is loss.py:1638's factor",
          abs(edm_scale_of(s) - (s**2 + 16.0**2) / (16.0 * s) ** 2) < 1e-12)
    check("c_skip is diffusion.py:602's",
          abs(c_skip_of(s) - 1.0 / (1.0 + (s / 16.0) ** 2)) < 1e-12)

    print("\n=== 4. read_heldout refuses a leaked arm ===")
    import json
    from pathlib import Path

    from kineidos.read_heldout import refuse_oracle

    tmp = Path(tempfile.mkdtemp(prefix="p010_check_oracle_"))
    named = tmp / "p009_oracle_20261008_120000"
    named.mkdir()
    relabelled = tmp / "p009_random_20261008_120000"
    relabelled.mkdir()
    (relabelled / "env.lock").write_text(json.dumps({"wp": {"mode": "oracle"}}))
    clean = tmp / "p009_random_20261008_130000"
    clean.mkdir()
    (clean / "env.lock").write_text(json.dumps({"wp": {"mode": "random"}}))
    for name, path in (("by directory name", named),
                       ("by env.lock's wp.mode", relabelled)):
        try:
            refuse_oracle(path)
            check(f"refused {name}", False, "it was accepted")
        except SystemExit as exc:
            check(f"refused {name}", "oracle" in str(exc).lower(),
                  str(exc)[:60])
    try:
        refuse_oracle(clean)
        check("and a real arm still passes", True)
    except SystemExit as exc:
        check("and a real arm still passes", False, str(exc)[:80])

    print("\n=== 4b. a zero-gradient parameter is not destroyed by the optimizer ===")
    check_optimizer_leaves_idle_params_alone(cfg_root=make_configs(
        "random", tempfile.mkdtemp(prefix="p010_check_adam_")))

    print("\n=== 5. no sbatch passes an empty value on the command line ===")
    check_no_empty_args()

    print("\n=== 6. the readout runs end to end on a synthetic fixture ===")
    check_readout_on_fixture()
    print("\n=== 7. the oracle branch's own layer (D2) ===")
    check_oracle_layer(cfg_root=make_configs(
        "random", tempfile.mkdtemp(prefix="p010_check_oracle_cfg_")))

    print("\n" + "=" * 62)
    print("验收:", "全部通过" if not FAILS else f"{len(FAILS)} 项失败 -> {FAILS}")
    return 0 if not FAILS else 1


def check_oracle_layer(*, cfg_root: Any) -> None:
    """research/kineidos-v3-diag-oracle's half, which is never merged.

    Everything here is CPU and seconds: a 3x768 matrix, a frame index, and the
    guard raising.  What it cannot check is that the probe is *off* by default
    in a real window -- that needs a trajectory, and p010_identity.sbatch is
    where it is checked, bit for bit against p010-base.
    """
    import hashlib

    import torch

    from kineidos.data import windows
    from kineidos.train.trainer import wp_token_dim_for
    from kineidos.wp_bridge import (MODES, ORACLE_TARGET_KEY,
                                    WorldParticleBridge)

    wp = cfg_root.wp
    for key, want, why in (
        ("oracle_source", "",
         "a default would decide for the reader which experiment ran"),
        ("oracle_seed", 20261008, "this number plus the sha256 reproduces h"),
        ("target_augmentation", True, "True is current behaviour"),
    ):
        got = getattr(wp, key, "<<missing>>")
        check(f"wp.{key} defaults to {want!r}", got == want,
              f"got {got!r}; {why}")

    check("'oracle' is a mode", "oracle" in MODES, str(MODES))
    check("and it gets the fusion's width",
          wp_token_dim_for("oracle") == 768, str(wp_token_dim_for("oracle")))

    # D2 item 6: the negative control is not optional, so the mode refuses to
    # be built without saying which of the two it is.
    try:
        WorldParticleBridge("oracle")
        check("oracle without a source is refused", False, "it was accepted")
    except ValueError as exc:
        check("oracle without a source is refused",
              "decoy" in str(exc), str(exc)[:70])

    # D2 item 7: the projection is generated, so the seed is what reproduces it
    # and the digest is what proves the seed did.
    a = WorldParticleBridge("oracle", oracle_source="target", oracle_seed=7)
    b = WorldParticleBridge("oracle", oracle_source="decoy", oracle_seed=7)
    c = WorldParticleBridge("oracle", oracle_source="target", oracle_seed=8)
    check("the projection is 3 -> 768",
          tuple(a.oracle_projection.shape) == (3, 768),
          str(tuple(a.oracle_projection.shape)))
    check("the same seed gives the same projection",
          torch.equal(a.oracle_projection, b.oracle_projection))
    check("a different seed gives a different one",
          not torch.equal(a.oracle_projection, c.oracle_projection))
    prov = a.oracle_provenance()
    digest = hashlib.sha256(
        a.oracle_projection.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()
    check("oracle_provenance's digest is of that matrix",
          prov.get("oracle_projection_sha256") == digest,
          str(prov.get("oracle_projection_sha256"))[:16])
    check("and it carries the seed and the source",
          prov.get("oracle_seed") == 7 and "oracle_source" in prov, str(prov))
    # It is a buffer, so freeze_trunk cannot make it trainable and the arm
    # cannot reshape the answer -- only transport it.
    check("the projection is a buffer, not a parameter",
          not any(p_.shape == a.oracle_projection.shape
                  for p_ in a.parameters()),
          f"{len(list(a.parameters()))} parameters on an oracle bridge")

    # D2 item 2, the guard that makes item 5 structural: every other mode
    # raises on the mere presence of the key.
    guarded = WorldParticleBridge("zero")
    try:
        guarded({ORACLE_TARGET_KEY: torch.zeros(4, 3)})
        check("a non-oracle mode refuses the oracle key", False,
              "it was accepted")
    except KeyError as exc:
        check("a non-oracle mode refuses the oracle key",
              "answer" in str(exc), str(exc)[:70])

    # The positive test, and the gap it fills is specific: section 6 item 5b
    # runs this branch in `random` mode, so none of the oracle code executes
    # there.  "Off equals base" cannot say anything about "on works", and the
    # arms were the oracle path's first real execution -- which is how a dtype
    # mismatch got three minutes into two of them.
    #
    # Under autocast, because that is the condition that broke it: the bridge
    # is called inside autocast(bfloat16) while its output is consumed with
    # autocast disabled, so an h that inherits the ambient dtype is wrong by
    # construction.
    feats_ok = {
        "wp_position_nm": torch.zeros(8, 5, 3),
        "wp_velocity_nm_per_ps": torch.zeros(8, 5, 3),
        "wp_frame_time_ns": torch.zeros(8),
        "wp_frame_mask": torch.ones(8, dtype=torch.bool),
        "wp_ref_pos_nm": torch.zeros(5, 3),
        "wp_canonicalized": torch.tensor(False),
        ORACLE_TARGET_KEY: torch.zeros(5, 3),
    }
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        h = a(feats_ok)
    check("oracle h is [N, 768]", tuple(h.shape) == (5, 768), str(tuple(h.shape)))
    check("and float32 even under autocast(bfloat16)",
          h.dtype == torch.float32,
          f"got {h.dtype}; the fusion's weights are float32 and the diffusion "
          f"head runs with autocast disabled (skip_amp), so a bf16 h fails in "
          f"_fuse_wp_tokens -- which is what killed jobs 2149800/2149801")

    # And the oracle mode refuses to run without it, rather than silently
    # falling back to something.
    try:
        a({"wp_position_nm": torch.zeros(8, 4, 3),
           "wp_velocity_nm_per_ps": torch.zeros(8, 4, 3),
           "wp_frame_time_ns": torch.zeros(8),
           "wp_frame_mask": torch.ones(8, dtype=torch.bool),
           "wp_ref_pos_nm": torch.zeros(4, 3),
           "wp_canonicalized": torch.tensor(False)})
        check("oracle mode refuses to run without the key", False,
              "it was accepted")
    except KeyError as exc:
        check("oracle mode refuses to run without the key",
              ORACLE_TARGET_KEY in str(exc), str(exc)[:70])

    print("  (the constructor set windows.oracle_source() = "
          f"{windows.oracle_source()!r}; resetting it)")
    windows.set_oracle_source("")
    try:
        windows.set_oracle_source("answer")
        check("an unknown oracle source is refused", False, "it was accepted")
    except ValueError as exc:
        check("an unknown oracle source is refused",
              "target" in str(exc), str(exc)[:60])

    # The decoy's gap and its determinism.  GAGU is 10000 frames at 0.1 ns, so
    # 50 ns is 500 frames and every target has somewhere to go.
    sample = _ns(sample_id="gagu_fake_r4", frame_interval_ns=0.1,
                 n_frames=10000)
    gaps = []
    for target in (0, 250, 5000, 9999):
        f = windows.decoy_frame_for(sample, target, 4)
        gaps.append(abs(f - target) * sample.frame_interval_ns)
        again = windows.decoy_frame_for(sample, target, 4)
        if f != again:
            check(f"decoy for target {target} is deterministic", False,
                  f"{f} then {again}")
    check("every decoy is at least 50 ns from its target",
          all(g >= 50.0 for g in gaps),
          ", ".join(f"{g:.0f} ns" for g in gaps))
    check("the decoy is deterministic in (sample, target, stride)", True,
          "crc32, not hash -- PYTHONHASHSEED is per process")
    check("a different stride gives a different decoy",
          windows.decoy_frame_for(sample, 5000, 4)
          != windows.decoy_frame_for(sample, 5000, 40),
          "two windows ending on the same frame are different examples")
    # A trajectory too short to have anywhere to put the decoy must say so,
    # not quietly pick a correlated frame.
    short = _ns(sample_id="short", frame_interval_ns=0.1, n_frames=300)
    try:
        windows.decoy_frame_for(short, 150, 4)
        check("a trajectory with no room refuses", False, "it returned a frame")
    except ValueError as exc:
        check("a trajectory with no room refuses",
              "50" in str(exc) or "correlated" in str(exc), str(exc)[:60])


if __name__ == "__main__":
    sys.exit(main())
