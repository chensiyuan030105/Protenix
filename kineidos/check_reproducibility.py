"""Acceptance for the seeding entry point and the checkpoint loader."""

from __future__ import annotations

import random
import sys

import numpy as np
import torch

from kineidos.checkpoint import FUSION_KEYS, load_checkpoint
from kineidos.seeding import set_all_seeds

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def draws() -> tuple:
    """One draw from each generator that decides anything here."""
    return (
        random.random(),
        float(np.random.rand()),
        float(torch.rand(1)),
        # scipy, the one that matters: this is the call Protenix's augmentation
        # makes on every training step and every sampling step.
        float(__import__("scipy.spatial.transform", fromlist=["Rotation"])
              .Rotation.random().as_matrix()[0, 0]),
    )


def main() -> int:
    print("=== 1. the seeding entry point covers every generator ===")
    set_all_seeds(7)
    a = draws()
    set_all_seeds(7)
    b = draws()
    set_all_seeds(8)
    c = draws()
    labels = ("python random", "numpy", "torch", "scipy Rotation.random")
    for i, lab in enumerate(labels):
        check(f"{lab} repeats under the same seed", a[i] == b[i])
    # Different seeds must actually differ, or "reproducible" would be trivially
    # true for a function that seeds nothing.
    check("different seeds give different draws",
          all(a[i] != c[i] for i in range(len(labels))),
          f"{sum(a[i] != c[i] for i in range(len(labels)))}/{len(labels)} differ")
    rec = set_all_seeds(7)
    check("the record names every generator",
          {"python_random", "numpy", "torch", "torch_cuda"} <= set(rec),
          str(sorted(rec)))

    print("\n=== 2. the checkpoint loader ===")

    class Tiny(torch.nn.Module):
        def __init__(self, fused: bool):
            super().__init__()
            self.shared = torch.nn.Linear(4, 4, bias=False)
            if fused:
                self.wp_fusion = torch.nn.Linear(4, 4, bias=False)

    base = Tiny(False)
    sd = {k: v.clone() for k, v in base.state_dict().items()}

    info = load_checkpoint(Tiny(False), sd)
    check("loads cleanly when nothing is new", info["tensors_loaded"] == 1)

    fused = Tiny(True)
    try:
        load_checkpoint(fused, sd)
        check("refuses an unaccounted-for missing key", False, "it did not raise")
    except RuntimeError as exc:
        check("refuses an unaccounted-for missing key",
              "missing and not expected" in str(exc), str(exc).splitlines()[0][:50])

    info = load_checkpoint(fused, sd, expect_new=["wp_fusion.weight"])
    check("accepts exactly the declared new key",
          info["left_at_initialisation"] == ["wp_fusion.weight"])

    # Equality, not containment: declaring a key that is in fact present means
    # the caller's model of the checkpoint is wrong, and that is worth an error.
    try:
        load_checkpoint(Tiny(False), sd, expect_new=["wp_fusion.weight"])
        check("refuses a declared key that is not missing", False, "it did not raise")
    except RuntimeError as exc:
        check("refuses a declared key that is not missing",
              "expected to be missing but present" in str(exc))

    extra = dict(sd)
    extra["stale.weight"] = torch.zeros(2)
    try:
        load_checkpoint(Tiny(False), extra)
        check("refuses an unexpected checkpoint key", False, "it did not raise")
    except RuntimeError as exc:
        check("refuses an unexpected checkpoint key", "unexpected in checkpoint" in str(exc))

    class Wrong(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = torch.nn.Linear(8, 8, bias=False)

    try:
        load_checkpoint(Wrong(), sd)
        check("refuses a shape mismatch", False, "it did not raise")
    except RuntimeError as exc:
        check("refuses a shape mismatch", "shapes differ" in str(exc))

    print(f"\n  the real fusion keys this guards: {len(FUSION_KEYS)}")
    for k in sorted(FUSION_KEYS):
        print(f"    {k}")

    print("\n" + "=" * 62)
    print("验收:", "全部通过" if not FAILS else f"{len(FAILS)} 项失败 -> {FAILS}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())
