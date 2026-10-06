"""One real GAGU window, shaped the way AF3Trainer's batch is.

The trainer hands the model three dictionaries -- input_feature_dict,
label_dict and label_full_dict -- that upstream's dataset builds in
protenix/data/pipeline/dataset.py.  We do not use that pipeline (it reads
mmCIF and expects experimental metadata), so the same three have to be
assembled here from a Window.

Two things this deliberately does NOT do, both of which an earlier draft did:

- **No batch dimension.**  Upstream's loader collates with `collate_fn_first`
  (protenix/utils/torch_utils.py:247), which returns `x[0]` -- the sample as
  the dataset built it, with no leading axis.  Batch size is 1 by construction
  anyway: the loss calls `feat_dict["resolution"].item()`.
- **No relative-position encoding and no `update_input_feature_dict`.**
  `Protenix.forward` calls both itself (protenix/model/protenix.py:946-949).
  kineidos/check_fusion.py calls them by hand because it drives DiffusionModule
  directly, bypassing `forward`; a batch for the whole model must not.

Kept apart from the acceptance scripts because it is the collate: P004.5's
trainer needs exactly this, and a check that builds its own batch would be
checking something the trainer never sees.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from kineidos.wp_bridge import wp_inputs_from_window

# Keys chain_permutation.correct_symmetric_chains reads off label_full_dict
# besides the coordinates (see its docstring).  They are per-atom annotations
# the featurizer already produced, so the "full" label is the cropped one plus
# these -- true here only because we never crop.
_CHAIN_PERM_KEYS = ("entity_mol_id", "mol_id", "mol_atom_index", "pae_rep_atom_mask")


def collate_window(window: Any) -> dict[str, Any]:
    """Window -> {input_feature_dict, label_dict, label_full_dict, basic}."""
    feats = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in window.features.items()
    }
    # The history window itself.  It rides in the feature dict so that the
    # trainer's to_device moves it with everything else, and so that the bridge
    # -- which is a submodule of the model (protenix.py's wp_bridge) -- can read
    # it inside forward rather than being called from outside, where DDP would
    # not reduce its gradients.
    feats.update(wp_inputs_from_window(window))

    label_dict = {k: v.clone() for k, v in window.labels.items()}
    label_full_dict = {k: v.clone() for k, v in window.labels.items()}
    for key in _CHAIN_PERM_KEYS:
        if key not in window.features:
            raise KeyError(
                f"the window has no {key!r}, which chain permutation reads off "
                f"label_full_dict; the featurizer should have produced it"
            )
        label_full_dict[key] = window.features[key].clone()

    return {
        "input_feature_dict": feats,
        "label_dict": label_dict,
        "label_full_dict": label_full_dict,
        # AF3Trainer reads one key out of `basic`: runner/train.py:522 takes
        # basic["pdb_id"] for per-structure eval metrics.  GAGU has no PDB id,
        # so the sample id stands in -- it is what identifies a trajectory here.
        "basic": {
            "pdb_id": window.sample_id,
            "sample_id": window.sample_id,
            "target_frame": window.target_frame,
            "delta_t_ns": window.delta_t_ns,
        },
    }


def collate_fn_window(batch: list[Any]) -> dict[str, Any]:
    """DataLoader collate: one window per batch, collated, not stacked.

    The counterpart of upstream's collate_fn_first (protenix/utils/torch_utils
    .py:247), which also takes batch[0] and keeps it unbatched.  Batch size is
    1 because ProtenixLoss calls feat_dict["resolution"].item(); raising here
    says so, instead of letting that .item() fail several frames deeper.
    """
    if len(batch) != 1:
        raise ValueError(
            f"batch_size must be 1 (the loss calls resolution.item()), got "
            f"{len(batch)}"
        )
    return collate_window(batch[0])


GAGU_ROOT = Path(
    "/mnt/xfs/home/mhg/Projects/ForSiyuan/RNA-WorldParticle-Workspace/datasets/"
    "processed/gagu_internal_loop_v0_1"
)
DEFAULT_SAMPLE = "gagu_100mM_K_agaguu_startI_r1"


def one_batch(sample: str = DEFAULT_SAMPLE, *, target_frame: int = 5000,
              stride: int = 100, k: int = 8) -> tuple[Any, dict[str, Any]]:
    """The default window used by the acceptance scripts."""
    from kineidos.data.gagu import GAGUProtenixAdapter
    from kineidos.data.windows import build_window

    loaded = GAGUProtenixAdapter(GAGU_ROOT / sample).load()
    window = build_window(loaded, target_frame=target_frame, stride=stride, k=k)
    return window, collate_window(window)
