"""The history-frame feature contract: what WorldParticle is told about an atom.

P011 D2, plan section 5.1.  One module, not a few lines inside the bridge,
because P008 E1's pretraining has to use *this* contract and not a second
implementation of it -- a pretrained checkpoint whose `other_feats` means
something slightly different from the fine-tuning side is a failure mode with
no error message.

Why it looks like this
----------------------

P009 10.7 measured, with forward hooks in eval mode, that WorldParticle's `h`
is 100.000% a cross-atom constant, and found the cause in the five channels the
bridge was feeding it:

    channel 0     1.0000      zero cross-atom std    WorldParticle's own bias
    channels 1-3  1.8-2.7e-4  the velocity components
    channel 4     0.1250      zero cross-atom std    the frame-time channel

`dense0_molecular` is a `Linear(5 -> 384)`, so two constants 400-5000x larger
than the only per-atom signal decided its output: the per-atom part came out at
1.57e-07 of the magnitude.  **Nothing in the input said which atom this was.**
That is not a WorldParticle bug -- its particle contract is
`(position, velocity, other_feats)`, the fluid-dynamics view in which a water
molecule is fully described by where it is and how fast it is going -- but RNA
atoms are not interchangeable, and `other_feats` was the one extensible slot
and we had spent all of it on a frame-time scalar.

So this module fills `other_feats` with per-atom identity, and the velocity
slot goes to zero.  Both halves follow STAR-MD, which is the only reference
implementation we have read end to end (P002's reproduction,
literature/topics/history-featurization.md):

  * **No velocity.**  Not one of the nine papers surveyed feeds a physical
    velocity, and STAR-MD's per-frame input is pairwise C-beta distances
    alone.  Mori-Zwanzig says why that is coherent rather than merely
    conventional: the history *is* the replacement for the momentum, not a
    supplement to it.  And ours is thin anyway -- at a 100 ps save interval the
    finite difference is 2e-4 nm/ps and close to noise (P008 1.4).
  * **Frame-independent identity.**  STAR-MD's `s_OF` + `pos_emb` + `aa_emb`
    (a learnable embedding of the 20 amino acids) are all constant across
    frames; only the C-beta distances vary.  The analogue here is element,
    nucleotide and atom name.
  * **Per-branch scale, before the concat.**  STAR-MD's A12 is the same bug we
    hit, at 220:1 rather than 5000:1, and its fix is to normalise each branch
    *before* concatenating: "a norm applied afterwards cannot undo the
    imbalance."  Here each branch is built at rms within about 2x of
    WorldParticle's `ones` channel (reported by `branch_rms`), and `wp-v3`'s
    `LocalFeatureExtractor` then LayerNorms the branch as a whole.

Everything is derived from Protenix's own `input_feature_dict`.  Nothing is
re-derived from the PDB: `ref_element`, `ref_atom_name_chars` and `restype` are
already there (`protenix/data/core/featurizer.py`), already checked against the
atom order the fusion point assumes, and a second parser would be a second
source of truth for "which atom is this".
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Index into `ref_element`'s 128 one-hot columns.  The column order is
# `get_all_elems()` (protenix/data/constants.py:395), i.e. RDKit's periodic
# table by atomic number starting at H=0, so C(6) is 5, N(7) is 6, O(8) is 7
# and P(15) is 14.  Spelled out rather than computed so that a change upstream
# shows up as a failed assertion here instead of as a silent relabelling.
ELEMENT_COLUMNS = {"C": 5, "N": 6, "O": 7, "P": 14}

# Index into `restype`'s 32 one-hot columns: RNA_STD_RESIDUES
# (protenix/data/constants.py:294) puts A at 21, G at 22, C at 23, U at 24.
NUCLEOTIDE_COLUMNS = {"A": 21, "G": 22, "C": 23, "U": 24}

ATOM_NAME_CHARS = 4 * 64          # ref_atom_name_chars is [N_atom, 4, 64]
ATOM_NAME_DIM = 8

# 1 frame index + 4 element + 4 nucleotide + 8 atom name.
IDENTITY_CHANNELS = len(ELEMENT_COLUMNS) + len(NUCLEOTIDE_COLUMNS) + ATOM_NAME_DIM
OTHER_FEATS_CHANNELS = 1 + IDENTITY_CHANNELS


class HistoryFeatureContract(nn.Module):
    """`other_feats` for one history frame: `[N_atom, OTHER_FEATS_CHANNELS]`.

    The identity part is frame-independent by construction, which is the point
    of D2 and also makes it cheap: it is computed once per window and reused
    for all K frames.  `forward` takes the frame's time channel and concatenates.

    Only one branch has parameters -- the atom-name embedding -- and it is the
    one that needs watching.  `Linear(4 * 64 -> 8)` on the flattened
    `ref_atom_name_chars` one-hot is exactly a lookup table: the 256-wide
    one-hot distinguishes (character position, character), so the layer holds a
    separate 8-vector per (position, char) and sums the four.  Its LayerNorm is
    not decoration either: it is the only learnable branch, `weight_decay` is 0
    under D16, and nothing else would stop it from growing until it swamped the
    one-hots -- which is A12 again, one level down.
    """

    def __init__(self, atom_name_dim: int = ATOM_NAME_DIM) -> None:
        super().__init__()
        self.atom_name_dim = int(atom_name_dim)
        self.channels = 1 + len(ELEMENT_COLUMNS) + len(NUCLEOTIDE_COLUMNS) \
            + self.atom_name_dim
        self.atom_name_embed = nn.Linear(ATOM_NAME_CHARS, self.atom_name_dim,
                                         bias=False)
        self.atom_name_norm = nn.LayerNorm(self.atom_name_dim)
        nn.init.xavier_uniform_(self.atom_name_embed.weight)
        # Filled by `identity()`; read by kineidos/probe_wp_content.py for the
        # plan's "各支 rms 同量级，报出来".
        self.branch_rms: dict[str, float] = {}

    # ------------------------------------------------------------- identity

    def identity(self, feats: dict) -> torch.Tensor:
        """`[N_atom, IDENTITY_CHANNELS]`, the same for every frame."""
        missing = [k for k in ("ref_element", "ref_atom_name_chars",
                               "restype", "atom_to_token_idx")
                   if k not in feats]
        if missing:
            raise KeyError(
                f"the feature dict is missing {missing}; the history feature "
                f"contract derives atom identity from Protenix's own features "
                f"rather than re-parsing the structure (plan section 5.1)"
            )

        element = feats["ref_element"]
        name_chars = feats["ref_atom_name_chars"]
        restype = feats["restype"]
        a2t = feats["atom_to_token_idx"]

        # Squeeze any leading singleton axes.  check_fusion drives
        # DiffusionModule directly and adds them; the trainer's collate does
        # not.  Both have to reach the same `h`, and a stray leading 1 would
        # otherwise propagate into the concat and change the shape silently.
        element = _drop_leading(element, 2)
        name_chars = _drop_leading(name_chars, 3)
        restype = _drop_leading(restype, 2)
        a2t = _drop_leading(a2t, 1)

        n_atom = element.shape[0]
        if name_chars.shape[0] != n_atom or a2t.shape[0] != n_atom:
            raise ValueError(
                f"per-atom features disagree on the atom count: ref_element "
                f"{element.shape[0]}, ref_atom_name_chars {name_chars.shape[0]}, "
                f"atom_to_token_idx {a2t.shape[0]}"
            )

        dtype = self.atom_name_embed.weight.dtype

        # --- element one-hot, C/N/O/P -----------------------------------
        cols = torch.tensor([ELEMENT_COLUMNS[k] for k in ("C", "N", "O", "P")],
                            device=element.device)
        elem = element.index_select(-1, cols).to(dtype)        # [N, 4]
        _assert_onehot(
            elem, "element",
            "every GAGU heavy atom is C, N, O or P. An all-zero row means an "
            "element outside that set reached the contract, and it would be "
            "fed to WorldParticle as 'none of the above' -- i.e. its identity "
            "silently dropped, which is the P009 10.7 bug all over again. Add "
            "the element to ELEMENT_COLUMNS rather than widening the "
            "tolerance.")

        # --- nucleotide one-hot, A/G/C/U, token level -> atom level -------
        ncols = torch.tensor([NUCLEOTIDE_COLUMNS[k] for k in ("A", "G", "C", "U")],
                             device=restype.device)
        nuc_token = restype.index_select(-1, ncols).to(dtype)  # [N_token, 4]
        _assert_onehot(
            nuc_token, "nucleotide",
            "every GAGU token is a standard RNA residue (A/G/C/U). See the "
            "element note above for why an all-zero row is not acceptable as "
            "'other'.")
        nuc = nuc_token.index_select(0, a2t.to(torch.long))    # [N_atom, 4]

        # --- atom name, learnable ----------------------------------------
        flat = name_chars.reshape(n_atom, -1).to(dtype)
        if flat.shape[-1] != ATOM_NAME_CHARS:
            raise ValueError(
                f"ref_atom_name_chars flattens to {flat.shape[-1]}, expected "
                f"{ATOM_NAME_CHARS} (4 characters x 64 codes)"
            )
        name = self.atom_name_norm(self.atom_name_embed(flat))  # [N_atom, 8]

        ident = torch.cat([elem, nuc, name], dim=-1)
        self.branch_rms.update({
            "element_onehot": _rms(elem),
            "nucleotide_onehot": _rms(nuc),
            "atom_name_embed": _rms(name),
        })
        return ident

    # -------------------------------------------------------------- forward

    def forward(self, time_feat: torch.Tensor, identity: torch.Tensor
                ) -> torch.Tensor:
        """`[1]` or `[]` frame time + `[N_atom, C]` identity -> `[N_atom, 1+C]`.

        The frame-time channel is `_frame_time_feature`'s output, expanded over
        atoms.  It stays a per-frame property deliberately: it says only where
        this frame sits in the window, and the *physical* dt reaches the model
        through the conditioning pathway in plan section 6, not through here.
        Mixing the two would put a quantity the model also gets from AdaLN into
        a channel that mean-pooling then destroys.
        """
        n_atom = identity.shape[0]
        t = time_feat.reshape(1, -1).to(identity.dtype).expand(n_atom, -1)
        out = torch.cat([t, identity], dim=-1)
        self.branch_rms["frame_index"] = _rms(t)
        self.branch_rms["other_feats_total"] = _rms(out)
        if out.shape[-1] != self.channels:
            raise ValueError(
                f"contract produced {out.shape[-1]} channels, declared "
                f"{self.channels}"
            )
        return out.contiguous()


# ---------------------------------------------------------------- helpers

def _drop_leading(t: torch.Tensor, want_dim: int) -> torch.Tensor:
    """Strip leading singleton axes down to `want_dim` dimensions."""
    while t.dim() > want_dim and t.shape[0] == 1:
        t = t[0]
    if t.dim() != want_dim:
        raise ValueError(
            f"expected a {want_dim}-d tensor after dropping leading singleton "
            f"axes, got shape {tuple(t.shape)}"
        )
    return t


def _assert_onehot(x: torch.Tensor, name: str, why: str) -> None:
    """Every row sums to exactly one.  Raises with `why` if not."""
    s = x.sum(dim=-1)
    bad = int((s != 1).sum())
    if bad:
        raise ValueError(
            f"{name} one-hot: {bad} of {s.numel()} rows do not sum to 1 "
            f"(min {float(s.min()):.3f}, max {float(s.max()):.3f}). {why}"
        )


def _rms(t: torch.Tensor) -> float:
    return float(t.detach().float().pow(2).mean().sqrt())
