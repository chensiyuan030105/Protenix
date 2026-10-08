"""The v1 metric kernels: bonds, clashes, RMSD, RMSF, and the GAGU loop probes.

Ported from `artifacts/reports/P007/calibrate/calibrate_md.py`, which produced
the MD-side calibration in `artifacts/reports/P007/md_calibration.md` on all 64
trajectories.  The algorithms are the same ones, deliberately: the numbers in
`thresholds.json` are the ceilings and floors the model is scored against, and a
metric computed one way for MD and another way for the model would compare two
different quantities.  Where this module reorganises the calibration script it
is into functions, not into different arithmetic.

Three things here are not obvious and each one has been got wrong before:

**The bond table comes from interatomic distances, never from MDTraj's
templates.**  G5 and A3 are terminal residue names MDTraj's RNA templates do not
carry, so the template path silently drops *every* O3'-P bond (P004 section 2.5)
-- and the maximum O3'-P deviation is one of the five metrics.  The distance
method on frame 0 gives 526 bonds, 2 connected components and 20 O3'-P bonds on
every GAGU context.

**Both chain assignments are tried for every RMSD.**  GAGU is a duplex of two
identical 11-mers, so an arm with no history (`none`, `zero`) draws each frame
independently and has no way to tell strand A from strand B.  Scoring under one
fixed assignment would charge such a sample for a relabelling rather than for a
structural error.  `chain_swap_permutation` builds the relabelling once from the
residue numbering; every RMSD here reports which assignment it used, so the swap
rate is a reported number rather than a hidden one (P007 section 3.0, item 4).

**The four contexts differ at four residues, not one.**  AgaguU, CgaguG, GgaguC
and UgaguA share residues 1, 4-7, 10 and 11 (and their 11-shifted twins) and
differ at 2, 3, 8 and 9: residue 9 is C in AgaguU and UgaguA but U in the other
two.  The calibration script's first run across all 64 trajectories died with
`KeyError (9, 'N4')` for exactly that reason, so anything that names a residue
here either names one of the shared ones or looks its type up.  The GAGU loop
itself -- G4, A5, G6, U7 and G15, A16, G17, U18 -- is shared, which is why all
of P007's loop observables are context independent.

Numpy only, no torch and no Protenix: these run in either environment, and the
CPU analysis of a rollout should not need the model's environment to be
importable.  lDDT is the exception and lives in `bench/model.py`, because it is
Protenix's own implementation rather than one of ours (P007 section 3.1).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

# Bondi radii, as the calibration used them.
VDW = {"C": 1.70, "N": 1.55, "O": 1.52, "P": 1.80}

BOND_CUTOFF_A = 1.9
"""Two heavy atoms closer than this in the reference frame are bonded.  1.9 A
sits above the longest covalent bond here (O3'-P peaks at 1.78 A over all 64
trajectories) and below the shortest non-bonded contact."""

CLASH_VDW_SCALE = 0.75
CLASH_ABS_A = 1.1
CLASH_MIN_GRAPH_DISTANCE = 4
"""Pairs separated by three bonds or fewer are excluded: 1-2, 1-3 and 1-4
neighbours are close by construction and would swamp any real clash."""

STRAND_LENGTH = 11
"""Residues per strand.  Strand A is 1-11, strand B is 12-22, and the duplex is
self-complementary, so residue r of A and residue r + 11 of B carry the same
residue name in all four contexts -- which is what makes the chain relabelling
a well-defined permutation of atom indices."""

BASE_ATOMS = {
    "A": ["N1", "C2", "N3", "C4", "C5", "C6", "N6", "N7", "C8", "N9"],
    "G": ["N1", "C2", "N2", "N3", "C4", "C5", "C6", "O6", "N7", "C8", "N9"],
    "C": ["N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"],
    "U": ["N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6"],
}

# The GAGU internal loop, in both strands.  Shared by all four contexts.
LOOP_G = (4, 6, 15, 17)
LOOP_A = (5, 16)
LOOP_U = (7, 18)
GG_PAIRS = ((4, 17), (6, 15))
"""The two G.G pairs whose edge-to-edge distance says the loop is still in
conformation I (P007 section 3.5)."""
FLIP_U = (7, 18)
"""The two uridines that are flipped out of the helix in conformation I."""
CONTROL_C = (10, 21)
"""Watson-Crick cytosines, paired in every conformation and present in all four
contexts -- the control for the SASA criterion.  Residues 9 / 20 are *not* C in
CgaguG and GgaguC; see the module docstring."""

SASA_FLIP_A2 = 100.0
PAIR_DISTANCE_A = 3.5
SYN_CHI_RANGE = (-30.0, 110.0)
GAMMA_TRANS_DEG = 120.0


# --------------------------------------------------------------- the molecule


def parse_pdb(path: str | Path) -> list[tuple[str, str, int, str]]:
    """(name, residue name, residue number, element) per ATOM record, in file
    order -- which is the order every other array here is in."""
    atoms = []
    for line in open(path):
        if line.startswith(("ATOM", "HETATM")):
            name = line[12:16].strip()
            resn = line[17:20].strip()
            resi = int(line[22:26])
            element = line[76:78].strip() or name[0]
            atoms.append((name, resn, resi, element))
    return atoms


@dataclass
class Topology:
    """Everything about the molecule that does not change between frames.

    Built once, from the reference frame, and then reused for every generated
    frame.  Rebuilding it per frame would be worse than slow: a generated frame
    with a stretched bond would get a *different bond table*, and the bond
    deviation it is supposed to measure would be defined away.
    """

    pdb_path: Path
    heavy: np.ndarray            # [N] indices into the full atom list
    names: np.ndarray            # [N]
    resn: np.ndarray             # [N]
    resi: np.ndarray             # [N]
    element: np.ndarray          # [N]
    ref_angstrom: np.ndarray     # [N, 3] the frame the bond table came from

    bond_i: np.ndarray
    bond_j: np.ndarray
    bond_d0: np.ndarray          # reference bond lengths
    n_components: int
    o3p_bond_idx: np.ndarray     # indices into bond_i / bond_j

    clash_i: np.ndarray
    clash_j: np.ndarray
    clash_vdw_thr: np.ndarray

    chain_swap: np.ndarray       # [N] permutation: strand A <-> strand B
    c1p_idx: np.ndarray          # [22] the C1' atoms, one per residue
    index: dict[tuple[int, str], int]

    @property
    def n_atoms(self) -> int:
        return len(self.names)

    def at(self, resi: int, name: str) -> int:
        try:
            return self.index[(int(resi), name)]
        except KeyError:
            raise KeyError(
                f"residue {resi} has no atom {name!r}; residue {resi} is "
                f"{self.resn[self.resi == resi][0] if (self.resi == resi).any() else 'absent'!r}. "
                f"The four GAGU contexts differ at residues 2, 3, 8 and 9 -- "
                f"name a shared residue or look its type up."
            ) from None

    def base_atoms(self, resi: int) -> list[int]:
        """The base-ring atoms of one residue, chosen by its actual residue
        name rather than by an assumption about the context."""
        kind = str(self.resn[self.resi == resi][0]).rstrip("35")
        if kind not in BASE_ATOMS:
            raise KeyError(f"residue {resi} is {kind!r}, not one of {sorted(BASE_ATOMS)}")
        return [self.at(resi, n) for n in BASE_ATOMS[kind]]


def build_topology(pdb_path: str | Path,
                   ref_angstrom: Optional[np.ndarray] = None,
                   *, heavy_only: bool = True) -> Topology:
    """The bond table, clash candidates and chain relabelling, from one frame.

    Args:
        pdb_path: the sample's PDB, which GAGU guarantees is frame 0 of the npz
            (`kineidos/data/gagu.py` asserts it to 1e-3 A).
        ref_angstrom: the frame to build the bond table from, [N_heavy, 3] in
            Angstroms.  Defaults to the PDB's own coordinates.  Pass the
            trajectory's frame 0 to be bit-identical with the calibration.
        heavy_only: drop hydrogens, as the model does (470 of 712 atoms).
    """
    pdb_path = Path(pdb_path)
    atoms = parse_pdb(pdb_path)
    heavy = np.array([i for i, a in enumerate(atoms)
                      if not heavy_only or a[3] != "H"], dtype=np.int64)
    table = [atoms[i] for i in heavy]
    n = len(table)

    names = np.array([a[0] for a in table])
    resn = np.array([a[1] for a in table])
    resi = np.array([a[2] for a in table], dtype=np.int64)
    element = np.array([a[3] for a in table])
    index = {(int(a[2]), a[0]): k for k, a in enumerate(table)}

    if ref_angstrom is None:
        coords = []
        for line in open(pdb_path):
            if line.startswith(("ATOM", "HETATM")):
                coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
        ref_angstrom = np.asarray(coords, dtype=np.float64)[heavy]
    ref_angstrom = np.asarray(ref_angstrom, dtype=np.float64)
    if ref_angstrom.shape != (n, 3):
        raise ValueError(f"ref_angstrom is {ref_angstrom.shape}, expected {(n, 3)}")

    dm = np.linalg.norm(ref_angstrom[:, None] - ref_angstrom[None], axis=-1)
    iu = np.triu_indices(n, 1)
    bonded = dm[iu] < BOND_CUTOFF_A
    bond_i, bond_j = iu[0][bonded], iu[1][bonded]

    adj: list[list[int]] = [[] for _ in range(n)]
    for i, j in zip(bond_i, bond_j):
        adj[i].append(j)
        adj[j].append(i)

    seen = np.zeros(n, bool)
    n_components = 0
    for start in range(n):
        if seen[start]:
            continue
        n_components += 1
        stack = [start]
        seen[start] = True
        while stack:
            v = stack.pop()
            for w in adj[v]:
                if not seen[w]:
                    seen[w] = True
                    stack.append(w)

    o3p_bond_idx = np.array(
        [k for k, (i, j) in enumerate(zip(bond_i, bond_j))
         if {names[i], names[j]} == {"O3'", "P"}], dtype=np.int64)

    # Pairs within three bonds, excluded from the clash count.
    close: set[tuple[int, int]] = set()
    for start in range(n):
        frontier = {start}
        seen_near = {start}
        for _ in range(CLASH_MIN_GRAPH_DISTANCE - 1):
            nxt: set[int] = set()
            for v in frontier:
                for w in adj[v]:
                    if w not in seen_near:
                        seen_near.add(w)
                        nxt.add(w)
            frontier = nxt
        for w in seen_near:
            if w > start:
                close.add((start, w))
    cand = np.array([(i, j) for i, j in zip(iu[0], iu[1]) if (i, j) not in close],
                    dtype=np.int64)
    clash_i, clash_j = cand[:, 0], cand[:, 1]
    clash_vdw_thr = CLASH_VDW_SCALE * (
        np.array([VDW[e] for e in element[clash_i]])
        + np.array([VDW[e] for e in element[clash_j]]))

    c1p_idx = np.array([k for k, a in enumerate(table) if a[0] == "C1'"], dtype=np.int64)

    return Topology(
        pdb_path=pdb_path, heavy=heavy, names=names, resn=resn, resi=resi,
        element=element, ref_angstrom=ref_angstrom,
        bond_i=bond_i, bond_j=bond_j,
        bond_d0=np.linalg.norm(ref_angstrom[bond_i] - ref_angstrom[bond_j], axis=-1),
        n_components=n_components, o3p_bond_idx=o3p_bond_idx,
        clash_i=clash_i, clash_j=clash_j, clash_vdw_thr=clash_vdw_thr,
        chain_swap=chain_swap_permutation(resi, names),
        c1p_idx=c1p_idx, index=index,
    )


def chain_swap_permutation(resi: np.ndarray, names: np.ndarray) -> np.ndarray:
    """The atom permutation that relabels strand A as strand B and back.

    GAGU's two strands are the same 11-mer, so residue r of strand A and residue
    r + 11 of strand B hold the same atom names.  `perm[k]` is the atom that
    plays atom k's role under the other labelling, and the permutation is its
    own inverse.

    Raises if the two strands do not match atom for atom: that would mean the
    duplex is not self-complementary after all, and then an arm without history
    really can be charged for mixing the strands up, rather than this being an
    artefact of the labelling.
    """
    lookup = {(int(r), nm): k for k, (r, nm) in enumerate(zip(resi, names))}
    perm = np.empty(len(resi), dtype=np.int64)
    for k, (r, nm) in enumerate(zip(resi, names)):
        other = int(r) + STRAND_LENGTH if int(r) <= STRAND_LENGTH else int(r) - STRAND_LENGTH
        if (other, nm) not in lookup:
            raise ValueError(
                f"atom {nm!r} of residue {r} has no counterpart at residue "
                f"{other}; the two strands are not atom-for-atom identical, so "
                f"there is no chain relabelling to try"
            )
        perm[k] = lookup[(other, nm)]
    if not np.array_equal(perm[perm], np.arange(len(perm))):
        raise ValueError("the chain relabelling is not an involution")
    return perm


# ------------------------------------------------------------------ geometry


def kabsch_rotation(mobile: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rotation R minimising ||mobile @ R - target||; both already centred.

    Same correction as `kineidos/window_align.py`: the reflection SVD can hand
    back is folded out, because superimposing a mirror image is not a pose
    change but a different molecule.
    """
    cov = mobile.T @ target
    u, _s, vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    return (vt.T @ np.diag([1.0, 1.0, d]) @ u.T).T


def kabsch_rmsd_pairs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """RMSD of a[k] against b[k] after optimal superposition, batched over k.

    Verbatim from the calibration: the batched SVD is what made 25 lags on
    10000 frames affordable, and the per-pair form is what the lag curve needs.
    """
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    h = np.einsum("kni,knj->kij", a, b)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(np.einsum("kij,kjl->kil", u, vt)))
    vt[:, 2, :] *= d[:, None]
    rot = np.einsum("kij,kjl->kil", u, vt)
    return np.sqrt(((np.einsum("kni,kij->knj", a, rot) - b) ** 2).sum(-1).mean(-1))


def kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    """One pair."""
    return float(kabsch_rmsd_pairs(np.asarray(a, float)[None],
                                   np.asarray(b, float)[None])[0])


def align_to(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Superpose every frame of x [T, N, 3] onto ref [N, 3]."""
    x = np.asarray(x, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    xc = x - x.mean(1, keepdims=True)
    rc = ref - ref.mean(0)
    h = np.einsum("tni,nj->tij", xc, rc)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(np.einsum("tij,tjk->tik", u, vt)))
    vt[:, 2, :] *= d[:, None]
    rot = np.einsum("tij,tjk->tik", u, vt)
    return np.einsum("tni,tij->tnj", xc, rot) + rc.mean(0)


def rmsf(x: np.ndarray) -> np.ndarray:
    """Per-atom RMS fluctuation about the mean of x [T, N, 3].

    The caller aligns first.  RMSF of an unaligned trajectory measures the
    tumbling of the box, not the flexibility of the molecule.
    """
    x = np.asarray(x, dtype=np.float64)
    return np.sqrt(((x - x.mean(0)) ** 2).sum(-1).mean(0))


@dataclass
class ChainAwareRMSD:
    rmsd: float
    swapped: bool
    rmsd_identity: float
    rmsd_swapped: float


def rmsd_chain_aware(pred: np.ndarray, true: np.ndarray,
                     topo: Topology) -> ChainAwareRMSD:
    """Kabsch RMSD under the better of the two chain assignments.

    Both are reported, not just the minimum, so that "the swap helped" can be
    told from "the swap was irrelevant" afterwards without rerunning anything.
    """
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    pair = kabsch_rmsd_pairs(np.stack([pred, pred[topo.chain_swap]]),
                             np.stack([true, true]))
    identity, swapped = float(pair[0]), float(pair[1])
    return ChainAwareRMSD(min(identity, swapped), swapped < identity,
                          identity, swapped)


def rmsd_chain_aware_batch(pred: np.ndarray, true: np.ndarray,
                           topo: Topology) -> tuple[np.ndarray, np.ndarray]:
    """`rmsd_chain_aware` over pred [S, N, 3] against one true frame.

    Returns (rmsd [S], swapped [S] bool).
    """
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    s = len(pred)
    tiled = np.repeat(true[None], s, axis=0)
    identity = kabsch_rmsd_pairs(pred, tiled)
    swapped = kabsch_rmsd_pairs(pred[:, topo.chain_swap], tiled)
    return np.minimum(identity, swapped), swapped < identity


def align_frame_to(pred: np.ndarray, ref: np.ndarray,
                   topo: Topology) -> tuple[np.ndarray, bool]:
    """Put one generated frame into `ref`'s pose, trying both chain labellings.

    This is the step P004 section 2.19 (fourth qualifier) makes unavoidable:
    sampling starts from pure noise with no orientation to read, so the output
    pose is arbitrary.  Feeding an unaligned frame back into `build_window`
    trips its inter-frame rotation guard on the first step, and any metric that
    read the raw coordinates would be measuring the sampler's random rotation.

    The permutation is applied to the returned frame when it wins, so atom
    identity along a rollout stays consistent: the next step compares against
    this frame, not against the one before the relabelling.
    """
    pred = np.asarray(pred, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    out, best, swapped = None, np.inf, False
    for perm, is_swap in ((np.arange(topo.n_atoms), False), (topo.chain_swap, True)):
        x = pred[perm]
        xc = x - x.mean(0)
        rc = ref - ref.mean(0)
        rot = kabsch_rotation(xc, rc)
        placed = xc @ rot + ref.mean(0)
        err = float(np.sqrt(((placed - ref) ** 2).sum(-1).mean()))
        if err < best:
            out, best, swapped = placed, err, is_swap
    return out, swapped


# ----------------------------------------------------------- validity, bonds


def bond_observables(pos_angstrom: np.ndarray, topo: Topology,
                     *, chunk: int = 200) -> dict[str, np.ndarray]:
    """Per-frame bond deviation, O3'-P deviation and clash counts.

    Returns arrays of length T:
        bond_mae            mean |bond length - reference| over all 526 bonds
        o3p_maxdev          max over the 20 O3'-P bonds of the same deviation
        o3p_min, o3p_max    the shortest and longest O3'-P bond in the frame
        clash_vdw_per1k     pairs closer than 0.75 * (r_i + r_j), per 1k atoms
        clash_abs11_per1k   pairs closer than 1.1 A, per 1k atoms
        clash_vdw_count     the same clashes as a count, not a rate
    """
    pos = np.asarray(pos_angstrom, dtype=np.float64)
    if pos.ndim == 2:
        pos = pos[None]
    t, n = pos.shape[0], pos.shape[1]
    if n != topo.n_atoms:
        raise ValueError(f"{n} atoms, topology has {topo.n_atoms}")
    out = {k: np.empty(t) for k in ("bond_mae", "o3p_maxdev", "o3p_min", "o3p_max",
                                    "clash_vdw_per1k", "clash_abs11_per1k",
                                    "clash_vdw_count", "clash_abs11_count")}
    for s in range(0, t, chunk):
        p = pos[s:s + chunk]
        db = np.linalg.norm(p[:, topo.bond_i] - p[:, topo.bond_j], axis=-1)
        dev = np.abs(db - topo.bond_d0)
        out["bond_mae"][s:s + chunk] = dev.mean(1)
        o = db[:, topo.o3p_bond_idx]
        out["o3p_maxdev"][s:s + chunk] = np.abs(
            o - topo.bond_d0[topo.o3p_bond_idx]).max(1)
        out["o3p_min"][s:s + chunk] = o.min(1)
        out["o3p_max"][s:s + chunk] = o.max(1)
        dc = np.linalg.norm(p[:, topo.clash_i] - p[:, topo.clash_j], axis=-1)
        n_vdw = (dc < topo.clash_vdw_thr).sum(1)
        n_abs = (dc < CLASH_ABS_A).sum(1)
        out["clash_vdw_count"][s:s + chunk] = n_vdw
        out["clash_abs11_count"][s:s + chunk] = n_abs
        out["clash_vdw_per1k"][s:s + chunk] = n_vdw / n * 1000.0
        out["clash_abs11_per1k"][s:s + chunk] = n_abs / n * 1000.0
    return out


def valid_frames(obs: dict[str, np.ndarray], *,
                 o3p_maxdev_max: float,
                 clash_vdw_count_max: float) -> np.ndarray:
    """The frame filter of P007 section 3.3, as a boolean mask.

    A frame is valid when its worst O3'-P bond deviates no more than MD's own
    worst (0.25 A over all 64 trajectories) and it has at most
    `clash_vdw_count_max` clashing pairs.  MD never clashes at all over 640,000
    frames; the non-zero tolerance follows STAR-MD's treatment of chain breaks.

    Both thresholds come from `thresholds.json`, which is frozen before any
    model is scored -- hence keyword-only arguments with no defaults: a default
    here would be a threshold invented at the call site.
    """
    return (obs["o3p_maxdev"] <= o3p_maxdev_max) & \
           (obs["clash_vdw_count"] <= clash_vdw_count_max)


# -------------------------------------------------------------- lag and RMSF


def rmsd_vs_lag(pos_angstrom: np.ndarray, lags: Sequence[int],
                *, max_pairs: Optional[int] = None,
                rng: Optional[np.random.Generator] = None
                ) -> dict[str, np.ndarray]:
    """RMSD between frames `lag` apart, as a function of lag.

    With `max_pairs=None` every available pair is used, which is what P007
    section 3.2 asks for on a 100-frame rollout.  The calibration sampled 300
    pairs per lag because its trajectories are 10000 frames long; pass
    `max_pairs=300` with the same `rng` to reproduce it.
    """
    pos = np.asarray(pos_angstrom, dtype=np.float64)
    t = len(pos)
    keep = [int(l) for l in lags if 0 < int(l) < t]
    mean = np.full(len(keep), np.nan)
    std = np.full(len(keep), np.nan)
    npairs = np.zeros(len(keep), dtype=np.int64)
    for k, lag in enumerate(keep):
        available = t - lag
        if max_pairs is None or available <= max_pairs:
            offsets = np.arange(available)
        else:
            if rng is None:
                raise ValueError("max_pairs needs an rng")
            offsets = rng.choice(available, size=max_pairs, replace=False)
        v = kabsch_rmsd_pairs(pos[offsets], pos[offsets + lag])
        mean[k], std[k], npairs[k] = v.mean(), v.std(), len(v)
    return {"lags": np.array(keep, dtype=np.int64), "mean": mean, "std": std,
            "n_pairs": npairs}


def rmsf_pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r between two per-atom RMSF profiles.

    A correlation, so it says nothing about scale -- which is why P007's metric
    5 (the amplitude ratio) exists alongside it.  Returns nan for a constant
    profile rather than letting numpy warn and hand back nan quietly.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# ----------------------------------------------------- the GAGU loop's state


def loop_observables(pos_angstrom: np.ndarray, topo: Topology,
                     *, sasa_stride: int = 1,
                     with_sasa: bool = True) -> dict[str, np.ndarray]:
    """The conformation-I probes of P007 section 3.5, per frame.

    Returns:
        gg_min      [T, 2] closest edge-atom distance for G4.G17 and G6.G15
        sasa        [T_sasa, 4] base SASA of U7, U18, C10, C21 in A^2
        sasa_frames the frame indices the SASA rows belong to
        chi         [T, 4] glycosidic torsion of G4, G6, G15, G17 in degrees
        gamma       [T, 2] gamma of G6 and G17
        flipped     [T_sasa, 2] U7 / U18 base SASA above the threshold
        paired      [T, 2] the two G.G pairs within 3.5 A

    chi is recorded and not scored: conformation I contains a syn/anti substate
    that switches between replicas (P007 section 1.3), so a classifier keyed on
    chi would call a substate switch a conformational transition.  The flag uses
    flipping-out and pairing, which MD holds at 1.00 and 0.91-1.00 for every one
    of the 32 trajectories started from I.
    """
    pos = np.asarray(pos_angstrom, dtype=np.float64)
    if pos.ndim == 2:
        pos = pos[None]
    t = len(pos)

    edge = ["N1", "N2", "O6", "N7"]

    def min_edge(r1: int, r2: int) -> np.ndarray:
        i = [topo.at(r1, n) for n in edge]
        j = [topo.at(r2, n) for n in edge]
        d = np.linalg.norm(pos[:, i][:, :, None] - pos[:, j][:, None, :], axis=-1)
        return d.reshape(t, -1).min(1)

    gg = np.stack([min_edge(*p) for p in GG_PAIRS], 1)

    out: dict[str, np.ndarray] = {
        "gg_min": gg,
        "paired": gg < PAIR_DISTANCE_A,
    }

    import mdtraj as md

    top = md.load(str(topo.pdb_path)).topology.subset(topo.heavy)
    traj = md.Trajectory(xyz=(pos / 10.0).astype(np.float32), topology=top)
    chi_idx = [[topo.at(r, "O4'"), topo.at(r, "C1'"), topo.at(r, "N9"), topo.at(r, "C4")]
               for r in LOOP_G]
    gam_idx = [[topo.at(r, "O5'"), topo.at(r, "C5'"), topo.at(r, "C4'"), topo.at(r, "C3'")]
               for r in (6, 17)]
    out["chi"] = np.degrees(md.compute_dihedrals(traj, chi_idx))
    out["gamma"] = np.degrees(md.compute_dihedrals(traj, gam_idx))

    if with_sasa:
        frames = np.arange(0, t, max(1, int(sasa_stride)))
        sasa = md.shrake_rupley(traj[frames], mode="atom") * 100.0
        cols = [topo.base_atoms(r) for r in (FLIP_U + CONTROL_C)]
        out["sasa"] = np.stack([sasa[:, c].sum(1) for c in cols], 1)
        out["sasa_frames"] = frames
        out["flipped"] = out["sasa"][:, :2] > SASA_FLIP_A2
    return out


def syn_fraction(chi: np.ndarray) -> np.ndarray:
    """Fraction of frames in the syn range, per column of `chi`."""
    lo, hi = SYN_CHI_RANGE
    return ((chi > lo) & (chi < hi)).mean(0)


def first_sustained_drop(series: np.ndarray, *, threshold: float = 0.8,
                         window: int = 10) -> Optional[int]:
    """First step at which a boolean series stays below `threshold` for
    `window` consecutive steps -- P007 section 3.5's hallucinated transition.

    `series` is per-step truth values (flipped out, or paired).  The returned
    index is the first step of the run, or None if it never happens.  A running
    mean over the window rather than a bare "10 falses in a row" so that a
    criterion flickering at 0.5 counts, which is the regime a collapsing
    structure actually passes through.
    """
    s = np.asarray(series, dtype=np.float64)
    if len(s) < window:
        return None
    kernel = np.ones(window) / window
    rolling = np.convolve(s, kernel, mode="valid")
    hit = np.flatnonzero(rolling < threshold)
    return int(hit[0]) if len(hit) else None


# ------------------------------------------------------------------ baselines


def mean_structure(pos_angstrom: np.ndarray) -> np.ndarray:
    """The trajectory-average structure, aligned to frame 0 first.

    This is the `s` baseline of P009 section 1: the best a predictor that
    ignores the history entirely can do.  Averaging unaligned frames would
    instead produce the average of a tumbling molecule, which is a blob.
    """
    return align_to(pos_angstrom, np.asarray(pos_angstrom, float)[0]).mean(0)


def summarise(values: Iterable[float]) -> dict[str, float]:
    """mean / std / median / min / max, nan-safe, for a report cell."""
    v = np.asarray(list(values), dtype=np.float64)
    v = v[np.isfinite(v)]
    if not len(v):
        return {"n": 0, "mean": float("nan"), "std": float("nan"),
                "median": float("nan"), "min": float("nan"), "max": float("nan")}
    return {"n": int(len(v)), "mean": float(v.mean()), "std": float(v.std()),
            "median": float(np.median(v)), "min": float(v.min()),
            "max": float(v.max())}
