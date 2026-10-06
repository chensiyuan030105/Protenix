"""One GAGU sample as Protenix features plus the raw trajectory.

Ported from the v1 adapter at
repos/research/kineidos-v1/legacy/protenix/data/gagu_adapter.py, which was
verified to still run unchanged against the v2.0.0 baseline before porting.
Window construction is deliberately *not* here -- P004.2 replaces the v1
fixed-stride history with random-stride sampling, per-frame relative time and a
validity mask, which live in kineidos/data/windows.py.

The PDBs are handled directly rather than converted to mmCIF: MDTraj writes
blank chain IDs and encodes terminal residues as ``G5``/``A3``.  The TER records
recover the two RNA chains, after which residue labels are normalised to
standard ``A/C/G/U``.

## Units are a hard boundary, and the v1 code got it wrong

The NPZ holds nanometres and nm/ps; the PDB holds Angstroms; Protenix works in
Angstroms.  The v1 adapter converted everything to Angstroms and then handed
WorldParticle the Angstrom array
(``features["world_particle_position"] = position``).  WorldParticle's lengths
are nanometres -- ``particle_radius`` is 0.0778 nm after the P004 calibration
(plan section 2.10) -- so Angstrom input makes every length ten times too large
relative to the cutoff, which is exactly the degenerate regime where all 470
atoms are isolated and `h` is the same vector for every particle.

Both unit systems are therefore kept under names that say which is which, and
``assert_plausible_units`` is called on load.  This class of bug is silent: the
only symptom is "WorldParticle learns nothing", indistinguishable from the
several other reasons that could happen.

Units carried:
    position_nm / velocity_nm_per_ps        -> WorldParticle
    position_angstrom / velocity_angstrom_per_ps, ref_pos, labels -> Protenix
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from biotite.structure import AtomArray, BondList, get_residue_starts
from biotite.structure.io import pdb

from protenix.data.core.featurizer import Featurizer
from protenix.data.tokenizer import AtomArrayTokenizer, TokenArray
from protenix.data.constants import RES_ATOMS_DICT
from protenix.data.core.parser import AddAtomArrayAnnot
from protenix.data.utils import make_dummy_feature


_NUCLEOTIDES = frozenset("ACGU")
_TERMINAL_RESIDUE = re.compile(r"^([ACGU])[35]$", re.IGNORECASE)


def _sample_paths(sample_dir: Path) -> tuple[Path, Path, Path]:
    """Return metadata, NPZ and PDB paths for one processed sample."""

    metadata_path = sample_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing GAGU metadata: {metadata_path}")
    with metadata_path.open() as handle:
        metadata = json.load(handle)
    sample_id = str(metadata.get("sample_id", sample_dir.name))
    npz_path = sample_dir / f"{sample_id}.npz"
    pdb_path = sample_dir / f"{sample_id}.pdb"
    if not npz_path.is_file() or not pdb_path.is_file():
        # Be helpful for hand-created samples whose filenames do not match
        # metadata.sample_id exactly.
        npz_candidates = sorted(sample_dir.glob("*.npz"))
        pdb_candidates = sorted(sample_dir.glob("*.pdb"))
        if len(npz_candidates) != 1 or len(pdb_candidates) != 1:
            raise FileNotFoundError(
                "GAGU sample must contain one .npz and one .pdb file: "
                f"{sample_dir}"
            )
        npz_path, pdb_path = npz_candidates[0], pdb_candidates[0]
    return metadata_path, npz_path, pdb_path


def _atom_count_before_first_ter(path: Path) -> int:
    """Get the number of ATOM/HETATM records before the first TER record."""

    count = 0
    with path.open() as handle:
        for line in handle:
            record = line[:6].strip().upper()
            if record == "TER":
                break
            if record in {"ATOM", "HETATM"}:
                count += 1
    if count == 0:
        raise ValueError(f"No ATOM records before the first TER in {path}")
    return count


def _normalise_residue_name(name: str) -> str:
    """Map MDTraj terminal names (G5/A3) to standard RNA residue names."""

    name = str(name).strip().upper()
    match = _TERMINAL_RESIDUE.match(name)
    if match:
        name = match.group(1)
    if name not in _NUCLEOTIDES:
        raise ValueError(f"GAGU adapter only supports A/C/G/U residues, got {name!r}")
    return name


def _safe_tokatom_indices(atom_array: AtomArray) -> np.ndarray:
    """Assign Protenix tokatom indices, tolerating explicit hydrogens.

    The production Protenix parser removes hydrogens before calling
    ``add_tokatom_idx``.  ``heavy_atoms_only=False`` is useful for inspecting
    the source trajectory, so unknown hydrogen names receive the documented
    fallback index 0 rather than raising a KeyError.
    """

    result = np.zeros(len(atom_array), dtype=np.int64)
    for idx, atom in enumerate(atom_array):
        atom_name_position = RES_ATOMS_DICT.get(str(atom.res_name), {})
        result[idx] = atom_name_position.get(str(atom.atom_name), 0)
    return result


def _clone_and_batch_features(features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Match Protenix's collate convention used by the existing smoke tests."""

    result = {}
    for key, value in features.items():
        if torch.is_tensor(value):
            result[key] = (
                value.clone()
                if key == "atom_to_token_idx"
                else value.unsqueeze(0).clone()
            )
        else:
            # atom_perm_list is a Python list consumed by the permutation
            # utilities and must survive the batching step unchanged.
            result[key] = copy.deepcopy(value)
    return result


def _assert_plausible_units(position_nm: np.ndarray, atom_array: AtomArray) -> None:
    """Fail loudly if the array handed around is not in nanometres.

    A factor of ten in either direction is the failure this whole module is
    organised against, and it cannot be detected downstream: WorldParticle just
    reports no neighbours, or every atom as a neighbour, and either way the only
    visible symptom is that nothing is learned.

    The test is the nearest-neighbour distance.  Covalently bonded heavy atoms
    in RNA sit around 0.13 nm (the plan's section 2.5 measures 0.1309 nm on this
    very dataset), so 0.05-0.3 nm is a generous window that Angstrom data
    (~1.3) and metre data would both miss.
    """
    frame = position_nm[0]
    # Sampling 64 atoms keeps this O(64 N) rather than O(N^2) -- it runs on
    # every load, and a unit error is uniform across atoms anyway.
    probe = frame[:: max(1, len(frame) // 64)]
    d = np.linalg.norm(probe[:, None, :] - frame[None, :, :], axis=-1)
    # Each probe atom's distance to itself is exactly zero, so masking zeros
    # drops the self pairs without having to track which column each probe came
    # from.  Exactly coincident atoms would be dropped too, which is also right.
    d[d == 0.0] = np.inf
    nearest = float(d.min())
    if not 0.05 <= nearest <= 0.3:
        raise ValueError(
            f"GAGU positions do not look like nanometres: nearest heavy-atom "
            f"distance is {nearest:.4f}, expected 0.05-0.3 nm "
            f"(bonded heavy atoms are ~0.13 nm). Angstroms would give ~1.3."
        )


def _assert_ref_pos_is_centred_source(atom_array: AtomArray,
                                      ref_pos: torch.Tensor) -> None:
    """Confirm `ref_pos` is this AtomArray's coordinates, centred per residue.

    The featurizer passes ref_pos through random_transform for each
    ref_space_uid (featurizer.py:403-413), and random_transform centralises by
    default (utils/geometry.py:50).  With ref_pos_augment=False that is the
    whole transform, so adding each residue's centroid back has to recover the
    input coordinates.

    This pins down what ref_pos actually is -- it is ~72 A away from the PDB
    coordinates purely from the centring, which is easy to mistake for a
    corrupted atom order -- and it would catch the featurizer reordering atoms,
    which would break the position-by-position concatenation with `h` silently.
    """
    coord = np.asarray(atom_array.coord, dtype=np.float64)
    got = ref_pos.numpy().astype(np.float64)
    recovered = np.empty_like(got)
    for uid in np.unique(atom_array.ref_space_uid):
        m = atom_array.ref_space_uid == uid
        recovered[m] = got[m] + coord[m].mean(axis=0)
    err = float(np.abs(recovered - coord).max())
    if err > 1e-3:
        raise AssertionError(
            f"ref_pos is not this AtomArray's coordinates centred per residue: "
            f"max |diff| = {err:.3e} A after adding the residue centroids back. "
            f"Either the featurizer reordered atoms or it applies a transform "
            f"beyond centring (check ref_pos_augment)."
        )


@dataclass
class GAGUSample:
    """Parsed GAGU sample and its static Protenix features."""

    sample_id: str
    sample_dir: Path
    metadata: dict[str, Any]
    atom_array: AtomArray
    token_array: TokenArray
    base_features: dict[str, torch.Tensor]
    coordinate_mask: torch.Tensor
    # Nanometres: what WorldParticle consumes, and what the npz natively holds.
    position_nm: np.ndarray
    velocity_nm_per_ps: np.ndarray
    heavy_atom_indices: np.ndarray
    pair_dim: int = 128
    frame_interval_ns: float = 0.1  # GAGU saves every 100 ps

    @property
    def position_angstrom(self) -> np.ndarray:
        """Angstroms: what Protenix consumes.  Derived rather than stored, so the
        two unit systems cannot drift apart."""
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

    @property
    def n_tokens(self) -> int:
        return len(self.token_array)

    @property
    def duration_ns(self) -> float:
        return self.n_frames * self.frame_interval_ns

    def ref_pos_nm(self) -> np.ndarray:
        """The reference conformer in nanometres: frame 0, which is what the PDB
        holds and what Protenix uses as ``ref_pos``.  The canonicalisation in
        kineidos/window_align.py anchors to this."""
        return self.position_nm[0]


class GAGUProtenixAdapter:
    """Load one processed GAGU sample into Protenix/World Particle tensors."""

    def __init__(
        self,
        sample_dir: str | Path,
        *,
        heavy_atoms_only: bool = True,
        pair_dim: int = 128,
        alignment_tolerance_angstrom: float = 1e-3,
        validate_alignment: bool = True,
    ) -> None:
        self.sample_dir = Path(sample_dir).expanduser().resolve()
        self.heavy_atoms_only = bool(heavy_atoms_only)
        self.pair_dim = int(pair_dim)
        self.alignment_tolerance_angstrom = float(alignment_tolerance_angstrom)
        self.validate_alignment = bool(validate_alignment)

    @staticmethod
    def iter_sample_dirs(dataset_root: str | Path) -> Iterator[Path]:
        """Yield processed GAGU sample directories in stable order."""

        root = Path(dataset_root).expanduser().resolve()
        for metadata_path in sorted(root.glob("*/metadata.json")):
            yield metadata_path.parent

    def _annotate_atom_array(
        self, atom_array: AtomArray, first_chain_atom_count: int
    ) -> tuple[AtomArray, np.ndarray]:
        """Recover chains and add the annotations expected by Featurizer."""

        raw_atom_indices = np.arange(len(atom_array), dtype=np.int64)
        chain_index = (raw_atom_indices >= first_chain_atom_count).astype(np.int64)
        if np.any(chain_index > 1):
            raise ValueError("GAGU sample contains more than two chains")

        # Normalize residue names before creating residue-level annotations.
        normalized_names = np.array(
            [_normalise_residue_name(name) for name in atom_array.res_name],
            dtype="U3",
        )
        atom_array.set_annotation("res_name", normalized_names)

        # MDTraj PDBs have blank chain IDs.  TER marks the chain boundary; the
        # two chains are equivalent copies of the same RNA sequence.
        chain_ids = np.where(chain_index == 0, "A", "B").astype("U2")
        atom_array.set_annotation("chain_id", chain_ids)
        atom_array.set_annotation("label_asym_id", chain_ids.copy())
        atom_array.set_annotation("auth_asym_id", chain_ids.copy())
        atom_array.set_annotation("label_entity_id", np.full(len(atom_array), "1", dtype="U4"))
        atom_array.set_annotation("auth_seq_id", atom_array.res_id.copy())
        atom_array.set_annotation("label_seq_id", atom_array.res_id.copy())

        # Reset residue numbering within each chain (1..11), as expected for
        # equivalent polymer copies in AF3 relative-position features.
        res_id = np.zeros(len(atom_array), dtype=np.int32)
        for chain in (0, 1):
            chain_mask = chain_index == chain
            chain_res_ids = atom_array.res_id[chain_mask]
            unique_res_ids = np.unique(chain_res_ids)
            remap = {old: new for new, old in enumerate(unique_res_ids, start=1)}
            res_id[chain_mask] = [remap[int(old)] for old in chain_res_ids]
        atom_array.set_annotation("res_id", res_id)

        n_atom = len(atom_array)
        atom_array.set_annotation("mol_type", np.full(n_atom, "rna", dtype="U7"))
        atom_array.set_annotation("chain_mol_type", np.full(n_atom, "rna", dtype=object))
        atom_array.set_annotation("asym_id_int", chain_index.copy())
        atom_array.set_annotation("entity_id_int", np.zeros(n_atom, dtype=np.int64))
        atom_array.set_annotation("sym_id_int", chain_index.copy())
        atom_array.set_annotation("mol_id", chain_index.copy())
        atom_array.set_annotation("entity_mol_id", np.zeros(n_atom, dtype=np.int64))
        mol_atom_index = np.zeros(n_atom, dtype=np.int64)
        for chain in (0, 1):
            chain_mask = chain_index == chain
            mol_atom_index[chain_mask] = np.arange(np.sum(chain_mask))
        atom_array.set_annotation("mol_atom_index", mol_atom_index)

        atom_array = AddAtomArrayAnnot.add_centre_atom_mask(atom_array)
        atom_array = AddAtomArrayAnnot.add_atom_mol_type_mask(atom_array)
        atom_array = AddAtomArrayAnnot.add_distogram_rep_atom_mask(atom_array)
        atom_array = AddAtomArrayAnnot.add_plddt_m_rep_atom_mask(atom_array)
        # The GAGU residues are already standard A/C/G/U.  Assigning this
        # directly avoids requiring Protenix's multi-terabyte CCD file for a
        # self-contained trajectory adapter.
        atom_array.set_annotation("cano_seq_resname", normalized_names.copy())
        atom_array.set_annotation("tokatom_idx", _safe_tokatom_indices(atom_array))
        atom_array = AddAtomArrayAnnot.add_modified_res_mask(atom_array)
        atom_array = AddAtomArrayAnnot.add_ref_space_uid(atom_array)

        # The PDB is the reference conformer.  Coordinates are already in Å.
        atom_array.set_annotation("ref_pos", atom_array.coord.copy())
        atom_array.set_annotation("ref_charge", np.zeros(n_atom, dtype=np.int64))
        atom_array.set_annotation("ref_mask", np.ones(n_atom, dtype=np.int64))
        atom_array.set_annotation("is_resolved", np.ones(n_atom, dtype=bool))
        atom_array.set_annotation("resolution", np.full(n_atom, -1.0, dtype=np.float32))
        return atom_array, chain_index

    def load(self) -> GAGUSample:
        metadata_path, npz_path, pdb_path = _sample_paths(self.sample_dir)
        with metadata_path.open() as handle:
            metadata = json.load(handle)
        sample_id = str(metadata.get("sample_id", self.sample_dir.name))
        coordinate_unit = metadata.get("coordinate_unit")
        velocity_unit = metadata.get("velocity_unit")
        if coordinate_unit not in (None, "nanometer", "nm"):
            raise ValueError(
                "GAGU adapter expects NPZ coordinates in nanometers, got "
                f"{coordinate_unit!r}"
            )
        if velocity_unit not in (None, "nanometer_per_ps", "nm_per_ps"):
            raise ValueError(
                "GAGU adapter expects NPZ velocities in nanometer/ps, got "
                f"{velocity_unit!r}"
            )

        with np.load(npz_path, allow_pickle=False) as trajectory:
            if "position" not in trajectory or "velocity" not in trajectory:
                raise KeyError(f"{npz_path} must contain position and velocity arrays")
            position_nm = np.asarray(trajectory["position"], dtype=np.float32)
            velocity_nm_per_ps = np.asarray(trajectory["velocity"], dtype=np.float32)
        if position_nm.ndim != 3 or position_nm.shape[-1] != 3:
            raise ValueError(f"GAGU position must be [frame, atom, 3], got {position_nm.shape}")
        if velocity_nm_per_ps.shape != position_nm.shape:
            raise ValueError(
                "GAGU velocity shape must match position: "
                f"{velocity_nm_per_ps.shape} != {position_nm.shape}"
            )
        if not np.isfinite(position_nm).all() or not np.isfinite(velocity_nm_per_ps).all():
            raise ValueError("GAGU trajectory contains NaN/Inf")

        raw_atom_array = pdb.get_structure(
            pdb.PDBFile.read(str(pdb_path)), model=1, altloc="first"
        )
        if len(raw_atom_array) != position_nm.shape[1]:
            raise ValueError(
                f"PDB/NPZ atom count mismatch: {len(raw_atom_array)} != {position_nm.shape[1]}"
            )
        first_chain_atom_count = _atom_count_before_first_ter(pdb_path)
        if first_chain_atom_count >= len(raw_atom_array):
            raise ValueError("GAGU PDB does not contain a second chain after TER")

        position_angstrom_all = position_nm * 10.0
        velocity_angstrom_per_ps_all = velocity_nm_per_ps * 10.0
        alignment_error = float(
            np.max(np.abs(position_angstrom_all[0] - raw_atom_array.coord))
        )
        if self.validate_alignment and alignment_error > self.alignment_tolerance_angstrom:
            raise ValueError(
                "GAGU frame-0 does not align with the PDB atom order: "
                f"max error {alignment_error:.6g} Å > {self.alignment_tolerance_angstrom}"
            )

        keep = np.ones(len(raw_atom_array), dtype=bool)
        if self.heavy_atoms_only:
            keep &= np.char.upper(raw_atom_array.element.astype("U3")) != "H"
        heavy_atom_indices = np.flatnonzero(keep).astype(np.int64)
        atom_array = raw_atom_array[keep]
        # MDTraj's PDB export has no CONECT records.  Protenix's featurizer
        # expects a BondList even when the polymer bond matrix is empty.
        if atom_array.bonds is None:
            atom_array.bonds = BondList(len(atom_array))
        # Chain boundary is counted in the unfiltered PDB atom order.
        first_chain_kept = int(np.sum(keep[:first_chain_atom_count]))
        atom_array, _ = self._annotate_atom_array(atom_array, first_chain_kept)

        # Kept in nanometres, the npz's own units.  Angstroms are a derived
        # property on GAGUSample so the two cannot drift apart; see the module
        # docstring for why that boundary is load-bearing.
        position_nm_heavy = position_nm[:, keep, :].copy()
        velocity_nm_heavy = velocity_nm_per_ps[:, keep, :].copy()
        _assert_plausible_units(position_nm_heavy, atom_array)
        token_array = AtomArrayTokenizer(atom_array).get_token_array()
        featurizer = Featurizer(
            cropped_token_array=token_array,
            cropped_atom_array=atom_array,
            ref_pos_augment=False,
            include_discont_poly_poly_bonds=True,
        )
        base_features = featurizer.get_all_input_features()
        base_features["profile"] = base_features["restype"].clone()
        base_features["deletion_mean"] = torch.zeros(len(token_array))
        # GAGU contains standard RNA residues and no chemically ambiguous
        # atom swaps.  Supplying an identity permutation keeps the feature
        # dictionary compatible with Protenix's symmetric-permutation path.
        atom_perm_list = []
        residue_starts = np.asarray(
            get_residue_starts(atom_array, add_exclusive_stop=True), dtype=np.int64
        )
        for start, stop in zip(residue_starts[:-1], residue_starts[1:]):
            atom_perm_list.extend(
                [[local_idx] for local_idx in range(int(stop - start))]
            )
        base_features["atom_perm_list"] = atom_perm_list
        # Keep the dictionary consumable by the full Protenix Pairformer too,
        # not only by DiffusionModule.  GAGU has no MSA/template databases, so
        # use the same deterministic dummy tensors as the standard pipeline.
        base_features = make_dummy_feature(
            features_dict=base_features, dummy_feats=["msa", "template"]
        )
        base_features["is_distillation"] = torch.tensor([False])
        coordinate_mask = torch.from_numpy(atom_array.is_resolved.astype(np.int64))

        _assert_ref_pos_is_centred_source(atom_array, base_features["ref_pos"])

        return GAGUSample(
            sample_id=sample_id,
            sample_dir=self.sample_dir,
            metadata=metadata,
            atom_array=atom_array,
            token_array=token_array,
            base_features=base_features,
            coordinate_mask=coordinate_mask,
            position_nm=position_nm_heavy,
            velocity_nm_per_ps=velocity_nm_heavy,
            heavy_atom_indices=heavy_atom_indices,
            pair_dim=self.pair_dim,
        )
