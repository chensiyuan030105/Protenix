"""History windows: random physical stride, relative time, validity mask.

The v1 adapter took a fixed stride of consecutive frames, which shows the model
one timescale only.  STAR-MD draws the physical stride independently per
training example and conditions on it through adaptive layernorm, and its stated
reason is not merely "tell the network the timescale" but that doing so
decouples physical duration from the number of frames in the context window: a
small window with strides spanning orders of magnitude exposes the model to
relative time deltas spanning orders of magnitude, so long-range dependencies
can be learned without paying for long sequences.

Ranges have to be recomputed for GAGU rather than copied from STAR-MD:

    frame interval   100 ps = 0.1 ns   (ATLAS is 10 ps, ten times finer)
    trajectory       10000 frames = 1 us
    K = 8 history frames, window spans K * stride frames, so
    stride <= 9999 / 8 = 1249  ->  dt <= 124.9 ns

STAR-MD's [0.01, 10] ns would be wrong twice over here: 0.01 ns is below the
save interval and simply cannot be sampled, and a 10 ns ceiling throws away an
order of magnitude of GAGU's reach.  This module uses LogUniform[0.1, 100] ns --
three decades, as STAR-MD has, with headroom under the 124.9 ns limit.

Three things are encoded, per the plan's section 2.8:

    dt                  scalar, to AdaLN alongside the diffusion noise level
    per-frame time      pooling over K frames destroys order, so each frame has
                        to say how old it is: -K*dt, ..., -2*dt, -dt
    validity mask       near the start of a trajectory fewer than K history
                        frames exist

Windows are canonicalised before WorldParticle sees them
(kineidos/window_align.py), which P004.1 established as necessary: `h` is
otherwise neither rotation invariant nor equivariant and carries the window's
arbitrary orientation into the conditioning signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from kineidos.window_align import canonicalize_window, inter_frame_rotation_deg

DT_MIN_NS = 0.1
# The upper end is a memory time, not a data-availability figure.  P004 used
# 100 ns because the trajectories could supply it; measured afterwards on GAGU
# (tau_half = 0.35 ns, kineidos.measure_decorrelation) that range leaves the
# history at most 5.7% of the no-history error to remove, against a measurement
# floor of 0.31-1.70% -- signal and noise at one magnitude, which is why P004's
# three arms read the same.  1 ns is 2.9 tau_half, the same place STAR-MD's
# upper end sits on its proteins (2.2), and rounding dt to an integer stride
# puts the expected available information at 15.2%.  P009 sections 2.1-2.4.
#
# Both datasets are built from these defaults, so this one constant moves the
# training windows and the held-out windows together.  A range set per dataset
# would make the held-out loss measure a different problem than the one trained.
DT_MAX_NS = 1.0


@dataclass
class Window:
    """One training example.

    WorldParticle's tensors are nanometres, Protenix's are Angstroms.  The field
    names carry the unit because mixing them is silent: WorldParticle reports no
    neighbours or all of them, and either way the symptom is only that nothing
    is learned.  See kineidos/data/gagu.py's module docstring.
    """

    sample_id: str
    target_frame: int
    history_frames: np.ndarray      # [K] trajectory indices, oldest first
    stride: int                     # frames between consecutive history frames
    delta_t_ns: float               # stride * frame_interval_ns

    wp_position_nm: torch.Tensor    # [K, N, 3] canonicalised
    wp_velocity_nm_per_ps: torch.Tensor
    wp_frame_time_ns: torch.Tensor  # [K] negative, oldest most negative
    wp_frame_mask: torch.Tensor     # [K] bool; False marks padding
    anchor_rmsd_to_ref_nm: float
    inter_frame_rotation_deg: float

    # The reference conformer in nanometres, and whether this window was put
    # into canonical pose against it.  Both are here for the bridge, which has
    # to verify that nothing transformed the window after the dataloader --
    # canonicalisation is idempotent, so re-aligning a canonical window must
    # leave it where it is, and any drift means something moved it.
    #
    # features["ref_pos"] cannot serve: the featurizer centres it per residue
    # (featurizer.py:403-413 through random_transform), so it is not the
    # conformer in global coordinates and Kabsch against it is meaningless.
    ref_pos_nm: torch.Tensor        # [N, 3]
    canonicalized: bool

    features: dict[str, torch.Tensor]        # Protenix, Angstroms
    labels: dict[str, torch.Tensor]          # Protenix, Angstroms


def sample_delta_t_ns(rng: np.random.Generator,
                      dt_min: float = DT_MIN_NS,
                      dt_max: float = DT_MAX_NS) -> float:
    """Draw dt ~ LogUniform[dt_min, dt_max].

    Uniform in log space, which is what "spanning orders of magnitude" requires:
    drawn uniformly in linear space, 90% of samples would sit in the top decade
    and the model would barely see short strides.
    """
    return float(np.exp(rng.uniform(np.log(dt_min), np.log(dt_max))))


def stride_for(delta_t_ns: float, frame_interval_ns: float,
               n_frames: int, k: int) -> int:
    """Frames per step for a requested dt, clamped to what the data supports.

    Rounded to at least 1: dt below the save interval is unreachable, and
    silently returning 0 would make every history frame the same frame.
    """
    stride = int(round(delta_t_ns / frame_interval_ns))
    max_stride = max(1, (n_frames - 1) // k)
    return max(1, min(stride, max_stride))


def build_window(
    sample,
    target_frame: int,
    *,
    stride: int,
    k: int = 8,
    canonicalize: bool = True,
    max_inter_frame_rotation_deg: float = 15.0,
) -> Window:
    """Assemble one window ending at `target_frame`.

    History frames are strictly before the target -- the task is to predict the
    target from its past, and including it would leak the answer.

    Args:
        sample: a GAGUSample.
        target_frame: the frame to be predicted.
        stride: frames between consecutive history frames.
        k: how many history frames.
        canonicalize: put the window in a canonical pose before WorldParticle
            sees it.  Leave on; off is for measuring what it buys.
        max_inter_frame_rotation_deg: raise if the molecule turns more than this
            between consecutive frames of the window.  Canonicalisation removes
            the window's global pose but not tumbling *within* it, and `h` is
            not rotation invariant, so an unfitted trajectory would quietly feed
            each frame's orientation into the conditioning signal.  GAGU sits at
            0.5-0.8 degrees because it was RMSD-fitted upstream; 15 is far above
            that and far below the 126.9 degrees of random orientations.

    Protenix's relative-position encoding is deliberately not applied here.  The
    v1 adapter called update_input_feature_dict and
    RelativePositionEncoding.generate_relp inside its window builder, but both
    want the batch dimension, so doing it per window means batching per window
    and then unbatching to collate.  It belongs in the collate function, next to
    the rest of the batching, and is left to whoever assembles batches
    (P004.3/P004.4).  No `add_relp` flag here: a parameter that silently does
    nothing is worse than its absence.
    """
    if not 0 <= target_frame < sample.n_frames:
        raise IndexError(f"target_frame {target_frame} outside "
                         f"[0, {sample.n_frames})")
    if k < 1 or stride < 1:
        raise ValueError(f"k and stride must be >= 1, got {k}, {stride}")

    wanted = target_frame - stride * np.arange(k, 0, -1)   # oldest first
    valid = wanted >= 0
    if not valid.any():
        raise ValueError(
            f"target_frame {target_frame} with stride {stride} leaves no history "
            f"frame at or after 0; the caller should not have offered this pair"
        )
    # Padding repeats the oldest frame that does exist.  The content is
    # arbitrary because the mask excludes it -- but it has to be a real
    # configuration rather than zeros, or the neighbour search in a padded frame
    # would collapse every atom onto the origin and cost O(N^2) pairs on data
    # that is then discarded.
    frames = np.where(valid, wanted, wanted[valid][0]).astype(np.int64)

    pos_nm = sample.position_nm[frames].astype(np.float64)
    vel_nm = sample.velocity_nm_per_ps[frames].astype(np.float64)

    # Measured before canonicalisation, which cannot change it: one rigid
    # transform applied to every frame leaves inter-frame angles untouched.
    turn = inter_frame_rotation_deg(pos_nm[valid]) if valid.sum() > 1 else 0.0
    if turn > max_inter_frame_rotation_deg:
        raise ValueError(
            f"{sample.sample_id} frame {target_frame}, stride {stride}: the "
            f"molecule turns {turn:.1f} deg between consecutive history frames, "
            f"over the {max_inter_frame_rotation_deg} deg limit. Canonicalisation "
            f"removes the window's global pose but not tumbling inside it, and "
            f"`h` is not rotation invariant, so this trajectory would feed each "
            f"frame's orientation into the conditioning signal. Fit the "
            f"trajectory upstream, or see plan section 2.9.5."
        )

    info = {"anchor_rmsd_to_ref": float("nan")}
    if canonicalize:
        pos_nm, vel_nm, info = canonicalize_window(
            pos_nm, vel_nm, sample.ref_pos_nm()
        )

    # Relative time, in physical units rather than frame counts, so a window
    # with stride 10 and one with stride 1000 are distinguishable.
    frame_time = -sample.frame_interval_ns * stride * np.arange(k, 0, -1)

    features = {
        k_: (v.clone() if torch.is_tensor(v) else v)
        for k_, v in sample.base_features.items()
    }
    labels = {
        "coordinate": torch.from_numpy(
            sample.position_angstrom[target_frame].copy()).float(),
        "coordinate_mask": sample.coordinate_mask.clone(),
    }

    window = Window(
        sample_id=sample.sample_id,
        target_frame=int(target_frame),
        history_frames=frames,
        stride=int(stride),
        delta_t_ns=float(stride * sample.frame_interval_ns),
        wp_position_nm=torch.from_numpy(pos_nm).float(),
        wp_velocity_nm_per_ps=torch.from_numpy(vel_nm).float(),
        wp_frame_time_ns=torch.from_numpy(frame_time).float(),
        wp_frame_mask=torch.from_numpy(valid.copy()),
        anchor_rmsd_to_ref_nm=float(info["anchor_rmsd_to_ref"]),
        inter_frame_rotation_deg=float(turn),
        ref_pos_nm=torch.from_numpy(sample.ref_pos_nm().copy()).float(),
        canonicalized=bool(canonicalize),
        features=features,
        labels=labels,
    )
    assert_atom_order(sample, window)
    return window


def assert_atom_order(sample, window: Window) -> None:
    """The atom-order contract, checked per window.

    `h` is concatenated onto `c_l` position by position, so one wrong index is
    wrong everywhere and raises nothing -- it just trains a model nobody can
    explain.  The plan's section 2 therefore asks for this on every data path,
    by coordinates rather than by shape, since a permuted array of the right
    length passes a shape check.

    Checked here: the AtomArray the Protenix features were built from still
    holds the same atoms, in the same order, as the trajectory arrays handed to
    WorldParticle.  That is the invariant that breaks if the hydrogen filter is
    ever changed in one place and not the other.

    Note what is *not* compared.  `features["ref_pos"]` is not the reference
    conformer in global coordinates: the featurizer runs it through
    random_transform per ref_space_uid (featurizer.py:403-413), and
    random_transform centralises by default (utils/geometry.py:50), so each
    residue is centred on its own origin.  With ref_pos_augment=False there is
    no rotation, but the centring alone puts it ~72 A away from the PDB
    coordinates -- an earlier version of this assertion compared the two and
    failed for that reason, not because anything was wrong.  `atom_array.coord`
    is the untransformed array and is what the contract is about.
    """
    n = window.wp_position_nm.shape[1]

    if len(sample.atom_array) != n:
        raise AssertionError(
            f"the feature AtomArray has {len(sample.atom_array)} atoms but "
            f"WorldParticle was given {n}"
        )
    err = float(np.abs(sample.atom_array.coord - sample.ref_pos_nm() * 10.0).max())
    if err > 1e-3:
        raise AssertionError(
            f"the feature AtomArray is not the trajectory's frame 0: max |diff| "
            f"= {err:.3e} A. The hydrogen filter or the atom order differs "
            f"between the Protenix features and the trajectory arrays."
        )
    if tuple(sample.base_features["ref_pos"].shape) != (n, 3):
        raise AssertionError(
            f"ref_pos is {tuple(sample.base_features['ref_pos'].shape)} but "
            f"WorldParticle was given {n} particles"
        )
    if window.labels["coordinate"].shape[0] != n:
        raise AssertionError(
            f"label has {window.labels['coordinate'].shape[0]} atoms, "
            f"WorldParticle was given {n}"
        )


class GAGUWindowDataset(torch.utils.data.Dataset):
    """Random-stride windows over one or more GAGU samples.

    Length is nominal: an epoch is `length` draws, not an enumeration of every
    (sample, target, stride) triple, because the stride is continuous and the
    product would be meaningless. Indices seed the draw, so an index always
    yields the same window -- otherwise two workers reading the same index would
    disagree and nothing would be reproducible.
    """

    def __init__(self, samples, *, k: int = 8, length: int = 10_000,
                 seed: int = 0, canonicalize: bool = True,
                 dt_min_ns: float = DT_MIN_NS, dt_max_ns: float = DT_MAX_NS):
        if not samples:
            raise ValueError("no samples")
        self.samples = list(samples)
        self.k = int(k)
        self.length = int(length)
        self.seed = int(seed)
        self.canonicalize = bool(canonicalize)
        self.dt_min_ns = float(dt_min_ns)
        self.dt_max_ns = float(dt_max_ns)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Window:
        # Seeded per index rather than from a shared generator: dataloader
        # workers each hold their own copy of this object, so a shared stream
        # would hand out different windows depending on how work was split.
        rng = np.random.default_rng((self.seed, idx))
        sample = self.samples[int(rng.integers(len(self.samples)))]

        dt = sample_delta_t_ns(rng, self.dt_min_ns, self.dt_max_ns)
        stride = stride_for(dt, sample.frame_interval_ns, sample.n_frames, self.k)
        # The target must leave at least one real history frame; everything
        # earlier is padded and masked.
        target = int(rng.integers(stride, sample.n_frames))

        return build_window(
            sample, target, stride=stride, k=self.k,
            canonicalize=self.canonicalize,
        )
