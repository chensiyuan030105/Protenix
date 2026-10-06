"""Put a history window into a canonical pose before WorldParticle sees it.

P004.1 measured WorldParticle's per-particle tokens `h` and found them neither
rotation invariant nor rotation equivariant: `ops.continuous_conv` maps each
neighbour's relative offset into a 4x4x4 kernel grid whose axes are fixed in the
lab frame, so rotating a configuration moves neighbours into different cells.
For WorldParticle's own domain that is a design choice -- fluids have a
preferred axis, gravity -- but a molecule in solution has no preferred axis, so
`h` ends up carrying the window's arbitrary orientation into the diffusion
head's conditioning signal.

The MD trajectory models all avoid this rather than train through it, and none
of them uses rotation augmentation: STAR-MD keeps Invariant Point Attention on
backbone rigids with pair features built from Cbeta distances; ConfRover passes
the temporal module a frame latent that is "invariant to global translation and
rotation"; MDGen tokenises trajectories as roto-translation offsets relative to
key frames, so a global rotation cancels algebraically.  This module takes the
MDGen-shaped route, applied to coordinates instead of to a tokenisation.

## Why the window's own oldest frame is not a sufficient reference

Aligning every frame of the window to the window's own oldest frame does *not*
produce an invariant input, which is worth spelling out because the result looks
right and fails quietly.

Write K(P, Q) for the rotation minimising ||P U - Q||.  Aligning frame k to the
window's oldest frame x_0 gives U_k = K(x_k, x_0) and x_k U_k.  Rotate the whole
window by R, so x_k -> x_k R^T and x_0 -> x_0 R^T as well.  Then U_k' = R U_k R^T
(substitute and the R^T factors cancel inside the norm), and the aligned frame
comes out as

    x_k R^T . R U_k R^T  =  x_k U_k R^T

-- still carrying R^T.  The reference rotated along with the data, so this step
removes inter-frame tumbling and leaves the global orientation untouched.

Anchoring to a *fixed* reference does work.  With V = K(x_0, ref) applied to
every frame in the window, a rotated window gives V' = K(x_0 R^T, ref) = R V and

    x_k R^T . R V  =  x_k V

which is exactly the unrotated result.  So the oldest frame decides *which*
transform to use and the fixed reference is what makes the pose canonical;
neither alone is enough.

`ref` is the sample's reference conformer -- for GAGU the PDB, which is npz
frame 0, and which Protenix already uses as ref_pos.  Its conformation drifts
away from late windows over a microsecond, which loosens the fit but cannot
affect invariance: the argument above holds for any fixed ref.

## Ordering, which also matters

WorldParticle is fed the canonicalised window; Protenix keeps its own random
augmentation on the target.  Canonicalising first and then applying a uniform
random rotation would hand WorldParticle a randomly rotated pose again and undo
all of this.  Protenix needs no help here: its loss superimposes the ground
truth onto the prediction before scoring (loss.py, Kabsch, under no_grad).
"""

from __future__ import annotations

import numpy as np


def kabsch_rotation(mobile: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rotation R minimising ||mobile @ R - target||, both already centred.

    Returns a proper rotation: the reflection that SVD may hand back is folded
    out by flipping the least significant singular direction, the standard
    correction.  A reflection would superimpose mirror images, which for a
    chiral molecule is not a pose change but a different molecule.
    """
    cov = mobile.T @ target
    u, _s, vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    correction = np.diag([1.0, 1.0, d])
    return (vt.T @ correction @ u.T).T


def canonicalize_window(
    pos: np.ndarray,
    vel: np.ndarray | None,
    ref: np.ndarray,
    anchor: int = 0,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, float]]:
    """Canonicalise a window so a global rigid motion of it leaves the result
    unchanged.

    Args:
        pos: window positions [F, N, 3], oldest frame first.
        vel: window velocities [F, N, 3], or None.  Velocities are vectors: they
            take the rotation and not the translation.
        ref: reference conformer [N, 3], the same atom order.
        anchor: which frame of the window supplies the transform.  0 is the
            oldest, the only choice that does not look into the window's future.

    Returns:
        (pos_aligned, vel_aligned, info) where info carries the anchor's RMSD to
        the reference -- worth logging, since a drifting fit is the expected
        behaviour over long trajectories and a sudden jump is not.
    """
    if pos.ndim != 3 or pos.shape[-1] != 3:
        raise ValueError(f"pos must be [F, N, 3], got {pos.shape}")
    if ref.shape != pos.shape[1:]:
        raise ValueError(f"ref {ref.shape} does not match a frame {pos.shape[1:]}")
    if vel is not None and vel.shape != pos.shape:
        raise ValueError(f"vel {vel.shape} does not match pos {pos.shape}")

    anchor_frame = pos[anchor]
    anchor_centroid = anchor_frame.mean(axis=0)
    ref_centroid = ref.mean(axis=0)

    rot = kabsch_rotation(anchor_frame - anchor_centroid, ref - ref_centroid)

    # One transform, from the anchor, applied to every frame.  Translating by
    # the anchor's centroid is what makes the result translation invariant: a
    # shifted window shifts its centroid identically, so the difference cancels.
    pos_aligned = (pos - anchor_centroid) @ rot + ref_centroid
    vel_aligned = None if vel is None else vel @ rot

    rmsd = float(
        np.sqrt(
            (((anchor_frame - anchor_centroid) @ rot - (ref - ref_centroid)) ** 2)
            .sum(axis=-1)
            .mean()
        )
    )
    return pos_aligned, vel_aligned, {"anchor_rmsd_to_ref": rmsd}
