#!/usr/bin/env python
"""Write a runs/<run_id>/env.lock describing exactly what a run executed.

P004 requires every run to be traceable back to a rebuildable code state.  The
run command names the code trees and the environment directory, but a directory
name does not pin its contents: packages can be installed into an environment
later, leaving an older command string meaning something different.  This file
closes that gap by recording the contents, not just the names.

Usage (from the workspace root):
    PYTHONPATH=repos/research/kineidos-v3:repos/research/wp-v2 \
    LAYERNORM_TYPE=torch ATTN_IMPL=sdpa \
    LD_LIBRARY_PATH=/mnt/xfs/home/mhg/anaconda3/envs/kineidos-v2-slurm/lib \
      .../envs/kineidos-v2-slurm/bin/python -m kineidos.env_lock runs/<run_id>

An optional second argument is the JSON that kineidos.seeding.set_all_seeds
returned, which is the only record of what the run's randomness was set to.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Shared objects placed into the environment by hand rather than by a package
# manager.  venv manages Python packages only, and open3d's torch ops link
# libEGL, which neither this host nor any wheel provides.  They are recorded
# here because nothing else would notice if they changed.
HAND_PLACED_LIBS = ("libEGL.so.1.1.0", "libGLdispatch.so.0.0.0")

# Environment variables that change which code path runs, not merely where code
# is found.  A run is not reproducible without them.
RELEVANT_ENV = (
    "PYTHONPATH", "LAYERNORM_TYPE", "ATTN_IMPL", "LD_LIBRARY_PATH",
    "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "PYTHONHASHSEED",
)


def find_workspace_root(start: Path | None = None) -> Path:
    """Walk up from `start` (default: cwd) to the directory holding both
    AGENTS.md and repos/.

    This file used to live at <workspace>/tools/ and derive the root from
    __file__'s grandparent.  It now sits inside a worktree, where that would
    resolve to the worktree instead -- and the worktree has no repos/, so the
    worktrees it recorded would silently come back as nulls.  Searching from
    cwd matches the documented convention of running from the workspace root.
    """
    here = (start or Path.cwd()).resolve()
    for d in (here, *here.parents):
        if (d / "AGENTS.md").is_file() and (d / "repos").is_dir():
            return d
    raise SystemExit(
        f"workspace root not found above {here}: expected a directory "
        "containing both AGENTS.md and repos/. Run from the workspace root."
    )


def _run(cmd: list[str], cwd: str | None = None) -> str | None:
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _code_trees(workspace: Path) -> dict[str, object]:
    """The code trees this process imports from, derived rather than named.

    This used to list repos/research/kineidos-v2 literally.  P009 moved the work
    to a v3 worktree, and the hardcoded version would have recorded v2's commit
    -- which is frozen at tag p009-base, the very commit P009 section 6.1's
    first condition compares the diff against.  The lock would have read exactly
    right while describing code that was not running.  Nothing would have
    noticed, because a plausible commit is indistinguishable from the correct
    one once it is written down.

    Derived from PYTHONPATH, which is what the command declared, and cross-
    checked against where `import kineidos` actually resolved: an editable
    install or a stale sys.path entry makes those two disagree, and that
    disagreement is the failure AGENTS.md's PYTHONPATH section exists for.
    """
    import kineidos

    resolved = Path(kineidos.__file__).resolve().parent.parent
    declared = []
    for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if not entry:
            continue
        path = Path(entry)
        declared.append((path if path.is_absolute() else workspace / path).resolve())

    trees: dict[str, object] = {}
    for path in declared + ([resolved] if resolved not in declared else []):
        record = _worktree(path)
        record["declared_on_pythonpath"] = path in declared
        record["imports_resolve_here"] = path == resolved
        trees[path.name] = record
    return trees


def _worktree(path: Path) -> dict[str, object]:
    """Commit and dirtiness of one worktree.  A dirty tree is recorded, not
    rejected: refusing to run would be the wrong trade during development, but
    a result from a dirty tree must never look like one from a clean tree."""
    p = str(path)
    status = _run(["git", "-C", p, "status", "--porcelain"])
    return {
        "path": p,
        "commit": _run(["git", "-C", p, "rev-parse", "HEAD"]),
        "branch": _run(["git", "-C", p, "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(status) if status is not None else None,
        "dirty_files": status.splitlines() if status else [],
    }


def collect(workspace: Path) -> dict[str, object]:
    prefix = Path(sys.prefix)
    libs = {}
    for name in HAND_PLACED_LIBS:
        f = prefix / "lib" / name
        libs[name] = (
            hashlib.sha256(f.read_bytes()).hexdigest() if f.is_file() else None
        )

    import torch  # imported here so the file is usable without torch installed

    record: dict[str, object] = {
        "written_at": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "python": {
            "version": sys.version.split()[0],
            "executable": sys.executable,
            "prefix": str(prefix),
            # More than one entry means the isolation broke: either the venv was
            # recreated with --system-site-packages, or PYTHONPATH reaches into
            # another environment.  Then `import torch` resolves by path order.
            "site_packages_on_path": [p for p in sys.path if p.endswith("site-packages")],
        },
        "torch": {
            "version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_names": [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ],
        },
        "env_vars": {k: os.environ.get(k) for k in RELEVANT_ENV},
        "hand_placed_libs": {
            "source": "conda-forge libegl 1.7.0 ha4b6fd6_5, copied from envs/mfm/lib on 2026-10-06",
            "sha256": libs,
        },
        "worktrees": _code_trees(workspace),
        "pip_freeze": (_run([sys.executable, "-m", "pip", "freeze"]) or "").splitlines(),
    }

    try:
        import open3d
        from open3d import _build_config

        record["open3d"] = {
            "version": open3d.__version__,
            "built_for_torch": _build_config["Pytorch_VERSION"],
            "built_for_cuda": _build_config["CUDA_VERSION"],
            # open3d's compiled torch ops are ABI-bound to one torch minor
            # version.  A mismatch does not warn: it fails at load with an
            # undefined c10 symbol, or silently never gets exercised.
            "matches_installed_torch": _build_config["Pytorch_VERSION"] == torch.__version__,
        }
    except Exception as exc:  # noqa: BLE001 - recording the failure is the point
        record["open3d"] = {"import_failed": repr(exc)}

    return record


def main() -> int:
    if len(sys.argv) not in (2, 3):
        print(__doc__)
        return 2
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    workspace = find_workspace_root()

    record = collect(workspace)
    if len(sys.argv) == 3:
        # The seeds a run was started with.  Recorded because no environment
        # variable controls them: Protenix's augmentation draws from numpy
        # (utils/geometry.py's random_transform, every training step and every
        # sampling step), WorldParticle's initialisers draw from torch, and a
        # run that logs only one of the two does not pin what it did.
        record["seeds"] = json.loads(sys.argv[2])
    (out_dir / "env.lock").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")

    print(f"wrote {out_dir / 'env.lock'}")
    for label, wt in record["worktrees"].items():
        flag = " [DIRTY]" if wt["dirty"] else ""
        print(f"  {label:12s} {str(wt['commit'])[:12]} ({wt['branch']}){flag}")
    o3d = record["open3d"]
    if o3d.get("matches_installed_torch") is False:
        print("  WARNING: open3d was built for torch "
              f"{o3d['built_for_torch']}, installed is {record['torch']['version']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
