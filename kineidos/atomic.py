"""Write a file so that a kill never leaves half of one.

`background` preempts with no grace period, so every file this project writes
is written by a process that may be SIGKILLed between any two bytes.  Three
places wrote in place and are now routed through here: the per-sigma jsonl,
the per-window held-out jsonl, and env.lock.

Two distinct failures, and the second is the one that cost the P006 session
two systems' equilibration output on 2026-10-08:

  * **a half-written new file.**  `runs/p010/sigma_grid` held two of these at
    3659 and 3720 rows against a complete arm's 11264, from preemptions.  A
    truncated arm does not look broken -- it looks like a complete arm over
    fewer windows, which silently shrinks the paired set for every arm.
  * **a destroyed old file.**  `open(path, "w")` truncates to zero the instant
    the handle opens, so a rescore that is then killed leaves nothing where
    good rows had been.  "The state cannot be trusted, so redo it" is a
    correct judgement; destroying the old artefact before the new one exists
    is not part of it, and the two get bundled together only because
    overwriting in place is one line shorter than writing aside.

### The temporary name, and what it can and cannot hide from

Measured, not reasoned about (python 3.12.14), because the P006 session and I
each had a different wrong belief here:

    pattern    glob.glob / shell            pathlib.Path.glob
    *.jsonl    skips dotfiles               MATCHES dotfiles
    <name>.*   matches `<name>.tmp.N`       matches `<name>.tmp.N`
    *          matches everything           matches everything

So `.tmp.<pid>.<name>.part` carries three parts for three reasons:

  * the **leading dot** hides it from `glob.glob` and from shell globs;
  * the **trailing `.part`** hides it from `pathlib.Path.glob("*.jsonl")`,
    which the dot does not -- and this project's readers are pathlib;
  * not sharing a prefix with the real name hides it from `<name>.*`, which a
    `<name>.tmp.<pid>` suffix scheme cannot do.

Nothing hides from `glob("*")`, so `is_temp` exists for any reader that
enumerates.  There is no such reader here today -- every glob in kineidos/ is
anchored -- and this is the hook for when somebody writes one.

The **pid** is not about hiding: a fixed temporary name is a contention point.
The standalone sweeps all write one shared directory, so two scorings of the
same arm and step would otherwise be two writers on one temporary file,
interleaving rows into something that looks like a complete arm.
"""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from typing import Any, Iterator, TextIO

TEMP_PREFIX = ".tmp."
TEMP_SUFFIX = ".part"


def temp_name_for(path: Path) -> Path:
    return path.with_name(f"{TEMP_PREFIX}{os.getpid()}.{path.name}{TEMP_SUFFIX}")


def is_temp(path: Path | str) -> bool:
    """Is this one of our half-written files.

    For a reader that enumerates a directory rather than matching a pattern --
    the one case no naming scheme can protect, so it has to filter.
    """
    name = Path(path).name
    return name.startswith(TEMP_PREFIX) and name.endswith(TEMP_SUFFIX)


def sweep_stale(path: Path) -> list[Path]:
    """Remove leftovers of earlier attempts at exactly this path.

    A SIGKILL -- which is what preemption is -- runs no `finally`, so the
    cleanup below cannot be relied on and leftovers do accumulate.  /mnt/xfs
    sits at 97%, and an 11k-row jsonl is ~4 MB.

    Only this path's leftovers, matched on the real name, so a concurrent
    writer's in-progress temporary is never touched.  Returns what it removed
    so the caller can say so: a leftover is evidence that a previous attempt
    died, and that is worth one line in a log even though the bytes are not
    worth keeping.
    """
    removed = []
    pattern = f"{TEMP_PREFIX}*.{path.name}{TEMP_SUFFIX}"
    for stale in sorted(path.parent.glob(pattern)):
        try:
            stale.unlink()
            removed.append(stale)
        except OSError:
            pass
    return removed


@contextlib.contextmanager
def atomic_open(path: Path, *, sweep: bool = True) -> Iterator[TextIO]:
    """A handle on a temporary file, renamed onto `path` on a clean exit.

    `os.replace` is one rename syscall, so `path` holds either the old
    complete file or the new complete file and never anything else.

    On an exception the temporary is removed rather than kept.  That is a
    deletion, so it is worth being explicit about why it is not the failure
    this module exists to prevent: the temporary is the uncommitted half of a
    write, not an artefact -- the artefact is whatever `path` already held, and
    that is untouched either way.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if sweep:
        sweep_stale(path)
    tmp = temp_name_for(path)
    handle = open(tmp, "w")
    try:
        yield handle
        handle.close()
        os.replace(tmp, path)
    except BaseException:
        handle.close()
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


def write_json(path: Path, record: Any, **dumps_kwargs: Any) -> None:
    """json.dumps into `path`, atomically.  Defaults match what this project
    has always written: two-space indent, sorted keys, trailing newline."""
    dumps_kwargs.setdefault("indent", 2)
    dumps_kwargs.setdefault("sort_keys", True)
    with atomic_open(path) as handle:
        handle.write(json.dumps(record, **dumps_kwargs) + "\n")
