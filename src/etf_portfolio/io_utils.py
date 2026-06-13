"""Atomic filesystem writes for crash-safe, half-written-file-free outputs.

Writers create a temporary file in the destination directory and ``os.replace``
it onto the final path only on success. ``os.replace`` is atomic for same-
filesystem moves on POSIX and Windows, so a reader never observes a partial
file and an interrupted/failed write leaves the previous output untouched
(see docs/engineering_standards.md §J).
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def atomic_path(path: Path | str) -> Iterator[Path]:
    """Yield a temp path in the destination dir; atomically replace ``path`` on success.

    Write to the yielded path. On clean exit the temp file replaces ``path``.
    On any exception the temp file is removed and ``path`` is left unchanged.
    """

    final_path = Path(path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temp_name = tempfile.mkstemp(
        dir=final_path.parent,
        prefix=f".{final_path.name}.",
        suffix=".tmp",
    )
    os.close(file_descriptor)
    temp_path = Path(temp_name)
    try:
        yield temp_path
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    else:
        os.replace(temp_path, final_path)


def atomic_write_bytes(path: Path | str, data: bytes) -> Path:
    """Atomically write ``data`` to ``path``."""

    with atomic_path(path) as temp_path:
        temp_path.write_bytes(data)
    return Path(path)


def atomic_write_text(path: Path | str, data: str, *, encoding: str = "utf-8") -> Path:
    """Atomically write ``data`` to ``path`` as text."""

    with atomic_path(path) as temp_path:
        temp_path.write_text(data, encoding=encoding)
    return Path(path)


__all__ = ["atomic_path", "atomic_write_bytes", "atomic_write_text"]
