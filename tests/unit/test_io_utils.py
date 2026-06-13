"""Tests for atomic output writes."""

from __future__ import annotations

from pathlib import Path

import pytest

from etf_portfolio.io_utils import atomic_path, atomic_write_bytes, atomic_write_text


def test_atomic_write_text_creates_file_and_parents(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "out.txt"
    atomic_write_text(target, "hello")
    assert target.read_text(encoding="utf-8") == "hello"


def test_atomic_write_bytes_roundtrip(tmp_path: Path) -> None:
    target = tmp_path / "out.bin"
    atomic_write_bytes(target, b"\x00\x01\x02")
    assert target.read_bytes() == b"\x00\x01\x02"


def test_atomic_path_leaves_original_on_exception(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"
    target.write_text("original", encoding="utf-8")

    with pytest.raises(RuntimeError), atomic_path(target) as temp_path:
        temp_path.write_text("partial", encoding="utf-8")
        raise RuntimeError("boom")

    # The previous content survives and no temp files are left behind.
    assert target.read_text(encoding="utf-8") == "original"
    assert list(tmp_path.glob(".out.txt.*")) == []


def test_atomic_path_no_temp_leftovers_on_success(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"
    with atomic_path(target) as temp_path:
        temp_path.write_text("done", encoding="utf-8")
    assert target.read_text(encoding="utf-8") == "done"
    assert list(tmp_path.glob(".out.txt.*")) == []
