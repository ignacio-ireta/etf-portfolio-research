"""Tests for CLI ergonomics: --version, exit-code mapping, and --log-file."""

from __future__ import annotations

from pathlib import Path

import pytest

from etf_portfolio import __version__, cli, errors


def test_version_flag_prints_version_and_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--version"])
    assert exit_info.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_main_maps_domain_error_to_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    def boom(*args: object, **kwargs: object) -> None:
        raise errors.InsufficientHistoryError("not enough history")

    monkeypatch.setattr(cli, "run_features", boom)
    assert cli.main(["features", "--config", "configs/base.yaml"]) == errors.EXIT_INFEASIBLE


def test_main_maps_config_error_to_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    def boom(*args: object, **kwargs: object) -> None:
        raise errors.ConfigError("bad config")

    monkeypatch.setattr(cli, "run_optimize", boom)
    assert cli.main(["optimize", "--config", "configs/base.yaml"]) == errors.EXIT_CONFIG


def test_main_maps_pipeline_interrupted_to_130(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    def interrupt(*args: object, **kwargs: object) -> None:
        raise errors.PipelineInterrupted("stopped")

    monkeypatch.setattr(cli, "run_all", interrupt)
    assert cli.main(["run-all", "--config", "configs/base.yaml"]) == errors.EXIT_INTERRUPTED


def test_main_log_file_flag_creates_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "run_features", lambda *args, **kwargs: None)

    log_path = tmp_path / "logs" / "run.log"
    assert (
        cli.main(["features", "--config", "c.yaml", "--log-file", str(log_path)]) == errors.EXIT_OK
    )
    assert log_path.exists()
