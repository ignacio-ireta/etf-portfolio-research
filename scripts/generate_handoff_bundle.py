"""Generate a curated handoff bundle archive and manifest."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from fnmatch import fnmatch
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

INCLUDE_EXACT_FILES = (
    Path("pyproject.toml"),
    Path("uv.lock"),
    Path("README.md"),
    Path("Makefile"),
    Path(".github/workflows/ci.yml"),
    Path("scripts/generate_handoff_bundle.py"),
    Path("data/metadata/etf_universe.csv"),
    Path("handoff/preflight_summary.json"),
    Path("reports/metrics/backtest_metrics.json"),
    Path("reports/html/latest_report.html"),
    Path("reports/excel/portfolio_results.xlsx"),
    Path("reports/excel/optimized_portfolios.xlsx"),
)

INCLUDE_GLOBS = (
    "data/processed/*.parquet",
    "handoff/*.txt",
    "reports/figures/*.png",
)

INCLUDE_DIRS = (
    Path("configs"),
    Path("src"),
    Path("tests"),
    Path("docs"),
)

EXCLUDE_PATH_PREFIXES = (
    "__pycache__",
    ".venv",
    "data/raw",
    "mlruns",
)

EXCLUDE_NAME_GLOBS = ("*.pyc",)

# Conservative secret filters to prevent accidental leakage.
SECRET_PATH_HINTS = (
    "secret",
    ".env",
    "credentials",
    "token",
    "api_key",
    "private_key",
)

MAX_HTML_BYTES = 1_000_000

HTML_SIZE_ALLOWLIST = {
    Path("reports/html/latest_report.html"),
}

VALIDATION_COMMANDS = (
    ("uv_sync", ("uv", "sync", "--group", "dev", "--frozen")),
    ("ruff_check", ("uv", "run", "ruff", "check", ".")),
    ("ruff_format_check", ("uv", "run", "ruff", "format", "--check", ".")),
    ("pytest", ("uv", "run", "pytest", "-q")),
    (
        "run_all",
        ("uv", "run", "etf-portfolio", "run-all", "--config", "configs/base.yaml"),
    ),
)

PROVENANCE_FRESHNESS_DIRS = (
    Path("src"),
    Path("tests"),
    Path("configs"),
    Path("docs"),
    Path("scripts"),
)

PROVENANCE_FRESHNESS_FILES = (
    Path("pyproject.toml"),
    Path("uv.lock"),
    Path("README.md"),
    Path("Makefile"),
)

FRESHNESS_LOG_NAMES = (
    "ruff_check",
    "ruff_format_check",
    "pytest",
    "run_all",
)

RUN_ID_PATTERN = re.compile(r"run-all-\d{8}T\d{6}Z-[0-9a-f]+")


@dataclass(frozen=True)
class CommandLog:
    name: str
    command: tuple[str, ...]
    path: Path
    started_utc: str
    finished_utc: str
    exit_code: int


class ValidationCommandError(RuntimeError):
    def __init__(self, message: str, logs: dict[str, CommandLog]) -> None:
        super().__init__(message)
        self.logs = logs


def _as_posix(path: Path) -> str:
    return path.as_posix()


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _mtime_utc(path: Path) -> str:
    return (
        datetime.fromtimestamp(path.stat().st_mtime, UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _git_commit(project_root: Path) -> str | None:
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=project_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    commit = completed.stdout.strip()
    if completed.returncode != 0 or not commit:
        return None
    return commit


def _is_excluded(path: Path) -> bool:
    posix_path = _as_posix(path)
    lower_posix_path = posix_path.lower()
    file_name = path.name.lower()

    for prefix in EXCLUDE_PATH_PREFIXES:
        if lower_posix_path == prefix or lower_posix_path.startswith(f"{prefix}/"):
            return True

    if any(fnmatch(path.name, pattern) for pattern in EXCLUDE_NAME_GLOBS):
        return True

    # Exclude obvious secrets even if discovered through a broad include.
    if any(hint in lower_posix_path for hint in SECRET_PATH_HINTS):
        return True

    if (
        path.suffix.lower() in {".html", ".htm"}
        and path.exists()
        and path not in HTML_SIZE_ALLOWLIST
    ):
        return path.stat().st_size > MAX_HTML_BYTES

    if file_name in {"secrets", "secrets.txt", "secrets.json"}:
        return True

    return False


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_files_in_dir(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file())


def collect_handoff_files(project_root: Path, *, latest_run_record: Path) -> list[Path]:
    selected: set[Path] = set()

    for relative in INCLUDE_EXACT_FILES:
        candidate = project_root / relative
        if candidate.is_file() and not _is_excluded(relative):
            selected.add(relative)

    for directory in INCLUDE_DIRS:
        for path in _iter_files_in_dir(project_root / directory):
            relative = path.relative_to(project_root)
            if not _is_excluded(relative):
                selected.add(relative)

    for pattern in INCLUDE_GLOBS:
        for path in sorted(project_root.glob(pattern)):
            if path.is_file():
                relative = path.relative_to(project_root)
                if not _is_excluded(relative):
                    selected.add(relative)

    run_record_relative = latest_run_record.relative_to(project_root)
    if latest_run_record.is_file() and not _is_excluded(run_record_relative):
        selected.add(run_record_relative)

    return sorted(selected, key=_as_posix)


def write_manifest(files: list[Path], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(f"./{_as_posix(path)}" for path in files)
    if payload:
        payload = f"{payload}\n"
    manifest_path.write_text(payload, encoding="utf-8")


def write_bundle(files: list[Path], bundle_path: Path, project_root: Path) -> None:
    if bundle_path.exists():
        bundle_path.unlink()
    with ZipFile(bundle_path, mode="w", compression=ZIP_DEFLATED) as archive:
        for relative in files:
            archive.write(project_root / relative, arcname=_as_posix(relative))


def refresh_validation_logs(project_root: Path) -> dict[str, CommandLog]:
    """Run acceptance validation commands and persist their logs under handoff/."""

    handoff_dir = project_root / "handoff"
    handoff_dir.mkdir(parents=True, exist_ok=True)
    logs: dict[str, CommandLog] = {}
    for name, command in VALIDATION_COMMANDS:
        started_utc = _utc_now_iso()
        completed = subprocess.run(
            command,
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        finished_utc = _utc_now_iso()
        log_path = handoff_dir / f"{name}.txt"
        log_payload = "\n".join(
            (
                f"Command: {' '.join(command)}",
                f"Working directory: {project_root}",
                f"Started UTC: {started_utc}",
                completed.stdout.rstrip(),
                f"Finished UTC: {finished_utc}",
                f"Exit code: {completed.returncode}",
            )
        )
        log_path.write_text(f"{log_payload}\n", encoding="utf-8")
        logs[name] = CommandLog(
            name=name,
            command=command,
            path=log_path,
            started_utc=started_utc,
            finished_utc=finished_utc,
            exit_code=completed.returncode,
        )
        if completed.returncode != 0:
            raise ValidationCommandError(
                f"Validation command failed; see {log_path}: {' '.join(command)}",
                logs,
            )
    return logs


def _provenance_input_files(project_root: Path) -> list[Path]:
    files: list[Path] = []
    for directory in PROVENANCE_FRESHNESS_DIRS:
        for path in _iter_files_in_dir(project_root / directory):
            relative = path.relative_to(project_root)
            if not _is_excluded(relative):
                files.append(path)
    for relative in PROVENANCE_FRESHNESS_FILES:
        path = project_root / relative
        if path.is_file() and not _is_excluded(relative):
            files.append(path)
    return sorted(files)


def _newest_file(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    return max(paths, key=lambda path: path.stat().st_mtime)


def _verify_log_freshness(
    project_root: Path,
    logs: dict[str, CommandLog] | None = None,
) -> dict[str, object]:
    newest_source = _newest_file(_provenance_input_files(project_root))
    if newest_source is None:
        raise RuntimeError("No source files found for handoff freshness validation.")

    newest_source_mtime = newest_source.stat().st_mtime
    stale_logs: list[str] = []
    missing_logs: list[str] = []
    freshness_logs: dict[str, dict[str, str]] = {}
    for name in FRESHNESS_LOG_NAMES:
        if logs is not None and name in logs:
            log_path = logs[name].path
        else:
            log_path = project_root / "handoff" / f"{name}.txt"
        log_relative = log_path.relative_to(project_root).as_posix()
        if not log_path.is_file():
            missing_logs.append(log_relative)
            continue
        freshness_logs[name] = {
            "path": log_relative,
            "mtime_utc": _mtime_utc(log_path),
        }
        if log_path.stat().st_mtime < newest_source_mtime:
            stale_logs.append(log_relative)

    if missing_logs or stale_logs:
        source = newest_source.relative_to(project_root).as_posix()
        failures: list[str] = []
        if missing_logs:
            failures.append(f"missing logs: {', '.join(missing_logs)}")
        if stale_logs:
            failures.append(f"stale logs: {', '.join(stale_logs)}")
        raise RuntimeError(
            "Validation logs are not fresh. "
            f"Newest provenance input is {source}; {'; '.join(failures)}."
        )

    return {
        "checked_source_roots": [path.as_posix() for path in PROVENANCE_FRESHNESS_DIRS],
        "checked_source_files": [path.as_posix() for path in PROVENANCE_FRESHNESS_FILES],
        "newest_source_file": newest_source.relative_to(project_root).as_posix(),
        "newest_source_timestamp": _mtime_utc(newest_source),
        "logs": freshness_logs,
    }


def _extract_run_all_run_id(log_path: Path) -> str:
    run_ids: list[str] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            run_id = payload.get("run_id")
            if isinstance(run_id, str) and RUN_ID_PATTERN.fullmatch(run_id):
                run_ids.append(run_id)
                continue
        run_ids.extend(RUN_ID_PATTERN.findall(line))
    if not run_ids:
        raise RuntimeError(f"Could not find a run-all run_id in {log_path}.")
    latest = run_ids[-1]
    if any(run_id != latest for run_id in run_ids):
        unique = ", ".join(sorted(set(run_ids)))
        raise RuntimeError(f"Run-all log contains multiple run_ids: {unique}.")
    return latest


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object in {path}.")
    return payload


def _latest_run_record(project_root: Path) -> Path:
    candidates = sorted((project_root / "reports/runs").glob("*.json"))
    if not candidates:
        raise RuntimeError("No run records found in reports/runs.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _verify_artifacts(project_root: Path, run_all_log_path: Path) -> dict[str, object]:
    run_id = _extract_run_all_run_id(run_all_log_path)
    metrics_path = project_root / "reports/metrics/backtest_metrics.json"
    metrics = _load_json(metrics_path)
    metrics_run_id = metrics.get("run_id")
    if metrics_run_id != run_id:
        raise RuntimeError(f"Metrics run_id {metrics_run_id!r} does not match {run_id!r}.")

    run_record_path = project_root / "reports/runs" / f"backtest_{run_id}.json"
    if not run_record_path.is_file():
        raise RuntimeError(f"Expected latest run record does not exist: {run_record_path}.")
    latest_run_record = _latest_run_record(project_root)
    if latest_run_record != run_record_path:
        raise RuntimeError(
            f"Latest run record by mtime is {latest_run_record}, not {run_record_path}."
        )

    metrics_run_record = metrics.get("run_record")
    expected_run_record = run_record_path.relative_to(project_root).as_posix()
    if metrics_run_record != expected_run_record:
        raise RuntimeError(
            f"Metrics run_record {metrics_run_record!r} does not match {expected_run_record!r}."
        )

    run_record = _load_json(run_record_path)
    if run_record.get("run_id") != run_id:
        raise RuntimeError(f"Run record run_id does not match {run_id!r}.")

    output_artifacts = run_record.get("output_artifacts")
    if not isinstance(output_artifacts, dict):
        raise RuntimeError(f"Run record has no output_artifacts object: {run_record_path}.")

    verified_artifacts: dict[str, dict[str, object]] = {}
    for name, artifact in output_artifacts.items():
        if not isinstance(artifact, dict):
            raise RuntimeError(f"Run record artifact {name!r} is malformed.")
        artifact_path_value = artifact.get("path")
        artifact_sha = artifact.get("sha256")
        if not isinstance(artifact_path_value, str) or not isinstance(artifact_sha, str):
            raise RuntimeError(f"Run record artifact {name!r} is missing path or sha256.")
        artifact_path = project_root / artifact_path_value
        if not artifact_path.is_file():
            raise RuntimeError(f"Run record artifact is missing: {artifact_path_value}.")
        current_sha = _file_sha256(artifact_path)
        if current_sha != artifact_sha:
            raise RuntimeError(
                f"Artifact hash mismatch for {artifact_path_value}: "
                f"{current_sha} != {artifact_sha}."
            )
        verified_artifacts[str(name)] = {
            "path": artifact_path_value,
            "sha256": current_sha,
            "mtime_utc": _mtime_utc(artifact_path),
        }

    return {
        "run_id": run_id,
        "run_all_log": run_all_log_path.relative_to(project_root).as_posix(),
        "metrics_path": metrics_path.relative_to(project_root).as_posix(),
        "run_record_path": run_record_path.relative_to(project_root).as_posix(),
        "verified_output_artifacts": verified_artifacts,
    }


def write_preflight_summary(
    project_root: Path,
    *,
    logs: dict[str, CommandLog] | None = None,
    freshness: dict[str, object] | None = None,
    artifacts: dict[str, object] | None = None,
    artifact_paths_included: list[str] | None = None,
    status: str,
    error: str | None = None,
) -> Path:
    logs = logs or {}
    freshness = freshness or {}
    artifacts = artifacts or {}
    validation_log_timestamps = {
        name: {
            "path": log.path.relative_to(project_root).as_posix(),
            "mtime_utc": _mtime_utc(log.path),
            "started_utc": log.started_utc,
            "finished_utc": log.finished_utc,
            "exit_code": log.exit_code,
        }
        for name, log in logs.items()
        if log.path.exists()
    }
    summary_path = project_root / "handoff/preflight_summary.json"
    payload = {
        "git_commit": _git_commit(project_root),
        "generated_at": _utc_now_iso(),
        "latest_run_id": artifacts.get("run_id"),
        "validation_log_timestamps": validation_log_timestamps,
        "newest_source_timestamp": freshness.get("newest_source_timestamp"),
        "newest_source_path": freshness.get("newest_source_file"),
        "artifact_paths_included": artifact_paths_included or [],
        "status": status,
        "validation_commands": {
            name: {
                "command": list(log.command),
                "path": log.path.relative_to(project_root).as_posix(),
                "started_utc": log.started_utc,
                "finished_utc": log.finished_utc,
                "exit_code": log.exit_code,
                "mtime_utc": _mtime_utc(log.path),
            }
            for name, log in logs.items()
        },
        "freshness": freshness,
        "artifacts": artifacts,
    }
    if error is not None:
        payload["error"] = error
    summary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary_path


def write_generation_log(project_root: Path, *, bundle_file_count: int | None = None) -> None:
    lines = [
        "Command: uv run python scripts/generate_handoff_bundle.py",
        f"Working directory: {project_root}",
        f"Finished UTC: {_utc_now_iso()}",
        "Exit code: 0",
    ]
    if bundle_file_count is not None:
        lines.append(f"Bundle file count: {bundle_file_count}")
    (project_root / "handoff/generate_handoff_bundle.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    project_root = Path.cwd()
    logs: dict[str, CommandLog] = {}
    freshness: dict[str, object] = {}
    artifacts: dict[str, object] = {}
    try:
        logs = refresh_validation_logs(project_root)
        freshness = _verify_log_freshness(project_root, logs)
        artifacts = _verify_artifacts(project_root, logs["run_all"].path)
        latest_run_record = project_root / str(artifacts["run_record_path"])
        files = collect_handoff_files(project_root, latest_run_record=latest_run_record)
        write_generation_log(project_root, bundle_file_count=len(files))
        files = collect_handoff_files(project_root, latest_run_record=latest_run_record)
        write_preflight_summary(
            project_root,
            logs=logs,
            freshness=freshness,
            artifacts=artifacts,
            artifact_paths_included=[_as_posix(path) for path in files],
            status="pass",
        )
        files = collect_handoff_files(project_root, latest_run_record=latest_run_record)
        write_manifest(files, project_root / "handoff/included_files.txt")
        files = collect_handoff_files(project_root, latest_run_record=latest_run_record)
        write_preflight_summary(
            project_root,
            logs=logs,
            freshness=freshness,
            artifacts=artifacts,
            artifact_paths_included=[_as_posix(path) for path in files],
            status="pass",
        )
        files = collect_handoff_files(project_root, latest_run_record=latest_run_record)
        write_bundle(files, project_root / "handoff_bundle.zip", project_root)
        print(
            f"Generated handoff bundle with {len(files)} files: "
            "handoff_bundle.zip and handoff/included_files.txt"
        )
        return 0
    except ValidationCommandError as exc:
        logs = exc.logs
        _write_failure_summary(project_root, logs, freshness, artifacts, exc)
        raise
    except Exception as exc:
        _write_failure_summary(project_root, logs, freshness, artifacts, exc)
        raise


def _write_failure_summary(
    project_root: Path,
    logs: dict[str, CommandLog],
    freshness: dict[str, object],
    artifacts: dict[str, object],
    error: Exception,
) -> None:
    try:
        write_preflight_summary(
            project_root,
            logs=logs,
            freshness=freshness,
            artifacts=artifacts,
            artifact_paths_included=[],
            status="fail",
            error=str(error),
        )
    except Exception as summary_error:
        print(f"Failed to write handoff/preflight_summary.json: {summary_error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
