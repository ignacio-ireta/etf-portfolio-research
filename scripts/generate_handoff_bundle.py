"""Generate a curated handoff bundle archive and manifest."""

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

INCLUDE_EXACT_FILES = (
    Path("pyproject.toml"),
    Path("uv.lock"),
    Path("README.md"),
    Path(".github/workflows/ci.yml"),
    Path("data/metadata/etf_universe.csv"),
    Path("reports/metrics/backtest_metrics.json"),
    Path("reports/excel/portfolio_results.xlsx"),
    Path("reports/excel/optimized_portfolios.xlsx"),
)

INCLUDE_GLOBS = (
    "handoff/*.txt",
    "reports/runs/*.json",
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
    "data/processed",
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


def _as_posix(path: Path) -> str:
    return path.as_posix()


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

    if path.suffix.lower() in {".html", ".htm"} and path.exists():
        return path.stat().st_size > MAX_HTML_BYTES

    if file_name in {"secrets", "secrets.txt", "secrets.json"}:
        return True

    return False


def _iter_files_in_dir(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file())


def collect_handoff_files(project_root: Path) -> list[Path]:
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


def main() -> int:
    project_root = Path.cwd()
    files = collect_handoff_files(project_root)
    write_manifest(files, project_root / "handoff/included_files.txt")
    write_bundle(files, project_root / "handoff_bundle.zip", project_root)
    print(
        f"Generated handoff bundle with {len(files)} files: "
        "handoff_bundle.zip and handoff/included_files.txt"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
