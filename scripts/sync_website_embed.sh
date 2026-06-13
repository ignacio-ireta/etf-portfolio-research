#!/usr/bin/env bash
set -euo pipefail

# Sync the CURATED ETF report artifact subset into the personal website repo.
#
# Usage:
#   scripts/sync_website_embed.sh [page-dir]
#
# [page-dir] is the showcase page directory (the one that holds the hand-authored
# index.html), e.g. website/projects/etf-portfolio-research. The script syncs into
# its reports/ subtree only.
#
# Local default target assumes these sibling clones:
#   ~/Projects/etf-portfolio-research
#   ~/Projects/ignacio-ireta.github.io
#
# CRITICAL CONTRAST WITH THE CDMX SYNC:
#   CDMX's sync does `rm -rf "$TARGET_DIR"` over the WHOLE embed directory, which
#   is safe there because projects/cdmx-map/ is 100% machine-generated (Vite dist).
#   For ETF that would DELETE the hand-authored index.html + embed.js. So here we
#   scope the wipe to the generated reports/ subtree only and never touch the page.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SOURCE_REPORTS="${SOURCE_REPORTS:-$REPO_ROOT/reports}"

# The showcase PAGE directory (holds the hand-authored index.html). Default to the
# sibling-clone path; CI passes "website/projects/etf-portfolio-research".
PAGE_DIR="${1:-${PAGE_DIR:-$REPO_ROOT/../ignacio-ireta.github.io/projects/etf-portfolio-research}}"

# The ONLY subtree this script owns. The wipe is scoped here, NOT to $PAGE_DIR, so
# the hand-authored index.html / embed.js beside it are never removed.
TARGET_REPORTS="$PAGE_DIR/reports"

# Safety guard: refuse to run if we'd be wiping anything other than a reports/ dir.
if [[ "$(basename "$TARGET_REPORTS")" != "reports" ]]; then
  echo "Refusing to run: target '$TARGET_REPORTS' does not end in /reports." >&2
  exit 1
fi

# Preconditions: the curated source artifacts must exist (reports/ is tracked).
for f in \
  "$SOURCE_REPORTS/html/latest_report.html" \
  "$SOURCE_REPORTS/html/frontier.html" \
  "$SOURCE_REPORTS/metrics/backtest_metrics.json" \
  "$SOURCE_REPORTS/metrics/validation_summary.json"; do
  if [[ ! -f "$f" ]]; then
    echo "Missing required artifact: $f" >&2
    echo "Are the committed reports present? (reports/ is tracked in git.)" >&2
    exit 1
  fi
done

if [[ ! -d "$PAGE_DIR" ]]; then
  echo "Target page dir does not exist: $PAGE_DIR" >&2
  echo "Expected the hand-authored index.html to already live there." >&2
  exit 1
fi

# Scoped, idempotent wipe of ONLY the generated reports/ subtree. This makes the
# sync deletion-aware (renamed/removed artifacts upstream disappear here too) while
# leaving $PAGE_DIR/index.html and $PAGE_DIR/embed.js intact.
rm -rf "$TARGET_REPORTS"
mkdir -p \
  "$TARGET_REPORTS/html" \
  "$TARGET_REPORTS/figures" \
  "$TARGET_REPORTS/excel" \
  "$TARGET_REPORTS/metrics" \
  "$TARGET_REPORTS/runs"

# Curated allow-list copy (NOT a blanket cp of reports/). Drop backtest_report.html:
# it is byte-identical to latest_report.html, so shipping it would waste ~7 MB.
cp "$SOURCE_REPORTS/html/latest_report.html" "$TARGET_REPORTS/html/"
cp "$SOURCE_REPORTS/html/frontier.html" "$TARGET_REPORTS/html/"
cp "$SOURCE_REPORTS"/figures/*.png "$TARGET_REPORTS/figures/"
cp "$SOURCE_REPORTS"/excel/*.xlsx "$TARGET_REPORTS/excel/"
cp "$SOURCE_REPORTS/metrics/backtest_metrics.json" "$TARGET_REPORTS/metrics/"
cp "$SOURCE_REPORTS/metrics/validation_summary.json" "$TARGET_REPORTS/metrics/"
cp "$SOURCE_REPORTS"/runs/*.json "$TARGET_REPORTS/runs/"

printf 'Synced curated reports: %s -> %s\n' "$SOURCE_REPORTS" "$TARGET_REPORTS"
printf 'Left untouched: %s/index.html, %s/embed.js (hand-authored)\n' "$PAGE_DIR" "$PAGE_DIR"
