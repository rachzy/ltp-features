#!/usr/bin/env python3
"""Compare every extracted star CSV with its confirmed candidate table.

Rows in both files are sorted by orbital period and matched by position. The
confirmed ``target`` column supplies the planet name displayed in reports.

Usage (from ``src/testing`` with the project venv active)::

    python compare_confirmed_and_extracted.py
    python compare_confirmed_and_extracted.py --verbose
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent
REPO_ROOT = SRC_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.compare_extracted_confirmed import (  # noqa: E402
    compare_extracted_confirmed,
    find_confirmed_csv,
)

EXTRACTED_NAME_RE = re.compile(r"^(?P<star>.+)_(?P<date>\d{8})\.csv$")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Match extracted feature CSVs against confirmed catalogs and "
            "print a candidates × properties % difference table."
        )
    )
    parser.add_argument(
        "--extracted-dir",
        type=Path,
        default=REPO_ROOT / "data" / "extracted",
        help="Directory of extracted feature CSVs (default: data/extracted)",
    )
    parser.add_argument(
        "--confirmed-dir",
        type=Path,
        default=REPO_ROOT / "data" / "confirmed",
        help="Directory of confirmed CSVs (default: data/confirmed)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the full per-candidate comparison report before the summary table",
    )
    parser.add_argument(
        "--minimal-columns",
        action="store_true",
        default=False,
        help=(
            "Print only the preferred period, depth, duration, max_ses and "
            "max_mes columns in the summary table"
        ),
    )
    return parser.parse_args(argv)


def _latest_extracted_by_star(extracted_dir: Path) -> dict[str, Path]:
    """Return the newest dated extracted candidate CSV per host star."""
    latest: dict[str, tuple[str, Path]] = {}
    for path in sorted(extracted_dir.glob("*.csv")):
        if not path.is_file():
            continue
        match = EXTRACTED_NAME_RE.match(path.name)
        if match is None:
            continue
        star = match.group("star")
        date = match.group("date")
        prev = latest.get(star)
        if prev is None or date > prev[0]:
            latest[star] = (date, path)
    return {star: path for star, (_date, path) in sorted(latest.items())}


# Shown as property columns; summary stats replace planet_radius_rearth.
EXCLUDED_PROPERTY_COLS = frozenset({"planet_radius_rearth"})
SUMMARY_COLS = frozenset({"mean_match_no_t0"})


def _format_pct_cell(value: float, *, signed: bool = True) -> str:
    if pd.isna(value):
        return "n/a"
    if value == float("inf") or value == float("-inf"):
        return "inf"
    if signed:
        return f"{value:+.2f}%"
    return f"{value:.2f}%"


def _match_score_no_t0(values: pd.Series) -> float:
    """0–100 match score from |%diff|: 100 = exact match, 0 = ≥100% off.

    Per property: ``max(0, 100 - |pct_diff|)``. Row score is the mean over
    finite property columns excluding ``t0`` (missing/n/a properties ignored).
    """
    cols = [c for c in values.index if c != "t0" and c not in SUMMARY_COLS]
    arr = pd.to_numeric(values[cols], errors="coerce").to_numpy(dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    scores = np.maximum(0.0, 100.0 - np.abs(finite))
    return float(np.mean(scores))


def build_pct_diff_table(
    pairs: list[tuple[str, Path, Path]],
    *,
    minimal_columns: bool,
    verbose: bool,
) -> pd.DataFrame:
    """Run comparisons and stack % diffs into a candidates × properties table."""
    rows: list[dict[str, float | str]] = []

    for star, extracted_path, confirmed_path in pairs:
        comparison = compare_extracted_confirmed(
            extracted_path,
            confirmed_path,
            print_report=verbose,
        )
        if comparison.empty:
            print(f"  {star}: no overlapping numeric columns — skipped")
            continue

        for _, candidate_comparison in comparison.groupby(
            "candidate_index", sort=True
        ):
            candidate = str(candidate_comparison.iloc[0]["candidate"])
            row: dict[str, float | str] = {"candidate": candidate}
            for _, feat_row in candidate_comparison.iterrows():
                feature = str(feat_row["feature"])
                if feature in EXCLUDED_PROPERTY_COLS:
                    continue
                row[feature] = float(feat_row["pct_diff"])
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    table = pd.DataFrame(rows).set_index("candidate")
    # Stable column order: prefer common transit properties first, then the rest.
    preferred = [
        "period_days",
        "depth_mean_per_transit",
        "duration_hours"
    ]

    if not minimal_columns:
        preferred += ["duration_days", "planet_radius_rjup", "t0"]

    property_cols = preferred
    # Preserve the established table shape even if every candidate lacks one
    # of the preferred properties; missing columns are displayed as ``n/a``.
    table = table.reindex(columns=property_cols)

    no_t0_cols = [c for c in property_cols if c != "t0"]
    table["mean_match_no_t0"] = (
        table[no_t0_cols].apply(_match_score_no_t0, axis=1)
        if no_t0_cols
        else float("nan")
    )
    return table.sort_values("mean_match_no_t0", ascending=False, na_position="last")


def _format_summary_table(table: pd.DataFrame) -> pd.DataFrame:
    display = pd.DataFrame(index=table.index)
    for col in table.columns:
        signed = col not in SUMMARY_COLS
        display[col] = table[col].map(lambda v, s=signed: _format_pct_cell(v, signed=s))
    return display


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.extracted_dir.is_dir():
        print(f"Extracted directory not found: {args.extracted_dir}", file=sys.stderr)
        return 1
    if not args.confirmed_dir.is_dir():
        print(f"Confirmed directory not found: {args.confirmed_dir}", file=sys.stderr)
        return 1

    extracted_by_star = _latest_extracted_by_star(args.extracted_dir)
    if not extracted_by_star:
        print(f"No extracted CSVs matching '*_YYYYMMDD.csv' in {args.extracted_dir}")
        return 0

    pairs: list[tuple[str, Path, Path]] = []
    unmatched: list[str] = []

    print(f"Scanning {len(extracted_by_star)} extracted star(s)…\n")
    for star, extracted_path in extracted_by_star.items():
        confirmed_path = find_confirmed_csv(star, args.confirmed_dir)
        if confirmed_path is None:
            unmatched.append(star)
            continue
        pairs.append((star, extracted_path, confirmed_path))
        print(f"  {star}: {extracted_path.name} ↔ {confirmed_path.name}")

    if unmatched:
        print(f"\nNo confirmed match ({len(unmatched)}): {', '.join(unmatched)}")

    if not pairs:
        print("\nNothing to compare.")
        return 0

    print(f"\nComparing candidates from {len(pairs)} matched star(s)…\n")
    table = build_pct_diff_table(
        pairs,
        verbose=args.verbose,
        minimal_columns=args.minimal_columns,
    )
    if table.empty:
        print("No overlapping properties across matched pairs.")
        return 0

    display = _format_summary_table(table)
    print("=" * 78)
    print(" % DIFFERENCE SUMMARY  (extracted vs confirmed)")
    print(" rows = candidates   |   columns = properties")
    print(
        " mean_match_no_t0 = mean max(0, 100-|%diff|) over properties except t0"
        "  (100% = perfect match; sorted desc)"
    )
    print("=" * 78)
    print(display.to_string())
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
