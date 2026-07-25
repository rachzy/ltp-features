#!/usr/bin/env python3
"""Compare extracted pipeline features against confirmed catalog values."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Columns where a large mismatch is often expected (different conventions / units).
KNOWN_CAVEATS = {
    "t0": (
        "Epoch zero-points often differ (e.g. BKJD ≈ BJD − 2454833 for Kepler). "
        "A ~2454833 day offset is usually a reference-frame difference, not a bad fit."
    ),
}


def _load_feature_row(path: str | Path) -> pd.Series:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{path} has no data rows")
    return df.iloc[0]


def _as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def percent_difference(extracted: float, confirmed: float) -> float:
    """Return ((extracted - confirmed) / |confirmed|) * 100."""
    if not np.isfinite(extracted) or not np.isfinite(confirmed):
        return float("nan")
    if confirmed == 0.0:
        return 0.0 if extracted == 0.0 else float("inf")
    return float((extracted - confirmed) / abs(confirmed) * 100.0)


def compare_feature_rows(
    extracted: pd.Series,
    confirmed: pd.Series,
) -> pd.DataFrame:
    """Build a comparison table for columns present in both rows."""
    shared = [c for c in confirmed.index if c in extracted.index]
    rows = []
    for col in shared:
        ext = _as_float(extracted[col])
        conf = _as_float(confirmed[col])
        if not np.isfinite(ext) and not np.isfinite(conf):
            continue
        pct = percent_difference(ext, conf)
        rows.append(
            {
                "feature": col,
                "extracted": ext,
                "confirmed": conf,
                "abs_diff": abs(ext - conf) if np.isfinite(ext) and np.isfinite(conf) else np.nan,
                "pct_diff": pct,
                "caveat": KNOWN_CAVEATS.get(col, ""),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=["feature", "extracted", "confirmed", "abs_diff", "pct_diff", "caveat"]
        )

    out = pd.DataFrame(rows)
    # Sort by |%| descending so the biggest caveats float to the top.
    out["_sort"] = out["pct_diff"].abs()
    out.loc[~np.isfinite(out["_sort"]), "_sort"] = np.inf
    out = out.sort_values("_sort", ascending=False).drop(columns="_sort").reset_index(drop=True)
    return out


def _fmt_num(value: float, width: int = 14) -> str:
    if not np.isfinite(value):
        return f"{'nan':>{width}}"
    if abs(value) >= 1e4 or (abs(value) > 0 and abs(value) < 1e-3):
        return f"{value:>{width}.6g}"
    return f"{value:>{width}.6f}".rstrip("0").rstrip(".").rjust(width)


def _fmt_pct(value: float, width: int = 10) -> str:
    if not np.isfinite(value):
        if np.isinf(value):
            return f"{'inf':>{width}}"
        return f"{'n/a':>{width}}"
    return f"{value:>+{width}.2f}%"


def _severity_marker(pct: float) -> str:
    if not np.isfinite(pct):
        return "·"
    ap = abs(pct)
    if ap >= 50:
        return "!!"
    if ap >= 10:
        return "!"
    if ap >= 1:
        return "~"
    return "ok"


def format_comparison_report(
    comparison: pd.DataFrame,
    extracted_path: str | Path,
    confirmed_path: str | Path,
) -> str:
    """Pretty multi-line report highlighting largest % mismatches first."""
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append(" EXTRACTED vs CONFIRMED COMPARISON")
    lines.append("=" * 78)
    lines.append(f" Extracted : {extracted_path}")
    lines.append(f" Confirmed : {confirmed_path}")
    lines.append("-" * 78)

    if comparison.empty:
        lines.append(" No overlapping numeric columns to compare.")
        lines.append("=" * 78)
        return "\n".join(lines)

    lines.append(
        f" Matching columns: {len(comparison)}"
        f"  |  sorted by |% difference| (largest caveats first)"
    )
    lines.append("")
    header = (
        f"{'':>2}  {'feature':<28} {'extracted':>14} {'confirmed':>14} "
        f"{'% diff':>10}  {'|Δ|':>12}"
    )
    lines.append(header)
    lines.append("-" * 78)

    for _, row in comparison.iterrows():
        mark = _severity_marker(row["pct_diff"])
        lines.append(
            f"{mark:>2}  {row['feature']:<28} "
            f"{_fmt_num(row['extracted'])} {_fmt_num(row['confirmed'])} "
            f"{_fmt_pct(row['pct_diff'])}  {_fmt_num(row['abs_diff'], 12)}"
        )

    caveats = comparison[comparison["caveat"].astype(bool)]
    if not caveats.empty:
        lines.append("")
        lines.append(" Known caveats")
        lines.append("-" * 78)
        for _, row in caveats.iterrows():
            lines.append(f" • {row['feature']}: {row['caveat']}")

    # Quick summary of severity buckets
    pcts = comparison["pct_diff"].to_numpy(dtype=float)
    finite = pcts[np.isfinite(pcts)]
    n_ok = int(np.sum(np.abs(finite) < 1))
    n_mild = int(np.sum((np.abs(finite) >= 1) & (np.abs(finite) < 10)))
    n_warn = int(np.sum((np.abs(finite) >= 10) & (np.abs(finite) < 50)))
    n_bad = int(np.sum(np.abs(finite) >= 50) + np.sum(~np.isfinite(pcts)))

    lines.append("")
    lines.append(" Severity legend:  ok <1%   ~ 1–10%   ! 10–50%   !! ≥50% / undefined")
    lines.append(
        f" Counts: ok={n_ok}  ~={n_mild}  !={n_warn}  !!={n_bad}"
    )
    lines.append("=" * 78)
    return "\n".join(lines)


def compare_extracted_confirmed(
    extracted_path: str | Path,
    confirmed_path: str | Path,
    *,
    print_report: bool = True,
) -> pd.DataFrame:
    """Compare matching columns between an extracted and a confirmed CSV.

    Parameters
    ----------
    extracted_path, confirmed_path:
        Wide two-row CSVs (header + one values row).
    print_report:
        If True, print a prettified percentage-difference report.

    Returns
    -------
    pd.DataFrame
        Per-feature comparison sorted by absolute percent difference.
    """
    extracted_path = Path(extracted_path)
    confirmed_path = Path(confirmed_path)

    extracted = _load_feature_row(extracted_path)
    confirmed = _load_feature_row(confirmed_path)
    comparison = compare_feature_rows(extracted, confirmed)

    if print_report:
        print(format_comparison_report(comparison, extracted_path, confirmed_path))

    return comparison


def find_confirmed_csv(
    planet_name: str,
    confirmed_dir: str | Path,
) -> Path | None:
    """Return `{planet}-confirmed.csv` if present (also accepts `confimed` typo)."""
    confirmed_dir = Path(confirmed_dir)
    if not confirmed_dir.is_dir():
        return None

    candidates = [
        confirmed_dir / f"{planet_name}-confirmed.csv",
        confirmed_dir / f"{planet_name}-confimed.csv",  # historical typo
    ]
    for path in candidates:
        if path.is_file():
            return path

    # Last resort: any file whose stem starts with the planet name and mentions confirm.
    matches = sorted(
        p
        for p in confirmed_dir.glob(f"{planet_name}*")
        if p.is_file() and "confirm" in p.stem.lower()
    )
    return matches[0] if matches else None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare matching columns between an extracted-features CSV and a "
            "confirmed-catalog CSV; print percentage differences."
        )
    )
    parser.add_argument("extracted", type=Path, help="Path to extracted features CSV")
    parser.add_argument("confirmed", type=Path, help="Path to confirmed features CSV")
    args = parser.parse_args(argv)
    compare_extracted_confirmed(args.extracted, args.confirmed)


if __name__ == "__main__":
    main()
