#!/usr/bin/env python3
"""Compare extracted pipeline features against confirmed catalog values."""

from __future__ import annotations

import argparse
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .target_names import host_star_name
    from .ephemeris import align_catalog_epoch
except ImportError:  # Allow direct execution: python src/utils/compare_extracted_confirmed.py
    from target_names import host_star_name
    from ephemeris import align_catalog_epoch

# Columns where a large mismatch is often expected (different conventions / units).
KNOWN_CAVEATS = {
    "t0": (
        "Epochs are converted to the mission time base and aligned modulo the "
        "period. Its % difference is the phase offset divided by transit duration, "
        "not a raw percentage of Julian date."
    ),
}

# Two rows describe the same planet when their periods agree this closely.
# Recovered periods land within ~0.01% in practice, so this is loose enough to
# survive a poor fit and far tighter than the spacing of real planets.
PERIOD_MATCH_TOLERANCE = 0.02
# A search can lock onto a small integer ratio of the true period (a P/3 alias,
# or a 2P/3 harmonic). Recognising those is more informative than reporting the
# planet as missed and the alias as an unrelated false positive.
ALIAS_MAX_NUMERATOR = 5
ALIAS_MAX_DENOMINATOR = 5


def _load_feature_rows(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{path} has no data rows")
    if "period_days" in df.columns:
        periods = pd.to_numeric(df["period_days"], errors="coerce")
        df = (
            df.assign(_period_sort=periods)
            .sort_values("_period_sort", kind="stable", na_position="last")
            .drop(columns="_period_sort")
        )
    return df.reset_index(drop=True)


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


def _alias_ratios() -> list[Fraction]:
    """Small rational period ratios, simplest first.

    Ordered so that a direct match is always considered before any alias, and
    milder aliases (1/2, 2) before wilder ones (1/5, 5).
    """
    ratios = {
        Fraction(numerator, denominator)
        for numerator in range(1, ALIAS_MAX_NUMERATOR + 1)
        for denominator in range(1, ALIAS_MAX_DENOMINATOR + 1)
    }
    ratios.discard(Fraction(1))
    return [Fraction(1)] + sorted(
        ratios, key=lambda ratio: (abs(np.log(float(ratio))), float(ratio))
    )


def _ratio_label(ratio: Fraction) -> str:
    """Human-readable form of an extracted/confirmed period ratio."""
    if ratio == 1:
        return "direct"
    numerator, denominator = ratio.numerator, ratio.denominator
    if denominator == 1:
        return f"{numerator}P"
    if numerator == 1:
        return f"P/{denominator}"
    return f"{numerator}P/{denominator}"


def _period_column(rows: pd.DataFrame) -> np.ndarray:
    if "period_days" not in rows.columns:
        return np.full(len(rows), np.nan)
    return pd.to_numeric(rows["period_days"], errors="coerce").to_numpy(dtype=float)


def _row_label(rows: pd.DataFrame, index: int, fallback: str) -> str:
    if "target" in rows.columns:
        value = rows.iloc[index].get("target", np.nan)
        if pd.notna(value) and str(value).strip():
            return str(value).strip()
    return fallback


def match_candidate_rows(
    extracted_rows: pd.DataFrame,
    confirmed_rows: pd.DataFrame,
    *,
    tolerance: float = PERIOD_MATCH_TOLERANCE,
) -> list[dict]:
    """Pair extracted and confirmed candidates by orbital period.

    Rows are matched on period proximity, not on sorted position. Positional
    pairing silently cross-matches unrelated planets as soon as one candidate
    is missed or one spurious candidate is added: every later row shifts by
    one, and the resulting epoch differences look catastrophic even when every
    recovered ephemeris is accurate to minutes.

    Returns one record per pairing and per leftover row, with ``kind`` in
    ``direct`` (same period), ``alias`` (a small integer ratio of it),
    ``missed`` (confirmed planet with no extracted counterpart) or ``extra``
    (extracted candidate matching nothing, i.e. a likely false positive).
    """
    extracted_periods = _period_column(extracted_rows)
    confirmed_periods = _period_column(confirmed_rows)
    ratios = _alias_ratios()

    pairings = []
    for extracted_index, extracted_period in enumerate(extracted_periods):
        if not (np.isfinite(extracted_period) and extracted_period > 0):
            continue
        for confirmed_index, confirmed_period in enumerate(confirmed_periods):
            if not (np.isfinite(confirmed_period) and confirmed_period > 0):
                continue
            for ratio in ratios:
                expected = float(ratio) * confirmed_period
                relative = abs(extracted_period - expected) / expected
                if relative > tolerance:
                    continue
                # Ratios are ordered simplest-first and are spaced far wider
                # than the tolerance, so the first hit is the only hit.
                pairings.append(
                    (
                        0 if ratio == 1 else 1,
                        relative,
                        extracted_index,
                        confirmed_index,
                        ratio,
                    )
                )
                break
    pairings.sort()

    matches: list[dict] = []
    claimed_extracted: set[int] = set()
    claimed_confirmed: set[int] = set()
    for _priority, relative, extracted_index, confirmed_index, ratio in pairings:
        if extracted_index in claimed_extracted or confirmed_index in claimed_confirmed:
            continue
        claimed_extracted.add(extracted_index)
        claimed_confirmed.add(confirmed_index)
        matches.append(
            {
                "kind": "direct" if ratio == 1 else "alias",
                "extracted_index": extracted_index,
                "confirmed_index": confirmed_index,
                "ratio": float(ratio),
                "ratio_label": _ratio_label(ratio),
                "period_rel_diff": float(relative),
                "target": _row_label(
                    confirmed_rows, confirmed_index, f"candidate-{confirmed_index + 1}"
                ),
                "extracted_period": float(extracted_periods[extracted_index]),
                "confirmed_period": float(confirmed_periods[confirmed_index]),
            }
        )

    for confirmed_index in range(len(confirmed_periods)):
        if confirmed_index in claimed_confirmed:
            continue
        matches.append(
            {
                "kind": "missed",
                "extracted_index": None,
                "confirmed_index": confirmed_index,
                "ratio": float("nan"),
                "ratio_label": "missed",
                "period_rel_diff": float("nan"),
                "target": _row_label(
                    confirmed_rows, confirmed_index, f"candidate-{confirmed_index + 1}"
                ),
                "extracted_period": float("nan"),
                "confirmed_period": float(confirmed_periods[confirmed_index]),
            }
        )

    for extracted_index in range(len(extracted_periods)):
        if extracted_index in claimed_extracted:
            continue
        extracted = extracted_rows.iloc[extracted_index]
        matches.append(
            {
                "kind": "extra",
                "extracted_index": extracted_index,
                "confirmed_index": None,
                "ratio": float("nan"),
                "ratio_label": "unmatched",
                "period_rel_diff": float("nan"),
                "target": None,
                "extracted_period": float(extracted_periods[extracted_index]),
                "confirmed_period": float("nan"),
                "extracted_max_mes": _as_float(extracted.get("max_mes", np.nan)),
                "extracted_depth": _as_float(
                    extracted.get("depth_mean_per_transit", np.nan)
                ),
            }
        )

    order = {"direct": 0, "alias": 1, "missed": 2, "extra": 3}
    matches.sort(
        key=lambda match: (
            order[match["kind"]],
            match["confirmed_period"]
            if np.isfinite(match["confirmed_period"])
            else match["extracted_period"],
        )
    )
    return matches


def compare_feature_rows(
    extracted: pd.Series,
    confirmed: pd.Series,
    *,
    alignment_period: float | None = None,
) -> pd.DataFrame:
    """Build a comparison table for columns present in both rows.

    ``alignment_period`` overrides the period used to fold the catalog epoch
    onto the extracted one. For an alias pairing the two periods differ, and
    only the shorter one carries transits both ephemerides agree on.
    """
    shared = [c for c in confirmed.index if c in extracted.index]
    rows = []
    for col in shared:
        ext = _as_float(extracted[col])
        conf = _as_float(confirmed[col])
        if not np.isfinite(ext) and not np.isfinite(conf):
            continue
        abs_diff = (
            abs(ext - conf) if np.isfinite(ext) and np.isfinite(conf) else np.nan
        )
        pct = percent_difference(ext, conf)

        if col == "t0" and np.isfinite(ext) and np.isfinite(conf):
            period = _as_float(alignment_period) if alignment_period is not None else np.nan
            if not np.isfinite(period):
                period = _as_float(extracted.get("period_days", np.nan))
            if not np.isfinite(period):
                period = _as_float(confirmed.get("period_days", np.nan))
            target = confirmed.get("target", None)
            conf = align_catalog_epoch(conf, ext, period, target=target)
            phase_delta = ext - conf
            abs_diff = abs(phase_delta) if np.isfinite(phase_delta) else np.nan
            duration = _as_float(extracted.get("duration_days", np.nan))
            if not np.isfinite(duration) or duration <= 0:
                duration = _as_float(confirmed.get("duration_days", np.nan))
            pct = (
                float(100.0 * phase_delta / duration)
                if np.isfinite(phase_delta)
                and np.isfinite(duration)
                and duration > 0
                else float("nan")
            )
        rows.append(
            {
                "feature": col,
                "extracted": ext,
                "confirmed": conf,
                "abs_diff": abs_diff,
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
    *,
    candidate: str | None = None,
    candidate_index: int | None = None,
    match: dict | None = None,
) -> str:
    """Pretty multi-line report highlighting largest % mismatches first."""
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append(" EXTRACTED vs CONFIRMED COMPARISON")
    lines.append("=" * 78)
    lines.append(f" Extracted : {extracted_path}")
    lines.append(f" Confirmed : {confirmed_path}")
    if candidate is not None:
        ordinal = f"row {candidate_index + 1}, " if candidate_index is not None else ""
        lines.append(f" Candidate : {ordinal}{candidate}")
    if match is not None:
        expected = (
            "the catalog period"
            if match["kind"] == "direct"
            else f"{match['ratio_label']} of the catalog period"
        )
        lines.append(
            f" Pairing   : {match['kind']} — extracted period is within "
            f"{match['period_rel_diff'] * 100:.3f}% of {expected}"
        )
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
    lines.append(f" Counts: ok={n_ok}  ~={n_mild}  !={n_warn}  !!={n_bad}")
    lines.append("=" * 78)
    return "\n".join(lines)


def compare_extracted_confirmed(
    extracted_path: str | Path,
    confirmed_path: str | Path,
    *,
    print_report: bool = True,
) -> pd.DataFrame:
    """Compare candidate rows from extracted and confirmed CSVs.

    Parameters
    ----------
    extracted_path, confirmed_path:
        Wide CSVs containing one row per candidate. Rows are paired by orbital
        period (see :func:`match_candidate_rows`), including small integer
        aliases; unpaired rows on either side are reported rather than forced
        onto a neighbour.
    print_report:
        If True, print a prettified percentage-difference report.

    Returns
    -------
    pd.DataFrame
        Per-feature comparisons for every paired candidate. ``candidate`` and
        ``candidate_index`` identify the confirmed row used for each pairing,
        and ``match_kind``/``ratio_label`` record how the pair was found. The
        full pairing, including missed and unmatched rows, is available as
        ``result.attrs["matches"]``.
    """
    extracted_path = Path(extracted_path)
    confirmed_path = Path(confirmed_path)

    extracted_rows = _load_feature_rows(extracted_path)
    confirmed_rows = _load_feature_rows(confirmed_path)
    matches = match_candidate_rows(extracted_rows, confirmed_rows)
    paired = [m for m in matches if m["kind"] in ("direct", "alias")]

    if print_report:
        missed = [m for m in matches if m["kind"] == "missed"]
        extra = [m for m in matches if m["kind"] == "extra"]
        if missed or extra:
            print(
                f"Paired {len(paired)} of {len(confirmed_rows)} confirmed "
                f"candidate(s) against {len(extracted_rows)} extracted; "
                f"{len(missed)} missed, {len(extra)} unmatched.\n"
            )

    comparisons: list[pd.DataFrame] = []
    for candidate_index, match in enumerate(paired):
        extracted = extracted_rows.iloc[match["extracted_index"]]
        confirmed = confirmed_rows.iloc[match["confirmed_index"]]
        candidate = match["target"]

        # A direct pairing grades the extracted ephemeris on its own period, so
        # the epoch residual is the timing error where the extracted epoch sits
        # and the period error stays in its own column rather than being
        # counted twice. An alias has no such choice: only the shorter of the
        # two periods carries transits that both ephemerides predict.
        alignment_period = (
            match["extracted_period"]
            if match["kind"] == "direct"
            else min(match["extracted_period"], match["confirmed_period"])
        )
        comparison = compare_feature_rows(
            extracted, confirmed, alignment_period=alignment_period
        )
        comparison.insert(0, "candidate_index", candidate_index)
        comparison.insert(1, "candidate", candidate)
        comparison.insert(2, "match_kind", match["kind"])
        comparison.insert(3, "ratio_label", match["ratio_label"])
        comparisons.append(comparison)

        if print_report:
            print(
                format_comparison_report(
                    comparison,
                    extracted_path,
                    confirmed_path,
                    candidate=candidate,
                    candidate_index=candidate_index,
                    match=match,
                )
            )

    if not comparisons:
        empty = pd.DataFrame(
            columns=[
                "candidate_index",
                "candidate",
                "match_kind",
                "ratio_label",
                "feature",
                "extracted",
                "confirmed",
                "abs_diff",
                "pct_diff",
                "caveat",
            ]
        )
        empty.attrs["matches"] = matches
        return empty

    result = pd.concat(comparisons, ignore_index=True)
    result.attrs["matches"] = matches
    return result


def find_confirmed_csv(
    star_name: str,
    confirmed_dir: str | Path,
    *,
    auto_fetch: bool = True,
    mission: str = "Kepler",
) -> Path | None:
    """Return the confirmed candidate table for a host star.

    If no confirmed CSV exists yet, this falls back to fetching literature
    data for the star from the KOI dataset (see ``get_literature_data.py``)
    and saving it into ``confirmed_dir`` before giving up. Pass
    ``auto_fetch=False`` to skip that fallback and only look on disk.
    """
    confirmed_dir = Path(confirmed_dir)
    star_name = host_star_name(star_name)

    if confirmed_dir.is_dir():
        candidates = [
            confirmed_dir / f"{star_name}-confirmed.csv",
            confirmed_dir / f"{star_name}-confimed.csv",  # historical typo
        ]
        for path in candidates:
            if path.is_file():
                return path

        # Last resort: any file whose stem starts with the star name and mentions confirm.
        matches = sorted(
            p
            for p in confirmed_dir.glob(f"{star_name}*")
            if p.is_file() and "confirm" in p.stem.lower()
        )
        if matches:
            return matches[0]

    if not auto_fetch:
        return None

    return _fetch_missing_confirmed_csv(star_name, confirmed_dir, mission=mission)


def _fetch_missing_confirmed_csv(
    star_name: str,
    confirmed_dir: Path,
    *,
    mission: str,
) -> Path | None:
    """Best-effort fallback: fetch literature data when no confirmed CSV exists.

    Imported lazily so the network-hitting ``get_literature_data`` module
    isn't a hard import-time dependency of ``utils`` and any failure here
    (missing module, network error, no matching KOI rows, ...) degrades to
    the pre-existing "no confirmed CSV" behavior instead of raising.
    """
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    try:
        from get_literature_data import LiteratureDataError, get_literature_data
    except ImportError as exc:
        print(f"Could not import get_literature_data to auto-fetch {star_name}: {exc}")
        return None

    print(
        f"No confirmed CSV for {star_name} in {confirmed_dir}; "
        f"fetching literature data from the {mission} KOI dataset…"
    )
    try:
        out_path, _ = get_literature_data(star_name, mission=mission, out_dir=confirmed_dir)
    except (LiteratureDataError, NotImplementedError) as exc:
        print(f"Could not auto-fetch confirmed data for {star_name}: {exc}")
        return None

    print(f"Saved fetched literature data to {out_path}")
    return out_path


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
