#!/usr/bin/env python3
"""Fetch confirmed-planet literature parameters from the NASA Exoplanet Archive.

Replaces manually copying values into ``data/confirmed/`` by querying the
Kepler Objects of Interest (KOI) "cumulative" table for every confirmed
planet of a host star and writing them out in the same wide format already
used by the hand-built confirmed CSVs.

Usage (from ``src/scripts`` with the project venv active)::

    python get_literature_data.py Kepler-5
    python get_literature_data.py Kepler-11b
    python get_literature_data.py Kepler-90 --out-dir /tmp/confirmed

Only ``--mission Kepler`` (the default) is implemented today; the flag exists
so K2/TESS support can be added later without changing the CLI surface.

Caveats inherited from the KOI cumulative table (not bugs in this script):
eccentricity is fixed to 0 by the default Kepler pipeline fit, and
``koi_impact``/``koi_incl``/``koi_dor``/``koi_ror`` reflect that pipeline fit
rather than a refined literature value, so they can disagree with
hand-curated CSVs sourced from discovery papers. ``max_ses``/``max_mes``
come straight from the same pipeline the existing CSVs already use, so those
two columns should match closely. Planets without an official KOI number
(e.g. Kepler-90i, found by a non-pipeline search) will not appear.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable
from urllib.parse import urlencode

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent
REPO_ROOT = SRC_DIR.parent

# Allow `python get_literature_data.py ...` from src/scripts without installing the package.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.ephemeris import MISSION_TIME_OFFSETS  # noqa: E402
from utils.target_names import host_star_name  # noqa: E402

DEFAULT_CONFIRMED_DIR = REPO_ROOT / "data" / "confirmed"

TAP_SYNC_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

# Columns pulled from the KOI table; only the koi_* ones below are mapped
# into the confirmed-CSV output, the rest are used for filtering/logging.
KOI_SELECT_COLUMNS = (
    "kepid",
    "kepoi_name",
    "kepler_name",
    "koi_disposition",
    "koi_period",
    "koi_time0bk",
    "koi_duration",
    "koi_depth",
    "koi_ror",
    "koi_impact",
    "koi_incl",
    "koi_dor",
    "koi_eccen",
    "koi_prad",
    "koi_srad",
    "koi_max_sngle_ev",
    "koi_max_mult_ev",
)

# KOI column -> confirmed-CSV column (unit-compatible, direct copy).
KOI_COLUMN_MAP = {
    "koi_period": "period_days",
    "koi_duration": "duration_hours",
    "koi_ror": "planet_to_star_radius_ratio",
    "koi_impact": "impact_parameter",
    "koi_incl": "inclination",
    "koi_dor": "scaled_semi_major_axis",
    "koi_eccen": "eccentricity",
    "koi_srad": "stellar_radius",
    "koi_max_sngle_ev": "max_ses",
    "koi_max_mult_ev": "max_mes",
}

# Column order matches the existing data/confirmed/*.csv files exactly.
CONFIRMED_COLUMNS = [
    "target",
    "period_days",
    "t0",
    "duration_days",
    "duration_hours",
    "depth_mean_per_transit",
    "planet_to_star_radius_ratio",
    "impact_parameter",
    "inclination",
    "scaled_semi_major_axis",
    "eccentricity",
    "planet_radius_rjup",
    "stellar_radius",
    "max_ses",
    "max_mes",
]

# koi_depth is reported in parts-per-million; the pipeline stores a fraction.
PPM_PER_UNITY = 1_000_000.0

# koi_prad is reported in Earth radii; the pipeline stores Jupiter radii.
# Equatorial radius ratio: 71,492 km (Jupiter) / 6,378.1 km (Earth).
EARTH_RADII_PER_JUPITER_RADIUS = 11.209


class LiteratureDataError(RuntimeError):
    """Raised when literature data can't be found/fetched for a target."""


def _build_koi_query(star: str, *, table: str = "cumulative") -> str:
    """Build the ADQL query for every KOI row belonging to ``star``.

    Matching is done against ``kepler_name`` with a trailing space (e.g.
    ``lower(kepler_name) like 'kepler-5 %'``) so that ``Kepler-5`` doesn't
    also match ``Kepler-50``. Comparison is done in lowercase because the
    archive's SQL backend doesn't support ``ILIKE``.
    """
    safe_star = star.strip().replace("'", "''").lower()
    columns = ", ".join(KOI_SELECT_COLUMNS)
    return f"select {columns} from {table} where lower(kepler_name) like '{safe_star} %'"


def fetch_koi_table(
    star: str,
    *,
    mission: str = "Kepler",
    base_url: str = TAP_SYNC_URL,
) -> pd.DataFrame:
    """Query the NASA Exoplanet Archive for every KOI row of a host star."""
    if mission.strip().lower() != "kepler":
        raise NotImplementedError(
            f"get_literature_data only supports mission='Kepler' for now (got {mission!r})"
        )

    query = _build_koi_query(star)
    url = f"{base_url}?{urlencode({'query': query, 'format': 'csv'})}"
    try:
        return pd.read_csv(url)
    except Exception as exc:  # network error, bad response, malformed CSV, ...
        raise LiteratureDataError(
            f"Failed to query the NASA Exoplanet Archive for {star!r}: {exc}"
        ) from exc


def koi_rows_to_confirmed(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert raw KOI rows into the wide confirmed-CSV row format."""
    if raw.empty:
        return pd.DataFrame(columns=CONFIRMED_COLUMNS)

    df = raw.copy()
    if "koi_disposition" in df.columns:
        df = df[df["koi_disposition"] == "CONFIRMED"]
    if "kepler_name" in df.columns:
        df = df[df["kepler_name"].notna()]
    if df.empty:
        return pd.DataFrame(columns=CONFIRMED_COLUMNS)

    out = pd.DataFrame(index=df.index)
    out["target"] = df["kepler_name"].str.strip().str.replace(" ", "", regex=False)

    for koi_col, confirmed_col in KOI_COLUMN_MAP.items():
        out[confirmed_col] = pd.to_numeric(df.get(koi_col), errors="coerce")

    out["depth_mean_per_transit"] = pd.to_numeric(df.get("koi_depth"), errors="coerce") / PPM_PER_UNITY
    out["duration_days"] = out["duration_hours"] / 24.0

    prad_rearth = pd.to_numeric(df.get("koi_prad"), errors="coerce")
    out["planet_radius_rjup"] = prad_rearth / EARTH_RADII_PER_JUPITER_RADIUS

    # koi_time0bk is a Kepler-relative BKJD epoch; store t0 as full BJD, like
    # the hand-curated confirmed CSVs do.
    time0bk = pd.to_numeric(df.get("koi_time0bk"), errors="coerce")
    out["t0"] = time0bk + MISSION_TIME_OFFSETS["kepler"]

    numeric_cols = [c for c in CONFIRMED_COLUMNS if c != "target"]
    out[numeric_cols] = out[numeric_cols].astype(float)

    out = out[CONFIRMED_COLUMNS]
    out = out.drop_duplicates(subset="target", keep="first")
    out = out.sort_values("period_days", kind="stable").reset_index(drop=True)
    return out


def get_literature_data(
    target: str,
    *,
    mission: str = "Kepler",
    out_dir: str | Path | None = None,
    fetch: Callable[..., pd.DataFrame] = fetch_koi_table,
) -> tuple[Path, pd.DataFrame]:
    """Fetch, convert, and save literature data for ``target``'s host star.

    ``target`` may name either the star (``Kepler-5``) or one of its planets
    (``Kepler-5b``); the saved file is always keyed by the star name.

    Returns the path written and the confirmed-format DataFrame saved to it.
    """
    star = host_star_name(target)
    raw = fetch(star, mission=mission)
    confirmed = koi_rows_to_confirmed(raw)
    if confirmed.empty:
        raise LiteratureDataError(
            f"No confirmed KOI rows found for host star '{star}' (mission={mission})."
        )

    out_dir = Path(out_dir) if out_dir is not None else DEFAULT_CONFIRMED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{star}-confirmed.csv"
    confirmed.to_csv(out_path, index=False)
    return out_path, confirmed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch confirmed literature parameters for a star (or one of its "
            "planets) from the NASA Exoplanet Archive KOI dataset and save "
            "them to data/confirmed/{star}-confirmed.csv."
        )
    )
    parser.add_argument(
        "target",
        help="Star or planet name, e.g. Kepler-5 or Kepler-5b",
    )
    parser.add_argument(
        "--mission",
        default="Kepler",
        help="Mission catalog to query (default: Kepler; only Kepler is supported today)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_CONFIRMED_DIR,
        help="Directory for the saved confirmed CSV (default: data/confirmed)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    star = host_star_name(args.target)

    print(f"\n=== Fetching {args.mission} literature data: {star} ===\n")
    try:
        out_path, confirmed = get_literature_data(
            args.target, mission=args.mission, out_dir=args.out_dir
        )
    except (LiteratureDataError, NotImplementedError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    planets = ", ".join(confirmed["target"])
    print(f"Saved {len(confirmed)} planet(s) to {out_path}: {planets}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
