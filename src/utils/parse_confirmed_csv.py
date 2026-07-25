#!/usr/bin/env python3
"""Convert confirmed parameter CSVs into a wide two-row format.

Confirmed files are tall tables (Parameter, Symbol, Value, ...). This script
pivots them into a single header row + single values row, renaming columns that
have equivalents in the extraction pipeline and snake_casing the rest.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

# Direct parameter-name → pipeline column mappings (unit-independent).
PARAMETER_MAP = {
    "orbital period": "period_days",
    "mid-transit epoch": "t0",
    "transit depth": "depth_mean_per_transit",
    "transit depth (approx.)": "depth_mean_per_transit",
    "transit depth (approx)": "depth_mean_per_transit",
}

# Units that identify duration_days / duration_hours for "Transit duration (total)".
_DURATION_DAY_UNITS = ("day", "days", "d")
_DURATION_HOUR_UNITS = ("hour", "hours", "hr", "hrs", "h")

# Units that identify planet_radius_rjup / planet_radius_rearth.
_RJUP_UNITS = ("r_jupiter", "rjupiter", "rjup", "rj", "r_jup", "jupiter")
_REARTH_UNITS = ("r_earth", "rearth", "r⊕", "re", "r_earth", "earth", "r⊕")


def _normalize_text(text: str) -> str:
    text = str(text).strip().lower()
    text = text.replace("★", "star").replace("⊕", "earth").replace("⊙", "sun")
    text = re.sub(r"\s+", " ", text)
    return text


def to_snake_case(name: str) -> str:
    """Convert a free-form parameter label to snake_case."""
    name = str(name).strip()
    name = name.replace("★", "star").replace("⊕", "earth").replace("⊙", "sun")
    name = name.replace("/", "_").replace("-", "_")
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", "_", name.strip().lower())
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "unnamed"


def _unit_token(unit: str) -> str:
    unit = _normalize_text(unit)
    unit = unit.replace(" ", "_")
    return unit


def map_column_name(parameter: str, unit: str = "") -> str:
    """Map a confirmed parameter (+ unit) to an extraction-pipeline column name."""
    param_key = _normalize_text(parameter)
    unit_key = _unit_token(unit)

    if param_key in PARAMETER_MAP:
        return PARAMETER_MAP[param_key]

    if param_key.startswith("transit duration"):
        if any(tok == unit_key or unit_key.startswith(tok) for tok in _DURATION_DAY_UNITS):
            return "duration_days"
        if any(tok == unit_key or unit_key.startswith(tok) for tok in _DURATION_HOUR_UNITS):
            return "duration_hours"
        # Fallback: inspect raw unit string
        if "day" in unit_key:
            return "duration_days"
        if "hour" in unit_key or unit_key in {"h", "hr", "hrs"}:
            return "duration_hours"
        return to_snake_case(parameter)

    if param_key == "planet radius":
        if any(tok in unit_key for tok in _RJUP_UNITS):
            return "planet_radius_rjup"
        if any(tok in unit_key for tok in _REARTH_UNITS) or "earth" in unit_key:
            return "planet_radius_rearth"
        return to_snake_case(parameter)

    return to_snake_case(parameter)


def parse_value(raw) -> float | str:
    """Coerce confirmed Value cells into floats when possible."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return float("nan")

    text = str(raw).strip()
    if text in {"", "—", "-", "–", "−", "nan", "NaN", "None", "none"}:
        return float("nan")

    # Strip approximate markers / surrounding whitespace.
    text = text.lstrip("~≈∼")
    text = text.replace(",", "")

    # Keep the first numeric token (handles values like "~89.9").
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if match:
        try:
            return float(match.group(0))
        except ValueError:
            pass
    return text


def parse_confirmed_csv(
    input_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Read a tall confirmed CSV and write a wide two-row CSV.

    Parameters
    ----------
    input_path:
        Path to a confirmed CSV with columns Parameter, Symbol, Value, ...
    output_path:
        Destination path. Defaults to overwriting ``input_path``.

    Returns
    -------
    Path
        Path of the written wide CSV.
    """
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path is not None else input_path

    df = pd.read_csv(input_path)
    if "Parameter" not in df.columns or "Value" not in df.columns:
        raise ValueError(
            f"{input_path} must contain 'Parameter' and 'Value' columns; "
            f"found {list(df.columns)}"
        )

    unit_col = "Unit" if "Unit" in df.columns else None
    wide: dict[str, float | str] = {}

    for _, row in df.iterrows():
        parameter = row["Parameter"]
        if pd.isna(parameter) or str(parameter).strip() == "":
            continue

        unit = str(row[unit_col]) if unit_col and not pd.isna(row[unit_col]) else ""
        col = map_column_name(str(parameter), unit)

        # Prefer the first occurrence if a name collides.
        if col in wide:
            continue

        wide[col] = parse_value(row["Value"])

    out_df = pd.DataFrame([wide])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    return output_path


def _default_confirmed_dir() -> Path:
    # src/utils/parse_confirmed_csv.py → repo root → data/confirmed
    return Path(__file__).resolve().parents[2] / "data" / "confirmed"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert confirmed tall-parameter CSVs to wide two-row CSVs "
            "compatible with the extraction pipeline output."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="Confirmed CSV path(s). If omitted, process all *.csv in --confirmed-dir.",
    )
    parser.add_argument(
        "--confirmed-dir",
        type=Path,
        default=_default_confirmed_dir(),
        help="Directory of confirmed CSVs (default: data/confirmed).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path (only valid with a single input file).",
    )
    args = parser.parse_args(argv)

    inputs = list(args.inputs)
    if not inputs:
        inputs = sorted(args.confirmed_dir.glob("*.csv"))
        if not inputs:
            raise SystemExit(f"No CSV files found in {args.confirmed_dir}")

    if args.output is not None and len(inputs) != 1:
        raise SystemExit("--output can only be used with a single input file")

    for path in inputs:
        out = args.output if args.output is not None else path
        written = parse_confirmed_csv(path, out)
        print(f"Wrote {written}")


if __name__ == "__main__":
    main()
