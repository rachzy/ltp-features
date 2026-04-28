# INTENDED TO BE USE WITH FLUX DATA
# DEPRECATED

#!/usr/bin/env python3
"""
Extract features from an exo-style CSV (LABEL + FLUX1..FLUXN) using the
array-based pipeline functions in src/pipeline.py.

Usage (from repo root with venv active):
  python pre_processing/helpers/extract_csv.py \
    --csv pre_processing/data/exoTrain.csv \
    --out pre_processing/batch_results.csv \
    --cadence-min 30 --label-col LABEL --flux-prefix FLUX --max-rows 128 --workers 8
"""

import os
import sys
import argparse
from collections import OrderedDict

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

THIS_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.join(THIS_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# Add the parent directory to sys.path to import pipeline
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from extract_feats import extract_features_from_arrays

# Desired final feature order for CSV outputs
DESIRED_FEATURE_ORDER = [
    "period_days",
    "t0",
    "duration_days",
    "duration_hours",
    "scale_mean",
    "scale_std",
    "scale_skewness",
    "scale_kurtosis",
    "scale_outlier_resistance",
    "local_noise",
    "depth_stability",
    "acf_lag_1h",
    "acf_lag_3h",
    "acf_lag_6h",
    "acf_lag_12h",
    "acf_lag_24h",
    "cadence_hours",
    "depth_mean_per_transit",
    "depth_std_per_transit",
    "npts_transit_median",
    "cdpp_3h",
    "cdpp_6h",
    "cdpp_12h",
    "SES_mean",
    "SES_std",
    "MES",
    "snr_global",
    "snr_per_transit_mean",
    "snr_per_transit_std",
    "resid_rms_global",
    "vshape_metric",
    "secondary_depth",
    "secondary_depth_snr",
    "secondary_depth_snr_log10",
    "secondary_depth_snr_capped",
    "secondary_to_primary_ratio",
    "secondary_is_eb_like",
    "odd_even_depth_ratio",
    "ingress_egress_asymmetry",
    "skewness_flux",
    "kurtosis_flux",
    "outlier_resistance",
    "planet_radius_rearth",
    "planet_radius_rjup",
]


def summarize_dataset_health(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate dataset-level noise/health metrics from per-row feature table.

    Returns a single-row DataFrame with robust statistics and completeness rates.
    """
    n_rows = len(df)
    out: OrderedDict = OrderedDict()
    out["rows"] = int(n_rows)
    if "error" in df.columns:
        err_rows = int(df["error"].notna().sum())
    else:
        err_rows = 0
    out["error_rows"] = err_rows
    out["error_rate"] = float(err_rows / n_rows) if n_rows > 0 else np.nan

    def add_stats(col: str):
        if col not in df.columns or n_rows == 0:
            out[f"{col}_finite_rate"] = np.nan
            out[f"{col}_median"] = np.nan
            out[f"{col}_p16"] = np.nan
            out[f"{col}_p84"] = np.nan
            out[f"{col}_mean"] = np.nan
            out[f"{col}_std"] = np.nan
            return
        s = pd.to_numeric(df[col], errors="coerce")
        finite = np.isfinite(s.values)
        out[f"{col}_finite_rate"] = (
            float(finite.sum() / n_rows) if n_rows > 0 else np.nan
        )
        if finite.any():
            v = s.values[finite]
            out[f"{col}_median"] = float(np.nanmedian(v))
            out[f"{col}_p16"] = float(np.nanpercentile(v, 16))
            out[f"{col}_p84"] = float(np.nanpercentile(v, 84))
            out[f"{col}_mean"] = float(np.nanmean(v))
            out[f"{col}_std"] = float(np.nanstd(v))
        else:
            out[f"{col}_median"] = np.nan
            out[f"{col}_p16"] = np.nan
            out[f"{col}_p84"] = np.nan
            out[f"{col}_mean"] = np.nan
            out[f"{col}_std"] = np.nan

    key_cols = [
        "local_noise",
        "cdpp_3h",
        "cdpp_6h",
        "cdpp_12h",
        "MES",
        "snr_global",
        "snr_per_transit_mean",
        "resid_rms_global",
        "cadence_hours",
        "duration_hours",
        "period_days",
        "npts_transit_median",
        "depth_mean_per_transit",
        "depth_std_per_transit",
    ]
    for c in key_cols:
        add_stats(c)

    # Simple data quality flags
    out["rows_with_low_points"] = (
        int((df.get("npts_transit_median", np.nan) < 3).sum())
        if "npts_transit_median" in df.columns
        else 0
    )
    out["rows_with_nan_depth"] = (
        int(
            pd.to_numeric(df.get("depth_mean_per_transit", np.nan), errors="coerce")
            .isna()
            .sum()
        )
        if "depth_mean_per_transit" in df.columns
        else 0
    )

    return pd.DataFrame([out])


def find_flux_columns(columns, flux_prefix: str) -> list:
    cols = []
    # Prefer no-dot style first (FLUX1..)
    for c in columns:
        if c.upper().startswith(flux_prefix.upper()):
            tail = c[len(flux_prefix) :]
            if tail.isdigit() or (tail.startswith(".") and tail[1:].isdigit()):
                cols.append(c)

    # Sort by numeric index
    def idx(name: str) -> int:
        tail = name[len(flux_prefix) :]
        return int(tail[1:] if tail.startswith(".") else tail)

    cols.sort(key=idx)
    return cols


def _row_worker(args):
    i, time, flux_values, label_value, label_col, refine_duration, use_tls = args
    out = OrderedDict()
    out["row_index"] = int(i)  # Keep for parallel processing order
    if label_col is not None:
        out["label"] = label_value
    try:
        # Replace non-finite by median before normalization
        med = np.nanmedian(flux_values)
        if not np.isfinite(med) or med == 0:
            med = 1.0
        flux_values = np.where(np.isfinite(flux_values), flux_values, med)
        flux_norm = flux_values / med
        feats = extract_features_from_arrays(
            time,
            flux_norm,
            verbose=False,
            refine_duration=refine_duration,
            use_tls=use_tls,
        )
        out.update(feats)
    except Exception as e:
        out["error"] = str(e)
    return out


def process_exo_csv(
    csv_path: str,
    output_path: str,
    cadence_minutes: float = 30.0,
    label_col: str = "LABEL",
    flux_prefix: str = "FLUX",
    max_rows: int | None = None,
    verbose: bool = True,
    n_workers: int | None = None,
    health_out: str | None = None,
    refine_duration: bool = True,
    use_tls: bool = False,
) -> pd.DataFrame:
    """Process an exo-style CSV and write per-row features to CSV."""
    df = pd.read_csv(csv_path)
    if max_rows is not None and max_rows > 0:
        df = df.head(max_rows)

    flux_cols = find_flux_columns(df.columns, flux_prefix=flux_prefix)
    if not flux_cols:
        raise ValueError(
            f"No flux columns found with prefix '{flux_prefix}'. Expected e.g. {flux_prefix}1..{flux_prefix}N or {flux_prefix}.1.."
        )

    n_points = len(flux_cols)
    dt_days = float(cadence_minutes) / 1440.0
    time = np.arange(n_points, dtype=float) * dt_days

    total = len(df)
    if n_workers is None:
        import os as _os

        n_workers = _os.cpu_count() or 1
    n_workers = max(1, int(n_workers))

    if verbose:
        mode = "parallel" if n_workers > 1 else "serial"
        print(f"Processing {total} rows in {mode} mode (workers={n_workers})")

    if n_workers == 1:
        results = []
        for i, row in df.iterrows():
            args = (
                i,
                time,
                pd.to_numeric(row[flux_cols], errors="coerce").values.astype(float),
                (
                    int(row[label_col])
                    if (label_col in df.columns and np.isfinite(row[label_col]))
                    else (row[label_col] if label_col in df.columns else None)
                ),
                (label_col if label_col in df.columns else None),
                refine_duration,
                use_tls,
            )
            results.append(_row_worker(args))
            if verbose and total >= 10 and (len(results) % max(1, total // 10) == 0):
                print(f"Processed {len(results)}/{total}")
        out_df = pd.DataFrame(results)
        # Remove row_index column and reorder: label first, then desired feature order
        if "row_index" in out_df.columns:
            out_df = out_df.drop(columns=["row_index"])
        cols = []
        if "label" in out_df.columns:
            cols.append("label")
        cols.extend([c for c in DESIRED_FEATURE_ORDER if c in out_df.columns])
        cols.extend([c for c in out_df.columns if c not in cols])
        out_df = out_df.reindex(columns=cols)
    else:
        results_by_index: list[OrderedDict | None] = [None] * total
        submitted = 0
        completed = 0
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = []
            for i, row in df.iterrows():
                args = (
                    i,
                    time,
                    pd.to_numeric(row[flux_cols], errors="coerce").values.astype(float),
                    (
                        int(row[label_col])
                        if (label_col in df.columns and np.isfinite(row[label_col]))
                        else (row[label_col] if label_col in df.columns else None)
                    ),
                    (label_col if label_col in df.columns else None),
                    refine_duration,
                    use_tls,
                )
                futures.append(ex.submit(_row_worker, args))
                submitted += 1
            for fut in as_completed(futures):
                res = fut.result()
                idx = res.get("row_index", None)
                if idx is not None and 0 <= idx < total:
                    results_by_index[idx] = res
                completed += 1
                if (
                    verbose
                    and total >= 10
                    and (completed % max(1, total // 10) == 0 or completed == total)
                ):
                    print(f"Completed {completed}/{total}")
        results = [r for r in results_by_index if r is not None]
        out_df = pd.DataFrame(results)
        # Remove row_index column and reorder: label first, then desired feature order
        if "row_index" in out_df.columns:
            out_df = out_df.drop(columns=["row_index"])
        cols = []
        if "label" in out_df.columns:
            cols.append("label")
        cols.extend([c for c in DESIRED_FEATURE_ORDER if c in out_df.columns])
        cols.extend([c for c in out_df.columns if c not in cols])
        out_df = out_df.reindex(columns=cols)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.to_csv(output_path, index=False)
    if verbose:
        print(f"Saved results to: {output_path}")
        print(f"Rows: {len(out_df)}  Columns: {len(out_df.columns)}")

    # Optional dataset-level health summary
    if health_out is not None:
        health_df = summarize_dataset_health(out_df)
        os.makedirs(os.path.dirname(health_out) or ".", exist_ok=True)
        if health_out.lower().endswith(".txt"):
            with open(health_out, "w") as f:
                f.write(health_df.to_string(index=False))
        else:
            health_df.to_csv(health_out, index=False)
        if verbose:
            print(f"Saved health summary to: {health_out}")
    return out_df


def parse_args():
    p = argparse.ArgumentParser(description="Extract features from exo-style CSV")
    p.add_argument(
        "--csv", required=True, help="Path to input CSV (exoTest/exoTrain style)"
    )
    p.add_argument("--out", required=True, help="Path to output CSV with features")
    p.add_argument(
        "--cadence-min",
        type=float,
        default=30.0,
        help="Cadence in minutes (default 30)",
    )
    p.add_argument(
        "--label-col", default="LABEL", help="Label column name (default LABEL)"
    )
    p.add_argument(
        "--flux-prefix", default="FLUX", help="Flux column prefix (default FLUX)"
    )
    p.add_argument(
        "--max-rows", type=int, default=None, help="Process only first N rows"
    )
    p.add_argument("--quiet", action="store_true", help="Reduce verbosity")
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count)",
    )
    p.add_argument(
        "--health-out",
        default=None,
        help="Optional path to dataset health summary (.csv or .txt)",
    )
    p.add_argument(
        "--no-refine",
        action="store_true",
        help="Disable two-pass BLS duration refinement",
    )
    p.add_argument(
        "--use-tls",
        action="store_true",
        help="Enable TransitLeastSquares refinement (adds runtime)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    process_exo_csv(
        csv_path=args.csv,
        output_path=args.out,
        cadence_minutes=args.cadence_min,
        label_col=args.label_col,
        flux_prefix=args.flux_prefix,
        max_rows=args.max_rows,
        verbose=not args.quiet,
        n_workers=args.workers,
        health_out=args.health_out,
        refine_duration=not args.no_refine,
        use_tls=args.use_tls,
    )


if __name__ == "__main__":
    main()
