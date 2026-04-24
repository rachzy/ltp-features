#!/usr/bin/env python3
"""
Extract features for a LightKurve target using src/pipeline.extract_all_features_v2.
Saves the downloaded light curve data to CSV and uses it for feature extraction.

Usage (from repo root with venv active):
  python pre_processing/helpers/extract_lk.py --target HAT-P-7 --mission Kepler --out pre_processing/hatp7_features.csv
  python pre_processing/helpers/extract_lk.py --target HAT-P-7 --mission Kepler --download-all --out pre_processing/hatp7_features.csv
"""

import os
import sys
import argparse
import json
import pandas as pd
from datetime import datetime
import lightkurve as lk

THIS_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.join(THIS_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# Add the parent directory to sys.path to import pipeline
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


from pipeline import extract_all_features_from_csv


def download_and_save_lightcurve(target: str, mission: str, sigma_clip: float, verbose: bool, download_all: bool = False):
    """Download light curve data and save to CSV file."""
    if verbose:
        print(f"Downloading light curve for {target} from {mission}...")
    
    # Search for light curves
    search_result = lk.search_lightcurve(target, mission=mission)
    
    # Download based on download_all parameter
    if download_all:
        if verbose:
            print(f"Downloading all {len(search_result)} available files...")
        lc_collection = search_result.download_all()
        # Combine all light curves into one
        lc = lc_collection.stitch()
    else:
        lc = search_result.download()
    
    lc = lc.remove_nans()
    lc = lc.normalize()
    lc = lc.remove_outliers(sigma=sigma_clip)
    
    # Create data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "lightkurve")
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate filename with current date
    current_date = datetime.now().strftime("%Y%m%d")
    star_name = target.replace(" ", "_").replace("-", "_")
    csv_filename = f"{star_name}_{current_date}.csv"
    csv_path = os.path.join(data_dir, csv_filename)
    
    # Save to CSV
    df = pd.DataFrame({
        'time': lc.time.value,
        'flux': lc.flux.value,
        'flux_err': lc.flux_err.value if hasattr(lc, 'flux_err') and lc.flux_err is not None else None
    })
    df.to_csv(csv_path, index=False)
    
    if verbose:
        print(f"Saved light curve data to: {csv_path}")
    
    return csv_path, lc


def run_lightkurve_extraction(target: str, mission: str, sigma_clip: float, verbose: bool, download_all: bool = False):
    """Download light curve, save to CSV, and extract features."""
    # Download and save the light curve
    csv_path, lc = download_and_save_lightcurve(target, mission, sigma_clip, verbose, download_all)
    
    # Extract features using the saved CSV data
    feats = extract_all_features_from_csv(csv_path=csv_path, verbose=verbose)
    return feats


def parse_args():
    p = argparse.ArgumentParser(description="Extract features using LightKurve search")
    p.add_argument("--target", required=True, help="Target name, e.g., HAT-P-7")
    p.add_argument("--lightcurve-path", help="Path to light curve data")
    p.add_argument("--mission", default="Kepler", help="Mission name (Kepler/TESS)")
    p.add_argument("--sigma-clip", type=float, default=5.0, help="Outlier sigma for cleaning")
    p.add_argument("--download-all", action="store_true", help="Download all available light curve files instead of just the first one")
    p.add_argument("--out", required=True, help="Path to output (csv or json)")
    p.add_argument("--quiet", action="store_true", help="Reduce verbosity")
    return p.parse_args()


def main():
    args = parse_args()
    if args.lightcurve_path:
        feats = extract_all_features_from_csv(csv_path=args.lightcurve_path, verbose=not args.quiet)
    else:
        feats = run_lightkurve_extraction(
            target=args.target,
            mission=args.mission,
            sigma_clip=args.sigma_clip,
            verbose=not args.quiet,
            download_all=args.download_all,
        )

    out_path = args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if out_path.lower().endswith(".json"):
        with open(out_path, "w") as f:
            json.dump({"target": args.target, "features": feats}, f, indent=2)
    else:
        df = pd.DataFrame([feats])
        df.insert(0, "target", args.target)
        df.to_csv(out_path, index=False)

    print(f"Saved features to: {out_path}")


if __name__ == "__main__":
    main()


