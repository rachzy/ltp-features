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

THIS_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.join(THIS_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# Add the parent directory to sys.path to import pipeline
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from extract_feats import extract_features_from_lightcurve
from download_and_clean import download_and_clean_lightcurve
from save import save_features
from extract_feats import extract_all_features_from_csv


def run_lightkurve_extraction(
    target: str,
    mission: str,
    sigma_clip: float,
    verbose: bool,
    lightkurve_out_path: str = None,
    download_all: bool = False,
):
    """Download light curve, save to CSV, and extract features."""
    # Download and save the light curve
    lc = download_and_clean_lightcurve(
        target, mission, sigma_clip, download_all, lightkurve_out_path, verbose
    )

    # Extract features using the saved CSV data
    feats = extract_features_from_lightcurve(lc, verbose=verbose)
    return feats


def run_csv_extraction(csv_path: str, verbose: bool = False):
    """Extract features from a CSV file."""

    feats = extract_all_features_from_csv(csv_path, verbose=verbose)
    return feats


def parse_args():
    p = argparse.ArgumentParser(description="Extract features using LightKurve search")
    p.add_argument("--target", help="Target name, e.g., HAT-P-7")
    p.add_argument("--input-lightkurve", help="Path to input lightkurve data")
    p.add_argument("--out-lightkurve", help="Path to lightkurve data")
    p.add_argument("--mission", default="Kepler", help="Mission name (Kepler/TESS)")
    p.add_argument(
        "--sigma-clip", type=float, default=5.0, help="Outlier sigma for cleaning"
    )
    p.add_argument(
        "--download-all",
        action="store_true",
        help="Download all available light curve files instead of just the first one",
    )
    p.add_argument("--out-features", help="Path to output (csv or json)")
    p.add_argument("--quiet", action="store_true", help="Reduce verbosity")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.target and not args.input_lightkurve:
        raise ValueError("Either --target or --input-lightkurve must be provided")

    if args.input_lightkurve:
        feats = run_csv_extraction(args.input_lightkurve, verbose=not args.quiet)
    else:
        feats = run_lightkurve_extraction(
            target=args.target,
            mission=args.mission,
            sigma_clip=args.sigma_clip,
            verbose=not args.quiet,
            lightkurve_out_path=args.out_lightkurve,
            download_all=args.download_all,
        )

    if args.out_features:
        save_features(feats, args.target, args.out_features, not args.quiet)


if __name__ == "__main__":
    main()
