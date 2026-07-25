#!/usr/bin/env python3
"""Extract pipeline features for a target and compare against confirmed values.

Usage (from ``src/testing`` with the project venv active)::

    python extract_and_compare.py Kepler-5b
    python extract_and_compare.py HAT-P-7 --mission Kepler --download-all
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parent
REPO_ROOT = SRC_DIR.parent

# Allow `python extract_and_compare.py ...` from src/testing without installing the package.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from download_and_clean import download_and_clean_lightcurve  # noqa: E402
from extract_feats import extract_features_from_lightcurve  # noqa: E402
from save import save_features  # noqa: E402
from utils.compare_extracted_confirmed import (  # noqa: E402
    compare_extracted_confirmed,
    find_confirmed_csv,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download a named light curve, extract features, and compare "
            "against data/confirmed/{target}-confirmed.csv when available."
        )
    )
    parser.add_argument(
        "target",
        help="LightKurve / catalog target name, e.g. Kepler-5b",
    )
    parser.add_argument(
        "--mission",
        default="Kepler",
        help="Mission name passed to lightkurve (default: Kepler)",
    )
    parser.add_argument(
        "--sigma-clip",
        type=float,
        default=5.0,
        help="Outlier sigma for cleaning (default: 5.0)",
    )
    parser.add_argument(
        "--download-all",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stitch all available light curves (default: true)",
    )
    parser.add_argument(
        "--refine-duration",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Refine BLS duration (default: true)",
    )
    parser.add_argument(
        "--use-tls",
        action="store_true",
        help="Use TLS instead of BLS where supported",
    )
    parser.add_argument(
        "--mask-eclipses",
        action="store_true",
        help="Mask eclipses in the light curve",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "data" / "extracted",
        help="Directory for the extracted features CSV (default: data/extracted)",
    )
    parser.add_argument(
        "--confirmed-dir",
        type=Path,
        default=REPO_ROOT / "data" / "confirmed",
        help="Directory of confirmed CSVs (default: data/confirmed)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce extraction verbosity",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    verbose = not args.quiet

    print(f"\n=== Extract & compare: {args.target} ({args.mission}) ===\n")

    lc = download_and_clean_lightcurve(
        target=args.target,
        mission=args.mission,
        sigma_clip=args.sigma_clip,
        all=args.download_all,
        verbose=verbose,
    )

    feats = extract_features_from_lightcurve(
        lc,
        verbose=verbose,
        refine_duration=args.refine_duration,
        use_tls=args.use_tls,
        mask_eclipses=args.mask_eclipses,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    extracted_path = args.out_dir / f"{args.target}_{datetime.now().strftime('%Y%m%d')}.csv"
    save_features(feats, args.target, str(extracted_path), verbose=verbose)
    print(f"\nExtracted features saved to: {extracted_path}")

    confirmed_path = find_confirmed_csv(args.target, args.confirmed_dir)
    if confirmed_path is None:
        print(
            f"\nNo confirmed CSV for {args.target} in {args.confirmed_dir} "
            f"(expected `{args.target}-confirmed.csv`). Skipping comparison."
        )
        return 0

    print(f"Comparing against confirmed catalog: {confirmed_path}\n")
    compare_extracted_confirmed(extracted_path, confirmed_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
