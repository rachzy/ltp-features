"""Regression tests for star-keyed, multi-candidate feature tables."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from save import save_features  # noqa: E402
from utils.compare_extracted_confirmed import (  # noqa: E402
    compare_extracted_confirmed,
)
from utils.target_names import host_star_name  # noqa: E402


class CandidateCollectionTests(unittest.TestCase):
    def test_host_star_name_removes_planet_suffix(self):
        cases = {
            "Kepler-186f": "Kepler-186",
            "HAT-P-7b": "HAT-P-7",
            "EPIC201111557.01": "EPIC201111557",
            "Kepler-186": "Kepler-186",
            "HAT-P-7": "HAT-P-7",
        }
        for target, expected in cases.items():
            with self.subTest(target=target):
                self.assertEqual(host_star_name(target), expected)

    def test_save_features_writes_period_sorted_unlabeled_rows(self):
        rows = [
            {"period_days": 20.0, "depth_mean_per_transit": 0.02},
            {"period_days": 3.0, "depth_mean_per_transit": 0.01},
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "Kepler-1_20260801.csv"
            save_features(rows, "Kepler-1", str(output))
            saved = pd.read_csv(output)

        self.assertEqual(saved["period_days"].tolist(), [3.0, 20.0])
        self.assertNotIn("target", saved.columns)
        self.assertEqual(len(saved), 2)

    def test_comparison_pairs_rows_by_period_order(self):
        extracted = pd.DataFrame(
            [
                {"period_days": 20.0, "duration_hours": 4.0},
                {"period_days": 3.0, "duration_hours": 2.0},
            ]
        )
        confirmed = pd.DataFrame(
            [
                {"target": "Test-1c", "period_days": 20.0, "duration_hours": 4.0},
                {"target": "Test-1b", "period_days": 3.0, "duration_hours": 2.0},
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            extracted_path = Path(tmp_dir) / "Test-1_20260801.csv"
            confirmed_path = Path(tmp_dir) / "Test-1-confirmed.csv"
            extracted.to_csv(extracted_path, index=False)
            confirmed.to_csv(confirmed_path, index=False)
            comparison = compare_extracted_confirmed(
                extracted_path,
                confirmed_path,
                print_report=False,
            )

        labels = comparison.groupby("candidate_index", sort=True)["candidate"].first()
        self.assertEqual(labels.tolist(), ["Test-1b", "Test-1c"])
        self.assertTrue((comparison["pct_diff"] == 0.0).all())


if __name__ == "__main__":
    unittest.main()
