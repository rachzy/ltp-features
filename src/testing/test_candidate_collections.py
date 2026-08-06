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
    match_candidate_rows,
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


class CandidateMatchingTests(unittest.TestCase):
    """Pairing must follow orbital period, never sorted row position."""

    @staticmethod
    def _frames(extracted_periods, confirmed):
        extracted = pd.DataFrame(
            [
                {"period_days": period, "duration_days": 0.2, "max_mes": 20.0}
                for period in extracted_periods
            ]
        )
        return extracted, pd.DataFrame(confirmed)

    def test_missed_planet_does_not_shift_every_later_row(self):
        # The pipeline misses the 14 d planet. Positional pairing would match
        # 60 d against it and cross-match everything after.
        extracted, confirmed = self._frames(
            [7.0, 60.0, 92.0],
            [
                {"target": "Test-2b", "period_days": 7.0},
                {"target": "Test-2i", "period_days": 14.0},
                {"target": "Test-2d", "period_days": 60.0},
                {"target": "Test-2e", "period_days": 92.0},
            ],
        )

        matches = match_candidate_rows(extracted, confirmed)
        by_target = {m["target"]: m for m in matches if m["target"]}

        self.assertEqual(by_target["Test-2b"]["kind"], "direct")
        self.assertEqual(by_target["Test-2d"]["kind"], "direct")
        self.assertEqual(by_target["Test-2e"]["kind"], "direct")
        self.assertEqual(by_target["Test-2i"]["kind"], "missed")
        self.assertEqual(by_target["Test-2d"]["extracted_period"], 60.0)

    def test_integer_ratio_aliases_are_labelled_not_reported_as_missed(self):
        extracted, confirmed = self._frames(
            [70.2, 210.6 * 2],
            [
                {"target": "Test-3g", "period_days": 210.6},
                {"target": "Test-3h", "period_days": 421.2},
            ],
        )

        matches = match_candidate_rows(extracted, confirmed)
        by_target = {m["target"]: m for m in matches if m["target"]}

        self.assertEqual(by_target["Test-3g"]["kind"], "alias")
        self.assertEqual(by_target["Test-3g"]["ratio_label"], "P/3")
        # 421.2 is exactly 2 x 210.6, so the greedy pass must not hand the
        # 421.2 d row to Test-3g as a 2P alias and strand Test-3h.
        self.assertEqual(by_target["Test-3h"]["kind"], "direct")

    def test_spurious_candidate_is_reported_as_extra_not_forced_onto_a_planet(self):
        extracted, confirmed = self._frames(
            [7.0, 121.3],
            [{"target": "Test-4b", "period_days": 7.0}],
        )

        matches = match_candidate_rows(extracted, confirmed)
        extra = [m for m in matches if m["kind"] == "extra"]

        self.assertEqual(len(extra), 1)
        self.assertAlmostEqual(extra[0]["extracted_period"], 121.3)
        self.assertIsNone(extra[0]["confirmed_index"])

    def test_period_beyond_tolerance_is_not_paired(self):
        extracted, confirmed = self._frames(
            [130.0],
            [{"target": "Test-5b", "period_days": 124.9}],
        )

        kinds = {m["kind"] for m in match_candidate_rows(extracted, confirmed)}
        self.assertEqual(kinds, {"missed", "extra"})


if __name__ == "__main__":
    unittest.main()
