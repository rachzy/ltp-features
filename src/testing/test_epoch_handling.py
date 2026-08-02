"""Regression tests for transit-epoch normalization and refinement."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from detrend_and_period import refine_transit_epoch  # noqa: E402
from utils.compare_extracted_confirmed import compare_feature_rows  # noqa: E402
from utils.ephemeris import (  # noqa: E402
    align_catalog_epoch,
    canonical_epoch,
    epoch_phase_offset,
)


class EpochNormalizationTests(unittest.TestCase):
    def test_integer_period_shifts_are_equivalent(self):
        period = 3.25
        epoch = 1.125
        reference = 100.0

        expected = canonical_epoch(epoch, period, reference)
        for cycle in (-100, -1, 0, 1, 100):
            shifted = canonical_epoch(epoch + cycle * period, period, reference)
            self.assertAlmostEqual(shifted, expected, places=10)
            self.assertAlmostEqual(
                epoch_phase_offset(epoch + cycle * period, epoch, period),
                0.0,
                places=10,
            )

    def test_kepler_bjd_is_aligned_to_bkjd(self):
        data_epoch = 121.3580004
        catalog_bjd = 2_454_954.358557
        period = 2.204735391

        aligned = align_catalog_epoch(
            catalog_bjd,
            data_epoch,
            period,
            target="Kepler-2b",
        )

        self.assertAlmostEqual(aligned, 121.358557, places=6)
        self.assertLess(abs(data_epoch - aligned), 0.001)

    def test_comparison_reports_epoch_error_relative_to_duration(self):
        extracted = pd.Series(
            {
                "period_days": 2.204735391,
                "t0": 121.3580004,
                "duration_days": 0.175,
            }
        )
        confirmed = pd.Series(
            {
                "target": "Kepler-2b",
                "period_days": 2.204735391,
                "t0": 2_454_954.358557,
                "duration_days": 0.164303,
            }
        )

        comparison = compare_feature_rows(extracted, confirmed)
        epoch_row = comparison.loc[comparison["feature"] == "t0"].iloc[0]

        self.assertLess(epoch_row["abs_diff"], 0.001)
        self.assertAlmostEqual(
            epoch_row["pct_diff"],
            100.0 * (121.3580004 - 121.358557) / 0.175,
            places=5,
        )


class EpochRefinementTests(unittest.TestCase):
    def test_refit_recovers_box_transit_phase(self):
        rng = np.random.default_rng(123)
        time = np.arange(0.0, 80.0, 0.02)
        period = 5.0
        duration = 0.20
        true_epoch = 1.237
        initial_epoch = true_epoch + 0.35 * duration
        phase_days = np.mod(time - true_epoch + 0.5 * period, period) - 0.5 * period
        flux = 1.0 + rng.normal(0.0, 2e-4, time.size)
        flux[np.abs(phase_days) < duration / 2.0] -= 0.01

        refined = refine_transit_epoch(
            time,
            flux,
            period,
            duration,
            initial_epoch,
            oversample=100,
        )

        initial_error = abs(epoch_phase_offset(initial_epoch, true_epoch, period))
        refined_error = abs(epoch_phase_offset(refined, true_epoch, period))
        self.assertLess(refined_error, initial_error)
        self.assertLessEqual(refined_error, 0.02)
        self.assertGreaterEqual(refined, time.min())
        self.assertLess(refined, time.min() + period)


if __name__ == "__main__":
    unittest.main()
