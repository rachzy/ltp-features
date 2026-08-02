"""Tests for ranked BLS peaks and iterative multi-candidate masking."""

from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from detrend_and_period import rank_independent_bls_candidates  # noqa: E402
from extract_feats import (  # noqa: E402
    _candidate_passes_mes,
    _periodic_mask,
    extract_features_from_arrays,
)


def _features(period, mes, t0=0.5, duration=0.1):
    return {
        "period_days": float(period),
        "t0": float(t0),
        "duration_days": float(duration),
        "max_mes": float(mes),
    }


class RankedPeakTests(unittest.TestCase):
    def test_local_maxima_are_ranked_and_nearby_peaks_clustered(self):
        periodogram = SimpleNamespace(
            period=np.array([1.000, 1.002, 1.004, 2.0, 3.0, 4.0, 5.0]),
            power=np.array([8.0, 1.0, 9.0, 1.0, 7.0, 1.0, 6.0]),
            duration=np.full(7, 0.1),
            transit_time=np.arange(7, dtype=float),
        )

        candidates = rank_independent_bls_candidates(periodogram)

        self.assertEqual([round(row["power"]) for row in candidates], [9, 7, 6])
        self.assertAlmostEqual(candidates[0]["period"], 1.004)
        self.assertAlmostEqual(candidates[-1]["period"], 5.0)

    def test_all_nonfinite_power_returns_no_candidates(self):
        periodogram = SimpleNamespace(
            period=np.array([1.0, 2.0]),
            power=np.full((2, 2), np.nan),
            duration=np.array([0.1, 0.2]),
            transit_time=np.array([0.0, 0.0]),
        )

        self.assertEqual(rank_independent_bls_candidates(periodogram), [])


class IterativeMaskingTests(unittest.TestCase):
    def test_mes_threshold_is_inclusive(self):
        self.assertTrue(_candidate_passes_mes({"max_mes": 7.1}))
        self.assertFalse(_candidate_passes_mes({"max_mes": 7.099}))
        self.assertFalse(_candidate_passes_mes({"max_mes": np.nan}))
        self.assertFalse(
            _candidate_passes_mes(
                {"max_mes": 20.0}, {"n_mes_events": 2}
            )
        )

    def test_periodic_mask_uses_requested_padding(self):
        time = np.array([0.40, 0.43, 0.50, 0.57, 0.60])
        narrow = _periodic_mask(time, 2.0, 0.5, 0.1, width_factor=1.0)
        padded = _periodic_mask(time, 2.0, 0.5, 0.1, width_factor=1.5)

        self.assertEqual(narrow.tolist(), [False, False, True, False, False])
        self.assertEqual(padded.tolist(), [False, True, True, True, False])

    def test_failed_peak_is_skipped_and_next_ranked_peak_can_pass(self):
        time = np.arange(0.0, 100.0, 0.02)
        flux = np.ones(time.size)
        global_results = [
            (
                _features(5.0, 20.0),
                {"search_candidates": [{"period": 5.0}], "n_mes_events": 10},
            ),
            (
                _features(11.0, 15.0),
                {"search_candidates": [{"period": 11.0}], "n_mes_events": 8},
            ),
            (
                _features(2.0, 5.0),
                {
                    "search_candidates": [
                        {"period": 2.0},
                        {"period": 13.0, "duration": 0.1, "t0": 0.5},
                    ],
                    "n_mes_events": 20,
                },
            ),
        ]
        global_index = 0

        def fake_single(_time, _flux, **kwargs):
            nonlocal global_index
            hint = kwargs.get("candidate_hint")
            if hint is not None:
                return (
                    _features(13.0, 9.0),
                    {"search_candidates": [hint], "n_mes_events": 7},
                    None,
                )
            features, info = global_results[global_index]
            global_index += 1
            return features, info, None

        with patch(
            "extract_feats._extract_single_candidate_from_arrays",
            side_effect=fake_single,
        ), patch(
            "extract_feats._quick_candidate_diagnostics",
            return_value={"max_mes": 9.0, "n_mes_events": 7},
        ), patch("extract_feats.MAX_TRANSIT_CANDIDATES", 3):
            with contextlib.redirect_stdout(io.StringIO()):
                rows = extract_features_from_arrays(time, flux)

        self.assertEqual([row["period_days"] for row in rows], [5.0, 11.0, 13.0])
        self.assertNotIn(2.0, [row["period_days"] for row in rows])

    def test_search_stops_when_no_peak_clears_mes_threshold(self):
        time = np.arange(0.0, 20.0, 0.02)
        flux = np.ones(time.size)
        failed = (
            _features(3.0, 7.09),
            {"search_candidates": [{"period": 3.0}], "n_mes_events": 6},
            None,
        )

        with patch(
            "extract_feats._extract_single_candidate_from_arrays",
            return_value=failed,
        ), patch("extract_feats.MAX_TRANSIT_CANDIDATES", 2):
            with contextlib.redirect_stdout(io.StringIO()):
                rows = extract_features_from_arrays(time, flux)

        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
