"""
(AI-Generated)
Tests for ranked BLS peaks and iterative multi-candidate masking.
"""

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

from detrend_and_period import (  # noqa: E402
    detrend_with_bls_mask,
    rank_independent_bls_candidates,
)
from extract_feats import (  # noqa: E402
    _candidate_passes_mes,
    _deep_events,
    _period_already_accepted,
    _periodic_mask,
    _unexplained_deep_event_mask,
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

    def test_period_dedup_ignores_phase_but_respects_tolerance(self):
        accepted = [_features(5.0, 20.0, t0=0.5, duration=0.1)]
        same_period_drifted_phase = _features(5.003, 18.0, t0=2.7, duration=0.1)
        different_period = _features(5.2, 18.0, t0=0.5, duration=0.1)

        self.assertTrue(_period_already_accepted(same_period_drifted_phase, accepted))
        self.assertFalse(_period_already_accepted(different_period, accepted))
        self.assertFalse(_period_already_accepted(same_period_drifted_phase, []))

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

    def test_same_period_drifted_phase_is_not_double_counted(self):
        """A residual re-detection at ~the accepted period must not become a 2nd row.

        Reproduces the Kepler-2 pattern: after a deep transit is accepted and
        masked, imperfect masking can leave a same-period, different-phase
        residual that the next iteration's fresh BLS pass "rediscovers" as an
        apparently independent, MES-qualified candidate. Without period-only
        dedup this gets double-counted as a second planet.
        """
        time = np.arange(0.0, 100.0, 0.02)
        flux = np.ones(time.size)
        global_results = [
            (
                _features(5.0, 20.0),
                {"search_candidates": [{"period": 5.0}], "n_mes_events": 10},
            ),
            (
                # Same period family as the accepted 5.0 d candidate (well
                # within tolerance) but a phase offset far outside
                # _same_ephemeris's duration-based tolerance, and a high MES
                # of its own - it must still be rejected on period alone.
                _features(5.003, 18.0, t0=2.7),
                {
                    "search_candidates": [
                        {"period": 5.003},
                        {"period": 11.0, "duration": 0.1, "t0": 0.5},
                    ],
                    "n_mes_events": 9,
                },
            ),
        ]
        global_index = 0

        def fake_single(_time, _flux, **kwargs):
            nonlocal global_index
            hint = kwargs.get("candidate_hint")
            if hint is not None:
                return (
                    _features(11.0, 15.0),
                    {"search_candidates": [hint], "n_mes_events": 8},
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
        ), patch("extract_feats.MAX_TRANSIT_CANDIDATES", 2):
            with contextlib.redirect_stdout(io.StringIO()):
                rows = extract_features_from_arrays(time, flux)

        periods = [row["period_days"] for row in rows]
        self.assertEqual(periods, [5.0, 11.0])
        self.assertNotIn(5.003, periods)

    def test_iterations_receive_full_series_plus_growing_gap_mask(self):
        """Later iterations must be masked, not sliced.

        Slicing the accepted cadences out leaves holes that read as data gaps
        to the noise model and fragment its segments; the extractor therefore
        keeps the series whole and marks the removed windows instead.
        """
        time = np.arange(0.0, 100.0, 0.02)
        flux = np.ones(time.size)
        seen = []
        results = [
            (_features(5.0, 20.0), {"search_candidates": [{"period": 5.0}],
                                    "n_mes_events": 10}),
            (_features(11.0, 15.0), {"search_candidates": [{"period": 11.0}],
                                     "n_mes_events": 8}),
        ]

        def fake_single(t, f, **kwargs):
            gap = kwargs.get("gap_mask")
            seen.append((np.asarray(t).size, 0 if gap is None else int(np.sum(gap))))
            return (*results[len(seen) - 1], None)

        with patch(
            "extract_feats._extract_single_candidate_from_arrays",
            side_effect=fake_single,
        ), patch("extract_feats.MAX_TRANSIT_CANDIDATES", 2):
            with contextlib.redirect_stdout(io.StringIO()):
                extract_features_from_arrays(time, flux)

        self.assertEqual(len(seen), 2)
        # Both iterations see every cadence ...
        self.assertEqual(seen[0][0], time.size)
        self.assertEqual(seen[1][0], time.size)
        # ... and only the mask grows.
        self.assertEqual(seen[0][1], 0)
        self.assertGreater(seen[1][1], 0)

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


class PeriodSearchRangeTests(unittest.TestCase):
    def test_search_range_reaches_a_third_of_the_baseline(self):
        # A flat 200 d ceiling made Kepler-90g (210.6 d) and h (331.6 d)
        # unsearchable over a ~1460 d baseline even though both show the
        # >= 3 transits a detection needs.
        time = np.arange(0.0, 1460.0, 0.02)
        flux = np.ones(time.size)
        captured = {}

        def fake_power(periods, durations, **kwargs):
            captured.setdefault("max_period", float(np.max(periods)))
            raise RuntimeError("stop after the global grid is built")

        with patch("detrend_and_period.BoxLeastSquares") as bls_cls:
            bls_cls.return_value = SimpleNamespace(power=fake_power)
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(RuntimeError):
                    detrend_with_bls_mask(time, flux)

        self.assertGreater(captured["max_period"], 200.0)
        self.assertAlmostEqual(captured["max_period"], 1460.0 / 3.0, delta=1.0)


class DeepEventSweepTests(unittest.TestCase):
    """Deep dips no accepted candidate explains are the raw material of weaves."""

    CADENCE = 0.02
    NOISE = 3e-4

    def _series(self, span=400.0, seed=0):
        rng = np.random.default_rng(seed)
        time = np.arange(0.0, span, self.CADENCE)
        return time, 1.0 + rng.normal(0.0, self.NOISE, time.size)

    @staticmethod
    def _add_box(time, flux, period, t0, duration, depth):
        phase = np.mod(time - t0 + 0.5 * period, period) - 0.5 * period
        flux[np.abs(phase) <= 0.5 * duration] -= depth

    @staticmethod
    def _active_after_masking(time, period, t0, duration, width=1.5):
        phase = np.mod(time - t0 + 0.5 * period, period) - 0.5 * period
        return np.abs(phase) > 0.5 * width * duration

    def test_isolated_deep_events_are_removed(self):
        # Two lone dips can never make a detection (that needs 3 transits), so
        # leaving them in only lets a later BLS pass weave a period through them.
        time, flux = self._series()
        for center in (123.0, 271.0):
            flux[np.abs(time - center) <= 0.15] -= 6e-3
        accepted = [{"period_days": 50.0, "t0": 10.0, "duration_days": 0.3}]

        with contextlib.redirect_stdout(io.StringIO()):
            removal = _unexplained_deep_event_mask(
                time,
                flux,
                self._active_after_masking(time, 50.0, 10.0, 0.3),
                accepted,
            )

        self.assertGreater(int(np.sum(removal)), 0)
        # Every removed cadence belongs to one of the two dips, so the sweep
        # took the events and nothing else.
        distances = np.minimum(
            np.abs(time[removal] - 123.0), np.abs(time[removal] - 271.0)
        )
        self.assertLess(float(np.max(distances)), 0.3)

    def test_detectable_periodic_group_is_left_for_a_later_iteration(self):
        # Six equal-depth events at a searchable period are a real planet, not
        # a false alarm: masking them would hide it rather than protect anything.
        time, flux = self._series()
        self._add_box(time, flux, 70.0, 30.0, 0.3, 6e-3)
        accepted = [{"period_days": 50.0, "t0": 10.0, "duration_days": 0.3}]

        with contextlib.redirect_stdout(io.StringIO()):
            removal = _unexplained_deep_event_mask(
                time,
                flux,
                self._active_after_masking(time, 50.0, 10.0, 0.3),
                accepted,
            )

        self.assertEqual(int(np.sum(removal)), 0)

    def test_residuals_of_an_epoch_drifted_accepted_candidate_are_removed(self):
        # The accepted epoch is off by half a duration, so the periodic mask
        # leaves a sliver of every transit behind. Those are not a new signal.
        time, flux = self._series()
        self._add_box(time, flux, 70.0, 30.0, 0.3, 6e-3)
        accepted = [{"period_days": 70.0, "t0": 30.15, "duration_days": 0.3}]

        with contextlib.redirect_stdout(io.StringIO()):
            removal = _unexplained_deep_event_mask(
                time,
                flux,
                self._active_after_masking(time, 70.0, 30.15, 0.3),
                accepted,
            )

        self.assertGreater(int(np.sum(removal)), 0)

    def test_shallow_transits_are_never_treated_as_deep_events(self):
        # A 500 ppm transit is ~1.7 sigma per cadence: detectable by folding,
        # far below the sweep's threshold, and must survive untouched.
        time, flux = self._series()
        self._add_box(time, flux, 13.0, 3.0, 0.2, 5e-4)

        self.assertEqual(_deep_events(time, flux, np.ones(time.size, bool)), [])

    def test_quarter_boundary_does_not_merge_two_events(self):
        # Consecutive array indices can straddle a 10 d gap; dips on either
        # side are separate events, not one long one.
        time = np.r_[
            np.arange(0.0, 50.0, self.CADENCE),
            np.arange(60.0, 110.0, self.CADENCE),
        ]
        rng = np.random.default_rng(1)
        flux = 1.0 + rng.normal(0.0, self.NOISE, time.size)
        trailing = (time >= 49.6) & (time < 50.0)   # end of the first block
        leading = (time >= 60.0) & (time <= 60.4)   # start of the second
        flux[trailing | leading] -= 6e-3

        events = _deep_events(time, flux, np.ones(time.size, bool))
        self.assertEqual(len(events), 2)


if __name__ == "__main__":
    unittest.main()
