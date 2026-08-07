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
from sesmes import compute_SES_MES, fold_statistics  # noqa: E402


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
        # 5.0 sits below the recovery bar as well, so nothing at all is
        # reported. A peak between the two bars is covered separately by
        # SubthresholdRecoveryTests.
        time = np.arange(0.0, 20.0, 0.02)
        flux = np.ones(time.size)
        failed = (
            _features(3.0, 5.0),
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


class HarmonicReconciliationTests(unittest.TestCase):
    """A search that locks onto P/2 must be corrected before anything is masked.

    Measured on Kepler-90b: accepted at 3.5042 d = P/2, whose mask covers every
    real transit of the 7.0084 d planet, making it unrecoverable and consuming
    9.8% of the light curve. max_mes was 12.44 at the true period against 7.95
    at the alias, so the discriminator was already available.
    """

    PERIOD = 7.0084
    DURATION = 0.16
    EPOCH = 1.5
    CADENCE = 0.02

    def _series(self, depth=1.2e-4, seed=3):
        rng = np.random.default_rng(seed)
        time = np.arange(0.0, 400.0, self.CADENCE)
        phase = (
            np.mod(time - self.EPOCH + 0.5 * self.PERIOD, self.PERIOD)
            - 0.5 * self.PERIOD
        )
        flux = 1.0 + rng.normal(0.0, 2e-4, time.size)
        flux[np.abs(phase) < self.DURATION / 2.0] -= depth
        return time, flux

    def _statistics(self, time, flux):
        phase = (
            np.mod(time - self.EPOCH + 0.5 * self.PERIOD, self.PERIOD)
            - 0.5 * self.PERIOD
        )
        result = compute_SES_MES(
            time, flux, self.PERIOD, self.EPOCH, self.DURATION,
            cadence_hours=self.CADENCE * 24.0,
            transit_mask=np.abs(phase) < self.DURATION / 2.0,
        )
        return result["statistics"]

    def _runner(self, statistics, first_period, first_epoch=None):
        """Fake extractor: global search returns first_period, hints echo back."""
        calls = []
        first_epoch = self.EPOCH if first_epoch is None else first_epoch

        def fake_single(t, f, **kwargs):
            hint = kwargs.get("candidate_hint")
            gap = kwargs.get("gap_mask")
            calls.append(
                {
                    "hint": hint,
                    "dealias": kwargs.get("dealias", True),
                    "gapped": 0 if gap is None else int(np.sum(gap)),
                }
            )
            if hint is None:
                period, epoch = first_period, first_epoch
            else:
                period, epoch = float(hint["period"]), float(hint["t0"])
            folded = fold_statistics(statistics, period, epoch, self.DURATION)
            feats = {
                "period_days": period,
                "t0": epoch,
                "duration_days": self.DURATION,
                "max_mes": float(folded["max_mes"]),
            }
            info = {
                "search_candidates": [{"period": period}],
                "n_mes_events": int(folded["max_mes_n_events"]),
                "sesmes_statistics": statistics,
            }
            return feats, info, None

        return fake_single, calls

    def _run(self, first_period, iterations=1, first_epoch=None):
        time, flux = self._series()
        statistics = self._statistics(time, flux)
        fake_single, calls = self._runner(statistics, first_period, first_epoch)
        with patch(
            "extract_feats._extract_single_candidate_from_arrays",
            side_effect=fake_single,
        ), patch("extract_feats.MAX_TRANSIT_CANDIDATES", iterations):
            with contextlib.redirect_stdout(io.StringIO()):
                rows = extract_features_from_arrays(time, flux)
        return rows, calls

    def test_accepted_half_period_alias_is_promoted_to_the_true_period(self):
        rows, calls = self._run(self.PERIOD / 2.0)

        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["period_days"], self.PERIOD, places=6)
        # The re-measure must run with de-aliasing off: it ranks on BLS power,
        # which is what preferred the alias in the first place.
        remeasures = [c for c in calls if c["hint"] is not None]
        self.assertTrue(remeasures)
        self.assertFalse(any(c["dealias"] for c in remeasures))

    def test_reconciled_period_builds_the_mask(self):
        """The point of reconciling before masking: half as many cadences go."""
        alias_rows, alias_calls = self._run(self.PERIOD / 2.0, iterations=2)
        self.assertAlmostEqual(alias_rows[0]["period_days"], self.PERIOD, places=6)
        reconciled_gap = alias_calls[-1]["gapped"]

        # Same loop with reconciliation disabled by an unreachable margin.
        time, flux = self._series()
        statistics = self._statistics(time, flux)
        fake_single, calls = self._runner(statistics, self.PERIOD / 2.0)
        with patch(
            "extract_feats._extract_single_candidate_from_arrays",
            side_effect=fake_single,
        ), patch("extract_feats.MAX_TRANSIT_CANDIDATES", 2), patch(
            "extract_feats.HARMONIC_ADOPT_MARGIN", 1e9
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                unreconciled_rows = extract_features_from_arrays(time, flux)
        self.assertAlmostEqual(
            unreconciled_rows[0]["period_days"], self.PERIOD / 2.0, places=6
        )
        self.assertGreater(calls[-1]["gapped"], 1.5 * reconciled_gap)

    def test_alias_epoch_on_an_empty_window_still_resolves(self):
        """The sub-phase scan, not the +-T14/2 offset scan, does this work.

        A P/2 alias predicts twice as many windows as there are transits, and
        its epoch can land on one of the empty ones. Doubling the period then
        only recovers the signal from the *other* starting sub-phase, half an
        alias period away - far outside anything the MES phase scan can reach.
        """
        alias_period = self.PERIOD / 2.0
        rows, _calls = self._run(
            alias_period, first_epoch=self.EPOCH + alias_period
        )

        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["period_days"], self.PERIOD, places=6)
        offset = abs(
            np.mod(rows[0]["t0"] - self.EPOCH + 0.5 * self.PERIOD, self.PERIOD)
            - 0.5 * self.PERIOD
        )
        self.assertLess(offset, 0.5 * self.DURATION)

    def test_true_period_is_not_dragged_to_a_harmonic(self):
        rows, _calls = self._run(self.PERIOD)

        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["period_days"], self.PERIOD, places=6)

    def test_adoption_requires_the_margin(self):
        time, flux = self._series()
        statistics = self._statistics(time, flux)
        fake_single, _calls = self._runner(statistics, self.PERIOD / 2.0)
        with patch(
            "extract_feats._extract_single_candidate_from_arrays",
            side_effect=fake_single,
        ), patch("extract_feats.MAX_TRANSIT_CANDIDATES", 1), patch(
            "extract_feats.HARMONIC_ADOPT_MARGIN", 1e9
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                rows = extract_features_from_arrays(time, flux)

        self.assertAlmostEqual(rows[0]["period_days"], self.PERIOD / 2.0, places=6)


class HarmonicDedupTests(unittest.TestCase):
    """Integer harmonics are one signal; near-resonant planets are not."""

    def test_near_two_to_one_resonant_planets_are_not_deduped(self):
        # Kepler-90b and i sit at a ratio of 2.0617 - only 3.1% from 2:1, and
        # the tightest real pair in the system. Deduping them would delete a
        # confirmed planet, so this is the regression that pins the tolerance.
        accepted = [{"period_days": 7.00821787}]
        self.assertFalse(
            _period_already_accepted({"period_days": 14.44912}, accepted)
        )
        self.assertFalse(
            _period_already_accepted({"period_days": 7.00821787}, [{"period_days": 14.44912}])
        )

    def test_other_kepler_90_pairs_survive(self):
        # d/e = 1.539 (2.6% from 3:2), i/d = 4.134 (3.4% from 4:1),
        # d/f = 2.091 (4.6% from 2:1), g/h = 1.575.
        periods = [59.7371443, 91.9401253, 14.44912, 124.922516, 210.601384,
                   331.597273]
        for index, period in enumerate(periods):
            others = [{"period_days": p} for p in periods[:index]]
            with self.subTest(period=period):
                self.assertFalse(
                    _period_already_accepted({"period_days": period}, others)
                )

    def test_exact_half_and_double_periods_are_deduped(self):
        accepted = [{"period_days": 7.00822}]
        for period in (3.50411, 14.01644, 7.00822, 2.33607, 21.02466):
            with self.subTest(period=period):
                self.assertTrue(
                    _period_already_accepted({"period_days": period}, accepted)
                )


class SubthresholdRecoveryTests(unittest.TestCase):
    """Peaks the search measured but the detection gate rejected.

    Kepler-90i is the case: the final iteration produced P=14.448764 (0.0024%
    from truth) at max_mes ~6.2 and discarded the fully measured row.
    """

    def _run(self, mes_values):
        time = np.arange(0.0, 60.0, 0.02)
        flux = np.ones(time.size)
        queue = list(mes_values)

        def fake_single(t, f, **kwargs):
            period, mes = queue.pop(0)
            return (
                _features(period, mes),
                {"search_candidates": [{"period": period}], "n_mes_events": 6},
                None,
            )

        with patch(
            "extract_feats._extract_single_candidate_from_arrays",
            side_effect=fake_single,
        ), patch("extract_feats.MAX_TRANSIT_CANDIDATES", 1):
            with contextlib.redirect_stdout(io.StringIO()) as out:
                rows = extract_features_from_arrays(time, flux)
        return rows, out.getvalue()

    def test_best_subthreshold_peak_is_reported_and_marked(self):
        rows, _ = self._run([(14.4488, 6.5)])

        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["period_days"], 14.4488)
        self.assertEqual(rows[0]["is_provisional_detection"], 1.0)
        self.assertEqual(rows[0]["mes_threshold_used"], 6.0)

    def test_peaks_below_the_recovery_bar_are_not_reported(self):
        rows, _ = self._run([(14.4488, 5.9)])

        self.assertEqual(rows, [])

    def test_recovery_respects_its_limit(self):
        rows, _ = self._run([(3.0, 6.9)])

        with patch("extract_feats.MAX_RECOVERED_CANDIDATES", 0):
            empty, _ = self._run([(3.0, 6.9)])
        self.assertEqual(len(rows), 1)
        self.assertEqual(empty, [])

    def test_quick_diagnostic_stubs_never_surface_as_rows(self):
        """Provisional-screen stubs carry 4 keys and would break the schema."""
        time = np.arange(0.0, 60.0, 0.02)
        flux = np.ones(time.size)
        primary = (
            _features(5.0, 5.0),
            {
                "search_candidates": [{"period": 5.0}, {"period": 9.0}],
                "n_mes_events": 6,
            },
            None,
        )

        with patch(
            "extract_feats._extract_single_candidate_from_arrays",
            return_value=primary,
        ), patch(
            "extract_feats._quick_candidate_diagnostics",
            return_value={"max_mes": 1.0, "n_mes_events": 6},
        ), patch("extract_feats.MAX_TRANSIT_CANDIDATES", 1):
            with contextlib.redirect_stdout(io.StringIO()):
                rows = extract_features_from_arrays(time, flux)

        # The 9.0 d peak was screened out with a stub; only the measured 5.0 d
        # row could ever be recovered, and it is under the recovery bar.
        self.assertEqual(rows, [])

    def test_recovered_row_is_not_reported_when_it_duplicates_an_accepted_one(self):
        time = np.arange(0.0, 60.0, 0.02)
        flux = np.ones(time.size)
        queue = [(5.0, 12.0), (10.0, 6.5)]

        def fake_single(t, f, **kwargs):
            period, mes = queue.pop(0)
            return (
                _features(period, mes),
                {"search_candidates": [{"period": period}], "n_mes_events": 6},
                None,
            )

        with patch(
            "extract_feats._extract_single_candidate_from_arrays",
            side_effect=fake_single,
        ), patch("extract_feats.MAX_TRANSIT_CANDIDATES", 2):
            with contextlib.redirect_stdout(io.StringIO()):
                rows = extract_features_from_arrays(time, flux)

        # 10.0 is 2x the accepted 5.0, so it describes the same signal.
        self.assertEqual([row["period_days"] for row in rows], [5.0])


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
