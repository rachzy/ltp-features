"""
(AI-Generated)
Tests for the time-domain duration-matched SES/MES estimator.
"""

from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cdpp import calculate_cdpp, duration_matched_statistics  # noqa: E402
from extract_feats import extract_features_from_arrays  # noqa: E402
from sesmes import compute_SES_MES  # noqa: E402


CADENCE_HOURS = 0.5
CADENCE_DAYS = CADENCE_HOURS / 24.0
DURATION_HOURS = 4.0
DURATION_DAYS = DURATION_HOURS / 24.0

EXPECTED_ARRAY_FEATURE_KEYS = [
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
    "max_ses",
    "max_mes",
    "snr_global",
    "snr_per_transit_mean",
    "snr_per_transit_std",
    "resid_rms_global",
    "vshape_metric",
    "secondary_depth",
    "secondary_depth_snr",
    "secondary_depth_snr_log",
    "secondary_depth_snr_capped",
    "odd_even_depth_ratio",
    "ingress_egress_asymmetry",
    "skewness_flux",
    "kurtosis_flux",
    "outlier_resistance",
    "planet_radius_rearth",
    "planet_radius_rjup",
]


def _transit_mask(time, period, epoch, duration_days=DURATION_DAYS):
    phase = np.mod(time - epoch + 0.5 * period, period) - 0.5 * period
    return np.abs(phase) <= duration_days / 2.0


def _robust_scale(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    center = np.median(values)
    return 1.4826 * np.median(np.abs(values - center))


class DurationMatchedStatisticsTests(unittest.TestCase):
    def test_extractor_keeps_feature_schema_and_order(self):
        rng = np.random.default_rng(77)
        time = np.arange(0.0, 120.0, 0.02)
        period = 5.0
        epoch = 1.0
        duration = 0.16
        mask = _transit_mask(time, period, epoch, duration)
        flux = 1.0 + rng.normal(0.0, 2e-4, time.size)
        flux[mask] -= 2e-3
        bls_info = {
            "best_period": period,
            "t0": epoch,
            "best_duration": duration,
        }

        with patch(
            "extract_feats.detrend_with_bls_mask",
            return_value=(flux, np.ones_like(flux), mask, bls_info),
        ), patch("extract_feats.MAX_TRANSIT_CANDIDATES", 1):
            with contextlib.redirect_stdout(io.StringIO()):
                row = extract_features_from_arrays(time, flux)[0]

        self.assertEqual(list(row), EXPECTED_ARRAY_FEATURE_KEYS)
        for name in ("SES_mean", "MES", "max_ses", "max_mes"):
            self.assertTrue(np.isfinite(row[name]), name)

    def test_white_noise_ses_is_centered_and_unit_scaled(self):
        rng = np.random.default_rng(101)
        time = np.arange(0.0, 300.0, CADENCE_DAYS)
        flux = 1.0 + rng.normal(0.0, 2e-4, time.size)

        statistics = duration_matched_statistics(
            time,
            flux,
            duration_hours=DURATION_HOURS,
            cadence_hours=CADENCE_HOURS,
        )
        background = statistics["SES"][statistics["noise_eligible"]]

        self.assertGreater(background.size, 1_000)
        self.assertLess(abs(float(np.median(background))), 0.1)
        self.assertGreater(_robust_scale(background), 0.8)
        self.assertLess(_robust_scale(background), 1.2)

    def test_injected_transits_recover_analytic_ses_and_mes(self):
        rng = np.random.default_rng(1234)
        time = np.arange(0.0, 300.0, CADENCE_DAYS)
        period = 10.0
        epoch = 2.037
        depth = 8e-4
        point_sigma = 2e-4
        transit_mask = _transit_mask(time, period, epoch)
        flux = 1.0 + rng.normal(0.0, point_sigma, time.size)
        flux[transit_mask] -= depth

        result = compute_SES_MES(
            time,
            flux,
            period,
            epoch,
            DURATION_DAYS,
            cadence_hours=CADENCE_HOURS,
            transit_mask=transit_mask,
        )

        expected_points = DURATION_DAYS / CADENCE_DAYS
        expected_event_ses = depth * np.sqrt(expected_points) / point_sigma
        expected_mes = expected_event_ses * np.sqrt(result["SES"].size)
        self.assertGreater(result["SES"].size, 20)
        self.assertAlmostEqual(
            float(np.mean(result["SES"])),
            expected_event_ses,
            delta=0.15 * expected_event_ses,
        )
        self.assertAlmostEqual(
            result["MES"], expected_mes, delta=0.15 * expected_mes
        )

        indices = result["event_indices"]
        statistics = result["statistics"]
        expected_from_components = np.sum(statistics["N"][indices]) / np.sqrt(
            np.sum(statistics["D"][indices])
        )
        self.assertAlmostEqual(result["MES"], expected_from_components, places=12)

    def test_local_noise_tracks_heteroskedastic_flux(self):
        rng = np.random.default_rng(202)
        time = np.arange(0.0, 400.0, CADENCE_DAYS)
        first_half = time < 200.0
        flux = np.ones(time.size)
        flux[first_half] += rng.normal(0.0, 1e-4, np.sum(first_half))
        flux[~first_half] += rng.normal(0.0, 4e-4, np.sum(~first_half))

        statistics = duration_matched_statistics(
            time,
            flux,
            duration_hours=DURATION_HOURS,
            cadence_hours=CADENCE_HOURS,
        )
        quiet = (
            (statistics["time"] > 50.0)
            & (statistics["time"] < 150.0)
            & statistics["valid"]
        )
        noisy = (
            (statistics["time"] > 250.0)
            & (statistics["time"] < 350.0)
            & statistics["valid"]
        )
        quiet_uncertainty = float(np.median(statistics["uncertainty"][quiet]))
        noisy_uncertainty = float(np.median(statistics["uncertainty"][noisy]))

        self.assertGreater(noisy_uncertainty / quiet_uncertainty, 3.0)
        self.assertLess(noisy_uncertainty / quiet_uncertainty, 5.0)

    def test_correlated_noise_exceeds_naive_white_noise_uncertainty(self):
        rng = np.random.default_rng(303)
        time = np.arange(0.0, 300.0, CADENCE_DAYS)
        point_sigma = 2e-4
        correlation = 0.8
        innovations = rng.normal(
            0.0,
            point_sigma * np.sqrt(1.0 - correlation**2),
            time.size,
        )
        correlated = np.empty(time.size)
        correlated[0] = innovations[0]
        for index in range(1, time.size):
            correlated[index] = correlation * correlated[index - 1] + innovations[index]

        statistics = duration_matched_statistics(
            time,
            1.0 + correlated,
            duration_hours=DURATION_HOURS,
            cadence_hours=CADENCE_HOURS,
        )
        naive = point_sigma / np.sqrt(DURATION_DAYS / CADENCE_DAYS)
        measured = float(np.nanmedian(statistics["uncertainty"]))

        self.assertGreater(measured, 1.5 * naive)

    def test_gaps_edges_and_zero_mad_are_not_given_weight(self):
        rng = np.random.default_rng(404)
        complete_time = np.arange(0.0, 100.0, CADENCE_DAYS)
        keep = ~((complete_time > 45.0) & (complete_time < 55.0))
        time = complete_time[keep]
        flux = 1.0 + rng.normal(0.0, 2e-4, time.size)
        statistics = duration_matched_statistics(
            time,
            flux,
            duration_hours=DURATION_HOURS,
            cadence_hours=CADENCE_HOURS,
        )

        near_left_gap = np.argmin(np.abs(statistics["time"] - 45.0))
        near_right_gap = np.argmin(np.abs(statistics["time"] - 55.0))
        self.assertFalse(statistics["valid"][near_left_gap])
        self.assertFalse(statistics["valid"][near_right_gap])
        self.assertFalse(statistics["valid"][0])
        self.assertFalse(statistics["valid"][-1])

        constant = duration_matched_statistics(
            complete_time,
            np.ones(complete_time.size),
            duration_hours=DURATION_HOURS,
            cadence_hours=CADENCE_HOURS,
        )
        self.assertFalse(np.any(constant["valid"]))
        self.assertFalse(np.any(np.isinf(constant["SES"])))

        invalid = duration_matched_statistics(
            complete_time,
            np.ones(complete_time.size),
            duration_hours=np.nan,
            cadence_hours=CADENCE_HOURS,
        )
        self.assertEqual(invalid["time"].size, 0)

    def test_phase_scan_recovers_shift_and_signed_mes(self):
        rng = np.random.default_rng(505)
        time = np.arange(0.0, 250.0, CADENCE_DAYS)
        period = 8.0
        supplied_epoch = 1.25
        true_offset = 0.25 * DURATION_DAYS
        true_epoch = supplied_epoch + true_offset
        mask = _transit_mask(time, period, true_epoch)
        noise = rng.normal(0.0, 1.5e-4, time.size)

        dimming_flux = 1.0 + noise
        dimming_flux[mask] -= 1.2e-3
        dimming = compute_SES_MES(
            time,
            dimming_flux,
            period,
            supplied_epoch,
            DURATION_DAYS,
            cadence_hours=CADENCE_HOURS,
            transit_mask=mask,
        )
        self.assertGreaterEqual(dimming["max_mes"], dimming["MES"])
        self.assertAlmostEqual(
            dimming["best_phase_offset"],
            true_offset,
            delta=CADENCE_DAYS,
        )

        brightening_flux = 1.0 + noise
        brightening_flux[mask] += 1.2e-3
        brightening = compute_SES_MES(
            time,
            brightening_flux,
            period,
            true_epoch,
            DURATION_DAYS,
            cadence_hours=CADENCE_HOURS,
            transit_mask=mask,
        )
        self.assertLess(brightening["MES"], 0.0)
        self.assertFalse(np.isinf(brightening["MES"]))

    def test_cdpp_uses_duration_matched_uncertainty(self):
        rng = np.random.default_rng(606)
        time = np.arange(0.0, 300.0, CADENCE_DAYS)
        point_sigma = 2e-4
        flux = 1.0 + rng.normal(0.0, point_sigma, time.size)
        cdpp = calculate_cdpp(
            flux,
            cadence_hours=CADENCE_HOURS,
            time=time,
        )

        expected_3h = point_sigma / np.sqrt(3.0 / CADENCE_HOURS) * 1e6
        expected_12h = point_sigma / np.sqrt(12.0 / CADENCE_HOURS) * 1e6
        self.assertAlmostEqual(cdpp["cdpp_3h"], expected_3h, delta=0.25 * expected_3h)
        self.assertAlmostEqual(
            cdpp["cdpp_12h"], expected_12h, delta=0.25 * expected_12h
        )
        self.assertGreater(cdpp["cdpp_3h"], cdpp["cdpp_12h"])


class GapMaskTests(unittest.TestCase):
    """Gapping an accepted candidate must not disturb the cadence grid.

    Slicing its cadences out leaves holes shorter than ``gap_cadences`` all
    over the series, which split ``segment_id`` and truncate the local noise
    windows. Passing the intact series plus ``gap_mask`` avoids that while
    still keeping the gapped windows out of the noise sample and the fold.
    """

    SIGMA = 1.5e-4
    P_ACCEPTED, DUR_ACCEPTED = 3.5485, 0.196
    P_TARGET, DUR_TARGET = 11.3, 0.145

    def _series(self, accepted_depth=0.007, seed=0):
        rng = np.random.default_rng(seed)
        time = np.arange(0.0, 400.0, CADENCE_DAYS)
        flux = 1.0 + rng.normal(0, self.SIGMA, time.size)
        accepted = _transit_mask(
            time, self.P_ACCEPTED, 1.0, self.DUR_ACCEPTED
        )
        flux = flux - accepted_depth * accepted
        # 1.5x padding, matching the extractor's TRANSIT_MASK_WIDTH
        gap = np.abs(
            ((time - 1.0 + 0.5 * self.P_ACCEPTED) % self.P_ACCEPTED)
            - 0.5 * self.P_ACCEPTED
        ) <= 0.75 * self.DUR_ACCEPTED
        return time, flux, gap

    def _median_uncertainty(self, time, flux, **kwargs):
        stats = duration_matched_statistics(
            time,
            flux,
            duration_hours=self.DUR_TARGET * 24.0,
            cadence_hours=CADENCE_HOURS,
            **kwargs,
        )
        unc = stats["uncertainty"][stats["noise_eligible"]]
        return float(np.median(unc)), stats

    def test_gap_mask_preserves_segments_where_slicing_fragments_them(self):
        time, flux, gap = self._series()
        # The padded window is wider than the 5-cadence segment-split
        # threshold, so slicing reads as a real data gap at every epoch.
        self.assertGreater(1.5 * self.DUR_ACCEPTED, 5.0 * CADENCE_DAYS)

        _, sliced = self._median_uncertainty(time[~gap], flux[~gap])
        _, gapped = self._median_uncertainty(time, flux, gap_mask=gap)

        def segments(t):
            return 1 + int(np.sum(np.diff(t) > 5.0 * CADENCE_DAYS))

        self.assertGreater(segments(time[~gap]), 50)
        self.assertEqual(segments(time), 1)
        self.assertTrue(np.all(np.isfinite(gapped["time"])))
        self.assertEqual(gapped["time"].size, time.size)
        self.assertLess(sliced["valid"].mean(), 1.0)

    def test_gap_mask_tracks_the_unmasked_baseline_closer_than_slicing(self):
        """Slicing biases the local MAD low; gapping leaves it near baseline.

        Averaged over seeds because the per-seed gap-vs-slice difference is
        ~1% and a single realization can invert it. The reference is the same
        curve with no accepted planet at all, not the analytic floor: the
        estimator carries its own few-percent bias from taking a MAD over
        overlapping boxes, which masking neither causes nor cures.
        """
        seeds = range(5)
        baseline, masked, sliced = [], [], []
        for seed in seeds:
            time, flux, gap = self._series(seed=seed)
            _, clean, _ = self._series(accepted_depth=0.0, seed=seed)
            baseline.append(self._median_uncertainty(time, clean)[0])
            masked.append(self._median_uncertainty(time, flux, gap_mask=gap)[0])
            sliced.append(self._median_uncertainty(time[~gap], flux[~gap])[0])
        baseline = float(np.mean(baseline))
        masked = float(np.mean(masked))
        sliced = float(np.mean(sliced))

        self.assertLess(abs(masked / baseline - 1.0), 0.01)
        self.assertLess(sliced, baseline)
        self.assertLess(
            abs(masked - baseline), abs(sliced - baseline)
        )

    def test_unmasked_accepted_transits_inflate_the_noise_estimate(self):
        """The other failure mode: leaving them in the noise sample entirely."""
        time, flux, gap = self._series()
        masked, _ = self._median_uncertainty(time, flux, gap_mask=gap)
        naive, _ = self._median_uncertainty(time, flux)
        self.assertGreater(naive, masked * 1.05)

    def test_gapped_boxes_are_excluded_from_the_fold(self):
        """Without this, a deep accepted transit leaks into the target's MES."""
        time, flux, gap = self._series()
        epoch = 5.0
        flux = flux - 0.0006 * _transit_mask(
            time, self.P_TARGET, epoch, self.DUR_TARGET
        )
        common = dict(cadence_hours=CADENCE_HOURS)

        contaminated = compute_SES_MES(
            time, flux, self.P_TARGET, epoch, self.DUR_TARGET, **common
        )["max_mes"]
        masked = compute_SES_MES(
            time, flux, self.P_TARGET, epoch, self.DUR_TARGET,
            gap_mask=gap, **common
        )["max_mes"]
        # Same curve with the accepted planet never injected at all.
        _, clean_flux, _ = self._series(accepted_depth=0.0)
        clean_flux = clean_flux - 0.0006 * _transit_mask(
            time, self.P_TARGET, epoch, self.DUR_TARGET
        )
        clean = compute_SES_MES(
            time, clean_flux, self.P_TARGET, epoch, self.DUR_TARGET, **common
        )["max_mes"]

        self.assertLess(abs(masked / clean - 1.0), 0.10)
        self.assertGreater(contaminated, clean * 1.15)

    def test_gap_mask_defaults_to_a_no_op(self):
        time, flux, _ = self._series()
        without, _ = self._median_uncertainty(time, flux)
        explicit, _ = self._median_uncertainty(
            time, flux, gap_mask=np.zeros(time.shape, dtype=bool)
        )
        self.assertAlmostEqual(without, explicit, places=12)

    def test_gap_mask_shape_is_validated(self):
        time, flux, _ = self._series()
        with self.assertRaises(ValueError):
            duration_matched_statistics(
                time,
                flux,
                duration_hours=DURATION_HOURS,
                cadence_hours=CADENCE_HOURS,
                gap_mask=np.zeros(time.size - 1, dtype=bool),
            )


if __name__ == "__main__":
    unittest.main()
