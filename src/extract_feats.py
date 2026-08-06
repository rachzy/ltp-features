import os
import time
import numpy as np
from scipy.stats import binned_statistic
from collections import OrderedDict
from scipy.stats import skew, kurtosis

from detrend_and_period import detrend_with_bls_mask
from folded_binned_metrics import folded_binned_metrics
from cdpp import calculate_cdpp
from sesmes import compute_SES_MES
from utils import (
    scaling_and_metrics,
    interp_cdpp,
    compute_secondary_depth,
    compute_odd_even_depth_ratio,
    compute_ingress_egress_asymmetry,
    compute_secondary_depth_snr,
)
from per_trans_stat import per_transit_stats_simple


MES_DETECTION_THRESHOLD = 7.1
PROVISIONAL_MES_THRESHOLD = 4.0
MAX_TRANSIT_CANDIDATES = 8
MAX_PEAKS_TO_VALIDATE = 20
MAX_REFINED_PEAKS_PER_ITERATION = 6
TRANSIT_MASK_WIDTH = 1.5
MAX_CUMULATIVE_MASK_FRACTION = 0.5
MIN_SEARCH_POINTS = 50
MIN_OBSERVED_TRANSIT_EVENTS = 3
PERIOD_DEDUP_TOLERANCE = 0.005

# Deep, unexplained single events are the raw material of spurious candidates:
# BLS happily threads a box train through a handful of them and reports a high
# MES for a period that no planet has. A cadence enters an event when it drops
# below CORE_SIGMA, and the event then extends outward while flux stays below
# EDGE_SIGMA so ingress/egress come along with the floor.
DEEP_EVENT_CORE_SIGMA = 8.0
DEEP_EVENT_EDGE_SIGMA = 3.0
DEEP_EVENT_MIN_CADENCES = 3
DEEP_EVENT_PAD_CADENCES = 2
# An event this far from an accepted candidate's predicted transit centre is a
# leftover of that candidate (drifted epoch, TTVs) rather than a new signal.
DEEP_EVENT_RESIDUAL_WIDTH = 3.0
# Two events belong to the same putative signal when their floors agree to
# within this relative tolerance.
DEEP_EVENT_DEPTH_GROUP_TOLERANCE = 0.5
DEEP_EVENT_MAX_MASK_FRACTION = 0.1


def _extract_single_candidate_from_arrays(
    tTime,
    flux,
    verbose=False,
    refine_duration=True,
    use_tls=False,
    mask_eclipses=False,
    candidate_hint=None,
    gap_mask=None,
):
    """Search and measure one candidate on the supplied light curve.

    ``gap_mask`` marks cadences already claimed by accepted candidates. The
    full series is still passed to the noise model alongside the mask, rather
    than sliced down to the surviving cadences, so the removed windows do not
    read as data gaps and fragment its segments.
    """
    start_time = time.time()
    print("Starting feature extraction from arrays...")

    feats = OrderedDict()

    mask0 = np.isfinite(tTime) & np.isfinite(flux)
    if np.sum(mask0) < 3:
        raise ValueError("too few valid points")
    time_arr = np.asarray(tTime)[mask0]
    flux_arr = np.asarray(flux)[mask0]
    if gap_mask is None:
        gap_arr = np.zeros(time_arr.shape, dtype=bool)
    else:
        gap_arr = np.asarray(gap_mask, dtype=bool)[mask0]

    print(f"Data loaded: {len(time_arr):,} valid points")
    if np.any(gap_arr):
        print(f"Gapped by accepted candidates: {int(np.sum(gap_arr)):,} cadences")
    print(f"Time range: {time_arr.min():.2f} to {time_arr.max():.2f} days")
    print(f"Flux range: {flux_arr.min():.6f} to {flux_arr.max():.6f}")

    flux_detr_full, trend_full, mask_transit, bls_info = detrend_with_bls_mask(
        time_arr,
        flux_arr,
        refine_duration=refine_duration,
        use_tls=use_tls,
        mask_eclipses=mask_eclipses,
        candidate_hint=candidate_hint,
        exclude_mask=gap_arr,
    )
    # Fold- and median-based features filter non-finite samples, so blanking
    # the gapped cadences gives them the same view they had when this function
    # received a pre-sliced array. The noise model instead gets the intact
    # series plus gap_arr, which is the whole point of the mask.
    flux_features = flux_detr_full.copy()
    flux_features[gap_arr] = np.nan
    kept = ~gap_arr
    period = float(bls_info.get("best_period", np.nan))
    t0 = float(bls_info.get("t0", np.nan))
    duration_days = float(bls_info.get("best_duration", np.nan))

    print(
        f"BLS results: P={period:.4f} days, T14={duration_days:.4f} days, t0={t0:.2f}"
    )

    # Kept for the caller's deep-event sweep: it needs a trend-free view of the
    # same cadences to judge which leftover dips are real events.
    bls_info["flux_detrended"] = flux_detr_full

    feats["period_days"] = period
    feats["t0"] = t0
    feats["duration_days"] = duration_days
    feats["duration_hours"] = duration_days * 24.0

    print("Computing scaling metrics...")
    # Compacted, not blanked: scaling_and_metrics imputes NaN to the median,
    # which would pull scale_std and the shape moments toward zero.
    flux_scaled, scaling_metrics = scaling_and_metrics(
        time_arr[kept], flux_detr_full[kept].copy()
    )
    feats["scale_mean"] = scaling_metrics.get("mean", np.nan)
    feats["scale_std"] = scaling_metrics.get("std", np.nan)
    feats["scale_skewness"] = scaling_metrics.get("skewness", np.nan)
    feats["scale_kurtosis"] = scaling_metrics.get("kurtosis", np.nan)
    feats["scale_outlier_resistance"] = scaling_metrics.get(
        "outlier_resistance", np.nan
    )

    binned = folded_binned_metrics(
        time_arr, flux_features, period, t0, lags_hours=(1, 3, 6, 12, 24)
    )
    feats["local_noise"] = binned.get("local_noise", np.nan)
    feats["depth_stability"] = binned.get("depth_stability", np.nan)
    acf_l = binned.get("acf_lags", {})
    feats["acf_lag_1h"] = acf_l.get(1, np.nan)
    feats["acf_lag_3h"] = acf_l.get(3, np.nan)
    feats["acf_lag_6h"] = acf_l.get(6, np.nan)
    feats["acf_lag_12h"] = acf_l.get(12, np.nan)
    feats["acf_lag_24h"] = acf_l.get(24, np.nan)
    feats["cadence_hours"] = binned.get("cadence_hours", np.nan)

    per = per_transit_stats_simple(time_arr, flux_features, period, t0, duration_days)
    depths = per.get("depths", np.array([]))
    npts_in_transit = per.get("npts_in_transit", np.array([]))
    print(f"Per-transit analysis: {len(depths)} transits analyzed")
    feats["depth_mean_per_transit"] = (
        float(np.nanmean(depths)) if depths.size else np.nan
    )
    feats["depth_std_per_transit"] = float(np.nanstd(depths)) if depths.size else np.nan
    feats["npts_transit_median"] = (
        float(np.nanmedian(npts_in_transit)) if npts_in_transit.size else np.nan
    )

    print("Computing CDPP...")
    cdpp = calculate_cdpp(
        flux_detr_full,
        cadence_hours=feats["cadence_hours"],
        time=time_arr,
        exclude_mask=mask_transit,
        gap_mask=gap_arr,
    )
    feats["cdpp_3h"] = cdpp.get("cdpp_3h", np.nan)
    feats["cdpp_6h"] = cdpp.get("cdpp_6h", np.nan)
    feats["cdpp_12h"] = cdpp.get("cdpp_12h", np.nan)

    duration_hours = feats["duration_hours"]
    cdpp_interp = interp_cdpp(cdpp, duration_hours)
    print("Computing SES/MES...")
    sesmes = compute_SES_MES(
        time_arr,
        flux_detr_full,
        period,
        t0,
        duration_days,
        cadence_hours=feats["cadence_hours"],
        transit_mask=mask_transit,
        gap_mask=gap_arr,
    )
    SES_arr = sesmes.get("SES", np.array([]))
    bls_info["n_mes_events"] = int(np.sum(np.isfinite(SES_arr)))
    MES_val = sesmes.get("MES", np.nan)
    max_ses = sesmes.get("max_ses", np.nan)
    max_mes = sesmes.get("max_mes", np.nan)
    feats["SES_mean"] = float(np.nanmean(SES_arr)) if SES_arr.size else np.nan
    feats["SES_std"] = float(np.nanstd(SES_arr)) if SES_arr.size else np.nan
    feats["MES"] = float(MES_val)
    feats["max_ses"] = float(max_ses)
    feats["max_mes"] = float(max_mes)
    print(
        f"SES/MES computed: MES={MES_val:.2f}, "
        f"max_ses={max_ses:.2f}, max_mes={max_mes:.2f}"
    )

    depth_mean = feats["depth_mean_per_transit"]
    if np.isfinite(depth_mean) and np.isfinite(cdpp_interp) and cdpp_interp > 0:
        feats["snr_global"] = float((depth_mean * 1e6) / cdpp_interp)
    else:
        feats["snr_global"] = np.nan

    if SES_arr.size:
        feats["snr_per_transit_mean"] = float(np.nanmean(SES_arr))
        feats["snr_per_transit_std"] = float(np.nanstd(SES_arr))
    else:
        feats["snr_per_transit_mean"] = np.nan
        feats["snr_per_transit_std"] = np.nan

    resid_global_full = flux_features - np.nanmedian(flux_features)
    feats["resid_rms_global"] = (
        float(np.nanstd(resid_global_full))
        if np.any(np.isfinite(resid_global_full))
        else np.nan
    )

    try:
        nbins = 200
        phase = ((time_arr - t0) / period) % 1.0
        phase = (phase + 0.5) % 1.0 - 0.5
        bins = np.linspace(-0.5, 0.5, nbins + 1)
        med_profile, _, _ = binned_statistic(
            phase, flux_features, statistic="median", bins=bins
        )
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        dur_phase = (duration_days / period) if (period > 0) else 0.05

        center_mask = np.abs(bin_centers) < (0.25 * dur_phase)
        shoulder_mask = (np.abs(bin_centers) > (0.5 * dur_phase)) & (
            np.abs(bin_centers) < dur_phase
        )

        center_flux = (
            np.nanmedian(med_profile[center_mask]) if np.any(center_mask) else np.nan
        )
        shoulder_flux = (
            np.nanmedian(med_profile[shoulder_mask])
            if np.any(shoulder_mask)
            else np.nan
        )

        depth_for_shape = abs(feats["depth_mean_per_transit"])
        num = shoulder_flux - center_flux

        if (
            np.isfinite(center_flux)
            and np.isfinite(shoulder_flux)
            and np.isfinite(depth_for_shape)
            and depth_for_shape > 0
        ):
            feats["vshape_metric"] = max(0.0, num / depth_for_shape)
        else:
            feats["vshape_metric"] = np.nan
    except Exception:
        feats["vshape_metric"] = np.nan

    feats["secondary_depth"] = compute_secondary_depth(
        time_arr, flux_features, period, t0, duration_days
    )

    # CDPP-based secondary SNR (cadence-invariant, ppm-consistent)
    cdpp_interp_for_duration = interp_cdpp(cdpp, duration_hours)
    if (
        np.isfinite(feats["secondary_depth"])
        and np.isfinite(cdpp_interp_for_duration)
        and cdpp_interp_for_duration > 0
    ):
        sec_snr = float((feats["secondary_depth"] * 1e6) / cdpp_interp_for_duration)
    else:
        sec_snr = np.nan
    feats["secondary_depth_snr"] = sec_snr
    feats["secondary_depth_snr_log"] = (
        float(np.log1p(sec_snr)) if np.isfinite(sec_snr) else np.nan
    )
    feats["secondary_depth_snr_capped"] = (
        float(np.clip(sec_snr, 0.0, 100.0)) if np.isfinite(sec_snr) else np.nan
    )

    # Step 4: Add EB/grazing discriminants
    print("Computing EB/grazing discriminants...")
    feats["odd_even_depth_ratio"] = compute_odd_even_depth_ratio(
        time_arr, flux_features, period, t0, duration_days
    )
    feats["ingress_egress_asymmetry"] = compute_ingress_egress_asymmetry(
        time_arr, flux_features, period, t0, duration_days
    )
    feats["secondary_depth_snr"] = compute_secondary_depth_snr(
        time_arr, flux_features, period, t0, duration_days, feats["local_noise"]
    )

    print(
        f"EB/Grazing features: odd_even_ratio={feats['odd_even_depth_ratio']:.3f}, "
        f"asymmetry={feats['ingress_egress_asymmetry']:.3f}, "
        f"sec_snr={feats['secondary_depth_snr']:.2f}"
    )

    finite_scaled = np.isfinite(flux_scaled)
    if np.sum(finite_scaled) > 2:
        feats["skewness_flux"] = float(skew(flux_scaled[finite_scaled]))
        feats["kurtosis_flux"] = float(kurtosis(flux_scaled[finite_scaled]))
    else:
        feats["skewness_flux"] = np.nan
        feats["kurtosis_flux"] = np.nan
    feats["outlier_resistance"] = (
        float(
            np.sum(np.abs(flux_scaled[finite_scaled]) > 5) / np.sum(finite_scaled) * 100
        )
        if np.sum(finite_scaled) > 0
        else np.nan
    )

    # Note: For CSV data, we don't have stellar radius information
    feats["planet_radius_rearth"] = np.nan
    feats["planet_radius_rjup"] = np.nan

    total_time = time.time() - start_time
    print(f"Feature extraction completed in {total_time:.2f} seconds")
    print(f"Total features extracted: {len(feats)}")

    if verbose:
        print("\n=== Extracted features (ordered) ===")
        for k, v in feats.items():
            print(f"{k}: {v}")

    return feats, bls_info, mask_transit


def _periodic_mask(time, period, t0, duration_days, width_factor=1.0):
    if not (
        np.isfinite(period)
        and period > 0
        and np.isfinite(t0)
        and np.isfinite(duration_days)
        and duration_days > 0
    ):
        return np.zeros(np.asarray(time).shape, dtype=bool)
    phase_days = np.mod(np.asarray(time) - t0 + 0.5 * period, period) - 0.5 * period
    return np.abs(phase_days) <= 0.5 * float(width_factor) * duration_days


def _contiguous_runs(flags, time, max_gap):
    """Return ``(start, stop)`` index pairs for each run of True in ``flags``.

    A run is also broken by a time gap wider than ``max_gap``: consecutive
    array indices can straddle a quarter boundary, and two dips on either side
    of it are separate events, not one long one.
    """
    indices = np.flatnonzero(flags)
    if indices.size == 0:
        return []
    time = np.asarray(time, dtype=float)
    gap = np.diff(time[indices])
    breaks = np.flatnonzero((np.diff(indices) > 1) | (gap > float(max_gap)))
    starts = np.r_[0, breaks + 1]
    stops = np.r_[breaks + 1, indices.size]
    return [
        (int(indices[s]), int(indices[e - 1]) + 1) for s, e in zip(starts, stops)
    ]


def _deep_events(time, flux, active):
    """Locate deep dips among the cadences still available to the search.

    Depths are measured against the robust baseline of the active series, so
    the scale is the same one the detection statistics use. Each event must
    hold at least ``DEEP_EVENT_MIN_CADENCES`` cadences below the core
    threshold, which keeps isolated outliers and cosmic rays out.
    """
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    active = np.asarray(active, dtype=bool)
    usable = active & np.isfinite(time) & np.isfinite(flux)
    if np.sum(usable) < 20:
        return []

    values = flux[usable]
    baseline = float(np.nanmedian(values))
    mad = float(np.nanmedian(np.abs(values - baseline)))
    scatter = 1.4826 * mad
    if not (np.isfinite(baseline) and np.isfinite(scatter) and scatter > 0):
        return []

    steps = np.diff(time[usable])
    steps = steps[np.isfinite(steps) & (steps > 0)]
    cadence = float(np.median(steps)) if steps.size else 0.02
    max_gap = max(3.0 * cadence, 0.1)

    core = usable & (flux < baseline - DEEP_EVENT_CORE_SIGMA * scatter)
    edge = usable & (flux < baseline - DEEP_EVENT_EDGE_SIGMA * scatter)
    if not np.any(core):
        return []

    events = []
    for start, stop in _contiguous_runs(edge, time, max_gap):
        span = slice(start, stop)
        core_in_span = core[span] & usable[span]
        if int(np.sum(core_in_span)) < DEEP_EVENT_MIN_CADENCES:
            continue
        floor = float(np.nanmedian(flux[span][core_in_span]))
        depth = baseline - floor
        if not (np.isfinite(depth) and depth > 0):
            continue
        events.append(
            {
                "start": start,
                "stop": stop,
                "center": float(np.nanmedian(time[span][usable[span]])),
                "depth": depth,
            }
        )
    return events


def _explained_by_accepted(
    center_time, accepted_rows, width_factor=DEEP_EVENT_RESIDUAL_WIDTH
):
    """True if ``center_time`` lands on a transit an accepted candidate predicts."""
    for row in accepted_rows:
        period = float(row.get("period_days", row.get("period", np.nan)))
        t0 = float(row.get("t0", np.nan))
        duration = float(row.get("duration_days", row.get("duration", np.nan)))
        if not (
            np.isfinite(period)
            and period > 0
            and np.isfinite(t0)
            and np.isfinite(duration)
            and duration > 0
        ):
            continue
        phase = np.mod(center_time - t0 + 0.5 * period, period) - 0.5 * period
        if abs(phase) <= 0.5 * float(width_factor) * duration:
            return True
    return False


def _group_events_by_depth(events, tolerance=DEEP_EVENT_DEPTH_GROUP_TOLERANCE):
    """Single-link events whose floors agree to within ``tolerance``."""
    ordered = sorted(events, key=lambda event: event["depth"])
    groups = []
    current = []
    for event in ordered:
        if current:
            previous = current[-1]["depth"]
            reference = max(event["depth"], previous)
            if reference > 0 and abs(event["depth"] - previous) > tolerance * reference:
                groups.append(current)
                current = []
        current.append(event)
    if current:
        groups.append(current)
    return groups


def _unexplained_deep_event_mask(
    time, flux, active, accepted_rows, min_events=MIN_OBSERVED_TRANSIT_EVENTS
):
    """Flag deep dips that no accepted candidate explains and no search can claim.

    Run after an accepted candidate's own windows are removed. Two kinds of
    leftover are withdrawn from the remaining search:

    * events sitting on an accepted candidate's predicted transit, which are
      that candidate's own transits escaping an epoch-drifted mask, and
    * groups of similar-depth events too small to ever be detected on their
      own, since a credible detection needs ``min_events`` transits.

    Anything else is left alone: a group of ``min_events`` or more equal-depth
    dips is a periodic signal a later iteration can still find, and masking it
    would hide a real planet rather than a false alarm.
    """
    time = np.asarray(time, dtype=float)
    active = np.asarray(active, dtype=bool)
    removal = np.zeros(active.shape, dtype=bool)
    events = _deep_events(time, flux, active)
    if not events:
        return removal

    doomed = []
    unclaimed = []
    for event in events:
        if _explained_by_accepted(event["center"], accepted_rows):
            doomed.append(event)
        else:
            unclaimed.append(event)

    residual_count = len(doomed)
    spared = 0
    for group in _group_events_by_depth(unclaimed):
        if len(group) >= int(min_events):
            spared += len(group)
            continue
        doomed.extend(group)

    if not doomed:
        return removal

    for event in doomed:
        start = max(0, event["start"] - DEEP_EVENT_PAD_CADENCES)
        stop = min(removal.size, event["stop"] + DEEP_EVENT_PAD_CADENCES)
        removal[start:stop] = True
    removal &= active

    fraction = float(np.sum(removal)) / float(max(1, active.size))
    if fraction > DEEP_EVENT_MAX_MASK_FRACTION:
        print(
            f"Deep-event sweep would mask {fraction:.1%} of cadences "
            f"(limit {DEEP_EVENT_MAX_MASK_FRACTION:.0%}); skipping"
        )
        return np.zeros(active.shape, dtype=bool)

    print(
        f"Deep-event sweep: {len(doomed)} event(s) removed "
        f"({residual_count} residual of accepted candidates, "
        f"{len(doomed) - residual_count} too few to be detectable), "
        f"{spared} left for later iterations, "
        f"{int(np.sum(removal)):,} cadences"
    )
    return removal


def _same_ephemeris(first, second, period_tolerance=PERIOD_DEDUP_TOLERANCE):
    first_period = float(first.get("period_days", first.get("period", np.nan)))
    second_period = float(second.get("period_days", second.get("period", np.nan)))
    first_t0 = float(first.get("t0", np.nan))
    second_t0 = float(second.get("t0", np.nan))
    first_duration = float(
        first.get("duration_days", first.get("duration", np.nan))
    )
    second_duration = float(
        second.get("duration_days", second.get("duration", np.nan))
    )
    if not (
        np.isfinite(first_period)
        and first_period > 0
        and np.isfinite(second_period)
        and second_period > 0
    ):
        return False
    relative_period_difference = abs(first_period - second_period) / min(
        first_period, second_period
    )
    if relative_period_difference >= period_tolerance:
        return False
    if not (np.isfinite(first_t0) and np.isfinite(second_t0)):
        return True
    reference_period = 0.5 * (first_period + second_period)
    phase_difference = abs(
        np.mod(first_t0 - second_t0 + 0.5 * reference_period, reference_period)
        - 0.5 * reference_period
    )
    duration_tolerance = max(
        value
        for value in (first_duration, second_duration, 0.02)
        if np.isfinite(value) and value > 0
    )
    return phase_difference <= duration_tolerance


def _period_already_accepted(
    candidate, accepted_rows, period_tolerance=PERIOD_DEDUP_TOLERANCE
):
    """True if ``candidate``'s period nearly matches an already-accepted one.

    Checked on period alone, ignoring phase (unlike ``_same_ephemeris``). A
    deep, many-cycle transit that survives masking imperfectly can resurface
    in a later iteration at numerically the same true period but a drifted
    epoch, since whatever residual structure escapes masking shifts from one
    iteration to the next - a phase-agreement requirement misses these.
    Two unrelated real transiting planets essentially never share a period
    this closely, so period proximity alone is a safe duplicate signal here.
    """
    candidate_period = float(
        candidate.get("period_days", candidate.get("period", np.nan))
    )
    if not (np.isfinite(candidate_period) and candidate_period > 0):
        return False
    for previous in accepted_rows:
        previous_period = float(
            previous.get("period_days", previous.get("period", np.nan))
        )
        if not (np.isfinite(previous_period) and previous_period > 0):
            continue
        relative_difference = abs(candidate_period - previous_period) / min(
            candidate_period, previous_period
        )
        if relative_difference < period_tolerance:
            return True
    return False


def _candidate_passes_mes(
    features,
    diagnostics=None,
    threshold=MES_DETECTION_THRESHOLD,
    min_events=MIN_OBSERVED_TRANSIT_EVENTS,
):
    max_mes = float(features.get("max_mes", np.nan))
    if diagnostics is None:
        observed_events = min_events
    else:
        observed_events = int(diagnostics.get("n_mes_events", 0))
    return (
        np.isfinite(max_mes)
        and max_mes >= float(threshold)
        and observed_events >= int(min_events)
    )


def _quick_candidate_diagnostics(time, flux, hint, gap_mask=None):
    period = float(hint.get("period", np.nan))
    duration = float(hint.get("duration", np.nan))
    t0 = float(hint.get("t0", np.nan))
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    positive_steps = np.diff(time)
    positive_steps = positive_steps[np.isfinite(positive_steps) & (positive_steps > 0)]
    cadence_hours = (
        float(np.median(positive_steps) * 24.0)
        if positive_steps.size
        else np.nan
    )
    transit_mask = _periodic_mask(time, period, t0, duration)
    result = compute_SES_MES(
        time,
        flux,
        period,
        t0,
        duration,
        cadence_hours=cadence_hours,
        transit_mask=transit_mask,
        gap_mask=gap_mask,
    )
    return {
        "max_mes": float(result.get("max_mes", np.nan)),
        "n_mes_events": int(np.sum(np.isfinite(result.get("SES", [])))),
    }


def extract_features_from_arrays(
    tTime, flux, verbose=False, refine_duration=True, use_tls=False, mask_eclipses=False
):
    """Iteratively extract MES-qualified transit candidates from one light curve.

    Each iteration computes one global BLS periodogram, ranks independent local
    maxima, and measures them in descending power order until one reaches the
    7.1 MES threshold. Accepted transit windows are padded and removed from the
    next iteration. This is a first masking implementation, not model
    subtraction or a joint multi-planet fit.

    A candidate whose period nearly matches an already-accepted one is treated
    as a duplicate regardless of phase: imperfectly masked deep transits can
    resurface at the same period with a drifted epoch in a later iteration.
    """
    time_values = np.asarray(tTime)
    flux_values = np.asarray(flux)
    finite = np.isfinite(time_values) & np.isfinite(flux_values)
    if np.sum(finite) < 3:
        raise ValueError("too few valid points")
    time_values = time_values[finite]
    flux_values = flux_values[finite]
    active = np.ones(time_values.size, dtype=bool)
    accepted_rows = []
    rejected_ephemerides = []

    for iteration in range(MAX_TRANSIT_CANDIDATES):
        if np.sum(active) < MIN_SEARCH_POINTS:
            break
        print(
            f"\n=== Transit search iteration {iteration + 1} "
            f"({np.sum(active):,} active cadences) ==="
        )
        gapped = ~active
        features, bls_info, _ = _extract_single_candidate_from_arrays(
            time_values,
            flux_values,
            verbose=verbose,
            refine_duration=refine_duration,
            use_tls=use_tls,
            mask_eclipses=mask_eclipses,
            gap_mask=gapped,
        )
        ranked_peaks = list(bls_info.get("search_candidates", []))
        attempts = [(features, bls_info, None)]
        attempts.extend(
            (None, None, peak)
            for peak in ranked_peaks[1:MAX_PEAKS_TO_VALIDATE]
        )

        accepted = None
        accepted_info = None
        refined_peaks = 1
        for measured_features, measured_info, hint in attempts:
            if measured_features is None:
                hint_ephemeris = {
                    "period_days": hint.get("period", np.nan),
                    "t0": hint.get("t0", np.nan),
                    "duration_days": hint.get("duration", np.nan),
                }
                if _period_already_accepted(hint_ephemeris, accepted_rows) or any(
                    _same_ephemeris(hint_ephemeris, previous)
                    for previous in rejected_ephemerides
                ):
                    continue
                quick_info = _quick_candidate_diagnostics(
                    time_values, flux_values, hint, gap_mask=gapped
                )
                quick_features = {
                    "period_days": hint.get("period", np.nan),
                    "t0": hint.get("t0", np.nan),
                    "duration_days": hint.get("duration", np.nan),
                    "max_mes": quick_info["max_mes"],
                }
                if not _candidate_passes_mes(
                    quick_features,
                    quick_info,
                    threshold=PROVISIONAL_MES_THRESHOLD,
                ):
                    rejected_ephemerides.append(quick_features)
                    print(
                        f"Skipped queued peak P={quick_features['period_days']:.6f} d: "
                        f"provisional max_mes={quick_features['max_mes']:.2f}, "
                        f"events={quick_info['n_mes_events']}"
                    )
                    continue
                if refined_peaks >= MAX_REFINED_PEAKS_PER_ITERATION:
                    print("Reached the queued-peak refinement budget")
                    break
                refined_peaks += 1
                (
                    measured_features,
                    measured_info,
                    _,
                ) = _extract_single_candidate_from_arrays(
                    time_values,
                    flux_values,
                    verbose=verbose,
                    refine_duration=refine_duration,
                    use_tls=use_tls,
                    mask_eclipses=mask_eclipses,
                    candidate_hint=hint,
                    gap_mask=gapped,
                )
            if _period_already_accepted(measured_features, accepted_rows) or any(
                _same_ephemeris(measured_features, previous)
                for previous in rejected_ephemerides
            ):
                continue
            if _candidate_passes_mes(measured_features, measured_info):
                accepted = measured_features
                accepted_info = measured_info
                break
            rejected_ephemerides.append(measured_features)
            observed_events = int(measured_info.get("n_mes_events", 0))
            print(
                f"Rejected BLS candidate P={measured_features['period_days']:.6f} d: "
                f"max_mes={measured_features['max_mes']:.2f}, "
                f"events={observed_events} "
                f"(requires MES >= {MES_DETECTION_THRESHOLD:.1f} and "
                f">= {MIN_OBSERVED_TRANSIT_EVENTS} events)"
            )

        if accepted is None:
            print("No remaining independent BLS peak passed the MES threshold")
            break

        accepted_rows.append(accepted)
        print(
            f"Accepted candidate {len(accepted_rows)}: "
            f"P={accepted['period_days']:.6f} d, "
            f"max_mes={accepted['max_mes']:.2f}"
        )
        remove_from_active = _periodic_mask(
            time_values,
            accepted["period_days"],
            accepted["t0"],
            accepted["duration_days"],
            width_factor=TRANSIT_MASK_WIDTH,
        ) & active
        if not np.any(remove_from_active):
            break
        active[remove_from_active] = False

        # The periodic mask only removes what this candidate predicts. Deep
        # dips left behind belong to something else, and BLS will weave a box
        # train through them next iteration if they stay in the search. The
        # sweep needs the detrended series: on raw flux, stellar variability
        # alone would drift below any fixed depth threshold.
        sweep_flux = (
            accepted_info.get("flux_detrended") if accepted_info is not None else None
        )
        if sweep_flux is not None and np.shape(sweep_flux) == time_values.shape:
            active[
                _unexplained_deep_event_mask(
                    time_values, sweep_flux, active, accepted_rows
                )
            ] = False

        masked_fraction = 1.0 - float(np.sum(active)) / float(active.size)
        if masked_fraction >= MAX_CUMULATIVE_MASK_FRACTION:
            print(
                f"Stopping after masking {masked_fraction:.1%} of valid cadences"
            )
            break

    return sorted(
        accepted_rows,
        key=lambda row: float(row.get("period_days", np.inf)),
    )


def extract_all_features_from_csv(
    csv_path,
    verbose=False,
    refine_duration=True,
    use_tls=False,
    mask_eclipses=False,
    include_ml_cutouts=False,
):
    """Extract candidate feature rows from a light-curve CSV."""
    import pandas as pd

    print(f"Loading light curve data from: {csv_path}")

    # Read the CSV file
    df = pd.read_csv(csv_path)
    time = df["time"].values
    flux = df["flux"].values

    print(f"CSV loaded: {len(time):,} data points")
    print(f"File size: {os.path.getsize(csv_path) / 1024 / 1024:.2f} MB")

    if verbose:
        print(f"Loaded light curve data from: {csv_path}")
        print(f"Data points: {len(time)}")

    # Use the same feature extraction logic as extract_all_features_v2
    feature_rows = extract_features_from_arrays(
        time,
        flux,
        verbose=verbose,
        refine_duration=refine_duration,
        use_tls=use_tls,
        mask_eclipses=mask_eclipses,
    )
    return feature_rows


def extract_features_from_lightcurve(
    lc, verbose=False, refine_duration=True, use_tls=False, mask_eclipses=False
):
    """Extract candidate feature rows from a LightCurve object."""
    time = lc.time.value
    flux = lc.flux.value

    # Use the internal function for feature extraction
    feature_rows = extract_features_from_arrays(
        time,
        flux,
        verbose=verbose,
        refine_duration=refine_duration,
        use_tls=use_tls,
        mask_eclipses=mask_eclipses,
    )

    # Add stellar radius information if available
    if "RADIUS" in lc.meta and np.isfinite(lc.meta["RADIUS"]):
        stellar_radius = lc.meta["RADIUS"]
        for feats in feature_rows:
            depth = feats["depth_mean_per_transit"]
            if np.isfinite(depth) and depth >= 0:
                Rp_over_Rs = np.sqrt(depth)
                feats["planet_radius_rearth"] = stellar_radius * 109.1 * Rp_over_Rs
                feats["planet_radius_rjup"] = stellar_radius * 9.731 * Rp_over_Rs

    return feature_rows
