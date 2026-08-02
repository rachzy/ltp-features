import time
import numpy as np
from scipy.interpolate import UnivariateSpline
from astropy.timeseries import BoxLeastSquares
from scipy.signal import find_peaks
from scipy.stats import binned_statistic
from utils.ephemeris import canonical_epoch, epoch_phase_offset

MAX_GLOBAL_PERIODS = 20_000
TARGET_POINT_PERIOD_EVALUATIONS = 200_000_000


def rank_independent_bls_candidates(
    periodogram,
    max_candidates=20,
    min_relative_separation=0.005,
    period_bands=16,
):
    """Rank local BLS maxima after clustering nearby and same-band peaks."""
    power = np.asarray(periodogram.power, dtype=float)
    periods = np.asarray(periodogram.period, dtype=float)
    if power.ndim == 1:
        power_per_period = power
        duration_indices = np.zeros(periods.size, dtype=int)
    else:
        finite_power = np.where(np.isfinite(power), power, -np.inf)
        power_per_period = np.max(finite_power, axis=1)
        duration_indices = np.argmax(finite_power, axis=1)

    finite = np.isfinite(power_per_period) & np.isfinite(periods) & (periods > 0)
    if not np.any(finite):
        return []
    safe_power = np.where(finite, power_per_period, -np.inf)
    peak_indices = find_peaks(safe_power)[0]
    edge_indices = []
    if safe_power.size == 1:
        edge_indices.append(0)
    elif safe_power.size > 1:
        if safe_power[0] > safe_power[1]:
            edge_indices.append(0)
        if safe_power[-1] > safe_power[-2]:
            edge_indices.append(safe_power.size - 1)
    best_index = int(np.argmax(safe_power))
    peak_indices = np.unique(
        np.concatenate((peak_indices, np.asarray(edge_indices, dtype=int), [best_index]))
    )
    ranked_indices = peak_indices[np.argsort(safe_power[peak_indices])[::-1]]

    duration_values = np.asarray(
        getattr(periodogram, "duration", np.array([np.nan])), dtype=float
    )
    transit_values = np.asarray(
        getattr(periodogram, "transit_time", np.array([np.nan])), dtype=float
    )

    def _value_for_index(values, period_index, duration_index):
        if values.ndim >= 2:
            return float(values[period_index, duration_index])
        flat = np.ravel(values)
        if flat.size == periods.size:
            return float(flat[period_index])
        if flat.size > duration_index:
            return float(flat[duration_index])
        return float(flat[0]) if flat.size else np.nan

    clustered = []
    for index in ranked_indices:
        period = float(periods[index])
        if any(
            abs(period - candidate["period"]) / min(period, candidate["period"])
            < min_relative_separation
            for candidate in clustered
        ):
            continue
        duration_index = int(duration_indices[index])
        clustered.append(
            {
                "period": period,
                "duration": _value_for_index(
                    duration_values, int(index), duration_index
                ),
                "t0": _value_for_index(transit_values, int(index), duration_index),
                "power": float(power_per_period[index]),
                "grid_index": int(index),
            }
        )

    finite_periods = periods[finite]
    band_edges = np.logspace(
        np.log10(np.min(finite_periods)),
        np.log10(np.max(finite_periods)),
        int(period_bands) + 1,
    )
    band_winners = []
    occupied_bands = set()
    for candidate in clustered:
        band = int(
            np.clip(
                np.searchsorted(band_edges, candidate["period"], side="right") - 1,
                0,
                len(band_edges) - 2,
            )
        )
        if band in occupied_bands:
            continue
        occupied_bands.add(band)
        band_winners.append(candidate)
    band_winners.sort(key=lambda candidate: candidate["power"], reverse=True)
    return band_winners[: int(max_candidates)]


def refine_transit_epoch(time, flux, period, duration, transit_time, oversample=50):
    """Refit transit phase at fixed period/duration on the final flux series.

    BLS transit times are quantized by the phase grid and the initial search is
    performed before the final spline detrending.  A one-period BLS pass on the
    final series removes those two avoidable sources of epoch error.  The
    returned epoch is canonicalized to the first predicted transit at or after
    the start of the supplied time series.
    """
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    valid = np.isfinite(time) & np.isfinite(flux)
    if (
        np.sum(valid) < 10
        or not np.isfinite(period)
        or period <= 0
        or not np.isfinite(duration)
        or duration <= 0
        or not np.isfinite(transit_time)
    ):
        return float(transit_time)

    t = time[valid]
    f = flux[valid]
    try:
        result = BoxLeastSquares(t, f).power(
            np.array([float(period)]),
            np.array([float(duration)]),
            oversample=max(10, int(oversample)),
        )
        index = int(np.nanargmax(result.power))
        refined = float(np.ravel(result.transit_time)[index])
        if not np.isfinite(refined):
            return float(transit_time)
        return canonical_epoch(refined, period, float(np.min(t)))
    except (TypeError, ValueError, FloatingPointError):
        return float(transit_time)

def detrend_with_bls_mask(
    tTime,
    flux,
    min_period=0.5,
    max_period=None,
    n_periods=2000,
    n_durations=200,
    oversample=10,
    bin_width=0.5,
    spline_s=0.001,
    max_iter=4,
    sigma=3.0,
    refine_duration=True,
    use_tls=False,
    mask_eclipses=False,
    eclipse_nsigma=10.0,
    eclipse_min_depth_abs=0.02,
    eclipse_min_group=6,
    eclipse_pad_points=2,
    eclipse_max_mask_fraction=0.8,
    candidate_hint=None,
):
    bls_start = time.time()
    print("Starting BLS detrending and period search...")

    def _duration_upper_bound(period, floor=0.01):
        """Keep duration fits from drifting to broad, non-physical windows."""
        if not np.isfinite(period) or period <= 0:
            return max(floor * 1.5, 0.2)
        duty_cap = 0.12 * float(period)
        # A central transit across a solar-density star lasts ~13 h * (P/1yr)^(1/3).
        # Allow 2.5x that to cover evolved, low-density hosts. This replaces a flat
        # 0.35 d cap that clipped real durations for P >~ 100 d (T14 grows with P).
        physical_cap = 2.5 * 0.5417 * (float(period) / 365.25) ** (1.0 / 3.0)
        return max(floor * 1.5, min(physical_cap, duty_cap))

    def _folded_profile_duration(time, flux_values, period, transit_time, floor):
        if not (np.isfinite(period) and period > 0 and np.isfinite(transit_time)):
            return np.nan
        mask = np.isfinite(time) & np.isfinite(flux_values)
        if np.sum(mask) < 50:
            return np.nan

        t = np.asarray(time)[mask]
        f = np.asarray(flux_values)[mask]
        phase = ((t - transit_time) / period + 0.5) % 1.0 - 0.5
        bins = np.linspace(-0.5, 0.5, 241)
        med, _, _ = binned_statistic(phase, f, statistic="median", bins=bins)
        centers = 0.5 * (bins[:-1] + bins[1:])
        finite = np.isfinite(med)
        if np.sum(finite) < 20:
            return np.nan

        oot = finite & (np.abs(centers) > 0.25)
        baseline = np.nanmedian(med[oot]) if np.any(oot) else np.nanmedian(med[finite])
        search = finite & (np.abs(centers) < 0.20)
        if np.sum(search) < 3 or not np.isfinite(baseline):
            return np.nan

        search_idx = np.flatnonzero(search)
        min_idx = int(search_idx[np.nanargmin(med[search])])
        low_flux = float(med[min_idx])
        depth = baseline - low_flux
        if not (np.isfinite(depth) and depth > 0):
            return np.nan

        half_level = baseline - 0.5 * depth
        in_half = finite & (med < half_level) & (np.abs(centers) < 0.20)
        if not in_half[min_idx]:
            return np.nan

        left = min_idx
        while left > 0 and in_half[left - 1]:
            left -= 1
        right = min_idx
        while right < len(in_half) - 1 and in_half[right + 1]:
            right += 1

        phase_width = max(1, right - left + 1) / float(len(med))
        duration = phase_width * float(period)
        upper = _duration_upper_bound(period, floor)
        return float(np.clip(duration, floor, upper))

    mask_valid = np.isfinite(tTime) & np.isfinite(flux)
    tTime = np.asarray(tTime)[mask_valid]
    flux = np.asarray(flux)[mask_valid]
    # Keep a copy of raw (pre-stitch) flux for eclipse detection
    _flux_raw_for_eclipse = np.asarray(flux).copy()

    # Pre-BLS stitching: per-segment robust normalization and tail winsorization
    try:
        diffs_all = np.diff(tTime)
        diffs_all = diffs_all[np.isfinite(diffs_all)]
        if diffs_all.size:
            dt_med = np.nanmedian(diffs_all)
        else:
            dt_med = 0.02
        # Detect large gaps as segment boundaries
        gap_thresh = max(5.0 * dt_med, 0.5)  # days
        split_idx = np.where(np.diff(tTime) > gap_thresh)[0]
        starts = np.r_[0, split_idx + 1]
        ends = np.r_[split_idx + 1, tTime.size]
        winsor_p = 0.3  # percent for winsorization per segment (0.3%)
        flux_stitched = flux.copy()
        medians = []
        for s, e in zip(starts, ends):
            seg = slice(s, e)
            fseg = flux_stitched[seg]
            if fseg.size == 0:
                medians.append(np.nan)
                continue
            lo = np.nanpercentile(fseg, winsor_p)
            hi = np.nanpercentile(fseg, 100.0 - winsor_p)
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                fseg = np.clip(fseg, lo, hi)
            m = np.nanmedian(fseg)
            if not np.isfinite(m) or m == 0:
                m = 1.0
            flux_stitched[seg] = flux_stitched[seg] / m
            medians.append(m)
        # Align all segments to a common baseline ~1.0
        gmed = np.nanmedian(flux_stitched)
        if np.isfinite(gmed) and gmed != 0:
            flux = flux_stitched / gmed
        else:
            flux = flux_stitched
        print(
            f"Pre-BLS stitching: {len(starts)} segments, winsor {winsor_p}% per segment"
        )
    except Exception as _e:
        print(f"Pre-BLS stitching skipped due to error: {_e}")

    print(f"BLS input: {len(tTime):,} valid points")

    # Eclipse masking before BLS
    def _mask_deep_eclipses(
        time,
        flux,
        nsigma=8.0,
        min_depth_abs=0.02,
        min_group=3,
        pad_points=1,
        max_mask_fraction=0.2,
    ):
        time = np.asarray(time)
        flux = np.asarray(flux)
        if time.size < 5:
            return np.zeros_like(time, dtype=bool)
        med = np.nanmedian(flux)
        mad = np.nanmedian(np.abs(flux - med))
        sigma_loc = 1.4826 * mad if (mad > 0 and np.isfinite(mad)) else np.nanstd(flux)
        if not np.isfinite(sigma_loc) or sigma_loc <= 0:
            sigma_loc = 1e-6
        thr_sigma = med - nsigma * sigma_loc
        thr_abs = med * (1.0 - float(min_depth_abs))
        # Flag as eclipse if either condition is met (OR), not AND
        dips = np.isfinite(flux) & ((flux < thr_sigma) | (flux < thr_abs))
        if not np.any(dips):
            return np.zeros_like(time, dtype=bool)
        idx = np.flatnonzero(dips)
        splits = np.where(np.diff(idx) > 1)[0]
        starts = np.r_[0, splits + 1]
        ends = np.r_[splits + 1, idx.size]
        mask = np.zeros_like(time, dtype=bool)
        for s, e in zip(starts, ends):
            i0 = int(idx[s])
            i1 = int(idx[e - 1])
            if (i1 - i0 + 1) >= int(min_group):
                j0 = max(0, i0 - int(pad_points))
                j1 = min(time.size, i1 + int(pad_points) + 1)
                mask[j0:j1] = True
        frac = float(np.sum(mask)) / float(time.size)
        if frac > float(max_mask_fraction):
            return np.zeros_like(time, dtype=bool)
        return mask

    if mask_eclipses:
        eclipse_mask = _mask_deep_eclipses(
            tTime,
            _flux_raw_for_eclipse,
            nsigma=eclipse_nsigma,
            min_depth_abs=eclipse_min_depth_abs,
            min_group=eclipse_min_group,
            pad_points=eclipse_pad_points,
            max_mask_fraction=eclipse_max_mask_fraction,
        )
    else:
        eclipse_mask = np.zeros_like(tTime, dtype=bool)

    time_bls = tTime[~eclipse_mask]
    flux_bls = flux[~eclipse_mask]
    if time_bls.size < 10:
        time_bls = tTime
        flux_bls = flux
        eclipse_mask[:] = False
    # Log masking statistics, including min/median/max of masked depths
    if np.any(eclipse_mask):
        masked_vals = _flux_raw_for_eclipse[eclipse_mask]
        print(
            f"Masked eclipses: {np.sum(eclipse_mask):,} points | depth med={float(np.nanmedian(1-masked_vals)):.5f}"
        )
    else:
        print("Masked eclipses: 0 points")

    span_days_full = float(tTime.max() - tTime.min())

    if max_period is None:
        # Require >= 3 transits for a credible detection (Kepler SOC rule).
        # Periods between span/3 and the baseline only add false-alarm territory
        # that competes with real signals for the global power maximum.
        max_period = min(span_days_full / 3.0, 200.0)

    print(f"Period search range: {min_period:.3f} to {max_period:.3f} days")

    # Build a cadence-aware duration grid. Floor the minimum duration to ~2 samples to accommodate LC data
    diffs = np.diff(time_bls)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size:
        dt_days = np.median(diffs)
    else:
        dt_days = 0.02

    # Widest duration searched: physical bound at the longest period, further
    # limited by the BLS requirement that durations stay below the minimum period.
    max_dur_cap = min(
        _duration_upper_bound(max_period),
        0.9 * min_period,  # BLS constraint
    )
    min_dur_floor = max(0.01, 1.5 * dt_days)
    # 2 samples (was 3): with long-cadence-only data 3*dt = 0.061 d exceeded
    # real grazing-transit durations (e.g. Kepler-447b T14 = 0.047 d).
    best_floor = max(0.01, 2.0 * dt_days)

    if min_dur_floor >= max_dur_cap:
        min_dur_floor = 0.5 * max_dur_cap

    durations = np.linspace(min_dur_floor, max_dur_cap, n_durations)

    # Remove low-frequency stellar variability from the BLS input only.
    # BLS otherwise runs on raw stitched flux; spotted/rotating stars (percent-level
    # modulation over days, e.g. Kepler-447) dominate the box search and the true
    # transit never surfaces. The window scales with the widest searched duration
    # so long transits stay well under half of any bin and survive the median.
    try:
        var_window = max(1.0, 3.0 * max_dur_cap)
        bins_v = np.arange(time_bls.min(), time_bls.max() + var_window, var_window)
        med_v, _, _ = binned_statistic(
            time_bls, flux_bls, statistic="median", bins=bins_v
        )
        centers_v = 0.5 * (bins_v[:-1] + bins_v[1:])
        good_v = np.isfinite(med_v)
        if np.sum(good_v) >= 3:
            trend_v = np.interp(time_bls, centers_v[good_v], med_v[good_v])
            ok_v = np.isfinite(trend_v) & (trend_v > 0)
            flux_bls = flux_bls.copy()
            flux_bls[ok_v] = flux_bls[ok_v] / trend_v[ok_v]
            print(
                f"Pre-BLS variability removal: window={var_window:.2f} d, "
                f"trend rms={float(np.nanstd(med_v[good_v])):.5f}"
            )
    except Exception as _e:
        print(f"Pre-BLS variability removal skipped: {_e}")

    # Stage-1 global search runs on flux binned well below the shortest searched
    # duration: box power is preserved, per-period cost drops, and the saved
    # budget buys a denser period grid.
    bin_stage1 = max(dt_days, min_dur_floor / 3.0)
    if bin_stage1 > 1.5 * dt_days and time_bls.size > 200000:
        edges1 = np.arange(time_bls.min(), time_bls.max() + bin_stage1, bin_stage1)
        f_b, _, _ = binned_statistic(
            time_bls, flux_bls, statistic="mean", bins=edges1
        )
        t_b, _, _ = binned_statistic(
            time_bls, time_bls, statistic="mean", bins=edges1
        )
        good_b = np.isfinite(f_b) & np.isfinite(t_b)
        time_stage1 = t_b[good_b]
        flux_stage1 = f_b[good_b]
        print(
            f"Stage-1 binning: {time_bls.size:,} -> {time_stage1.size:,} points "
            f"(bin {bin_stage1:.4f} d)"
        )
    else:
        time_stage1 = time_bls
        flux_stage1 = flux_bls

    # Duration-aware period grid. Coherent folding over the full baseline needs
    # log-period steps of ~ eps * duration / span (the required spacing is
    # proportional to the signal duration, so short-duty signals need dense
    # grids). The exact need usually exceeds any sane budget, so size the grid
    # by a compute budget (periods x points) and treat the result as stage 1;
    # the local refinement passes downstream supply the final precision.
    d_ref = max(min_dur_floor, 0.03)
    n_needed = int(
        np.ceil(np.log(max_period / min_period) * span_days_full / (3.0 * d_ref))
    )
    n_budget = int(
        np.clip(round(TARGET_POINT_PERIOD_EVALUATIONS / max(1, time_stage1.size)), n_periods, MAX_GLOBAL_PERIODS)
    )
    n_periods_eff = int(np.clip(n_needed, n_periods, n_budget))
    period_grid = np.logspace(
        np.log10(min_period), np.log10(max_period), n_periods_eff
    )

    print(f"Duration search range: {min_dur_floor:.4f} to {max_dur_cap:.4f} days")
    print(
        f"Search grid: {n_periods_eff} periods × {n_durations} durations = {n_periods_eff * n_durations:,} combinations"
        f" (needed {n_needed:,}, budget {n_budget:,})"
    )

    # Full-resolution BLS is kept for the refinement passes and transit mask;
    # the global periodogram runs on the stage-1 (possibly binned) series.
    bls = BoxLeastSquares(time_bls, flux_bls)
    bls_global = (
        bls
        if time_stage1 is time_bls
        else BoxLeastSquares(time_stage1, flux_stage1)
    )
    if candidate_hint is None:
        print("Computing BLS periodogram...")
        periodogram = bls_global.power(period_grid, durations, oversample=oversample)
        print("BLS periodogram computed successfully")
        search_candidates = rank_independent_bls_candidates(periodogram)

        if np.ndim(periodogram.power) == 1:
            idx_best = np.nanargmax(periodogram.power)
            best_period = periodogram.period[idx_best]
            best_duration = (
                periodogram.duration[idx_best]
                if hasattr(periodogram, "duration")
                else durations[0]
            )
            t0 = (
                periodogram.transit_time[idx_best]
                if hasattr(periodogram, "transit_time")
                else tTime[0]
            )
        else:
            power_per_period = np.nanmax(periodogram.power, axis=1)
            idx_best = int(np.nanargmax(power_per_period))
            best_period = periodogram.period[idx_best]
            dur_idx = int(np.nanargmax(periodogram.power[idx_best, :]))
            best_duration = (
                periodogram.duration[dur_idx]
                if hasattr(periodogram, "duration")
                else durations[dur_idx]
            )
            t0 = (
                periodogram.transit_time[idx_best]
                if hasattr(periodogram, "transit_time")
                else tTime[0]
            )
    else:
        best_period = float(candidate_hint.get("period", np.nan))
        best_duration = float(candidate_hint.get("duration", np.nan))
        t0 = float(candidate_hint.get("t0", np.nan))
        if not np.isfinite(best_duration) or best_duration <= 0:
            best_duration = float(np.median(durations))
        if not np.isfinite(t0):
            t0 = float(tTime[0])
        print(
            f"Refining queued BLS peak: P={best_period:.6f} days, "
            f"T14={best_duration:.4f} days"
        )
        periodogram = bls.power(
            np.array([best_period]),
            np.array([best_duration]),
            oversample=oversample,
        )
        search_candidates = [dict(candidate_hint)]
    print("dt_days =", dt_days)
    print("dur grid:", durations[0], durations[-1], len(durations))
    print("best before refine:", best_duration)

    print(
        f"Initial BLS result: P={best_period:.4f} days, T14={best_duration:.4f} days, t0={t0:.2f}"
    )

    best_cap = _duration_upper_bound(best_period, best_floor)
    if np.isfinite(best_duration):
        if best_duration < best_floor:
            best_duration = best_floor
        if np.isfinite(best_cap) and best_duration > best_cap:
            print(
                f"Clipping initial duration from {best_duration:.4f} to {best_cap:.4f} days"
            )
            best_duration = best_cap
        profile_duration = _folded_profile_duration(
            tTime, flux, best_period, t0, best_floor
        )
        if (
            np.isfinite(profile_duration)
            and profile_duration > best_floor
            and profile_duration < 0.75 * best_duration
        ):
            capped_duration = min(best_duration, 1.35 * profile_duration)
            print(
                f"Folded profile suggests narrower duration: {profile_duration:.4f} days; "
                f"capping to {capped_duration:.4f} days"
            )
            best_duration = capped_duration

    # Optional two-pass refinement for more accurate duration
    if refine_duration and np.isfinite(best_period) and np.isfinite(best_duration):
        print("Starting period/duration refinement...")
        # Second pass: focused search around detected period with proper duration range
        per_lo = best_period * 0.98
        per_hi = best_period * 1.02
        dur_lo = best_floor
        dur_hi = _duration_upper_bound(best_period, best_floor)

        if dur_hi > dur_lo:
            durations_refined = np.linspace(dur_lo, dur_hi, n_durations)
            periods_refined = np.logspace(
                np.log10(per_lo), np.log10(per_hi), max(256, n_periods // 4)
            )

            periodogram_refined = bls.power(
                periods_refined, durations_refined, oversample=oversample
            )

            if np.ndim(periodogram_refined.power) == 1:
                idx_best_refined = np.nanargmax(periodogram_refined.power)
                best_period_refined = periodogram_refined.period[idx_best_refined]
                best_duration_refined = (
                    periodogram_refined.duration[idx_best_refined]
                    if hasattr(periodogram_refined, "duration")
                    else durations_refined[0]
                )
                t0_refined = (
                    periodogram_refined.transit_time[idx_best_refined]
                    if hasattr(periodogram_refined, "transit_time")
                    else t0
                )
            else:
                power_per_period_refined = np.nanmax(periodogram_refined.power, axis=1)
                idx_best_refined = int(np.nanargmax(power_per_period_refined))
                best_period_refined = periodogram_refined.period[idx_best_refined]
                dur_idx_refined = int(
                    np.nanargmax(periodogram_refined.power[idx_best_refined, :])
                )
                best_duration_refined = (
                    periodogram_refined.duration[dur_idx_refined]
                    if hasattr(periodogram_refined, "duration")
                    else durations_refined[dur_idx_refined]
                )
                t0_refined = (
                    periodogram_refined.transit_time[idx_best_refined]
                    if hasattr(periodogram_refined, "transit_time")
                    else t0
                )

            # Use refined values if they're reasonable
            if (
                np.isfinite(best_period_refined)
                and np.isfinite(best_duration_refined)
                and best_period_refined > 0
                and best_duration_refined > 0
            ):
                best_period = best_period_refined
                best_duration = best_duration_refined
                t0 = t0_refined
                periodogram = periodogram_refined  # Update for consistency

                # Optional micro-zoom on duration at fixed period to reduce grid quantization
                try:
                    dur_half_span = max(0.15 * best_duration, best_floor)
                    d_lo = max(best_floor, best_duration - dur_half_span)
                    d_hi = min(
                        dur_hi,
                        _duration_upper_bound(best_period, best_floor),
                        best_duration + dur_half_span,
                    )
                    if np.isfinite(d_lo) and np.isfinite(d_hi) and d_hi > d_lo:
                        durations_zoom = np.linspace(d_lo, d_hi, max(64, n_durations))
                        periodogram_zoom = bls.power(
                            np.array([best_period]),
                            durations_zoom,
                            oversample=oversample,
                        )
                        # power has shape (len(durations),) when period array len == 1
                        idx_zoom = int(np.nanargmax(periodogram_zoom.power))
                        dur_zoom = (
                            periodogram_zoom.duration[idx_zoom]
                            if hasattr(periodogram_zoom, "duration")
                            else durations_zoom[idx_zoom]
                        )
                        if np.isfinite(dur_zoom) and dur_zoom > 0:
                            best_duration = float(dur_zoom)
                except Exception:
                    pass

                # Period de-aliasing: test P/k (k=2..20), 2*P, and m*P (m up to M)
                try:
                    print("Running period de-aliasing...")
                    cand_periods = [float(best_period) * 2.0]
                    for _k in range(2, 21):
                        cand_periods.append(float(best_period) / float(_k))
                    # Upward multiples based on baseline; a candidate m*P must
                    # still leave >= 3 transits in the data (Kepler SOC rule).
                    span_days = float(tTime.max() - tTime.min())
                    if best_period > 0 and np.isfinite(span_days):
                        max_m = int(
                            min(10, np.floor(span_days / (3.0 * best_period)))
                        )
                        for m_mult in range(2, max_m + 1):
                            cand_periods.append(float(best_period) * float(m_mult))
                    cand_results = []
                    tmin = float(tTime.min())
                    tmax = float(tTime.max())
                    # Compute baseline power once for acceptance tests
                    if np.ndim(periodogram.power) == 1:
                        base_power = float(np.nanmax(periodogram.power))
                    else:
                        base_power = float(
                            np.nanmax(np.nanmax(periodogram.power, axis=1))
                        )
                    for Pc in cand_periods:
                        if not (
                            np.isfinite(Pc)
                            and Pc >= min_period
                            and Pc <= max_period
                        ):
                            continue
                        per_lo_c = Pc * 0.995
                        per_hi_c = Pc * 1.005
                        dur_lo_c = best_floor
                        dur_hi_c = _duration_upper_bound(Pc, best_floor)
                        if not (
                            np.isfinite(dur_lo_c)
                            and np.isfinite(dur_hi_c)
                            and dur_hi_c > dur_lo_c
                        ):
                            continue
                        durations_c = np.linspace(
                            dur_lo_c, dur_hi_c, max(32, n_durations // 2)
                        )
                        periods_c = np.logspace(
                            np.log10(per_lo_c),
                            np.log10(per_hi_c),
                            max(64, n_periods // 16),
                        )
                        pg_c = bls.power(periods_c, durations_c, oversample=oversample)
                        if np.ndim(pg_c.power) == 1:
                            idx_c = np.nanargmax(pg_c.power)
                            P_c_best = float(pg_c.period[idx_c])
                            D_c_best = (
                                float(pg_c.duration[idx_c])
                                if hasattr(pg_c, "duration")
                                else float(durations_c[0])
                            )
                            t0_c_best = (
                                float(pg_c.transit_time[idx_c])
                                if hasattr(pg_c, "transit_time")
                                else float(t0)
                            )
                            power_c = float(pg_c.power[idx_c])
                        else:
                            power_per_P_c = np.nanmax(pg_c.power, axis=1)
                            idxP = int(np.nanargmax(power_per_P_c))
                            P_c_best = float(pg_c.period[idxP])
                            dur_idx_c = int(np.nanargmax(pg_c.power[idxP, :]))
                            D_c_best = (
                                float(pg_c.duration[dur_idx_c])
                                if hasattr(pg_c, "duration")
                                else float(durations_c[dur_idx_c])
                            )
                            t0_c_best = (
                                float(pg_c.transit_time[idxP])
                                if hasattr(pg_c, "transit_time")
                                else float(t0)
                            )
                            power_c = float(power_per_P_c[idxP])
                        # Duty cycle prior: 0.001 <= D/P <= 0.12
                        duty = D_c_best / P_c_best if P_c_best > 0 else np.nan
                        if not (
                            np.isfinite(duty)
                            and duty >= 0.001
                            and duty <= 0.12
                            and D_c_best <= _duration_upper_bound(P_c_best, best_floor)
                        ):
                            continue
                        # Prevent drift to shorter aliases unless power improves markedly
                        if (
                            P_c_best < float(best_period)
                            and power_c < 1.25 * base_power
                        ):
                            continue
                        n_epochs = int(max(1, np.floor((tmax - tmin) / P_c_best)))
                        cand_results.append(
                            {
                                "P": P_c_best,
                                "D": D_c_best,
                                "t0": t0_c_best,
                                "power": power_c,
                                "epochs": n_epochs,
                            }
                        )
                    if cand_results:
                        # Include current best for fair comparison
                        n_epochs_best = int(
                            max(1, np.floor((tmax - tmin) / float(best_period)))
                        )
                        duty_best = (
                            float(best_duration / best_period)
                            if best_period > 0
                            else np.nan
                        )
                        if (
                            np.isfinite(duty_best)
                            and duty_best >= 0.001
                            and duty_best <= 0.12
                            and best_duration
                            <= _duration_upper_bound(best_period, best_floor)
                        ):
                            cand_results.append(
                                {
                                    "P": float(best_period),
                                    "D": float(best_duration),
                                    "t0": float(t0),
                                    "power": base_power,
                                    "epochs": n_epochs_best,
                                }
                            )
                        # Sort by power, then prefer fewer epochs (longer periods) to avoid short aliases
                        cand_results.sort(
                            key=lambda r: (r["power"], -r["epochs"]), reverse=True
                        )
                        top = cand_results[0]
                        accept = False
                        if top["P"] > float(best_period):
                            # Accept longer period if power nearly as good
                            accept = top["power"] >= 0.98 * base_power
                        elif top["P"] < float(best_period):
                            # Accept shorter only with strong power gain
                            accept = top["power"] >= 1.25 * base_power
                        if accept and (top["P"] != float(best_period)):
                            print(
                                f"De-alias selected P={top['P']:.6f} (from {best_period:.6f}), epochs={top['epochs']}"
                            )
                            best_period = top["P"]
                            best_duration = top["D"]
                            t0 = top["t0"]
                except Exception as _e:
                    print(f"De-aliasing skipped due to error: {_e}")

                # Fine period refinement. The grids above quantize the period at
                # ~1e-4 day level; over a long baseline that residual error drifts
                # the transit in phase and smears the fold, which broadens every
                # downstream duration estimate (BLS box, profile, trapezoid).
                try:
                    span_days = float(tTime.max() - tTime.min())
                    if (
                        np.isfinite(best_period)
                        and best_period > 0
                        and np.isfinite(span_days)
                        and span_days > 2.0 * best_period
                    ):
                        # Step small enough that phase drift across the whole
                        # baseline stays below ~2% of the transit duration.
                        dP_target = 0.02 * best_duration * best_period / span_days
                        half_window = 1e-3 * best_period
                        n_fine = int(
                            np.clip(np.ceil(2.0 * half_window / dP_target), 128, 2000)
                        )
                        periods_fine = np.linspace(
                            best_period - half_window,
                            best_period + half_window,
                            n_fine,
                        )
                        dur_hi_f = _duration_upper_bound(best_period, best_floor)
                        d_lo_f = max(best_floor, 0.5 * best_duration)
                        d_hi_f = min(dur_hi_f, 1.5 * best_duration)
                        if not (np.isfinite(d_hi_f) and d_hi_f > d_lo_f):
                            d_lo_f, d_hi_f = best_floor, dur_hi_f
                        durations_fine = np.linspace(d_lo_f, d_hi_f, 64)
                        pg_f = bls.power(
                            periods_fine, durations_fine, oversample=oversample
                        )
                        if np.ndim(pg_f.power) == 1:
                            idx_f = int(np.nanargmax(pg_f.power))
                            P_f = float(pg_f.period[idx_f])
                            D_f = (
                                float(pg_f.duration[idx_f])
                                if hasattr(pg_f, "duration")
                                else float(durations_fine[0])
                            )
                            t0_f = (
                                float(pg_f.transit_time[idx_f])
                                if hasattr(pg_f, "transit_time")
                                else float(t0)
                            )
                        else:
                            power_per_P_f = np.nanmax(pg_f.power, axis=1)
                            idx_f = int(np.nanargmax(power_per_P_f))
                            P_f = float(pg_f.period[idx_f])
                            dur_idx_f = int(np.nanargmax(pg_f.power[idx_f, :]))
                            D_f = (
                                float(pg_f.duration[dur_idx_f])
                                if hasattr(pg_f, "duration")
                                else float(durations_fine[dur_idx_f])
                            )
                            t0_f = (
                                float(pg_f.transit_time[idx_f])
                                if hasattr(pg_f, "transit_time")
                                else float(t0)
                            )
                        if (
                            np.isfinite(P_f)
                            and P_f > 0
                            and np.isfinite(D_f)
                            and D_f > 0
                        ):
                            print(
                                f"Fine period refinement: P={P_f:.6f} "
                                f"(dP={P_f - best_period:+.2e}), T14={D_f:.4f}"
                            )
                            best_period = P_f
                            best_duration = D_f
                            t0 = t0_f
                except Exception as _e:
                    print(f"Fine period refinement skipped: {_e}")

                profile_duration = _folded_profile_duration(
                    tTime, flux, best_period, t0, best_floor
                )
                if (
                    np.isfinite(profile_duration)
                    and profile_duration > best_floor
                    and profile_duration < 0.75 * best_duration
                ):
                    capped_duration = min(best_duration, 1.35 * profile_duration)
                    print(
                        f"Folded profile suggests narrower duration: {profile_duration:.4f} days; "
                        f"capping to {capped_duration:.4f} days"
                    )
                    best_duration = capped_duration

                # Trapezoid refit at fixed period to refine T14 (duration)
                try:
                    t14_lo = max(best_floor, 0.5 * best_duration)
                    t14_hi = min(
                        _duration_upper_bound(best_period, best_floor),
                        1.6 * best_duration,
                    )
                    if np.isfinite(t14_lo) and np.isfinite(t14_hi) and t14_hi > t14_lo:
                        # Fold to phase time (days) around 0
                        phase = ((tTime - t0) / best_period + 0.5) % 1.0 - 0.5
                        x_days = phase * best_period
                        # Limit to neighborhood to improve robustness
                        win = max(2.0 * best_duration, 4.0 * best_floor)
                        sel = (
                            np.isfinite(x_days)
                            & np.isfinite(flux)
                            & (np.abs(x_days) <= win)
                        )
                        if np.sum(sel) >= 50:
                            xb = x_days[sel]
                            yb = flux[sel]
                            # Median bin to reduce noise
                            nb = min(200, max(60, int(np.sqrt(xb.size))))
                            bins = np.linspace(-win, win, nb + 1)
                            ymed, _, _ = binned_statistic(
                                xb, yb, statistic="median", bins=bins
                            )
                            xc = 0.5 * (bins[:-1] + bins[1:])
                            mask_med = np.isfinite(ymed) & np.isfinite(xc)
                            xc = xc[mask_med]
                            ymed = ymed[mask_med]


                            def _shape_trap(x, T14, T12):
                                half = 0.5 * T14
                                T12 = max(1e-6, min(T12, half))
                                flat_half = max(0.0, half - T12)
                                s = np.zeros_like(x)
                                # ingress
                                m = (x >= -half) & (x < -flat_half)
                                s[m] = (x[m] + half) / T12
                                # flat
                                m = (x >= -flat_half) & (x <= flat_half)
                                s[m] = 1.0
                                # egress
                                m = (x > flat_half) & (x <= half)
                                s[m] = (half - x[m]) / T12
                                # Outside remains 0
                                s = np.clip(s, 0.0, 1.0)
                                return s

                            best_loss = np.inf
                            best_t14 = best_duration
                            best_t14_idx = -1
                            best_x0 = 0.0
                            # Explore T14 and T12/T14 ratios (grazing→triangular up to 0.5)
                            t14_grid = np.linspace(t14_lo, t14_hi, 100)
                            ratio_grid = np.linspace(0.1, 0.45, 8)
                            # BLS quantizes transit_time on its internal phase bins
                            # (~duration/oversample), so the fold center can be off
                            # by up to ~0.005-0.01 days. A symmetric trapezoid forced
                            # onto an off-center fold widens by ~2x that offset, so
                            # fit a small center offset x0 alongside T14/T12.
                            x0_span = max(
                                0.1 * best_duration, 2.0 * (bins[1] - bins[0])
                            )
                            x0_grid = np.linspace(-x0_span, x0_span, 11)
                            one = np.ones_like(ymed)
                            for i_t14, T14 in enumerate(t14_grid):
                                for r in ratio_grid:
                                    T12 = r * T14
                                    for x0 in x0_grid:
                                        s = _shape_trap(xc - x0, T14, T12)
                                        A = np.column_stack([one, -s])
                                        try:
                                            coef, _, _, _ = np.linalg.lstsq(
                                                A, ymed, rcond=None
                                            )
                                            b0, d0 = float(coef[0]), float(coef[1])
                                            yhat = b0 - d0 * s
                                            resid = ymed - yhat
                                            loss = float(np.nanmean(resid * resid))
                                            if (
                                                np.isfinite(loss)
                                                and loss < best_loss
                                                and d0 > 0
                                            ):
                                                best_loss = loss
                                                best_t14 = float(T14)
                                                best_t14_idx = i_t14
                                                best_x0 = float(x0)
                                        except Exception:
                                            continue
                            # A best fit at the top edge usually means the model wants an
                            # even wider non-physical window; keep the BLS duration instead.
                            edge_idx = max(0, len(t14_grid) - 2)
                            if (
                                np.isfinite(best_t14)
                                and best_t14 > 0
                                and best_t14_idx < edge_idx
                            ):
                                best_duration = best_t14
                                if np.isfinite(best_x0) and best_x0 != 0.0:
                                    t0 = float(t0 + best_x0)
                                print(
                                    f"Trapezoid refit: T14={best_t14:.4f} days, "
                                    f"center offset={best_x0:+.4f} days"
                                )
                            elif best_t14_idx >= edge_idx:
                                print(
                                    "Trapezoid duration refit hit upper bound; keeping BLS duration"
                                )
                except Exception:
                    pass

                # Optional TLS refinement around the selected period
                if use_tls:
                    try:
                        from transitleastsquares import transitleastsquares

                        print("Running TLS refinement...")
                        per_lo_tls = best_period * 0.985
                        per_hi_tls = best_period * 1.015
                        y_tls = flux / (
                            np.nanmedian(flux)
                            if np.isfinite(np.nanmedian(flux))
                            else 1.0
                        )
                        model = transitleastsquares(tTime, y_tls)
                        tls_res = model.power(
                            period_min=per_lo_tls,
                            period_max=per_hi_tls,
                            oversampling_factor=5,
                            use_threads=8,
                            show_progress_bar=False,
                        )
                        if (
                            np.isfinite(tls_res.period)
                            and np.isfinite(tls_res.duration)
                            and tls_res.SDE > 0
                        ):
                            P_tls = float(tls_res.period)
                            D_tls = float(tls_res.duration)
                            duty_tls = D_tls / P_tls if P_tls > 0 else np.nan
                            if (
                                np.isfinite(duty_tls)
                                and duty_tls >= 0.0005
                                and duty_tls <= 0.12
                                and D_tls <= _duration_upper_bound(P_tls, best_floor)
                            ):
                                best_period = P_tls
                                best_duration = D_tls
                                t0 = float(tls_res.T0)
                                tls_used = True
                                print(
                                    f"TLS selected P={best_period:.6f}, T14={best_duration:.6f}"
                                )
                    except Exception as _e:
                        print(f"TLS refinement skipped: {_e}")

    mask_transit = bls.transit_mask(tTime, best_period, best_duration, t0)
    print(
        f"Transit mask created: {np.sum(mask_transit):,} transit points, {np.sum(~mask_transit):,} out-of-transit points"
    )

    flux_work = flux.copy()
    mask_use = ~mask_transit

    print("Starting iterative spline detrending...")
    for iteration in range(max_iter):
        bins = np.arange(tTime.min(), tTime.max() + bin_width, bin_width)
        digitized = np.digitize(tTime, bins)
        bin_means = []
        bin_times = []
        for i in range(1, len(bins)):
            sel = (digitized == i) & mask_use
            if np.any(sel):
                bin_means.append(np.median(flux_work[sel]))
                bin_times.append(np.median(tTime[sel]))
        bin_means = np.array(bin_means)
        bin_times = np.array(bin_times)

        good = np.isfinite(bin_means) & np.isfinite(bin_times)
        if good.sum() < 5:
            print(
                f"Spline iteration {iteration+1}: insufficient good points ({good.sum()}), stopping"
            )
            break
        spline = UnivariateSpline(bin_times[good], bin_means[good], s=spline_s, k=2)
        trend = spline(tTime)
        resid = flux_work - trend

        mad = np.nanmedian(np.abs(resid - np.nanmedian(resid)))
        thresh = -sigma * 1.4826 * mad
        mask_use = mask_use & (resid > thresh)

        print(
            f"Spline iteration {iteration+1}: {np.sum(mask_use):,} points used for fitting"
        )

    if np.sum(mask_use) < 3:
        # Global guard: ensure increasing x and adapt spline degree
        t_fit = np.asarray(tTime)
        f_fit = np.asarray(flux)
        ord_idx = np.argsort(t_fit)
        t_fit = t_fit[ord_idx]
        f_fit = f_fit[ord_idx]
        if t_fit.size:
            keep = np.hstack(([True], np.diff(t_fit) > 0))
            if not np.all(keep):
                t_fit = t_fit[keep]
                f_fit = f_fit[keep]
        k = 3
        if t_fit.size <= k:
            k = max(1, t_fit.size - 1)
        spline = UnivariateSpline(t_fit, f_fit, s=spline_s, k=k)
        trend = spline(tTime)
    else:
        # Final trend: fit spline on cadence-binned OOT medians to avoid huge-N spline fitting
        bins_final = np.arange(tTime.min(), tTime.max() + bin_width, bin_width)
        digitized_final = np.digitize(tTime, bins_final)
        bin_means_final = []
        bin_times_final = []
        for i in range(1, len(bins_final)):
            sel_bin = (digitized_final == i) & mask_use
            if np.any(sel_bin):
                bin_means_final.append(np.median(flux[sel_bin]))
                bin_times_final.append(np.median(tTime[sel_bin]))
        bin_means_final = np.array(bin_means_final)
        bin_times_final = np.array(bin_times_final)
        good_final = np.isfinite(bin_means_final) & np.isfinite(bin_times_final)
        if np.sum(good_final) >= 5:
            spline = UnivariateSpline(
                bin_times_final[good_final],
                bin_means_final[good_final],
                s=spline_s,
                k=2,
            )
            trend = spline(tTime)
        else:
            # Fallback: subsample OOT points if bins are insufficient
            t_fit = np.asarray(tTime[mask_use])
            f_fit = np.asarray(flux[mask_use])
            ord_idx = np.argsort(t_fit)
            t_fit = t_fit[ord_idx]
            f_fit = f_fit[ord_idx]
            if t_fit.size:
                keep = np.hstack(([True], np.diff(t_fit) > 0))
                if not np.all(keep):
                    t_fit = t_fit[keep]
                    f_fit = f_fit[keep]
            if t_fit.size > 50000:
                step = max(1, t_fit.size // 50000)
                t_fit = t_fit[::step]
                f_fit = f_fit[::step]
            k = 3
            if t_fit.size <= k:
                k = max(1, t_fit.size - 1)
            spline = UnivariateSpline(t_fit, f_fit, s=spline_s, k=k)
            trend = spline(tTime)
    flux_detr = flux / trend

    # Refit phase after detrending.  Earlier BLS passes see stitched/raw flux,
    # and their transit_time is quantized by the BLS phase grid.  SES/MES and
    # every downstream transit window use this final, duration-aware epoch.
    t0_before_final_refit = float(t0)
    t0 = refine_transit_epoch(
        tTime,
        flux_detr,
        best_period,
        best_duration,
        t0,
        oversample=max(50, 5 * int(oversample)),
    )
    phase_shift = epoch_phase_offset(t0, t0_before_final_refit, best_period)
    if np.isfinite(phase_shift):
        print(
            f"Final epoch refinement: dt0={phase_shift:+.6f} days "
            f"({phase_shift * 24.0 * 60.0:+.2f} min)"
        )
    mask_transit = bls.transit_mask(
        tTime, best_period, best_duration, t0
    )

    bls_info = {
        "best_period": float(best_period),
        "best_duration": float(best_duration),
        "t0": float(t0),
        "power": periodogram,
        "mask_transit": mask_transit,
        "time": tTime,
        "flux": flux,
        "search_candidates": search_candidates,
    }

    flux_detr_full = np.full(mask_valid.shape, np.nan, dtype=float)
    trend_full = np.full(mask_valid.shape, np.nan, dtype=float)
    flux_detr_full[mask_valid] = flux_detr
    trend_full[mask_valid] = trend

    bls_time = time.time() - bls_start
    print(f"BLS detrending completed in {bls_time:.2f} seconds")
    print(
        f"Final result: P={best_period:.4f} days, T14={best_duration:.4f} days, t0={t0:.2f}"
    )


    return flux_detr_full, trend_full, mask_transit, bls_info


# ========================================================================================================================

# def detrend_with_bls_mask(
#     time, flux,
#     min_period=0.5, max_period=None,
#     n_periods=2000, n_durations=50,
#     oversample=10,
#     bin_width=0.5, spline_s=0.001,
#     max_iter=5, sigma=3.0,
#     n_top_candidates=50
# ):
#     """
#     BLS detrend com busca global robusta e retorno de múltiplos períodos candidatos.
#     Mantém o comportamento original, mas retorna lista de períodos ordenados por power.
#     """

#     mask_valid = np.isfinite(time) & np.isfinite(flux)
#     time = np.asarray(time)[mask_valid]
#     flux = np.asarray(flux)[mask_valid]

#     if max_period is None:
#         max_period = (time.max() - time.min()) / 3.0

#     durations = np.linspace(0.01, 0.2, n_durations)
#     period_grid = np.logspace(np.log10(min_period), np.log10(max_period), n_periods)

#     bls = BoxLeastSquares(time, flux)
#     periodogram = bls.power(period_grid, durations, oversample=oversample)

#     # --- Seleção dos candidatos ---
#     if np.ndim(periodogram.power) == 1:
#         power = periodogram.power
#         t0s = periodogram.transit_time
#         durs = periodogram.duration
#     else:
#         power = np.nanmax(periodogram.power, axis=1)
#         dur_idx = np.nanargmax(periodogram.power, axis=1)
#         durs = periodogram.duration[dur_idx]
#         t0s = periodogram.transit_time[dur_idx]

#     sort_idx = np.argsort(power)[::-1]
#     top_idx = sort_idx[:n_top_candidates]
#     candidate_periods = periodogram.period[top_idx]
#     candidate_powers = power[top_idx]
#     candidate_durations = durs[top_idx]
#     candidate_t0s = t0s[top_idx]

#     # --- Melhor candidato (maior power) ---
#     idx_best = top_idx[0]
#     best_period = periodogram.period[idx_best]
#     best_duration = durs[idx_best]
#     t0 = t0s[idx_best]

#     print(f"[INFO] Melhor período: {best_period:.6f} d, duração {best_duration:.4f} d, t0={t0:.4f}")
#     print(f"[INFO] Top {n_top_candidates} candidatos ordenados por power extraídos.")

#     # --- Detrending usando o melhor ---
#     mask_transit = bls.transit_mask(time, best_period, best_duration, t0)
#     flux_work = flux.copy()
#     mask_use = ~mask_transit

#     for _ in range(max_iter):
#         bins = np.arange(time.min(), time.max() + bin_width, bin_width)
#         digitized = np.digitize(time, bins)
#         bin_means, bin_times = [], []
#         for i in range(1, len(bins)):
#             sel = (digitized == i) & mask_use
#             if np.any(sel):
#                 bin_means.append(np.median(flux_work[sel]))
#                 bin_times.append(np.median(time[sel]))
#         bin_means = np.array(bin_means)
#         bin_times = np.array(bin_times)
#         good = np.isfinite(bin_means) & np.isfinite(bin_times)
#         if good.sum() < 5:
#             break
#         spline = UnivariateSpline(bin_times[good], bin_means[good], s=spline_s)
#         trend = spline(time)
#         resid = flux_work - trend
#         mad = np.nanmedian(np.abs(resid - np.nanmedian(resid)))
#         thresh = -sigma * 1.4826 * mad
#         mask_use = mask_use & (resid > thresh)

#     if np.sum(mask_use) < 3:
#         spline = UnivariateSpline(time, flux, s=spline_s)
#     else:
#         spline = UnivariateSpline(time[mask_use], flux[mask_use], s=spline_s)

#     trend = spline(time)
#     flux_detr = flux / trend

#     # --- Empacotamento dos resultados ---
#     bls_info = {
#         "best_period": float(best_period),
#         "best_duration": float(best_duration),
#         "t0": float(t0),
#         "periodogram": periodogram,
#         "mask_transit": mask_transit,
#         "time": time,
#         "flux": flux,
#         "candidate_periods": candidate_periods,
#         "candidate_powers": candidate_powers,
#         "candidate_durations": candidate_durations,
#         "candidate_t0s": candidate_t0s
#     }

#     flux_detr_full = np.full(mask_valid.shape, np.nan, dtype=float)
#     trend_full = np.full(mask_valid.shape, np.nan, dtype=float)
#     flux_detr_full[mask_valid] = flux_detr
#     trend_full[mask_valid] = trend

#     return flux_detr_full, trend_full, mask_transit, bls_info

# ========================================================================================================================

# def detrend_with_bls_mask(time, flux,
#                           min_period=0.5, max_period=None,
#                           n_periods=2000, n_durations=50,
#                           oversample=10,
#                           bin_width=0.5, spline_s=0.001,
#                           max_iter=5, sigma=3.0):
#     mask_valid = np.isfinite(time) & np.isfinite(flux)
#     time = np.asarray(time)[mask_valid]
#     flux = np.asarray(flux)[mask_valid]
#     print("Detrend version 1")
#     if max_period is None:
#         max_period = (time.max() - time.min()) / 3.0

#     durations = np.linspace(0.01, 0.2, n_durations)
#     period_grid = np.logspace(np.log10(min_period), np.log10(max_period), n_periods)

#     bls = BoxLeastSquares(time, flux)
#     periodogram = bls.power(period_grid, durations, oversample=oversample)

#     if np.ndim(periodogram.power) == 1:
#         idx_best = np.nanargmax(periodogram.power)
#         best_period = periodogram.period[idx_best]
#         best_duration = periodogram.duration[idx_best] if hasattr(periodogram, "duration") else durations[0]
#         t0 = periodogram.transit_time[idx_best] if hasattr(periodogram, "transit_time") else time[0]
#     else:
#         power_per_period = np.nanmax(periodogram.power, axis=1)
#         idx_best = int(np.nanargmax(power_per_period))
#         best_period = periodogram.period[idx_best]
#         dur_idx = int(np.nanargmax(periodogram.power[idx_best, :]))
#         best_duration = periodogram.duration[dur_idx] if hasattr(periodogram, "duration") else durations[dur_idx]
#         t0 = periodogram.transit_time[idx_best] if hasattr(periodogram, "transit_time") else time[0]

#     mask_transit = bls.transit_mask(time, best_period, best_duration, t0)

#     flux_work = flux.copy()
#     mask_use = ~mask_transit

#     for _ in range(max_iter):
#         bins = np.arange(time.min(), time.max() + bin_width, bin_width)
#         digitized = np.digitize(time, bins)
#         bin_means = []
#         bin_times = []
#         for i in range(1, len(bins)):
#             sel = (digitized == i) & mask_use
#             if np.any(sel):
#                 bin_means.append(np.median(flux_work[sel]))
#                 bin_times.append(np.median(time[sel]))
#         bin_means = np.array(bin_means)
#         bin_times = np.array(bin_times)

#         good = np.isfinite(bin_means) & np.isfinite(bin_times)
#         if good.sum() < 5:
#             break
#         spline = UnivariateSpline(bin_times[good], bin_means[good], s=spline_s)
#         trend = spline(time)
#         resid = flux_work - trend

#         mad = np.nanmedian(np.abs(resid - np.nanmedian(resid)))
#         thresh = -sigma * 1.4826 * mad
#         mask_use = mask_use & (resid > thresh)

#     if np.sum(mask_use) < 3:
#         spline = UnivariateSpline(time, flux, s=spline_s)
#     else:
#         spline = UnivariateSpline(time[mask_use], flux[mask_use], s=spline_s)
#     trend = spline(time)
#     flux_detr = flux / trend

#     bls_info = {
#         "best_period": float(best_period),
#         "best_duration": float(best_duration),
#         "t0": float(t0),
#         "power": periodogram,
#         "mask_transit": mask_transit,
#         "time": time,
#         "flux": flux
#     }

#     flux_detr_full = np.full(mask_valid.shape, np.nan, dtype=float)
#     trend_full = np.full(mask_valid.shape, np.nan, dtype=float)
#     flux_detr_full[mask_valid] = flux_detr
#     trend_full[mask_valid] = trend

#     return flux_detr_full, trend_full, mask_transit, bls_info

# ========================================================================================================================

# def detrend_with_bls_mask(time, flux,
#                           min_period=0.5, max_period=None,
#                           n_periods=2000, n_durations=50,
#                           oversample=10,
#                           bin_width=0.5, spline_s=0.001,
#                           max_iter=5, sigma=3.0):
#     # Filtra os valores válidos
#     mask_valid = np.isfinite(time) & np.isfinite(flux)
#     time = np.asarray(time)[mask_valid]
#     flux = np.asarray(flux)[mask_valid]

#     # Ordena o tempo e reorganiza o fluxo correspondente
#     sorted_indices = np.argsort(time)
#     time = time[sorted_indices]
#     flux = flux[sorted_indices]

#     # Calcula o período máximo se não fornecido
#     if max_period is None:
#         max_period = (time.max() - time.min()) / 3.0

#     # Define a grade de períodos e durações
#     durations = np.linspace(0.01, 0.2, n_durations)
#     period_grid = np.logspace(np.log10(min_period), np.log10(max_period), n_periods)

#     # Realiza a análise de Box Least Squares
#     bls = BoxLeastSquares(time, flux)
#     periodogram = bls.power(period_grid, durations, oversample=oversample)

#     # Identifica o melhor período, duração e tempo de início
#     if np.ndim(periodogram.power) == 1:
#         idx_best = np.nanargmax(periodogram.power)
#         best_period = periodogram.period[idx_best]
#         best_duration = periodogram.duration[idx_best] if hasattr(periodogram, "duration") else durations[0]
#         t0 = periodogram.transit_time[idx_best] if hasattr(periodogram, "transit_time") else time[0]
#     else:
#         power_per_period = np.nanmax(periodogram.power, axis=1)
#         idx_best = int(np.nanargmax(power_per_period))
#         best_period = periodogram.period[idx_best]
#         dur_idx = int(np.nanargmax(periodogram.power[idx_best, :]))
#         best_duration = periodogram.duration[dur_idx] if hasattr(periodogram, "duration") else durations[dur_idx]
#         t0 = periodogram.transit_time[idx_best] if hasattr(periodogram, "transit_time") else time[0]

#     # Cria a máscara de trânsito
#     mask_transit = bls.transit_mask(time, best_period, best_duration, t0)

#     # Inicializa o trabalho com o fluxo
#     flux_work = flux.copy()
#     mask_use = ~mask_transit

#     # Itera para refinar a máscara de outliers
#     for _ in range(max_iter):
#         bins = np.arange(time.min(), time.max() + bin_width, bin_width)
#         digitized = np.digitize(time, bins)
#         bin_means = []
#         bin_times = []
#         for i in range(1, len(bins)):
#             sel = (digitized == i) & mask_use
#             if np.any(sel):
#                 bin_means.append(np.median(flux_work[sel]))
#                 bin_times.append(np.median(time[sel]))
#         bin_means = np.array(bin_means)
#         bin_times = np.array(bin_times)

#         good = np.isfinite(bin_means) & np.isfinite(bin_times)
#         if good.sum() < 5:
#             break
#         spline = UnivariateSpline(bin_times[good], bin_means[good], s=spline_s, k=1)
#         trend = spline(time)
#         resid = flux_work - trend

#         mad = np.nanmedian(np.abs(resid - np.nanmedian(resid)))
#         thresh = -sigma * 1.4826 * mad
#         mask_use = mask_use & (resid > thresh)

#     # Ajusta o spline final
#     if np.sum(mask_use) < 3:
#         spline = UnivariateSpline(time, flux, s=spline_s)
#     else:
#         spline = UnivariateSpline(time[mask_use], flux[mask_use], s=spline_s)
#     trend = spline(time)
#     flux_detr = flux / trend

#     # Armazena as informações do BLS
#     bls_info = {
#         "best_period": float(best_period),
#         "best_duration": float(best_duration),
#         "t0": float(t0),
#         "power": periodogram,
#         "mask_transit": mask_transit,
#         "time": time,
#         "flux": flux
#     }

#     # Preenche os resultados finais com NaN para os valores inválidos
#     flux_detr_full = np.full(mask_valid.shape, np.nan, dtype=float)
#     trend_full = np.full(mask_valid.shape, np.nan, dtype=float)
#     flux_detr_full[mask_valid] = flux_detr
#     trend_full[mask_valid] = trend

#     return flux_detr_full, trend_full, mask_transit, bls_info

# ========================================================================================================================

# def detrend_with_bls_mask(time, flux,
#                           min_period=0.5, max_period=None,
#                           n_periods=2000, n_durations=50,
#                           oversample=10,
#                           window_frac=0.02, polyorder=2,
#                           max_iter=5, sigma=3.0,
#                           verbose=False):
#     # Filtra os valores válidos
#     mask_valid = np.isfinite(time) & np.isfinite(flux)
#     time = np.asarray(time)[mask_valid]
#     flux = np.asarray(flux)[mask_valid]

#     # Ordena
#     sorted_indices = np.argsort(time)
#     time = time[sorted_indices]
#     flux = flux[sorted_indices]

#     timespan = time.max() - time.min()
#     if timespan <= 0:
#         raise ValueError("time span inválido")

#     # Ajuste seguro de max_period: metade do baseline por padrão (mais permissivo que /3)
#     if max_period is None:
#         max_period = max(min_period * 2.0, timespan / 2.0)

#     # grade de durações (em unidades de dias) e períodos (dias)
#     durations = np.linspace(0.01, 0.2, n_durations)
#     period_grid = np.logspace(np.log10(min_period), np.log10(max_period), n_periods)

#     # Executa BLS (USAMOS OS DADOS NÃO-DESLINED)
#     bls = BoxLeastSquares(time, flux)
#     periodogram = bls.power(period_grid, durations, oversample=oversample)
#     # show_top_bls_peaks(periodogram, n_peaks=50)


#     # Analisa a forma da power array e encontra o melhor período/duração
#     if np.ndim(periodogram.power) == 1:
#         idx_best = int(np.nanargmax(periodogram.power))
#         best_period = float(periodogram.period[idx_best])
#         best_duration = float(periodogram.duration[idx_best]) if hasattr(periodogram, "duration") else float(durations[0])
#         # transit_time pode não existir como esperado; estimamos t0 abaixo
#         t0_est = None
#     else:
#         power_per_period = np.nanmax(periodogram.power, axis=1)
#         idx_best = int(np.nanargmax(power_per_period))
#         best_period = float(periodogram.period[idx_best])
#         dur_idx = int(np.nanargmax(periodogram.power[idx_best, :]))
#         # extrai duração; se atributo duration existir, use-o; senão use duracoes da grade
#         try:
#             best_duration = float(periodogram.duration[dur_idx])
#         except Exception:
#             best_duration = float(durations[dur_idx])
#         t0_est = None

#     # Opcional: mostrar os 5 picos mais altos (diagnóstico)
#     if verbose:
#         # para 2D power, colapsa por periodo
#         if np.ndim(periodogram.power) == 2:
#             power_c = np.nanmax(periodogram.power, axis=1)
#         else:
#             power_c = periodogram.power
#         top_idx = np.argsort(power_c)[-5:][::-1]
#         print("time span (days):", timespan)
#         print("period grid min/max: %.3f / %.3f" % (period_grid.min(), period_grid.max()))
#         print("top BLS periods (days) and power:")
#         for ii in top_idx:
#             print(f"  {periodogram.period[ii]:.6f} -> power {power_c[ii]:.6g}")
#         print("chosen best_period:", best_period, "best_duration:", best_duration)

#     # Estima t0 por folding simples (mais robusto que confiar em transit_time)
#     try:
#         phase = ((time - time[0]) / best_period) % 1.0
#         # centraliza em -0.5..0.5
#         phase = (phase + 0.5) % 1.0 - 0.5
#         nbins = max(100, int(len(time) / 1000))  # bins adaptativos (mín 100)
#         bins = np.linspace(-0.5, 0.5, nbins + 1)
#         bin_median, _, _ = binned_statistic(phase, flux, statistic="median", bins=bins)
#         min_bin = np.nanargmin(bin_median)
#         bin_centers = 0.5 * (bins[:-1] + bins[1:])
#         phase_min = bin_centers[min_bin]
#         # transforma de volta para tempo: queremos o t0 mais próximo do início
#         t0 = float(time[0] + ((phase_min + 0.5) % 1.0) * best_period)
#     except Exception:
#         t0 = float(time[0])

#     # Cria a máscara de trânsito com a melhor solução
#     mask_transit = bls.transit_mask(time, best_period, best_duration, t0)

#     # Preparação para detrending com SavGol
#     flux_work = flux.copy()
#     mask_use = ~mask_transit

#     # janela adaptativa: fração do n_pts (garante impar e <= n_pts)
#     win_len = int(len(time) * window_frac)
#     if win_len % 2 == 0:
#         win_len += 1
#     win_len = max(win_len, polyorder + 1)
#     if win_len > len(time):
#         win_len = len(time) if (len(time) % 2 == 1) else (len(time) - 1)

#     if verbose:
#         print("savgol window length:", win_len, "polyorder:", polyorder)

#     # Itera para refinar máscara de outliers (aplica SavGol apenas aos pontos selecionados)
#     for _ in range(max_iter):
#         # se mask_use tiver poucos pontos, computa trend sobre todo o conjunto (fallback)
#         if np.sum(mask_use) <= win_len:
#             trend_full = savgol_filter(flux, window_length=win_len, polyorder=min(polyorder, max(1, win_len-1)), mode="interp")
#             resid = flux_work - trend_full
#         else:
#             # Para evitar chamar savgol com um array muito pequeno, construímos um trend interpolado:
#             t_sel = time[mask_use]
#             f_sel = flux_work[mask_use]
#             # se f_sel muito pequeno em tamanho, use fallback
#             if len(f_sel) <= win_len:
#                 trend_sel = savgol_filter(flux, window_length=win_len, polyorder=min(polyorder, max(1, win_len-1)), mode="interp")
#                 resid = flux_work - trend_sel
#             else:
#                 trend_sel = savgol_filter(f_sel, window_length=win_len if win_len <= len(f_sel) else (len(f_sel)-1 if (len(f_sel)-1)%2==1 else len(f_sel)), polyorder=min(polyorder, max(1, win_len-1)), mode="interp")
#                 # interpola trend_sel nos tempos originais
#                 trend_full = np.interp(time, t_sel, trend_sel, left=np.nan, right=np.nan)
#                 # para bins sem valor, preenche com trend do filtro aplicado a todo o fluxo
#                 mask_nan = ~np.isfinite(trend_full)
#                 if np.any(mask_nan):
#                     fallback_trend = savgol_filter(flux, window_length=win_len, polyorder=min(polyorder, max(1, win_len-1)), mode="interp")
#                     trend_full[mask_nan] = fallback_trend[mask_nan]
#                 resid = flux_work - trend_full

#         mad = np.nanmedian(np.abs(resid - np.nanmedian(resid)))
#         thresh = -sigma * 1.4826 * mad
#         # atualiza mask_use: remove pontos que estão abaixo do threshold (prováveis trânsitos/outliers negativos)
#         mask_use = mask_use & (resid > thresh)

#     # Filtro final aplicado em todo o fluxo
#     final_trend = savgol_filter(flux, window_length=win_len, polyorder=min(polyorder, max(1, win_len-1)), mode="interp")
#     flux_detr = flux / final_trend

#     bls_info = {
#         "best_period": float(best_period),
#         "best_duration": float(best_duration),
#         "t0": float(t0),
#         "power": periodogram,
#         "mask_transit": mask_transit,
#         "time": time,
#         "flux": flux
#     }

#     # Reconstrói vetores com NaNs na posição original
#     flux_detr_full = np.full(mask_valid.shape, np.nan, dtype=float)
#     trend_full = np.full(mask_valid.shape, np.nan, dtype=float)
#     flux_detr_full[mask_valid] = flux_detr
#     trend_full[mask_valid] = final_trend

#     return flux_detr_full, trend_full, mask_transit, bls_info

# ========================================================================================================================

# def detrend_with_tls(time, flux, min_period=0.5, max_period=None, n_periods=2000,
#                       n_durations=50, oversample=10, window_frac=0.02, polyorder=2,
#                       max_iter=5, sigma=3.0, verbose=False):
#     # Filtra os valores válidos
#     mask_valid = np.isfinite(time) & np.isfinite(flux)
#     time = np.asarray(time)[mask_valid]
#     flux = np.asarray(flux)[mask_valid]

#     # Ordena os dados
#     sorted_indices = np.argsort(time)
#     time = time[sorted_indices]
#     flux = flux[sorted_indices]

#     # Determina o intervalo de tempo
#     timespan = time.max() - time.min()
#     if timespan <= 0:
#         raise ValueError("Intervalo de tempo inválido")

#     # Ajuste seguro de max_period
#     if max_period is None:
#         max_period = max(min_period * 2.0, timespan / 2.0)

#     # Executa TLS
#     flux_flat = pre_detrend_flux(time, flux, window_length_days=5, polyorder=3)
#     model = transitleastsquares(time, flux_flat)
#     results = model.power(period_max=max_period)
#     # Exibe os resultados principais
#     print(f"Período: {results.period:.5f} dias")
#     print(f"Profundidade do trânsito: {results.depth:.5f}")
#     print(f"Melhor duração: {results.duration:.5f} dias")
#     print(f"Eficiência de detecção do sinal (SDE): {results.SDE:.2f}")

#     if verbose:
#         # Exibe os 5 principais picos
#         top_idx = np.argsort(results.power)[-5:][::-1]
#         print("Top 5 picos:")
#         for idx in top_idx:
#             print(f"Período: {results.periods[idx]:.5f} dias, SDE: {results.power[idx]:.2f}")

#     # Estima o tempo de início do trânsito (t0)
#     phase = ((time - time[0]) / results.period) % 1.0
#     phase = (phase + 0.5) % 1.0 - 0.5
#     nbins = max(100, int(len(time) / 1000))
#     bins = np.linspace(-0.5, 0.5, nbins + 1)
#     bin_median, _, _ = binned_statistic(phase, flux, statistic="median", bins=bins)
#     min_bin = np.nanargmin(bin_median)
#     bin_centers = 0.5 * (bins[:-1] + bins[1:])
#     phase_min = bin_centers[min_bin]
#     t0 = float(time[0] + ((phase_min + 0.5) % 1.0) * results.period)

#     # Cria a máscara de trânsito
#     # mask_transit = model.transit_mask(
#     #     time, results.period, results.duration, t0
#     # )

#     # Detrend com Savitzky-Golay
#     flux_detrended = flux / savgol_filter(flux, window_length=11, polyorder=3)

#     return flux_detrended, {
#         "period": results.period,
#         "duration": results.duration,
#         "depth": results.depth,
#         "t0": t0,
#         "SDE": results.SDE
#     }

# def pre_detrend_flux(time, flux, window_length_days=5.0, polyorder=3):
#     """
#     Pré-detrending do fluxo usando Savitzky-Golay.

#     time: array de tempos em dias
#     flux: array de fluxos normalizados
#     window_length_days: janela do filtro em dias
#     polyorder: ordem do polinômio para ajuste

#     Retorna:
#         flux_detrended: flux detrended
#     """
#     time = np.asarray(time)
#     flux = np.asarray(flux)

#     # calcula o tamanho da janela em número de pontos
#     diffs = np.diff(time)
#     median_dt = np.median(diffs[np.isfinite(diffs)])
#     window_length_pts = int(round(window_length_days / median_dt))

#     # o window_length do Savitzky-Golay precisa ser ímpar
#     if window_length_pts % 2 == 0:
#         window_length_pts += 1

#     # garante tamanho mínimo de 5 pontos
#     window_length_pts = max(window_length_pts, 5)

#     flux_trend = savgol_filter(flux, window_length=window_length_pts, polyorder=polyorder)
#     flux_detrended = flux / flux_trend  # ou flux - flux_trend, dependendo da escala

#     return flux_detrended
