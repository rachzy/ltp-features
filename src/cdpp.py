import numpy as np

# 4 - Compute CDPP


def _robust_sigma(values):
    """Return a MAD-based standard deviation with a finite-sample fallback."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return np.nan
    center = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - center))
    if np.isfinite(mad) and mad > 0:
        return float(1.4826 * mad)
    sigma = np.nanstd(values, ddof=1)
    return float(sigma) if np.isfinite(sigma) and sigma > 0 else np.nan


def _duration_window_means(
    time,
    residual_flux,
    exclude_mask,
    cadence_hours,
    duration_hours,
    min_coverage,
):
    """Measure box-template amplitudes in valid, transit-free time windows."""
    duration_days = float(duration_hours) / 24.0
    cadence_days = float(cadence_hours) / 24.0
    if duration_days <= 0 or cadence_days <= 0:
        return np.array([], dtype=float)

    order = np.argsort(time)
    time = np.asarray(time, dtype=float)[order]
    residual_flux = np.asarray(residual_flux, dtype=float)[order]
    exclude_mask = np.asarray(exclude_mask, dtype=bool)[order]

    finite_time = np.isfinite(time)
    time = time[finite_time]
    residual_flux = residual_flux[finite_time]
    exclude_mask = exclude_mask[finite_time]
    if time.size < 2 or time[-1] - time[0] < duration_days:
        return np.array([], dtype=float)

    expected_points = max(1.0, duration_hours / cadence_hours)
    minimum_points = max(2, int(np.ceil(min_coverage * expected_points)))
    # Half-overlapping windows avoid depending on an arbitrary starting phase.
    step_days = max(cadence_days, duration_days / 2.0)
    last_start = time[-1] - duration_days
    starts = np.arange(time[0], last_start + 0.5 * step_days, step_days)

    window_means = []
    for start in starts:
        left = int(np.searchsorted(time, start, side="left"))
        right = int(np.searchsorted(time, start + duration_days, side="left"))
        if right - left < minimum_points:
            continue
        # Discard the entire window when it overlaps a known transit. Merely
        # removing transit cadences would change the effective template length.
        if np.any(exclude_mask[left:right]):
            continue
        values = residual_flux[left:right]
        values = values[np.isfinite(values)]
        if values.size < minimum_points:
            continue
        window_means.append(float(np.mean(values)))

    return np.asarray(window_means, dtype=float)


def calculate_cdpp(
    flux,
    cadence_hours,
    durations=(3.0, 6.0, 12.0),
    time=None,
    exclude_mask=None,
    min_coverage=0.8,
    min_windows=5,
):
    """Estimate duration-matched photometric precision in ppm.

    The scatter of transit-duration box averages is an empirical uncertainty
    for a box-shaped event. Unlike a per-cadence RMS divided by ``sqrt(N)``, it
    retains the effect of time-correlated noise within each duration. Windows
    touching ``exclude_mask`` are rejected so detected transits do not inflate
    their own noise estimate.

    This is a robust time-domain approximation to CDPP, not a reproduction of
    Kepler TPS's adaptive wavelet whitening.
    """
    flux = np.asarray(flux, dtype=float)
    if flux.ndim != 1:
        raise ValueError("flux must be a one-dimensional array")
    if not np.isfinite(cadence_hours) or cadence_hours <= 0:
        return {f"cdpp_{int(dur)}h": np.nan for dur in durations}
    if not 0 < min_coverage <= 1:
        raise ValueError("min_coverage must be in the interval (0, 1]")

    if time is None:
        time = np.arange(flux.size, dtype=float) * float(cadence_hours) / 24.0
    else:
        time = np.asarray(time, dtype=float)
        if time.shape != flux.shape:
            raise ValueError("time and flux must have the same shape")

    if exclude_mask is None:
        exclude_mask = np.zeros(flux.shape, dtype=bool)
    else:
        exclude_mask = np.asarray(exclude_mask, dtype=bool)
        if exclude_mask.shape != flux.shape:
            raise ValueError("exclude_mask and flux must have the same shape")

    baseline_mask = np.isfinite(flux) & np.isfinite(time) & ~exclude_mask
    if np.sum(baseline_mask) < 2:
        return {f"cdpp_{int(dur)}h": np.nan for dur in durations}
    median = np.nanmedian(flux[baseline_mask])
    if not np.isfinite(median) or median == 0:
        return {f"cdpp_{int(dur)}h": np.nan for dur in durations}
    residual_flux = flux / median - 1.0

    cdpp_results = {}
    for duration in durations:
        duration = float(duration)
        means = _duration_window_means(
            time,
            residual_flux,
            exclude_mask,
            cadence_hours=float(cadence_hours),
            duration_hours=duration,
            min_coverage=float(min_coverage),
        )
        cdpp = _robust_sigma(means) if means.size >= min_windows else np.nan
        cdpp_results[f"cdpp_{int(duration)}h"] = (
            float(cdpp * 1e6) if np.isfinite(cdpp) else np.nan
        )

    return cdpp_results
