import numpy as np

# 4 - Compute duration-matched noise and CDPP


def _empty_statistics():
    empty_float = np.array([], dtype=float)
    empty_bool = np.array([], dtype=bool)
    return {
        "time": empty_float,
        "depth": empty_float,
        "raw_depth": empty_float,
        "uncertainty": empty_float,
        "N": empty_float,
        "D": empty_float,
        "SES": empty_float,
        "coverage": empty_float,
        "valid": empty_bool,
        "noise_eligible": empty_bool,
        "cadence_days": np.nan,
        "duration_days": np.nan,
    }


def _robust_location_scale(values):
    """Return the median and Gaussian-equivalent MAD, without RMS fallback."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return np.nan, np.nan
    location = float(np.median(values))
    mad = float(np.median(np.abs(values - location)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 0:
        scale = np.nan
    return location, scale


def _sample_anchor_indices(time, valid, segment_id, step_days):
    """Select valid statistic samples separated by at least ``step_days``."""
    selected = []
    for segment in np.unique(segment_id):
        candidates = np.flatnonzero(valid & (segment_id == segment))
        if candidates.size == 0:
            continue
        last_time = -np.inf
        for index in candidates:
            if time[index] - last_time >= step_days * (1.0 - 1e-12):
                selected.append(int(index))
                last_time = time[index]
    return np.asarray(selected, dtype=int)


def _independent_span_count(time, segment_id, duration_days):
    """Approximate the number of independent duration-sized noise samples."""
    count = 0.0
    for segment in np.unique(segment_id):
        values = time[segment_id == segment]
        if values.size:
            count += max(1.0, (values[-1] - values[0]) / duration_days)
    return count


def _local_anchor_noise(
    time,
    raw_depth,
    segment_id,
    anchor_indices,
    duration_days,
    noise_window_durations,
    min_local_anchors,
    min_independent_samples,
):
    """Estimate robust local depth location and scale at sparse anchors."""
    if anchor_indices.size == 0:
        return np.array([]), np.array([])

    anchor_time = time[anchor_indices]
    anchor_depth = raw_depth[anchor_indices]
    anchor_segment = segment_id[anchor_indices]
    global_location, global_scale = _robust_location_scale(anchor_depth)
    independent_count = _independent_span_count(
        anchor_time, anchor_segment, duration_days
    )
    if independent_count < min_independent_samples:
        global_location = np.nan
        global_scale = np.nan

    locations = np.full(anchor_indices.size, np.nan, dtype=float)
    scales = np.full(anchor_indices.size, np.nan, dtype=float)
    half_window = 0.5 * float(noise_window_durations) * duration_days

    for segment in np.unique(anchor_segment):
        positions = np.flatnonzero(anchor_segment == segment)
        segment_times = anchor_time[positions]
        segment_depths = anchor_depth[positions]
        for local_position, output_position in enumerate(positions):
            center = segment_times[local_position]
            left = int(
                np.searchsorted(segment_times, center - half_window, side="left")
            )
            right = int(
                np.searchsorted(segment_times, center + half_window, side="right")
            )
            if right - left >= min_local_anchors:
                location, scale = _robust_location_scale(segment_depths[left:right])
            else:
                location, scale = global_location, global_scale
            if not np.isfinite(scale) or scale <= 0:
                location, scale = global_location, global_scale
            locations[output_position] = location
            scales[output_position] = scale

    return locations, scales


def duration_matched_statistics(
    time,
    flux,
    duration_hours,
    cadence_hours=None,
    exclude_mask=None,
    gap_mask=None,
    min_coverage=0.8,
    gap_cadences=5.0,
    noise_window_durations=30.0,
    min_local_anchors=20,
    min_independent_samples=5,
):
    """Build a local, duration-matched transit detection statistic.

    A positive ``depth`` represents a flux deficit. The box-template amplitude
    is evaluated at every observed cadence. Its local median and MAD are
    estimated from half-duration-spaced anchors, excluding known transit
    windows from the noise sample only. The returned ``N`` and ``D`` components
    can be combined coherently across predicted transit events.

    ``exclude_mask`` marks the candidate's own transit windows: those cadences
    stay foldable but are kept out of the noise sample. ``gap_mask`` marks
    cadences to treat as unobserved (Kepler TPS "gapping" of already-detected
    transits) - any box touching one is invalidated, so it contributes to
    neither the noise model nor the fold. Passing ``gap_mask`` is strongly
    preferred over deleting those cadences from ``time``/``flux``: deletion
    leaves sub-``gap_cadences`` holes that fragment ``segment_id``, truncating
    the local noise windows and biasing the MAD low. Note ``gap_cadences``
    (segment-splitting threshold) is unrelated to ``gap_mask``.

    This is a time-domain approximation to Kepler TPS. It intentionally does
    not reproduce TPS's adaptive wavelet whitening.
    """
    result = _empty_statistics()
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    if time.ndim != 1 or flux.ndim != 1:
        raise ValueError("time and flux must be one-dimensional arrays")
    if time.shape != flux.shape:
        raise ValueError("time and flux must have the same shape")
    if not np.isfinite(duration_hours) or duration_hours <= 0:
        return result
    if not 0 < min_coverage <= 1:
        raise ValueError("min_coverage must be in the interval (0, 1]")
    if not np.isfinite(gap_cadences) or gap_cadences <= 1:
        raise ValueError("gap_cadences must be greater than one")

    if exclude_mask is None:
        exclude_mask = np.zeros(time.shape, dtype=bool)
    else:
        exclude_mask = np.asarray(exclude_mask, dtype=bool)
        if exclude_mask.shape != time.shape:
            raise ValueError("exclude_mask, time, and flux must have the same shape")

    if gap_mask is None:
        gap_mask = np.zeros(time.shape, dtype=bool)
    else:
        gap_mask = np.asarray(gap_mask, dtype=bool)
        if gap_mask.shape != time.shape:
            raise ValueError("gap_mask, time, and flux must have the same shape")

    finite = np.isfinite(time) & np.isfinite(flux)
    if np.sum(finite) < 2:
        return result
    order = np.argsort(time[finite], kind="stable")
    time = time[finite][order]
    flux = flux[finite][order]
    exclude_mask = exclude_mask[finite][order]
    gap_mask = gap_mask[finite][order]

    positive_steps = np.diff(time)
    positive_steps = positive_steps[np.isfinite(positive_steps) & (positive_steps > 0)]
    measured_cadence = (
        float(np.median(positive_steps)) if positive_steps.size else np.nan
    )
    supplied_cadence = (
        float(cadence_hours) / 24.0
        if cadence_hours is not None
        and np.isfinite(cadence_hours)
        and cadence_hours > 0
        else np.nan
    )
    cadence_days = (
        supplied_cadence if np.isfinite(supplied_cadence) else measured_cadence
    )
    duration_days = float(duration_hours) / 24.0
    if not np.isfinite(cadence_days) or cadence_days <= 0:
        return result

    result["cadence_days"] = cadence_days
    result["duration_days"] = duration_days

    segment_id = np.zeros(time.size, dtype=int)
    if time.size > 1:
        segment_id[1:] = np.cumsum(np.diff(time) > gap_cadences * cadence_days)

    baseline_values = flux[~exclude_mask & ~gap_mask]
    if baseline_values.size < 2:
        return result
    baseline = float(np.median(baseline_values))
    if not np.isfinite(baseline) or baseline == 0:
        return result
    deficit = 1.0 - flux / baseline

    half_duration = 0.5 * duration_days
    left = np.searchsorted(time, time - half_duration, side="left")
    right = np.searchsorted(time, time + half_duration, side="right")
    counts = right - left
    expected_points = max(1.0, duration_days / cadence_days)
    minimum_points = max(2, int(np.ceil(min_coverage * expected_points)))

    prefix_deficit = np.concatenate(([0.0], np.cumsum(deficit)))
    prefix_excluded = np.concatenate(
        ([0], np.cumsum(exclude_mask.astype(np.int64)))
    )
    prefix_gapped = np.concatenate(([0], np.cumsum(gap_mask.astype(np.int64))))
    raw_depth = (prefix_deficit[right] - prefix_deficit[left]) / np.maximum(counts, 1)
    overlaps_excluded = (prefix_excluded[right] - prefix_excluded[left]) > 0
    # Any overlap with a gapped cadence invalidates the box outright. Gapped
    # windows hold another signal's transits, so a partially overlapping box
    # would carry that depth into this candidate's fold.
    overlaps_gap = (prefix_gapped[right] - prefix_gapped[left]) > 0

    right_index = np.maximum(right - 1, 0)
    same_segment = segment_id[left] == segment_id[right_index]
    box_valid = (counts >= minimum_points) & same_segment & ~overlaps_gap
    raw_depth[~box_valid] = np.nan
    coverage = counts.astype(float) / expected_points
    coverage[~box_valid] = np.nan

    noise_candidate = box_valid & ~overlaps_excluded & np.isfinite(raw_depth)
    anchor_indices = _sample_anchor_indices(
        time,
        noise_candidate,
        segment_id,
        step_days=max(cadence_days, half_duration),
    )
    anchor_locations, anchor_scales = _local_anchor_noise(
        time,
        raw_depth,
        segment_id,
        anchor_indices,
        duration_days,
        noise_window_durations=float(noise_window_durations),
        min_local_anchors=int(min_local_anchors),
        min_independent_samples=int(min_independent_samples),
    )

    local_location = np.full(time.size, np.nan, dtype=float)
    local_scale = np.full(time.size, np.nan, dtype=float)
    if anchor_indices.size:
        anchor_segment = segment_id[anchor_indices]
        finite_anchor_noise = np.isfinite(anchor_locations) & np.isfinite(anchor_scales)
        global_location, global_scale = _robust_location_scale(
            raw_depth[anchor_indices]
        )
        independent_count = _independent_span_count(
            time[anchor_indices], anchor_segment, duration_days
        )
        if independent_count < min_independent_samples:
            global_location = np.nan
            global_scale = np.nan

        for segment in np.unique(segment_id):
            output_positions = np.flatnonzero(segment_id == segment)
            source_positions = np.flatnonzero(
                (anchor_segment == segment) & finite_anchor_noise
            )
            if source_positions.size:
                source_indices = anchor_indices[source_positions]
                local_location[output_positions] = np.interp(
                    time[output_positions],
                    time[source_indices],
                    anchor_locations[source_positions],
                )
                local_scale[output_positions] = np.interp(
                    time[output_positions],
                    time[source_indices],
                    anchor_scales[source_positions],
                )
            elif np.isfinite(global_scale):
                local_location[output_positions] = global_location
                local_scale[output_positions] = global_scale

    # The local MAD is calibrated on normally covered boxes. Only incomplete
    # boxes need extra uncertainty; inclusive endpoints can make regular boxes
    # contain one point more than the duration/cadence ratio and must not make
    # them look artificially more precise than the local noise model.
    effective_points = np.minimum(np.maximum(counts, 1), expected_points)
    coverage_factor = np.sqrt(expected_points / effective_points)
    uncertainty = local_scale * coverage_factor
    depth = raw_depth - local_location
    valid = (
        box_valid
        & np.isfinite(depth)
        & np.isfinite(uncertainty)
        & (uncertainty > 0)
    )
    depth[~valid] = np.nan
    uncertainty[~valid] = np.nan

    denominator = np.full(time.size, np.nan, dtype=float)
    numerator = np.full(time.size, np.nan, dtype=float)
    ses = np.full(time.size, np.nan, dtype=float)
    denominator[valid] = 1.0 / np.square(uncertainty[valid])
    numerator[valid] = depth[valid] * denominator[valid]
    ses[valid] = numerator[valid] / np.sqrt(denominator[valid])

    result.update(
        {
            "time": time,
            "depth": depth,
            "raw_depth": raw_depth,
            "uncertainty": uncertainty,
            "N": numerator,
            "D": denominator,
            "SES": ses,
            "coverage": coverage,
            "valid": valid,
            "noise_eligible": valid & ~overlaps_excluded,
        }
    )
    return result


def calculate_cdpp(
    flux,
    cadence_hours,
    durations=(3.0, 6.0, 12.0),
    time=None,
    exclude_mask=None,
    gap_mask=None,
    min_coverage=0.8,
    min_windows=5,
):
    """Return median local duration-matched uncertainty in ppm.

    CDPP is summarized from the same time-dependent uncertainty used by the
    SES/MES estimator. Known transit windows are excluded from the noise model
    and from the summary, but the statistic remains a time-domain approximation
    rather than a reproduction of Kepler TPS wavelet CDPP.

    ``gap_mask`` marks already-detected transits to treat as unobserved; see
    ``duration_matched_statistics`` for why gapping beats deleting cadences.
    """
    flux = np.asarray(flux, dtype=float)
    if flux.ndim != 1:
        raise ValueError("flux must be a one-dimensional array")
    if time is None:
        if not np.isfinite(cadence_hours) or cadence_hours <= 0:
            return {f"cdpp_{int(dur)}h": np.nan for dur in durations}
        time = np.arange(flux.size, dtype=float) * float(cadence_hours) / 24.0

    results = {}
    for duration in durations:
        duration = float(duration)
        series = duration_matched_statistics(
            time,
            flux,
            duration_hours=duration,
            cadence_hours=cadence_hours,
            exclude_mask=exclude_mask,
            gap_mask=gap_mask,
            min_coverage=min_coverage,
            min_independent_samples=min_windows,
        )
        eligible = series["noise_eligible"]
        uncertainty = series["uncertainty"][eligible]
        value = float(np.median(uncertainty) * 1e6) if uncertainty.size else np.nan
        results[f"cdpp_{int(duration)}h"] = value
    return results
