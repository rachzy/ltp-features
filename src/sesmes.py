import numpy as np

from cdpp import duration_matched_statistics

# 5 - Compute coherent SES and MES


def _empty_result():
    return {
        "SES": np.array([], dtype=float),
        "MES": np.nan,
        "max_ses": np.nan,
        "max_mes": np.nan,
        "best_phase_offset": np.nan,
        "event_times": np.array([], dtype=float),
        "event_indices": np.array([], dtype=int),
        "statistics": None,
    }


def _nearest_eligible_indices(series_time, eligible, event_times, tolerance_days):
    """Return unique nearest eligible cadence indices for predicted events."""
    if series_time.size == 0 or event_times.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    eligible_indices = np.flatnonzero(eligible)
    if eligible_indices.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    eligible_times = series_time[eligible_indices]
    insertion = np.searchsorted(eligible_times, event_times, side="left")
    chosen_indices = []
    chosen_times = []
    previous = -1
    for event_time, position in zip(event_times, insertion):
        candidates = []
        if position < eligible_times.size:
            candidates.append(int(position))
        if position > 0:
            candidates.append(int(position - 1))
        if not candidates:
            continue
        eligible_position = min(
            candidates, key=lambda value: abs(eligible_times[value] - event_time)
        )
        index = int(eligible_indices[eligible_position])
        if abs(series_time[index] - event_time) > tolerance_days * (1.0 + 1e-9):
            continue
        if index == previous:
            continue
        chosen_indices.append(index)
        chosen_times.append(float(event_time))
        previous = index
    return np.asarray(chosen_indices, dtype=int), np.asarray(chosen_times, dtype=float)


def _event_centers(time_min, time_max, period, epoch):
    first_cycle = int(np.ceil((time_min - epoch) / period - 1e-12))
    last_cycle = int(np.floor((time_max - epoch) / period + 1e-12))
    if last_cycle < first_cycle:
        return np.array([], dtype=float)
    cycles = np.arange(first_cycle, last_cycle + 1, dtype=np.int64)
    return epoch + cycles.astype(float) * period


def _fold_components(statistics, period, epoch, tolerance_days):
    series_time = statistics["time"]
    if series_time.size == 0:
        return np.nan, np.array([], dtype=float), np.array([], dtype=int), np.array([])
    predicted = _event_centers(series_time[0], series_time[-1], period, epoch)
    indices, sampled_times = _nearest_eligible_indices(
        series_time,
        statistics["valid"],
        predicted,
        tolerance_days,
    )
    if indices.size == 0:
        return np.nan, sampled_times, indices, np.array([], dtype=float)

    numerator = statistics["N"][indices]
    denominator = statistics["D"][indices]
    denominator_sum = float(np.sum(denominator))
    mes = (
        float(np.sum(numerator) / np.sqrt(denominator_sum))
        if np.isfinite(denominator_sum) and denominator_sum > 0
        else np.nan
    )
    event_ses = numerator / np.sqrt(denominator)
    return mes, sampled_times, indices, event_ses


def _phase_offsets(duration_days, cadence_days):
    step = max(cadence_days / 2.0, np.finfo(float).eps)
    offsets = np.arange(
        -duration_days / 2.0,
        duration_days / 2.0 + 0.5 * step,
        step,
        dtype=float,
    )
    if offsets.size == 0:
        offsets = np.array([0.0], dtype=float)
    if not np.any(np.isclose(offsets, 0.0, atol=step * 1e-9, rtol=0.0)):
        offsets = np.append(offsets, 0.0)
    return np.unique(np.sort(offsets))


def _scan_phase_offsets(statistics, period, epoch, duration_days, tolerance_days, mes):
    """Return the ``(best_mes, best_offset)`` of the nearby phase scan."""
    cadence_days = statistics["cadence_days"]
    best_mes = mes
    best_offset = 0.0 if np.isfinite(mes) else np.nan
    for offset in _phase_offsets(float(duration_days), cadence_days):
        candidate_mes, _, _, _ = _fold_components(
            statistics,
            float(period),
            float(epoch) + float(offset),
            tolerance_days,
        )
        if not np.isfinite(candidate_mes):
            continue
        improves = not np.isfinite(best_mes) or candidate_mes > best_mes
        tied_closer_to_bls_epoch = (
            np.isfinite(best_mes)
            and np.isclose(candidate_mes, best_mes, rtol=1e-12, atol=1e-12)
            and abs(float(offset)) < abs(float(best_offset))
        )
        if improves or tied_closer_to_bls_epoch:
            best_mes = candidate_mes
            best_offset = float(offset)
    return best_mes, best_offset


def fold_statistics(statistics, period, epoch, duration_days=None):
    """Re-fold a prepared statistics object at a trial ephemeris.

    ``statistics`` is what ``compute_SES_MES`` returns under ``"statistics"``.
    ``duration_matched_statistics`` takes neither a period nor an epoch - the
    box width is the only ephemeris-dependent thing baked into it - so any
    trial sharing that width can be scored for the cost of a fold, with no
    noise model to rebuild. Harmonics of one signal share T14, which makes
    ``P``, ``2P``, ``3P``, ``P/2`` and ``P/3`` all valid against one object.

    Returns the exact-ephemeris fold plus the same nearby-phase optimum
    ``compute_SES_MES`` reports, and the event count *at that optimum*:
    comparing candidates on ``max_mes`` means their event counts have to come
    from the same offset.
    """
    out = {
        "SES": np.array([], dtype=float),
        "MES": np.nan,
        "max_mes": np.nan,
        "best_phase_offset": np.nan,
        "event_times": np.array([], dtype=float),
        "event_indices": np.array([], dtype=int),
        "max_mes_n_events": 0,
    }
    if statistics is None or statistics["time"].size == 0:
        return out
    if not (
        np.isfinite(period)
        and period > 0
        and np.isfinite(epoch)
    ):
        return out

    if duration_days is None:
        duration_days = statistics["duration_days"]
    duration_days = float(duration_days)
    if not (np.isfinite(duration_days) and duration_days > 0):
        return out

    tolerance_days = statistics["cadence_days"] / 2.0
    mes, event_times, event_indices, event_ses = _fold_components(
        statistics, float(period), float(epoch), tolerance_days
    )
    best_mes, best_offset = _scan_phase_offsets(
        statistics, period, epoch, duration_days, tolerance_days, mes
    )

    if best_offset == 0.0:
        best_ses = event_ses
    elif np.isfinite(best_offset):
        _, _, _, best_ses = _fold_components(
            statistics, float(period), float(epoch) + best_offset, tolerance_days
        )
    else:
        best_ses = np.array([], dtype=float)
    n_events = int(np.sum(np.isfinite(best_ses)))

    out.update(
        {
            "SES": event_ses,
            "MES": mes,
            "max_mes": best_mes,
            "best_phase_offset": best_offset,
            "event_times": event_times,
            "event_indices": event_indices,
            "max_mes_n_events": n_events,
        }
    )
    return out


def compute_SES_MES(
    time,
    flux,
    period,
    t0,
    duration_days,
    cadence_hours=None,
    transit_mask=None,
    gap_mask=None,
    min_coverage=0.8,
):
    """Compute time-domain SES and coherent MES for a BLS candidate.

    ``max_ses`` is the largest positive cadence-level SES at the candidate
    duration. ``MES`` folds the matched-filter numerator and denominator at the
    exact BLS ephemeris. ``max_mes`` optimizes only a nearby phase offset while
    keeping the BLS period and duration fixed; it is not a full TPS periodogram
    maximum.

    ``gap_mask`` marks cadences belonging to already-accepted candidates. Pass
    the full series plus this mask rather than a pre-sliced array: deleting the
    cadences fragments the noise model's segments and biases its MAD low.

    The returned ``SES`` array contains the exact-ephemeris event statistics so
    the extractor's per-transit summary columns retain their established shape.
    """
    result = _empty_result()
    if (
        not np.isfinite(period)
        or period <= 0
        or not np.isfinite(t0)
        or not np.isfinite(duration_days)
        or duration_days <= 0
    ):
        return result

    statistics = duration_matched_statistics(
        time,
        flux,
        duration_hours=float(duration_days) * 24.0,
        cadence_hours=cadence_hours,
        exclude_mask=transit_mask,
        gap_mask=gap_mask,
        min_coverage=min_coverage,
    )
    result["statistics"] = statistics
    if statistics["time"].size == 0:
        return result

    positive_ses = statistics["SES"]
    positive_ses = positive_ses[np.isfinite(positive_ses) & (positive_ses > 0)]
    max_ses = float(np.max(positive_ses)) if positive_ses.size else np.nan

    folded = fold_statistics(statistics, float(period), float(t0), duration_days)
    folded.pop("max_mes_n_events", None)
    result.update(folded)
    result["max_ses"] = max_ses
    return result
