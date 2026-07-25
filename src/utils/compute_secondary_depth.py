import numpy as np


def compute_secondary_depth(time, flux_detr, period, t0, dur_days):
    mask = np.isfinite(time) & np.isfinite(flux_detr)
    if np.sum(mask) < 10:
        return np.nan
    t = np.asarray(time)[mask]
    f = np.asarray(flux_detr)[mask]
    phase_0to1 = ((t - t0) / period) % 1.0
    sec_center = 0.5
    sec_half = 1.5 * (dur_days / period) if (period > 0) else 0.05
    sel = (phase_0to1 > (sec_center - sec_half)) & (
        phase_0to1 < (sec_center + sec_half)
    )
    if not np.any(sel):
        return np.nan
    baseline = np.nanmedian(f)
    # robust depth using low percentile to reduce single-point outlier influence
    sec_low = np.nanpercentile(f[sel], 5.0)
    return float(baseline - sec_low)
