import numpy as np

from .compute_secondary_depth import compute_secondary_depth


def compute_secondary_depth_snr(time, flux_detr, period, t0, dur_days, local_noise):
    """Compute secondary depth signal-to-noise ratio."""
    secondary_depth = compute_secondary_depth(time, flux_detr, period, t0, dur_days)

    if (
        not np.isfinite(secondary_depth)
        or not np.isfinite(local_noise)
        or local_noise <= 0
    ):
        return np.nan

    # Estimate number of points in secondary eclipse region
    mask = np.isfinite(time) & np.isfinite(flux_detr)
    if np.sum(mask) < 3:
        return np.nan

    t = np.asarray(time)[mask]
    phase_0to1 = ((t - t0) / period) % 1.0
    sec_center = 0.5
    sec_half = 1.5 * (dur_days / period) if (period > 0) else 0.05
    sel = (phase_0to1 > (sec_center - sec_half)) & (
        phase_0to1 < (sec_center + sec_half)
    )
    n_sec_points = np.sum(sel)

    if n_sec_points <= 0:
        return np.nan

    # SNR = depth / (noise / sqrt(n_points))
    snr = secondary_depth / (local_noise / np.sqrt(n_sec_points))
    return float(snr)
