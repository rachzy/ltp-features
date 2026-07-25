import numpy as np


def compute_odd_even_depth_ratio(time, flux_detr, period, t0, dur_days):
    """Compute odd-even depth ratio to detect eclipsing binaries."""
    mask = np.isfinite(time) & np.isfinite(flux_detr)
    if np.sum(mask) < 6:
        return np.nan
    t = np.asarray(time)[mask]
    f = np.asarray(flux_detr)[mask]

    # Get epochs
    epochs = np.floor((t - t0) / period + 0.5).astype(int)
    unique_epochs = np.unique(epochs)

    if len(unique_epochs) < 2:
        return np.nan

    odd_depths = []
    even_depths = []
    half_window = 1.5 * dur_days

    for e in unique_epochs:
        center_time = t0 + e * period
        sel = (t >= center_time - half_window) & (t <= center_time + half_window)
        if np.sum(sel) < 3:
            continue

        t_e = t[sel]
        f_e = f[sel]
        in_tr_mask = np.abs(t_e - center_time) < (dur_days / 2.0)

        if np.sum(~in_tr_mask) >= 3:
            baseline = float(np.nanmedian(f_e[~in_tr_mask]))
        else:
            baseline = float(np.nanmedian(f_e))

        if np.any(in_tr_mask):
            in_vals = f_e[in_tr_mask]
            min_robust = float(np.nanpercentile(in_vals, 10.0))
            depth = baseline - min_robust

            if e % 2 == 0:
                even_depths.append(depth)
            else:
                odd_depths.append(depth)

    if len(odd_depths) == 0 or len(even_depths) == 0:
        return np.nan

    odd_mean = np.nanmean(odd_depths)
    even_mean = np.nanmean(even_depths)

    if even_mean == 0 or not np.isfinite(odd_mean) or not np.isfinite(even_mean):
        return np.nan

    return float(odd_mean / even_mean)
