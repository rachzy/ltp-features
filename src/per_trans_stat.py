import numpy as np

# 6 - Compute per-transit statistics

def per_transit_stats_simple(time, flux, period, t0, transit_duration_days, window_factor=2.0, min_points=3):
    time = np.asarray(time); flux = np.asarray(flux)
    mask = np.isfinite(time) & np.isfinite(flux)
    if np.sum(mask) < min_points:
        return {'per_transit': [], 'depths': np.array([]), 'npts_in_transit': np.array([])}
    t = time[mask]; f = flux[mask]
    if transit_duration_days is None or transit_duration_days <= 0:
        transit_duration_days = 0.1
    epochs = np.floor((t - t0) / period + 1e-12).astype(int)
    unique_epochs = np.unique(epochs)
    per_transit = []; depths = []; npts_list = []
    half_window = window_factor * transit_duration_days
    for e in unique_epochs:
        center_time = t0 + e * period
        sel = (t >= center_time - half_window) & (t <= center_time + half_window)
        if np.sum(sel) < min_points:
            continue
        t_e = t[sel]; f_e = f[sel]
        in_tr_mask = np.abs(t_e - center_time) < (transit_duration_days / 2.0)
        if np.sum(~in_tr_mask) >= max(3, int(0.3*np.sum(sel))):
            baseline = float(np.nanmedian(f_e[~in_tr_mask]))
        else:
            baseline = float(np.nanmedian(f_e))
        min_in = float(np.nanmin(f_e[in_tr_mask])) if np.any(in_tr_mask) else float(np.min(f_e))
        depth_i = baseline - min_in
        model = np.full_like(f_e, baseline)
        if np.any(in_tr_mask):
            model[in_tr_mask] = baseline - depth_i
        resid = f_e - model
        mad = np.nanmedian(np.abs(resid - np.nanmedian(resid)))
        sigma = 1.4826 * mad if (mad > 0 and np.isfinite(mad)) else np.nanstd(resid)
        chi2 = float(np.nansum((resid / sigma)**2)) if (np.isfinite(sigma) and sigma > 0) else np.nan
        per_transit.append({'epoch': int(e), 'center_time': float(center_time), 'npts': int(np.sum(sel)),
                             'baseline': baseline, 'depth': float(depth_i), 'resid_rms': float(np.nanstd(resid)), 'chi2': chi2})
        depths.append(depth_i); npts_list.append(int(np.sum(sel)))
    depths = np.array(depths) if len(depths)>0 else np.array([])
    npts_arr = np.array(npts_list) if len(npts_list)>0 else np.array([])
    return {'per_transit': per_transit, 'depths': depths, 'npts_in_transit': npts_arr}