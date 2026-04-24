import numpy as np
from scipy.stats import binned_statistic
from statsmodels.tsa.stattools import acf as sm_acf
from scipy.stats import binned_statistic
# 3 - Compute folded and binned metrics 

def folded_binned_metrics(time, flux, period, t0, lags_hours=(1,3,6,12,24), nbins=200):
    mask = np.isfinite(time) & np.isfinite(flux)
    time_u = time[mask]
    flux_u = flux[mask]
    if len(time_u) < 10:
        return {"local_noise": np.nan, "depth_stability": np.nan, "acf_lags": {h: np.nan for h in lags_hours}, "cadence_hours": np.nan}

    
    diffs = np.diff(time_u)
    diffs = diffs[np.isfinite(diffs)]
    diffs_small = diffs[diffs < 1.0]  
    dt_days = np.median(diffs_small) if len(diffs_small) > 0 else np.median(diffs)
    cadence_hours = dt_days * 24.0 if dt_days > 0 else np.nan

    phase = ((time_u - t0) / period) % 1.0
    phase = (phase + 0.5) % 1.0 - 0.5

    bins = np.linspace(-0.5, 0.5, nbins + 1)
    med, _, _ = binned_statistic(phase, flux_u, statistic="median", bins=bins)
    bin_centers = 0.5*(bins[:-1] + bins[1:])
    out_idx = np.abs(bin_centers) > 0.25
    baseline = np.nanmedian(med[out_idx]) if np.any(out_idx) else np.nanmedian(med)
    depth = baseline - np.nanmin(med)

    if np.isfinite(depth) and depth > 0:
        half = baseline - depth/2
        in_half = med < half
        if np.any(in_half):
            left = np.argmax(in_half)
            right = len(in_half) - np.argmax(in_half[::-1]) - 1
            dur_bins = max(1, right - left + 1)
            dur_phase = dur_bins / nbins
        else:
            dur_phase = max(0.05, (med.size / nbins) * 1.0)
    else:
        dur_phase = 0.05

    in_transit_mask = np.abs(phase) < (1.5 * dur_phase)
    oot_mask = ~in_transit_mask

    if np.any(oot_mask):
        mad_oot = np.nanmedian(np.abs(flux_u[oot_mask] - np.nanmedian(flux_u[oot_mask])))
        local_noise = 1.4826 * mad_oot if mad_oot > 0 else np.nanstd(flux_u[oot_mask])
    else:
        local_noise = np.nanstd(flux_u)

    epochs = np.floor((time_u - t0) / period).astype(int)
    depths = []
    for e in np.unique(epochs):
        sel = epochs == e
        if np.sum(sel) < 3:
            continue
        ph_e = phase[sel]; fl_e = flux_u[sel]
        oot_e = np.abs(ph_e) > (1.5 * dur_phase)
        baseline_e = np.nanmedian(fl_e[oot_e]) if np.any(oot_e) else np.nanmedian(fl_e)
        depths.append(baseline_e - np.nanmin(fl_e))
    depths = np.array(depths)
    depth_stability = (np.nanstd(depths)/np.nanmean(depths)) if (depths.size > 1 and np.nanmean(depths)!=0) else (0.0 if depths.size==1 else np.nan)

    try:
        x = flux_u - np.nanmean(flux_u)
        acf_vals = sm_acf(x, nlags=min(int(len(x)/2), 10000), fft=True, missing='drop')
    except Exception:
        acf_vals = np.array([np.nan])

    acf_lags = {}
    for h in lags_hours:
        if np.isnan(cadence_hours) or cadence_hours <= 0:
            acf_lags[h] = np.nan
        else:
            lag = int(round(h / cadence_hours))
            acf_lags[h] = float(acf_vals[lag]) if lag < len(acf_vals) else np.nan

    return {"local_noise": float(local_noise),
            "depth_stability": float(depth_stability),
            "acf_lags": acf_lags,
            "cadence_hours": float(cadence_hours)}