import numpy as np
from scipy import stats

def scaling_and_metrics(time, flux_detr):
    flux = np.asarray(flux_detr).copy()
    flux[np.isnan(flux)] = np.nanmedian(flux)
    median = np.nanmedian(flux)
    mad = np.nanmedian(np.abs(flux - median))
    std = np.nanstd(flux)
    if mad < 0.001:
        flux_scaled = (flux - median) / std if std > 0 else flux - median
    else:
        flux_scaled = (flux - median) / (1.4826 * mad)
    finite = np.isfinite(flux_scaled)
    scaling_metrics = {
        'mean': np.nanmean(flux_scaled),
        'std': np.nanstd(flux_scaled),
        'skewness': stats.skew(flux_scaled[finite]) if np.sum(finite) > 2 else np.nan,
        'kurtosis': stats.kurtosis(flux_scaled[finite]) if np.sum(finite) > 2 else np.nan,
        'outlier_resistance': np.sum(np.abs(flux_scaled[finite]) > 5) / np.sum(finite) * 100
    }
    return flux_scaled, scaling_metrics

def calculate_detection_rate(flux_scaled, n_sigma=3):
    median = np.nanmedian(flux_scaled)
    mad = np.nanmedian(np.abs(flux_scaled - median))
    threshold = median - n_sigma * 1.4826 * mad
    finite = np.isfinite(flux_scaled)
    detection_rate = np.sum(flux_scaled[finite] < threshold) / np.sum(finite) * 100
    return {"adaptive_threshold": float(threshold), "detection_rate": float(detection_rate)}

def compute_secondary_depth(time, flux_detr, period, t0, dur_days):
    mask = np.isfinite(time) & np.isfinite(flux_detr)
    if np.sum(mask) < 10:
        return np.nan
    t = np.asarray(time)[mask]; f = np.asarray(flux_detr)[mask]
    phase_0to1 = ((t - t0)/period) % 1.0
    sec_center = 0.5
    sec_half = 1.5 * (dur_days / period) if (period > 0) else 0.05
    sel = (phase_0to1 > (sec_center - sec_half)) & (phase_0to1 < (sec_center + sec_half))
    if not np.any(sel):
        return np.nan
    baseline = np.nanmedian(f)
    # robust depth using low percentile to reduce single-point outlier influence
    sec_low = np.nanpercentile(f[sel], 5.0)
    return float(baseline - sec_low)

def compute_odd_even_depth_ratio(time, flux_detr, period, t0, dur_days):
    """Compute odd-even depth ratio to detect eclipsing binaries."""
    mask = np.isfinite(time) & np.isfinite(flux_detr)
    if np.sum(mask) < 6:
        return np.nan
    t = np.asarray(time)[mask]; f = np.asarray(flux_detr)[mask]
    
    # Get epochs
    epochs = np.floor((t - t0) / period + 0.5).astype(int)
    unique_epochs = np.unique(epochs)
    
    if len(unique_epochs) < 2:
        return np.nan
    
    odd_depths = []; even_depths = []
    half_window = 1.5 * dur_days
    
    for e in unique_epochs:
        center_time = t0 + e * period
        sel = (t >= center_time - half_window) & (t <= center_time + half_window)
        if np.sum(sel) < 3:
            continue
            
        t_e = t[sel]; f_e = f[sel]
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

def compute_ingress_egress_asymmetry(time, flux_detr, period, t0, dur_days, nbins=200):
    """Compute ingress/egress asymmetry to detect grazing transits and EBs."""
    mask = np.isfinite(time) & np.isfinite(flux_detr)
    if np.sum(mask) < 10:
        return np.nan
    t = np.asarray(time)[mask]; f = np.asarray(flux_detr)[mask]
    
    # Phase fold
    phase = ((t - t0) / period) % 1.0
    phase = (phase + 0.5) % 1.0 - 0.5
    
    # Bin the folded light curve
    bins = np.linspace(-0.5, 0.5, nbins + 1)
    med_profile, _, _ = stats.binned_statistic(phase, f, statistic="median", bins=bins)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    
    # Find transit region
    dur_phase = (dur_days / period) if (period > 0) else 0.05
    transit_mask = np.abs(bin_centers) < (1.5 * dur_phase)
    
    if not np.any(transit_mask):
        return np.nan
    
    # Find baseline
    oot_mask = np.abs(bin_centers) > (2.0 * dur_phase)
    if np.any(oot_mask):
        baseline = np.nanmedian(med_profile[oot_mask])
    else:
        baseline = np.nanmedian(med_profile)
    
    # Find minimum
    transit_profile = med_profile[transit_mask]
    transit_centers = bin_centers[transit_mask]
    min_idx = np.nanargmin(transit_profile)
    min_phase = transit_centers[min_idx]
    
    # Define ingress and egress regions
    ingress_mask = (transit_centers >= -1.5 * dur_phase) & (transit_centers <= min_phase)
    egress_mask = (transit_centers >= min_phase) & (transit_centers <= 1.5 * dur_phase)
    
    if not (np.any(ingress_mask) and np.any(egress_mask)):
        return np.nan
    
    # Compute slopes
    ingress_flux = transit_profile[ingress_mask]
    ingress_phase = transit_centers[ingress_mask]
    egress_flux = transit_profile[egress_mask]
    egress_phase = transit_centers[egress_mask]
    
    # Linear fit for slopes
    try:
        if len(ingress_flux) >= 2:
            ingress_slope = np.polyfit(ingress_phase, ingress_flux, 1)[0]
        else:
            ingress_slope = 0.0
            
        if len(egress_flux) >= 2:
            egress_slope = np.polyfit(egress_phase, egress_flux, 1)[0]
        else:
            egress_slope = 0.0
            
        # Asymmetry metric: difference in absolute slopes
        asymmetry = abs(ingress_slope) - abs(egress_slope)
        return float(asymmetry)
        
    except Exception:
        return np.nan

def compute_secondary_depth_snr(time, flux_detr, period, t0, dur_days, local_noise):
    """Compute secondary depth signal-to-noise ratio."""
    secondary_depth = compute_secondary_depth(time, flux_detr, period, t0, dur_days)
    
    if not np.isfinite(secondary_depth) or not np.isfinite(local_noise) or local_noise <= 0:
        return np.nan
    
    # Estimate number of points in secondary eclipse region
    mask = np.isfinite(time) & np.isfinite(flux_detr)
    if np.sum(mask) < 3:
        return np.nan
    
    t = np.asarray(time)[mask]
    phase_0to1 = ((t - t0) / period) % 1.0
    sec_center = 0.5
    sec_half = 1.5 * (dur_days / period) if (period > 0) else 0.05
    sel = (phase_0to1 > (sec_center - sec_half)) & (phase_0to1 < (sec_center + sec_half))
    n_sec_points = np.sum(sel)
    
    if n_sec_points <= 0:
        return np.nan
    
    # SNR = depth / (noise / sqrt(n_points))
    snr = secondary_depth / (local_noise / np.sqrt(n_sec_points))
    return float(snr)

def interp_cdpp(cdpp_dict, duration_hours):
    if cdpp_dict is None or duration_hours is None or not np.isfinite(duration_hours):
        return np.nan
    c3 = cdpp_dict.get('cdpp_3h', np.nan)
    c6 = cdpp_dict.get('cdpp_6h', np.nan)
    c12 = cdpp_dict.get('cdpp_12h', np.nan)
    vals = np.array([c for c in (c3,c6,c12) if np.isfinite(c)])
    if vals.size == 0:
        return np.nan
    if not np.isfinite(c3): c3 = np.nanmean(vals)
    if not np.isfinite(c6): c6 = np.nanmean(vals)
    if not np.isfinite(c12): c12 = np.nanmean(vals)
    h = duration_hours
    if h <= 3:
        return float(c3)
    elif h <= 6:
        return float(c3 + (c6-c3)*(h-3)/(6-3))
    elif h <= 12:
        return float(c6 + (c12-c6)*(h-6)/(12-6))
    else:
        return float(c12)
