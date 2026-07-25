import numpy as np
from scipy import stats


def compute_ingress_egress_asymmetry(time, flux_detr, period, t0, dur_days, nbins=200):
    """Compute ingress/egress asymmetry to detect grazing transits and EBs."""
    mask = np.isfinite(time) & np.isfinite(flux_detr)
    if np.sum(mask) < 10:
        return np.nan
    t = np.asarray(time)[mask]
    f = np.asarray(flux_detr)[mask]

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
