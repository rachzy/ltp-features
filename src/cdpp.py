import numpy as np
from scipy.ndimage import uniform_filter1d

# 4 - Compute CDPP 

def calculate_cdpp(flux, cadence_hours, durations=[3.0, 6.0, 12.0]):
    flux = np.asarray(flux)
    flux = np.where(np.isfinite(flux), flux, np.nanmedian(flux))
    median = np.nanmedian(flux)
    if median == 0 or not np.isfinite(median):
        median = 1.0
    flux_norm = flux / median

    cdpp_results = {}
    points_per_hour = 1.0 / cadence_hours
    npts = len(flux_norm)

    for dur in durations:
        window = int(round(dur * points_per_hour))
        if window < 2:
            window = 2
        if window >= npts:
            window = max(2, npts // 2)
        smooth = uniform_filter1d(flux_norm, size=window, mode="nearest")
        resid = flux_norm - smooth
        cdpp_val = np.nanstd(resid) * 1e6 if np.any(np.isfinite(resid)) else np.nan
        cdpp_results[f'cdpp_{int(dur)}h'] = float(cdpp_val)

    return cdpp_results