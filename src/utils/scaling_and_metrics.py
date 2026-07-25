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
        "mean": np.nanmean(flux_scaled),
        "std": np.nanstd(flux_scaled),
        "skewness": stats.skew(flux_scaled[finite]) if np.sum(finite) > 2 else np.nan,
        "kurtosis": stats.kurtosis(flux_scaled[finite]) if np.sum(finite) > 2 else np.nan,
        "outlier_resistance": np.sum(np.abs(flux_scaled[finite]) > 5)
        / np.sum(finite)
        * 100,
    }
    return flux_scaled, scaling_metrics
