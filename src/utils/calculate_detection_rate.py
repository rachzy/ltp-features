import numpy as np


def calculate_detection_rate(flux_scaled, n_sigma=3):
    median = np.nanmedian(flux_scaled)
    mad = np.nanmedian(np.abs(flux_scaled - median))
    threshold = median - n_sigma * 1.4826 * mad
    finite = np.isfinite(flux_scaled)
    detection_rate = np.sum(flux_scaled[finite] < threshold) / np.sum(finite) * 100
    return {
        "adaptive_threshold": float(threshold),
        "detection_rate": float(detection_rate),
    }
