import numpy as np


def interp_cdpp(cdpp_dict, duration_hours):
    if cdpp_dict is None or duration_hours is None or not np.isfinite(duration_hours):
        return np.nan
    c3 = cdpp_dict.get("cdpp_3h", np.nan)
    c6 = cdpp_dict.get("cdpp_6h", np.nan)
    c12 = cdpp_dict.get("cdpp_12h", np.nan)
    vals = np.array([c for c in (c3, c6, c12) if np.isfinite(c)])
    if vals.size == 0:
        return np.nan
    if not np.isfinite(c3):
        c3 = np.nanmean(vals)
    if not np.isfinite(c6):
        c6 = np.nanmean(vals)
    if not np.isfinite(c12):
        c12 = np.nanmean(vals)
    h = duration_hours
    if h <= 3:
        return float(c3)
    elif h <= 6:
        return float(c3 + (c6 - c3) * (h - 3) / (6 - 3))
    elif h <= 12:
        return float(c6 + (c12 - c6) * (h - 6) / (12 - 6))
    else:
        return float(c12)
