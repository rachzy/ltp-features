import numpy as np

# 5 - Compute SES and MES

def compute_SES_MES(depths, local_noise, npts_in_transit, cdpp_dict=None, duration_hours=None, method='auto'):
    """
    depths: array (flux units, e.g. relative flux like 0.0067)
    local_noise: sigma per-point (same flux units)
    npts_in_transit: array of ints (n points used per transit)
    cdpp_dict: dict with keys 'cdpp_3h','cdpp_6h','cdpp_12h' (ppm)
    duration_hours: approximate transit duration in hours (float)
    method: 'auto'|'cdpp'|'point_sigma'
    Returns: {'SES': array, 'MES': float}
    """
    depths = np.asarray(depths, dtype=float)
    npts = np.asarray(npts_in_transit, dtype=float)

    if depths.size == 0 or npts.size == 0:
        return {'SES': np.array([]), 'MES': np.nan}

    def cdpp_interp(h):
        if not cdpp_dict or duration_hours is None:
            return np.nan
        c3 = cdpp_dict.get('cdpp_3h', np.nan)
        c6 = cdpp_dict.get('cdpp_6h', np.nan)
        c12 = cdpp_dict.get('cdpp_12h', np.nan)
        if not np.isfinite(c3) or not np.isfinite(c6) or not np.isfinite(c12):
            return np.nan
        if h <= 3:
            return c3
        elif h <= 6:
            return c3 + (c6 - c3) * (h - 3) / (6 - 3)
        elif h <= 12:
            return c6 + (c12 - c6) * (h - 6) / (12 - 6)
        else:
            return c12

    use_cdpp = False
    if method == 'cdpp':
        use_cdpp = True
    elif method == 'auto':
        use_cdpp = (cdpp_dict is not None and duration_hours is not None and np.isfinite(cdpp_interp(duration_hours)))

    if use_cdpp:
        cdpp_est = float(cdpp_interp(duration_hours))
        depths_ppm = depths * 1e6
        SES = np.full_like(depths_ppm, np.nan, dtype=float)
        if np.isfinite(cdpp_est) and cdpp_est > 0:
            SES = depths_ppm / cdpp_est     
    else:
        SES = np.full_like(depths, np.nan, dtype=float)
        if local_noise is not None and np.isfinite(local_noise) and local_noise > 0:
            valid = (npts > 0) & np.isfinite(depths)
            SES[valid] = depths[valid] / (local_noise / np.sqrt(npts[valid]))

    ses_valid = SES[np.isfinite(SES)]
    MES = float(np.sqrt(np.nansum(ses_valid**2))) if ses_valid.size > 0 else np.nan

    return {'SES': SES, 'MES': MES}