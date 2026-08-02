import numpy as np

# 5 - Compute SES and MES


def compute_SES_MES(
    depths,
    local_noise,
    npts_in_transit,
    cdpp_dict=None,
    duration_hours=None,
    method="auto",
):
    """
    depths: array (flux units, e.g. relative flux like 0.0067)
    local_noise: sigma per-point (same flux units)
    npts_in_transit: array of ints (n points used per transit)
    cdpp_dict: dict with keys 'cdpp_3h','cdpp_6h','cdpp_12h' (ppm)
    duration_hours: approximate transit duration in hours (float)
    method: 'auto'|'cdpp'|'point_sigma'

    Returns
    -------
    dict
        ``SES`` contains the signed per-event statistics. ``MES`` preserves
        the pipeline's historical root-sum-square aggregation. ``max_ses`` is
        the strongest signed event, and ``max_mes`` is the signed,
        inverse-variance-weighted statistic for the BLS-selected candidate.

        ``max_mes`` is not yet a full Kepler TPS MES periodogram maximum. It
        describes the winning candidate returned by the current BLS search.
    """
    depths = np.asarray(depths, dtype=float)
    npts = np.asarray(npts_in_transit, dtype=float)

    if depths.size == 0 or npts.size == 0:
        return {
            "SES": np.array([]),
            "MES": np.nan,
            "max_ses": np.nan,
            "max_mes": np.nan,
        }
    if depths.shape != npts.shape:
        raise ValueError("depths and npts_in_transit must have the same shape")

    def cdpp_interp(h):
        if not cdpp_dict or duration_hours is None:
            return np.nan
        c3 = cdpp_dict.get("cdpp_3h", np.nan)
        c6 = cdpp_dict.get("cdpp_6h", np.nan)
        c12 = cdpp_dict.get("cdpp_12h", np.nan)
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
    if method == "cdpp":
        use_cdpp = True
    elif method == "auto":
        use_cdpp = (
            cdpp_dict is not None
            and duration_hours is not None
            and np.isfinite(cdpp_interp(duration_hours))
        )

    depth_uncertainty = np.full_like(depths, np.nan, dtype=float)
    if use_cdpp:
        cdpp_est = float(cdpp_interp(duration_hours))
        SES = np.full_like(depths, np.nan, dtype=float)
        if np.isfinite(cdpp_est) and cdpp_est > 0:
            depth_uncertainty[np.isfinite(depths)] = cdpp_est / 1e6
    else:
        SES = np.full_like(depths, np.nan, dtype=float)
        if local_noise is not None and np.isfinite(local_noise) and local_noise > 0:
            valid = (npts > 0) & np.isfinite(depths)
            depth_uncertainty[valid] = local_noise / np.sqrt(npts[valid])

    valid = (
        np.isfinite(depths)
        & np.isfinite(depth_uncertainty)
        & (depth_uncertainty > 0)
    )
    SES[valid] = depths[valid] / depth_uncertainty[valid]

    ses_valid = SES[np.isfinite(SES)]
    MES = float(np.sqrt(np.nansum(ses_valid**2))) if ses_valid.size > 0 else np.nan

    max_ses = float(np.nanmax(ses_valid)) if ses_valid.size > 0 else np.nan

    if np.any(valid):
        weights = 1.0 / np.square(depth_uncertainty[valid])
        weight_sum = float(np.sum(weights))
        max_mes = (
            float(np.sum(weights * depths[valid]) / np.sqrt(weight_sum))
            if np.isfinite(weight_sum) and weight_sum > 0
            else np.nan
        )
    else:
        max_mes = np.nan

    return {
        "SES": SES,
        "MES": MES,
        "max_ses": max_ses,
        "max_mes": max_mes,
    }
