import json
import os
import time
import numpy as np
from scipy.stats import binned_statistic
from collections import OrderedDict
from scipy.stats import skew, kurtosis
from scipy.stats import binned_statistic

from detrend_and_period import detrend_with_bls_mask
from folded_binned_metrics import folded_binned_metrics
from cdpp import calculate_cdpp
from sesmes import compute_SES_MES
from utils import (
    scaling_and_metrics,
    interp_cdpp,
    compute_secondary_depth,
    compute_odd_even_depth_ratio,
    compute_ingress_egress_asymmetry,
    compute_secondary_depth_snr,
)
from per_trans_stat import per_transit_stats_simple


def extract_features_from_arrays(
    tTime, flux, verbose=False, refine_duration=True, use_tls=False
):
    """Internal function to extract features from time and flux arrays."""
    start_time = time.time()
    print("Starting feature extraction from arrays...")

    feats = OrderedDict()

    mask0 = np.isfinite(tTime) & np.isfinite(flux)
    if np.sum(mask0) < 3:
        raise ValueError("too few valid points")
    time_arr = np.asarray(tTime)[mask0]
    flux_arr = np.asarray(flux)[mask0]

    print(f"Data loaded: {len(time_arr):,} valid points")
    print(f"Time range: {time_arr.min():.2f} to {time_arr.max():.2f} days")
    print(f"Flux range: {flux_arr.min():.6f} to {flux_arr.max():.6f}")

    flux_detr_full, trend_full, mask_transit, bls_info = detrend_with_bls_mask(
        time_arr, flux_arr, refine_duration=refine_duration, use_tls=use_tls
    )
    period = float(bls_info.get("best_period", np.nan))
    t0 = float(bls_info.get("t0", np.nan))
    duration_days = float(bls_info.get("best_duration", np.nan))

    print(
        f"BLS results: P={period:.4f} days, T14={duration_days:.4f} days, t0={t0:.2f}"
    )

    feats["period_days"] = period
    feats["t0"] = t0
    feats["duration_days"] = duration_days
    feats["duration_hours"] = duration_days * 24.0

    print("Computing scaling metrics...")
    flux_scaled, scaling_metrics = scaling_and_metrics(time_arr, flux_detr_full.copy())
    feats["scale_mean"] = scaling_metrics.get("mean", np.nan)
    feats["scale_std"] = scaling_metrics.get("std", np.nan)
    feats["scale_skewness"] = scaling_metrics.get("skewness", np.nan)
    feats["scale_kurtosis"] = scaling_metrics.get("kurtosis", np.nan)
    feats["scale_outlier_resistance"] = scaling_metrics.get(
        "outlier_resistance", np.nan
    )

    binned = folded_binned_metrics(
        time_arr, flux_detr_full, period, t0, lags_hours=(1, 3, 6, 12, 24)
    )
    feats["local_noise"] = binned.get("local_noise", np.nan)
    feats["depth_stability"] = binned.get("depth_stability", np.nan)
    acf_l = binned.get("acf_lags", {})
    feats["acf_lag_1h"] = acf_l.get(1, np.nan)
    feats["acf_lag_3h"] = acf_l.get(3, np.nan)
    feats["acf_lag_6h"] = acf_l.get(6, np.nan)
    feats["acf_lag_12h"] = acf_l.get(12, np.nan)
    feats["acf_lag_24h"] = acf_l.get(24, np.nan)
    feats["cadence_hours"] = binned.get("cadence_hours", np.nan)

    per = per_transit_stats_simple(time_arr, flux_detr_full, period, t0, duration_days)
    depths = per.get("depths", np.array([]))
    npts_in_transit = per.get("npts_in_transit", np.array([]))
    print(f"Per-transit analysis: {len(depths)} transits analyzed")
    feats["depth_mean_per_transit"] = (
        float(np.nanmean(depths)) if depths.size else np.nan
    )
    feats["depth_std_per_transit"] = float(np.nanstd(depths)) if depths.size else np.nan
    feats["npts_transit_median"] = (
        float(np.nanmedian(npts_in_transit)) if npts_in_transit.size else np.nan
    )

    print("Computing CDPP...")
    cdpp = calculate_cdpp(flux_detr_full, cadence_hours=feats["cadence_hours"])
    feats["cdpp_3h"] = cdpp.get("cdpp_3h", np.nan)
    feats["cdpp_6h"] = cdpp.get("cdpp_6h", np.nan)
    feats["cdpp_12h"] = cdpp.get("cdpp_12h", np.nan)

    duration_hours = feats["duration_hours"]
    cdpp_interp = interp_cdpp(cdpp, duration_hours)
    print("Computing SES/MES...")
    sesmes = compute_SES_MES(
        depths,
        feats["local_noise"],
        npts_in_transit,
        cdpp_dict=cdpp,
        duration_hours=duration_hours,
        method="auto",
    )
    SES_arr = sesmes.get("SES", np.array([]))
    MES_val = sesmes.get("MES", np.nan)
    feats["SES_mean"] = float(np.nanmean(SES_arr)) if SES_arr.size else np.nan
    feats["SES_std"] = float(np.nanstd(SES_arr)) if SES_arr.size else np.nan
    feats["MES"] = float(MES_val)
    print(f"SES/MES computed: MES={MES_val:.2f}")

    depth_mean = feats["depth_mean_per_transit"]
    if np.isfinite(depth_mean) and np.isfinite(cdpp_interp) and cdpp_interp > 0:
        feats["snr_global"] = float((depth_mean * 1e6) / cdpp_interp)
    else:
        feats["snr_global"] = np.nan

    if SES_arr.size:
        feats["snr_per_transit_mean"] = float(np.nanmean(SES_arr))
        feats["snr_per_transit_std"] = float(np.nanstd(SES_arr))
    else:
        feats["snr_per_transit_mean"] = np.nan
        feats["snr_per_transit_std"] = np.nan

    resid_global_full = flux_detr_full - np.nanmedian(flux_detr_full)
    feats["resid_rms_global"] = (
        float(np.nanstd(resid_global_full))
        if np.any(np.isfinite(resid_global_full))
        else np.nan
    )

    try:
        nbins = 200
        phase = ((time_arr - t0) / period) % 1.0
        phase = (phase + 0.5) % 1.0 - 0.5
        bins = np.linspace(-0.5, 0.5, nbins + 1)
        med_profile, _, _ = binned_statistic(
            phase, flux_detr_full, statistic="median", bins=bins
        )
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        dur_phase = (duration_days / period) if (period > 0) else 0.05

        center_mask = np.abs(bin_centers) < (0.25 * dur_phase)
        shoulder_mask = (np.abs(bin_centers) > (0.5 * dur_phase)) & (
            np.abs(bin_centers) < dur_phase
        )

        center_flux = (
            np.nanmedian(med_profile[center_mask]) if np.any(center_mask) else np.nan
        )
        shoulder_flux = (
            np.nanmedian(med_profile[shoulder_mask])
            if np.any(shoulder_mask)
            else np.nan
        )

        depth_for_shape = abs(feats["depth_mean_per_transit"])
        num = shoulder_flux - center_flux

        if (
            np.isfinite(center_flux)
            and np.isfinite(shoulder_flux)
            and np.isfinite(depth_for_shape)
            and depth_for_shape > 0
        ):
            feats["vshape_metric"] = max(0.0, num / depth_for_shape)
        else:
            feats["vshape_metric"] = np.nan
    except Exception:
        feats["vshape_metric"] = np.nan

    feats["secondary_depth"] = compute_secondary_depth(
        time_arr, flux_detr_full, period, t0, duration_days
    )

    # CDPP-based secondary SNR (cadence-invariant, ppm-consistent)
    cdpp_interp_for_duration = interp_cdpp(cdpp, duration_hours)
    if (
        np.isfinite(feats["secondary_depth"])
        and np.isfinite(cdpp_interp_for_duration)
        and cdpp_interp_for_duration > 0
    ):
        sec_snr = float((feats["secondary_depth"] * 1e6) / cdpp_interp_for_duration)
    else:
        sec_snr = np.nan
    feats["secondary_depth_snr"] = sec_snr
    feats["secondary_depth_snr_log"] = (
        float(np.log1p(sec_snr)) if np.isfinite(sec_snr) else np.nan
    )
    feats["secondary_depth_snr_capped"] = (
        float(np.clip(sec_snr, 0.0, 100.0)) if np.isfinite(sec_snr) else np.nan
    )

    # Step 4: Add EB/grazing discriminants
    print("Computing EB/grazing discriminants...")
    feats["odd_even_depth_ratio"] = compute_odd_even_depth_ratio(
        time_arr, flux_detr_full, period, t0, duration_days
    )
    feats["ingress_egress_asymmetry"] = compute_ingress_egress_asymmetry(
        time_arr, flux_detr_full, period, t0, duration_days
    )
    feats["secondary_depth_snr"] = compute_secondary_depth_snr(
        time_arr, flux_detr_full, period, t0, duration_days, feats["local_noise"]
    )

    print(
        f"EB/Grazing features: odd_even_ratio={feats['odd_even_depth_ratio']:.3f}, "
        f"asymmetry={feats['ingress_egress_asymmetry']:.3f}, "
        f"sec_snr={feats['secondary_depth_snr']:.2f}"
    )

    finite_scaled = np.isfinite(flux_scaled)
    if np.sum(finite_scaled) > 2:
        feats["skewness_flux"] = float(skew(flux_scaled[finite_scaled]))
        feats["kurtosis_flux"] = float(kurtosis(flux_scaled[finite_scaled]))
    else:
        feats["skewness_flux"] = np.nan
        feats["kurtosis_flux"] = np.nan
    feats["outlier_resistance"] = (
        float(
            np.sum(np.abs(flux_scaled[finite_scaled]) > 5) / np.sum(finite_scaled) * 100
        )
        if np.sum(finite_scaled) > 0
        else np.nan
    )

    # Note: For CSV data, we don't have stellar radius information
    feats["planet_radius_rearth"] = np.nan
    feats["planet_radius_rjup"] = np.nan

    total_time = time.time() - start_time
    print(f"Feature extraction completed in {total_time:.2f} seconds")
    print(f"Total features extracted: {len(feats)}")

    if verbose:
        print("\n=== Extracted features (ordered) ===")
        for k, v in feats.items():
            print(f"{k}: {v}")

    return feats


def extract_all_features_from_csv(csv_path, verbose=False, include_ml_cutouts=False):
    """Extract features from a CSV file containing light curve data."""
    import pandas as pd

    print(f"Loading light curve data from: {csv_path}")

    # Read the CSV file
    df = pd.read_csv(csv_path)
    time = df["time"].values
    flux = df["flux"].values

    print(f"CSV loaded: {len(time):,} data points")
    print(f"File size: {os.path.getsize(csv_path) / 1024 / 1024:.2f} MB")

    if verbose:
        print(f"Loaded light curve data from: {csv_path}")
        print(f"Data points: {len(time)}")

    # Use the same feature extraction logic as extract_all_features_v2
    feats = extract_features_from_arrays(time, flux, verbose=verbose)
    return feats


def extract_features_from_lightcurve(
    lc, verbose=False, refine_duration=True, use_tls=False
):
    time = lc.time.value
    flux = lc.flux.value

    # Use the internal function for feature extraction
    feats = extract_features_from_arrays(
        time, flux, verbose=verbose, refine_duration=refine_duration, use_tls=use_tls
    )

    # Add stellar radius information if available
    if "RADIUS" in lc.meta and np.isfinite(lc.meta["RADIUS"]):
        stellar_radius = lc.meta["RADIUS"]
        Rp_over_Rs = np.sqrt(feats["depth_mean_per_transit"])
        feats["planet_radius_rearth"] = stellar_radius * 109.1 * Rp_over_Rs
        feats["planet_radius_rjup"] = stellar_radius * 9.95 * Rp_over_Rs

    return feats
