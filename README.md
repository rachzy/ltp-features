# LTP - Features

## Description

**Lightcurve Transit Photometry (LTP) - Features** is a python project for extracting features from light curve data useful for detecting exoplanets through the [Transit Photometry](https://en.wikipedia.org/wiki/Methods_of_detecting_exoplanets#Transit_photometry) method.

At first, it was intended to be pre-processing pipeline for digesting features to be consumed by ML model, but keep in mind that ,this is **NOT** a ML model, although it can be used as a pipeline for features digestion.

### What is Transit Photometry

When a planet passes in front of its host star along our line of sight, it blocks a small fraction of the star’s light. Photometers record a periodic dip in brightness whose depth scales roughly with the squared ratio of planet radius to stellar radius, and whose spacing in time reveals the orbital period. Surveys such as Kepler and TESS monitor many stars for these repeating dimmings; candidates from such data still need careful vetting (stellar activity, eclipsing binaries, instrumental effects) before they can be treated as planet detections.

<img width="487" height="271" alt="image" src="https://github.com/user-attachments/assets/0554c00d-14d5-44e3-9967-f115ecfa5ebc" />
<figcaption>
   Transit Photometry Example - <a href="https://www.apus.edu/academic-community/space-studies/exoplanet-transit-photometry/" target="_blank">Apus - Exoplanet Transit Photometry</a>
</figcaption>

### What are light curves

A **light curve** is the brightness of a star (or other source) measured over time—typically flux in detector units or normalized flux versus time in days. Space-based transit missions produce long, evenly or nearly evenly sampled series per target, often with millions of points. This repository treats those time and flux arrays as the raw input: cleaning, detrending, period search (e.g. BLS/TLS), folding at the trial period, and numerical summaries that describe shape, noise, and consistency of the signal.

<img width="288" height="180" alt="image" src="https://github.com/user-attachments/assets/e67310c7-315c-41ae-8c86-334a47d6caac" />
<figcaption>
   Lightcurve Example - <a href="https://imagine.gsfc.nasa.gov/features/yba/M31_velocity/lightcurve/lightcurve_more.html" target="_blank">Nasa - Imagine the Universe!</a>
</figcaption>

## Pipeline and Extraction

As explained above, light curves are basically (but not just) gigantic CSVs containing flux data of brightness variation of a star.

In order to know what to look for and what to calculate in this data, we use the following resources:

- [NASA Exoplanet Archive for the Kepler Mission](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative)
- [Kepler Science Data Processing Pipeline](https://github.com/nasa/kepler-pipeline)

The full pipeline flow is explained below.

### 1. Choose a target

According to lightkurve's official documentation, a target can be any one of the following

> - The name of the object as a string, e.g. “Kepler-10”.
> - The KIC or EPIC identifier as an integer, e.g. 11904151.
> - A coordinate string in decimal format, e.g. “285.67942179 +50.24130576”.
> - A coordinate string in sexagesimal format, e.g. “19:02:43.1 +50:14:28.7”.
> - An astropy.coordinates.SkyCoord object.

A specific mission can also be specified. We use "Kepler" by default.

### 2. Download and clean light curve data

After choosing the target, we first have to download its curve data.
Let's use **Kepler-5b** as example in this case:

```python
lc = lk.search_lightcurve("Kepler-5b", mission="Kepler")
lc = lc.download_all() # Download all available data for this target (recommended for more precise data)
lc = lc.stitch() # Stitch all downloaded curves into a single one
```

Now that we have the light curve, we can use some lightkurve native functions to do some data cleaning:

```python
lc = lc.remove_nans().normalize().remove_outliers(sigma=5.0)
```

With that, we're ready to throw it into our pipeline!

### 3. Pass the light curve into feature extraction

`extract_features_from_lightcurve` in `src/extract_feats.py` reads time and flux from the Lightkurve object and delegates to the same path as CSV input:

```python
time = lc.time.value
flux = lc.flux.value
feats = extract_features_from_arrays(time, flux, ...)
```

If `RADIUS` is present in `lc.meta`, stellar radius is used to fill planet-radius features; otherwise those fields stay empty.

### 4. Detrending and period search

`detrend_with_bls_mask` in `src/detrend_and_period.py` runs first: Box Least Squares (BLS), optional TLS refinement, iterative detrending, and transit masking. It returns detrended flux plus `best_period`, `t0`, and transit duration used everywhere below.

```python
flux_detr, trend, mask_transit, bls_info = detrend_with_bls_mask(
    time_arr, flux_arr, refine_duration=True, use_tls=True
)
```

### 5. Scaling metrics

`scaling_and_metrics` in `src/utils.py` standardizes the detrended flux and records summary statistics (mean, standard deviation, skewness, kurtosis, outlier resistance) into the feature dict.

### 6. Folded and binned metrics

`folded_binned_metrics` in `src/folded_binned_metrics.py` folds the series in phase at the BLS period and `t0`, builds a median phase profile, estimates a transit width in phase, then computes:

- **Cadence** from median short time steps (fed into CDPP later).
- **`local_noise`**: robust scatter (MAD) using out-of-transit points.
- **`depth_stability`**: how much per-epoch transit depths vary relative to the global folded depth.
- **`acf_lags`**: flux autocorrelation at configured hour lags (e.g. 1–24 h).

```python
binned = folded_binned_metrics(
    time_arr, flux_detr_full, period, t0, lags_hours=(1, 3, 6, 12, 24)
)
```

### 7. Per-transit statistics

`per_transit_stats_simple` in `src/per_trans_stat.py` walks each transit epoch, estimates a baseline outside the transit window, and collects per-transit depths and in-transit point counts. Those arrays feed SES/MES and several downstream features.

**Execution order vs. in-file labels:** Inside `extract_features_from_arrays`, this block runs _before_ CDPP even though `per_trans_stat.py` is tagged `# 6` and `cdpp.py` is `# 4`. Treat the `# N` lines in source files as module tags, not strict pipeline ordering.

### 8. CDPP

`calculate_cdpp` in `src/cdpp.py` median-normalizes the detrended flux, applies a moving uniform smooth whose window length matches 3 h, 6 h, and 12 h in samples (using `cadence_hours` from the folded metrics step), and stores residual RMS values in parts per million (`cdpp_3h`, `cdpp_6h`, `cdpp_12h`). Those values feed interpolated CDPP, global SNR vs depth, and secondary-depth SNR logic in the same extractor.

```python
cdpp = calculate_cdpp(flux_detr_full, cadence_hours=feats["cadence_hours"])
```

### 9. SES, MES, and remaining shape / vetting features

`compute_SES_MES` in `src/sesmes.py` (in-file comment: `# 5 - Compute SES and MES`) combines per-transit depths, local noise, point counts, and the CDPP dictionary into per-transit SES and an aggregate MES, which appear in the feature dict as SES statistics and MES.

The same extraction pass then adds folded **v-shape** metrics, **secondary eclipse** depth (and CDPP-based SNR variants), **odd/even depth ratio**, **ingress/egress asymmetry**, global residual RMS, and **skewness/kurtosis** on scaled flux, all still anchored to the BLS period, epoch, and duration from step 4.

## How to run the project

1. **Python environment** — Use Python 3.9+ (or whatever your stack expects), create a virtual environment, and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r src/requirements.txt
   ```

### CLI

From the repository root, run the CLI so `src` stays on the import path:

```bash
python src/cli/extract_lk.py --target HAT-P-7 --mission Kepler --out-features out/hatp7_features.csv
```

Use `--input-lightkurve path/to.csv` instead of `--target` if you already have a saved light curve file. Optional flags include `--mission` (e.g. `TESS`), `--sigma-clip`, `--download-all`, `--out-lightkurve` to write the downloaded/cleaned curve, and `--quiet`.

### Notebooks

Exploratory workflows live under `src/` (for example `lightcurve_analysis.ipynb`). The script `src/cli/extract_csv.py` is marked deprecated but may still reflect the batch CSV feature layout.

You need network access when downloading data through Lightkurve; first-time use may also pull mission-specific calibration dependencies.

## Credits

Great part of the code in this repository was originally meant for the Gatonautas team project for [Nasa Space Apps Challenge 2025](https://www.nasa.gov/nasa-space-apps-challenge-2025/), in which the author of this repository ([rachzy](github.com/rachzy)) actively participated and was responsible for the pre-processing pipeline.
