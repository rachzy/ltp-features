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

### Pipeline vs confirmed catalog (sanity check)

Against a small set of confirmed Kepler planets, we compare features written to `data/extracted/` with catalog values in `data/confirmed/` using `src/testing/compare_confirmed_and_extracted.py`. Files are keyed by host star, contain one row per transit candidate, and are sorted and matched by `period_days`. Confirmed files retain a `target` column for the planet label; extracted files do not require one. Each table cell is the percent difference `((extracted − confirmed) / |confirmed|) × 100`, sorted by match quality.

| Candidate   | period_days | depth_mean_per_transit | duration_hours | max_ses | max_mes | duration_days | planet_radius_rjup |       t0 | mean_match_no_t0 |
| ----------- | ----------: | ---------------------: | -------------: | ------: | ------: | ------------: | -----------------: | -------: | ---------------: |
| Kepler-15b  |      +0.00% |                 -5.33% |         -1.64% |  +3.33% | -26.95% |        -1.64% |             +2.37% |    -0.02% |           94.10% |
| Kepler-5b   |      -0.00% |                 -4.04% |         +3.01% | +18.47% | -16.89% |        +3.01% |             +0.05% |    +1.16% |           93.50% |
| Kepler-11g  |      -0.00% |                 +8.49% |         +4.69% |  +3.58% | -10.95% |        +4.69% |            +13.18% |    +1.95% |           93.49% |
| Kepler-186e |      -0.00% |                -11.70% |         +2.08% |  -4.95% | -30.27% |        +2.07% |             +1.69% |    +6.59% |           92.46% |
| Kepler-11f  |      -0.01% |                 +1.71% |        +12.74% |  +7.82% | -10.77% |       +12.74% |             +8.15% |    +0.66% |           92.30% |
| Kepler-11e  |      -0.01% |                 +0.79% |        +13.42% | +22.37% |  -5.49% |       +13.45% |             -0.19% |    +2.65% |           92.04% |
| Kepler-186b |      +0.00% |                -23.67% |         -6.72% |  -2.85% | -16.16% |        -6.74% |             -4.95% |    -6.46% |           91.27% |
| Kepler-7b   |      +0.00% |                 -3.82% |         +2.81% |  +6.44% | -45.05% |        +2.81% |             +0.43% |    -0.45% |           91.23% |
| Kepler-186d |      +0.00% |                 -4.37% |         +4.84% | +11.08% | -35.40% |        +4.84% |             +2.28% |    -2.06% |           91.03% |
| Kepler-2b   |      -0.00% |                 -2.11% |         +6.75% |  -3.32% | -41.20% |        +6.75% |             +2.82% |    +0.64% |           91.01% |
| Kepler-8b   |      -0.00% |                 -3.55% |         +7.21% | +11.19% | -32.46% |        +7.21% |             -6.35% |    +1.40% |           90.29% |
| Kepler-11d  |      +0.01% |                 +2.21% |         +5.70% | +43.18% |  +2.00% |        +5.68% |             +9.96% |    +0.47% |           90.18% |
| Kepler-12b  |      +0.00% |                 -4.36% |         +3.82% | +40.37% | -19.89% |        +3.82% |             -0.72% |    +0.92% |           89.58% |
| Kepler-4b   |      +0.01% |                -18.90% |        +12.45% | +25.62% |  -4.81% |       +12.45% |            +10.99% |    +0.82% |           87.82% |
| Kepler-11c  |      +0.01% |                 -2.34% |         +8.79% | +75.46% |  +5.79% |        +8.81% |             +7.66% |    -0.44% |           84.45% |
| Kepler-186c |      +0.00% |                -16.18% |        -12.54% | +67.00% | -16.04% |       -12.53% |             -0.03% |    -2.65% |           82.24% |
| Kepler-11b  |      +0.00% |                 -2.33% |        +17.56% |+219.76% |  -1.32% |       +17.56% |             +8.72% |    +1.08% |           78.93% |
| Kepler-186f |      -0.29% |                -40.34% |       +191.91% |+498.33% |+134.20% |       +192.12% |            -23.24% | +2004.95% |           33.73% |

<figcaption>
   Percent difference of pipeline-extracted features vs confirmed KOI catalog values (sorted by <code>mean_match_no_t0</code>)
</figcaption>

#### Column meanings

| Column                             | Meaning                                                                                                          |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `period_days`                      | Orbital period from BLS/TLS vs catalog period                                                                    |
| `t0`                               | Transit epoch phase error after mission zero-point conversion and modulo-period alignment, reported as % of transit duration |
| `duration_days` / `duration_hours` | Transit duration in days and hours                                                                               |
| `depth_mean_per_transit`           | Mean per-transit depth vs catalog depth                                                                          |
| `planet_radius_rjup`               | Inferred planet radius (Jupiter radii); `n/a` when stellar radius was unavailable                                |
| `mean_match_no_t0`                 | Mean of `max(0, 100 − \|%diff\|)` over all columns except `t0` (100% = perfect match, 0% = ≥100% off on average) |

**What this suggests:** for quiet, deep hot-Jupiter hosts (Kepler-5b, HAT-P-7, Kepler-7b) the pipeline recovers period and duration to within a few percent and scores above ~96%. Their epochs also agree closely after converting BJD to BKJD and aligning transit numbers. Shallower or harder targets (aliases, grazing geometry, long-period / multi-planet systems such as Kepler-12b, Kepler-186f, Kepler-447) diverge more on depth, radius, and sometimes period—useful stress cases rather than failures of the comparison script itself.

## Using the pipeline to extract lightcurve data

The full pipeline flow and how to use it is explained below.

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
Let's use its host star, **Kepler-5**, as the example in this case:

```python
lc = lk.search_lightcurve(
    "Kepler-5", mission="Kepler", author="Kepler", exptime="long"
)
lc = lc.download_all() # Download all available data for this target (recommended for more precise data)
lc = lc.stitch() # Stitch all downloaded curves into a single one
```

The repository's downloader applies that product filtering automatically before
stitching. Kepler and K2 default to their official long-cadence products; TESS
defaults to SPOC short-cadence products. Use the CLI's `--author` and
`--exptime` options when a different, but still internally consistent, product
set is needed.

Now that we have the light curve, we can use some lightkurve native functions to do some data cleaning:

```python
lc = lc.remove_nans().normalize().remove_outliers(sigma=5.0)
```

With that, we're ready to throw it into our pipeline!

### 3. Pass the light curve into feature extraction

`extract_features_from_lightcurve` in `src/extract_feats.py` reads time and flux from the Lightkurve object and delegates to the same path as CSV input. Every extraction entry point returns a list with one feature dictionary per MES-qualified transit candidate, sorted by period.

```python
time = lc.time.value
flux = lc.flux.value
candidate_rows = extract_features_from_arrays(time, flux, ...)
```

If `RADIUS` is present in `lc.meta`, stellar radius is used to fill planet-radius features; otherwise those fields stay empty.

Candidate discovery is currently an experimental iterative masking search. On
each iteration it computes a BLS periodogram, clusters nearby local maxima,
keeps representatives across the searched period range, and evaluates them in
descending BLS power. Rejecting a candidate rejects only that period, epoch,
and duration neighborhood, so the next independent peak can still be tested.
An accepted candidate must have `max_mes >= 7.1` and at least three observed
events. Its predicted transit windows are then masked at 1.5 times the fitted
duration before the next periodogram is calculated.

The 7.1 threshold is a first experimental stopping rule, not a calibrated
Kepler TCE decision boundary: this pipeline's time-domain MES is not
numerically equivalent to Kepler TPS. The search masks cadences rather than
subtracting a fitted transit model, stops after eight accepted candidates or
50% cumulative masking, and does not yet perform a final joint fit or common
re-detrending pass.

### 4. Detrending and period search

`detrend_with_bls_mask` in `src/detrend_and_period.py` runs first: Box Least Squares (BLS), optional TLS refinement, iterative detrending, and transit masking. After detrending, it repeats a fixed-period/fixed-duration BLS phase fit and canonicalizes `t0` to the first predicted transit in the data interval. It returns detrended flux plus `best_period`, `t0`, and transit duration used everywhere below.

```python
flux_detr, trend, mask_transit, bls_info = detrend_with_bls_mask(
    time_arr, flux_arr, refine_duration=True, use_tls=True
)
```

### 5. Scaling metrics

`scaling_and_metrics` in `src/utils/scaling_and_metrics.py` standardizes the detrended flux and records summary statistics (mean, standard deviation, skewness, kurtosis, outlier resistance) into the feature dict.

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

`per_transit_stats_simple` in `src/per_trans_stat.py` walks each transit epoch, estimates a baseline outside the transit window, and collects per-transit depths and the number of samples actually inside each transit. Those median-based measurements remain inputs to the depth and shape features; SES/MES now use the duration-matched time series described below.

**Execution order vs. in-file labels:** Inside `extract_features_from_arrays`, this block runs _before_ CDPP even though `per_trans_stat.py` is tagged `# 6` and `cdpp.py` is `# 4`. Treat the `# N` lines in source files as module tags, not strict pipeline ordering.

### 8. CDPP

`calculate_cdpp` in `src/cdpp.py` builds box-template depth series lasting 3 h, 6 h, and 12 h. It estimates a robust, time-dependent uncertainty in a local window spanning 30 template durations, retains correlated noise measured at the transit timescale, deweights partial coverage, and prevents the BLS transit windows from contaminating the noise model. Each CDPP feature is the median valid local uncertainty in ppm. These values are a time-domain duration-matched approximation, not an exact reproduction of Kepler's adaptive wavelet whitening.

```python
cdpp = calculate_cdpp(
    flux_detr_full,
    cadence_hours=feats["cadence_hours"],
    time=time_arr,
    exclude_mask=mask_transit,
)
```

### 9. SES, MES, and remaining shape / vetting features

`compute_SES_MES` in `src/sesmes.py` evaluates the exact BLS duration at every eligible cadence. For each measurement it constructs a local depth uncertainty and the coherent matched-filter components `N = depth / uncertainty²`, `D = 1 / uncertainty²`, and `SES = N / sqrt(D)`. `max_ses` is the largest positive cadence-level SES. `MES` folds `N` and `D` at the final BLS ephemeris, while `max_mes` searches only a nearby phase window at that fixed period and duration. `SES_mean` and related per-transit features summarize the exact-ephemeris event SES values. These are scientifically coherent time-domain approximations, not numerical reproductions of Kepler TPS wavelet statistics or a full MES period search.

This change replaces the previous root-sum-square `MES` and global-CDPP SES semantics without changing the saved column names. Existing extracted CSVs and trained artifacts should be regenerated before they are compared or combined with new output.

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

Use `--input-lightkurve path/to.csv` instead of `--target` if you already have a saved light curve file. Optional flags include `--mission` (e.g. `TESS`), `--author`, `--exptime`, `--sigma-clip`, `--download-all`, `--out-lightkurve` to write the downloaded/cleaned curve, and `--quiet`.

### Notebooks

Exploratory workflows live under `src/` (for example `lightcurve_analysis.ipynb`). The script `src/cli/extract_csv.py` is marked deprecated but may still reflect the batch CSV feature layout.

You need network access when downloading data through Lightkurve; first-time use may also pull mission-specific calibration dependencies.

## Credits

Great part of the code in this repository was originally meant for the Gatonautas team project for [Nasa Space Apps Challenge 2025](https://www.nasa.gov/nasa-space-apps-challenge-2025/), in which the author of this repository ([rachzy](github.com/rachzy)) actively participated and was responsible for the pre-processing pipeline.
