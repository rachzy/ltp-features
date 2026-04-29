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

### How do we know what features to extract

As explained above, light curves are basically (but not just) gigantic CSVs containing flux data of brightness variation of a star.

In order to know what to look for and what to calculate in this data, we use the following resources:

- [NASA Exoplanet Archive for the Kepler Mission](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative)
- [Kepler Science Data Processing Pipeline](https://github.com/nasa/kepler-pipeline)

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
