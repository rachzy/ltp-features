# Preprocessing Kepler Light Curves with Grazing and Inconsistent Transits for Machine Learning

## 1. Challenges
- **Grazing transits** often produce *V-shaped* light curves rather than U-shaped, making planet radius and impact parameter estimation uncertain.  
- **Inconsistent flux** or variability arises from stellar activity, instrumental noise, or data gaps in Kepler’s long-cadence sampling.  
- These issues complicate the detection and validation of exoplanets using ML classifiers because of weak, irregular, or asymmetric dips.

## 2. Data Cleaning and Detrending
Common strategies include:
- **Removal of systematics** via tools like *KeplerCBV*, *Wotan*, or *Lightkurve’s RegressionCorrector*.
- **Flattening** the light curve to isolate the transit signal using high-pass filters or Savitzky–Golay smoothing.
- **Sigma clipping** or *Gaussian Process regression* to mitigate outliers and stellar variability.

## 3. Normalization and Scaling
- **Median normalization**: divide flux by its median to set baseline at 1.
- **Z-score scaling**: used when preparing for ML to standardize flux across different light curves.
- **Min–max scaling** for CNN-based inputs to maintain consistent amplitude ranges.

## 4. Handling Grazing and Partial Transits
- **Model-based fitting** (e.g., `batman`, `exoplanet`, `PyTransit`) helps identify whether the shape is due to grazing geometry or systematics.  
- **Filtering based on shape metrics** (like transit depth ratio or skewness) before ML classification can improve dataset consistency.

## 5. Preprocessing for ML Pipelines
- **Phase folding** at the candidate period to align transits.  
- **Segment extraction**: keeping a window of ±1.5 transit durations around the event.  
- **Augmentation**: jittering noise or varying phase to improve generalization.  
- **Padding and resampling**: ensuring uniform input length for deep models.

## 6. Tools and Libraries
| Tool | Purpose | Reference |
|------|----------|------------|
| **Lightkurve** | Detrending, normalization, and folding of Kepler/TESS data | (Lightkurve Collab., 2018) |
| **Wotan** | Time-series detrending and outlier removal | Hippke et al., 2019 |
| **Eleanor** | Simplified Kepler/TESS light curve extraction | Feinstein et al., 2019 |
| **Exoplanet** | Probabilistic transit modeling | Foreman-Mackey et al., 2021 |

## 7. References
- Hippke, M. et al. (2019). *Wotan: Comprehensive time-series detrending in Python.*  
- Lightkurve Collaboration (2018). *Lightkurve: Kepler and TESS time series analysis in Python.*  
- Feinstein, A. D. et al. (2019). *Eleanor: An open-source tool for Kepler and TESS light curves.*  
- Foreman-Mackey, D. et al. (2021). *exoplanet: Probabilistic modeling of transit and RV data.*
