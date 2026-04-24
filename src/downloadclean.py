import numpy as np
import lightkurve as lk
from datetime import datetime
import pandas as pd
from pathlib import Path



# 1 - Dowload and clean light curve

def download_and_save_lightcurve(target: str, mission: str, sigma_clip: float, verbose: bool):
    """Download light curve data once; reuse cached CSV on subsequent calls."""
    data_dir = Path(__file__).resolve().parent.parent / "data" / "lightkurve"
    data_dir.mkdir(parents=True, exist_ok=True)

    star_name = target.replace(" ", "_").replace("-", "_")
    existing_files = sorted(data_dir.glob(f"{star_name}_*.csv"))
    if existing_files:
        csv_path = existing_files[-1]
        if verbose:
            print(f"Loading cached light curve from: {csv_path}")
        df = pd.read_csv(csv_path)
        flux_err = None
        if "flux_err" in df.columns and df["flux_err"].notna().any():
            flux_err = df["flux_err"].to_numpy()
        lc = lk.LightCurve(
            time=df["time"].to_numpy(),
            flux=df["flux"].to_numpy(),
            flux_err=flux_err,
        )
        return str(csv_path), lc

    if verbose:
        print(f"Downloading light curve for {target} from {mission}...")
    lc = lk.search_lightcurve(target, mission=mission).download()
    # lc = lc.download_all()
    # lc = lc.stitch()
    
    lc = lc.remove_nans().normalize().remove_outliers(sigma=sigma_clip)
    print(f"Downloaded {len(lc.time)} data points.")
    current_date = datetime.now().strftime("%Y%m%d")
    csv_path = data_dir / f"{star_name}_{current_date}.csv"
    df = pd.DataFrame({
        "time": lc.time.value,
        "flux": lc.flux.value,
        "flux_err": lc.flux_err.value if hasattr(lc, "flux_err") and lc.flux_err is not None else np.nan,
    })
    df.to_csv(csv_path, index=False)
    if verbose:
        print(f"Saved light curve data to: {csv_path}")
    return str(csv_path), lc

def download_and_clean(target, mission, sigma_clip=5.0):
    lc = lk.search_lightcurve(target, mission=mission).download()
    lc = lc.remove_nans()
    lc = lc.normalize()
    lc = lc.remove_outliers(sigma=sigma_clip)
    return lc
