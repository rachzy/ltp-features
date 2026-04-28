import lightkurve as lk
from pathlib import Path
from save import save_lightkurve


# 1 - Dowload and clean light curve


def download_and_clean_lightcurve(
    target: str,
    mission: str,
    sigma_clip: float = 5.0,
    all: bool = False,
    savePath: str = None,
    verbose: bool = False,
):
    """Download light curve data once; reuse cached CSV on subsequent calls."""
    data_dir = Path(__file__).resolve().parent.parent / "data" / "lightkurve"
    data_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Downloading light curve for {target} from {mission}...")
    lc = lk.search_lightcurve(target, mission=mission)

    if all:
        if verbose:
            print(f"Downloading all {len(lc)} available files...")
        lc = lc.download_all()
        lc = lc.stitch()
    else:
        lc = lc.download()

    lc = lc.remove_nans().normalize().remove_outliers(sigma=sigma_clip)
    print(f"Downloaded {len(lc.time)} data points.")
    if savePath:
        save_lightkurve(lc, target, savePath)

    return lc
