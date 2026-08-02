import lightkurve as lk
from pathlib import Path
from typing import Optional, Tuple, Union
from save import save_lightkurve
from numpy import inf


# 1 - Dowload and clean light curve


def default_product_selection(mission: str) -> Tuple[Optional[str], Optional[str]]:
    """Return a consistent official product and exposure class per mission."""
    mission_key = str(mission).strip().casefold()
    if mission_key == "kepler":
        return "Kepler", "long"
    if mission_key == "k2":
        return "K2", "long"
    if mission_key == "tess":
        return "SPOC", "short"
    return None, None


def _normalize_exptime(
    exptime: Optional[Union[str, float]],
) -> Optional[Union[str, float]]:
    """Let CLI callers pass either Lightkurve classes or seconds."""
    if not isinstance(exptime, str):
        return exptime
    value = exptime.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return value


def download_and_clean_lightcurve(
    target: str,
    mission: str,
    sigma_upper: float = 5.0,
    all: bool = False,
    savePath: str = None,
    verbose: bool = False,
    author: str = None,
    exptime: Optional[Union[str, float]] = None,
):
    """Download one consistent light-curve product class and clean it.

    Unless explicitly overridden, Kepler/K2 use official long-cadence products
    and TESS uses SPOC short-cadence products.  Filtering happens before
    ``download_all`` so stitching cannot mix exposure times or pipeline
    authors.
    """
    data_dir = Path(__file__).resolve().parent.parent / "data" / "lightkurve"
    data_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Downloading light curve for {target} from {mission}...")
    default_author, default_exptime = default_product_selection(mission)
    selected_author = author if author is not None else default_author
    selected_exptime = _normalize_exptime(
        exptime if exptime is not None else default_exptime
    )
    search_kwargs = {"mission": mission}
    if selected_author is not None:
        search_kwargs["author"] = selected_author
    if selected_exptime is not None:
        search_kwargs["exptime"] = selected_exptime

    search_result = lk.search_lightcurve(target, **search_kwargs)
    if len(search_result) == 0:
        raise ValueError(
            f"No {mission} light curves found for {target!r} with "
            f"author={selected_author!r} and exptime={selected_exptime!r}. "
            "Override author/exptime to select another consistent product."
        )

    if verbose:
        print(
            f"Selected {len(search_result)} product(s): "
            f"author={selected_author or 'any'}, "
            f"exptime={selected_exptime or 'any'}"
        )

    if all:
        if verbose:
            print(f"Downloading all {len(search_result)} selected files...")
        lc = search_result.download_all()
        lc = lc.stitch()
    else:
        lc = search_result.download()

    lc = lc.remove_nans().normalize().remove_outliers(sigma_upper=sigma_upper, sigma_lower=inf)
    print(f"Downloaded {len(lc.time)} data points.")
    if savePath:
        save_lightkurve(lc, target, savePath)

    return lc
