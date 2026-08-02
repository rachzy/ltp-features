"""Helpers for comparing and normalizing periodic transit epochs."""

from __future__ import annotations

import numpy as np


MISSION_TIME_OFFSETS = {
    "kepler": 2_454_833.0,
    "k2": 2_454_833.0,
    "tess": 2_457_000.0,
}


def canonical_epoch(epoch: float, period: float, reference_time: float) -> float:
    """Return the epoch equivalent to ``epoch`` in ``[reference, reference + P)``.

    A transit epoch is only defined modulo the orbital period.  Canonicalizing
    it makes results stable when different search/refinement stages happen to
    return different members of the same ``t0 + n * period`` sequence.
    """
    epoch = float(epoch)
    period = float(period)
    reference_time = float(reference_time)
    if not (
        np.isfinite(epoch)
        and np.isfinite(period)
        and period > 0
        and np.isfinite(reference_time)
    ):
        return float("nan")
    return float(reference_time + np.mod(epoch - reference_time, period))


def epoch_phase_offset(epoch: float, reference_epoch: float, period: float) -> float:
    """Return the shortest signed ``epoch - reference_epoch`` offset modulo P."""
    epoch = float(epoch)
    reference_epoch = float(reference_epoch)
    period = float(period)
    if not (
        np.isfinite(epoch)
        and np.isfinite(reference_epoch)
        and np.isfinite(period)
        and period > 0
    ):
        return float("nan")
    return float(np.mod(epoch - reference_epoch + 0.5 * period, period) - 0.5 * period)


def _mission_from_target(target: str | None) -> str | None:
    if target is None:
        return None
    name = str(target).strip().casefold()
    if name.startswith(("kepler", "koi", "kic")):
        return "kepler"
    if name.startswith(("k2", "epic")):
        return "k2"
    if name.startswith(("tess", "toi", "tic")):
        return "tess"
    return None


def align_catalog_epoch(
    catalog_epoch: float,
    data_epoch: float,
    period: float,
    *,
    target: str | None = None,
) -> float:
    """Express a catalog epoch in the data epoch's zero point and transit cycle.

    Kepler light curves normally use BKJD while catalog exports are sometimes
    stored as full BJD; TESS has the analogous BTJD convention.  The conversion
    is applied only when one value is a full Julian date and the other is a
    reduced date.  The converted catalog epoch is then shifted by an integer
    number of periods to the transit nearest ``data_epoch``.
    """
    catalog_epoch = float(catalog_epoch)
    data_epoch = float(data_epoch)
    period = float(period)
    if not (
        np.isfinite(catalog_epoch)
        and np.isfinite(data_epoch)
        and np.isfinite(period)
        and period > 0
    ):
        return float("nan")

    mission = _mission_from_target(target)
    offset = MISSION_TIME_OFFSETS.get(mission) if mission is not None else None
    catalog_is_jd = abs(catalog_epoch) > 1_000_000.0
    data_is_jd = abs(data_epoch) > 1_000_000.0
    if offset is not None and catalog_is_jd != data_is_jd:
        catalog_epoch += -offset if catalog_is_jd else offset

    delta = epoch_phase_offset(catalog_epoch, data_epoch, period)
    return float(data_epoch + delta)
