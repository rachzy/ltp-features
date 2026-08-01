"""Helpers for keeping file and search targets keyed by host star."""

from __future__ import annotations

import re


_LETTERED_PLANET_RE = re.compile(r"^(?P<star>.*\d)[b-z]$", re.IGNORECASE)
_NUMBERED_CANDIDATE_RE = re.compile(r"^(?P<star>(?:EPIC|KIC|TIC)\s*\d+)\.\d+$", re.IGNORECASE)


def host_star_name(target: str) -> str:
    """Return the host-star portion of a planet or candidate identifier.

    Examples include ``Kepler-186f`` -> ``Kepler-186`` and
    ``EPIC201111557.01`` -> ``EPIC201111557``. Names that already identify a
    star, such as ``Kepler-186`` or ``HAT-P-7``, are returned unchanged.
    """
    name = str(target).strip()

    numbered_match = _NUMBERED_CANDIDATE_RE.fullmatch(name)
    if numbered_match:
        return numbered_match.group("star")

    lettered_match = _LETTERED_PLANET_RE.fullmatch(name)
    if lettered_match:
        return lettered_match.group("star")

    return name
