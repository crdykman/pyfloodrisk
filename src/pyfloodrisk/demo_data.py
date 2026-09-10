"""Demo data helpers: paths, station listing, and climate data loading."""

from __future__ import annotations

import importlib.resources
import re
from pathlib import Path

import pandas as pd


#: ARR temporal-pattern region bundled for each demo station.  The Data Hub
#: serves patterns per region, not per gauge, so this is the mapping from a
#: station to the ``data/tps/<region>`` directories it draws patterns from:
#: ``<region>`` holds the point patterns and ``Areal_<region>`` the areal
#: ones.
DEMO_TP_REGIONS: dict[str, str] = {
    "117002A": "WT",     # Wet Tropics, QLD
    "405214": "MB",      # Murray Basin, VIC
}

#: Burst durations (hours) the bundled *areal* temporal patterns cover.  ARR
#: publishes areal patterns for long bursts only.
AREAL_TP_DURATIONS_H: tuple[int, ...] = (12, 18, 24, 36, 48, 72, 96, 120, 144, 168)

#: Burst durations (hours) taken from the bundled *point* patterns.  The point
#: files go down to 10 minutes, but a burst shorter than this is not useful on
#: catchments of a few hundred km2, so the short end is cut here.
POINT_TP_DURATIONS_H: tuple[int, ...] = (6, 9)

#: Burst duration (hours) at and above which the areal patterns are used.
#: Below it -- and down to ``min(POINT_TP_DURATIONS_H)`` -- the point patterns
#: are used instead, because ARR publishes no areal pattern that short.
#: 12 h exists in both files; the areal one wins, being the right kind for a
#: catchment rather than a gauge.
AREAL_TP_FROM_H: float = 12.0

#: Every burst duration (hours) the bundled patterns support, point and areal.
TP_DURATIONS_H: tuple[int, ...] = POINT_TP_DURATIONS_H + AREAL_TP_DURATIONS_H

#: ARR areal-reduction-factor region for each demo station, for
#: :class:`~pyfloodrisk.dffa.ifd.ARR2019ARF`.
#:
#: **These are read off the ARR ARF region map by eye and you should confirm
#: them.**  The region is a map lookup, not a formula on the coordinates, and
#: getting it wrong changes every design depth: 117002A sits on the tropical
#: Queensland coast near Townsville (-19.24, 146.63) and 405214 in the
#: Victorian uplands near Mansfield (-37.16, 146.11).
DEMO_ARF_REGIONS: dict[str, str] = {
    "117002A": "East Coast North",
    "405214": "Southern Temperate",
}


def demo_paths() -> dict[str, Path]:
    """Return absolute paths to the bundled demo data directories.

    Returns
    -------
    dict with keys ``root``, ``climate``, ``tps``, ``ifd``, ``storms``,
    ``stations``.
    """
    with importlib.resources.path("pyfloodrisk.data", "__init__.py") as p:
        root = p.parent
    return {
        "root": root,
        "climate": root / "GR4H_climatedata",
        "tps": root / "tps",
        "ifd": root / "ifd",
        "storms": root / "storms",
        "stations": root / "hrs_station_details.csv",
    }


def station_tp_region(station: str) -> str:
    """Return the bundled temporal-pattern region for a demo station."""
    try:
        return DEMO_TP_REGIONS[station]
    except KeyError:
        raise KeyError(
            f"no bundled temporal-pattern region for station {station}; "
            f"known stations are {sorted(DEMO_TP_REGIONS)}"
        ) from None


def station_arf_region(station: str) -> str:
    """Return the ARR areal-reduction region for a demo station.

    See :data:`DEMO_ARF_REGIONS` -- these are map lookups worth confirming.
    """
    try:
        return DEMO_ARF_REGIONS[station]
    except KeyError:
        raise KeyError(
            f"no ARF region recorded for station {station}; known stations "
            f"are {sorted(DEMO_ARF_REGIONS)}"
        ) from None


def _parse_dates(values) -> pd.Series:
    """Parse a demo date column, whichever way round the bundled file writes it.

    The bundled climate files are not consistent: one is ISO
    (``2010-07-06 16:00:00``), the other is day-first
    (``1/01/2012 0:00``).  ISO is tried first because it is unambiguous;
    anything else is read day-first, which is the Australian convention
    these files come in.
    """
    values = pd.Series(values)
    try:
        return pd.to_datetime(values, format="ISO8601")
    except (ValueError, TypeError):
        return pd.to_datetime(values, dayfirst=True)


def list_demo_stations() -> list[str]:
    """Return station IDs available in the bundled demo data."""
    climate_dir = demo_paths()["climate"]
    pattern = re.compile(r"^GR4H_climatedata_(.+)_hr\.csv$")
    return sorted(
        m.group(1)
        for f in climate_dir.iterdir()
        if (m := pattern.match(f.name))
    )


def load_demo_station_data(stations: list[str] | str | None = None) -> pd.DataFrame:
    """Load hourly climate data for one or more stations.

    Parameters
    ----------
    stations:
        Station id or list of ids. Defaults to all bundled stations.

    Returns
    -------
    DataFrame with columns ``Station``, ``Date``, ``PET``, ``PREC``, ``Q``.
    """
    if stations is None:
        stations = list_demo_stations()
    if isinstance(stations, str):
        stations = [stations]

    climate_dir = demo_paths()["climate"]
    frames: list[pd.DataFrame] = []
    for station in stations:
        path = climate_dir / f"GR4H_climatedata_{station}_hr.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing demo climate file for station {station}.")
        raw = pd.read_csv(path, header=0)
        # Columns are matched by name, not position: the bundled files do not
        # agree on the order (one is prec,qt,pet) and getting this wrong
        # silently swaps rainfall for evaporation.
        names = {str(c).strip().lower(): c for c in raw.columns}
        missing = [c for c in ("prec", "pet", "qt") if c not in names]
        if missing:
            raise ValueError(
                f"climate file for station {station} is missing column(s) "
                f"{missing}; got {list(raw.columns)}")
        tbl = pd.DataFrame({
            "Date": _parse_dates(raw[raw.columns[0]]).dt.tz_localize("UTC"),
            "PET": raw[names["pet"]].to_numpy(float),
            "PREC": raw[names["prec"]].to_numpy(float),
            "Q": raw[names["qt"]].to_numpy(float),
        })
        tbl.insert(0, "Station", station)
        frames.append(tbl)

    return pd.concat(frames, ignore_index=True)


def catchment_data(station: str) -> float:
    """Return the catchment area (km2) for a bundled demo station.

    Parameters
    ----------
    station:
        Station id.

    Returns
    -------
    Catchment area in km2.
    """
    metadata = {
        "117002A": {"area": 255.2},
        "405214" : {"area": 357.4},
    }
    return metadata[station]["area"]
