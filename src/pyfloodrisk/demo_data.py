"""Demo data helpers: paths, station listing, and climate data loading."""

from __future__ import annotations

import importlib.resources
import re
from pathlib import Path

import pandas as pd


def demo_paths() -> dict[str, Path]:
    """Return absolute paths to the bundled demo data directories.

    Returns
    -------
    dict with keys ``root``, ``climate``, ``storms``, ``stations``.
    """
    with importlib.resources.path("pyfloodrisk.data", "__init__.py") as p:
        root = p.parent
    return {
        "root": root,
        "climate": root / "GR4H_climatedata",
        "storms": root / "storms",
        "stations": root / "hrs_station_details.csv",
    }


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
        tbl = pd.read_csv(path, header=0)
        tbl.columns = ["Date", "PET", "PREC", "Q"]
        tbl["Date"] = pd.to_datetime(tbl["Date"], format="%m/%d/%Y %H:%M", utc=True)
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
        "421026": {"area": 149.689},
        "418005": {"area": 237.7235},
    }
    return metadata[station]["area"]
