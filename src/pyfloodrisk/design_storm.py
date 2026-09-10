"""Build design storms from bundled temporal-pattern increment files."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .demo_data import catchment_data, demo_paths, station_tp_region


# Pre-burst depths for the six standard pre-burst bands (mm)
_PREBURST_24H = [0.0, 0.6, 1.0, 1.3, 8.2, 13.4]


def build_design_storm(
    station: str = "117002A",
    duration_hours: int = 12,
    timestep_minutes: int = 30,
    intensity_mm: float = 107.0,
    aep_band_index: int = 3,
    preburst_index: int = 6,
    preburst_factor: float = 1.0,
) -> dict[str, Any]:
    """Build a design storm from the bundled temporal-pattern increment files.

    Parameters
    ----------
    station:
        Station id.
    duration_hours:
        Storm duration in hours.  The bundled *areal* patterns only cover
        12-168 h, so 12 is the shortest available.
    timestep_minutes:
        Temporal-pattern sub-hourly timestep in minutes.  The bundled areal
        patterns are on a 30-minute step.
    intensity_mm:
        Total storm depth (mm) across the design duration.
    aep_band_index:
        1-based index into the unique AEP values found in the increment
        file.  **Ignored for areal patterns**, which are published per
        standard catchment area with no AEP dependence -- there the
        ensemble is chosen by the station's own catchment area.
    preburst_index:
        1-based index into the pre-burst depth table.
    preburst_factor:
        Multiplier applied to the pre-burst depth.

    Returns
    -------
    dict with keys:

    * ``station``
    * ``intensity_mm``
    * ``preburst_depth``
    * ``temporal_patterns`` – hourly DataFrame (No, Date, Percent)
    * ``rainfall_matrix`` – wide DataFrame (No, pre, <hour columns>)
    * ``rainfall_long`` – long DataFrame (No, Time, Rain)
    """
    root = demo_paths()["root"]
    increment_path = _pick_increment_file(root, station)
    tmp_tp = _read_increment_file(increment_path)

    step_rain = duration_hours * (60 // timestep_minutes)
    duration_minutes = duration_hours * 60

    tmp_tp_dur = tmp_tp[tmp_tp["Duration"] == duration_minutes].copy()
    if tmp_tp_dur.empty:
        available = sorted(pd.to_numeric(tmp_tp["Duration"], errors="coerce")
                           .dropna().unique() / 60.0)
        raise ValueError(
            f"No temporal patterns for a {duration_hours} h storm in "
            f"{increment_path.name}; it covers {available} h. Areal patterns "
            "only go down to 12 h.")

    # Point patterns are published per AEP band, areal patterns per standard
    # catchment area with no AEP dependence at all -- so which column selects
    # the ensemble depends on which kind of file this is.
    key_col = "AEP" if "AEP" in tmp_tp_dur.columns else "Area"
    if key_col == "AEP":
        aep_values = tmp_tp_dur["AEP"].unique().tolist()
        if aep_band_index > len(aep_values):
            raise ValueError("Requested AEP band index is out of range.")
        tp_sub = aep_values[aep_band_index - 1]
    else:
        areas = np.unique(
            pd.to_numeric(tmp_tp_dur["Area"], errors="coerce").dropna())
        tp_sub = min(areas, key=lambda a: abs(a - float(catchment_data(station))))

    inc_cols = [c for c in tmp_tp_dur.columns if c.startswith("Inc")][:step_rain]
    tp_selected = tmp_tp_dur[tmp_tp_dur[key_col] == tp_sub][
        ["EventID", "Duration", "TimeStep", "Region", key_col] + inc_cols
    ].dropna(axis=1, how="all").reset_index(drop=True)
    tp_selected.insert(0, "No", range(1, len(tp_selected) + 1))

    # Rename Inc columns to 1-based integers
    rename = {c: str(i + 1) for i, c in enumerate(inc_cols)}
    tp_wide = tp_selected.rename(columns=rename)
    percent_cols = [str(i + 1) for i in range(step_rain)]

    tp_long = tp_wide[["No"] + percent_cols].melt(
        id_vars="No", var_name="Time", value_name="Percent"
    )
    tp_long["Time"] = tp_long["Time"].astype(int)
    tp_long = tp_long.sort_values(["No", "Time"]).reset_index(drop=True)

    # Assign sub-hourly datetimes then aggregate to hourly
    origin = pd.Timestamp("2000-01-01 00:00:00", tz="UTC")
    datetime_sub = pd.date_range(
        origin, periods=step_rain, freq=f"{timestep_minutes}min"
    )
    time_to_dt = dict(zip(range(1, step_rain + 1), datetime_sub))
    tp_long["Date"] = tp_long["Time"].map(time_to_dt)
    tp_long["hour_key"] = tp_long["Date"].dt.floor("h")

    tp_hourly = (
        tp_long.groupby(["No", "hour_key"], as_index=False)["Percent"].sum()
    )
    tp_hourly.rename(columns={"hour_key": "Date"}, inplace=True)
    tp_hourly = tp_hourly.sort_values(["No", "Date"]).reset_index(drop=True)

    # Pre-burst depth
    if preburst_index > len(_PREBURST_24H):
        raise ValueError("Requested preburst index is out of range.")
    preburst_depth = _PREBURST_24H[preburst_index - 1] * preburst_factor

    # Build wide rainfall matrix (No, pre, hour1, hour2, …)
    tp_pivot = tp_hourly.pivot(index="No", columns="Date", values="Percent")
    rain_matrix = tp_pivot.values / 100.0 * intensity_mm
    n_patterns = rain_matrix.shape[0]

    col_names = ["pre"] + [str(c) for c in tp_pivot.columns]
    rain_full = np.hstack([
        np.full((n_patterns, 1), preburst_depth),
        rain_matrix,
    ])
    rainfall_matrix = pd.DataFrame(rain_full, columns=col_names)
    rainfall_matrix.insert(0, "No", tp_pivot.index.tolist())

    # Long form
    rainfall_long = rainfall_matrix.melt(
        id_vars="No", var_name="Time", value_name="Rain"
    )
    rainfall_long["Time"] = range(len(rainfall_long))  # sequential index
    rainfall_long = (
        rainfall_long.sort_values(["No", "Time"]).reset_index(drop=True)
    )

    return {
        "station": station,
        "intensity_mm": intensity_mm,
        "preburst_depth": preburst_depth,
        "temporal_patterns": tp_hourly,
        "rainfall_matrix": rainfall_matrix,
        "rainfall_long": rainfall_long,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pick_increment_file(root: Path, station: str, kind: str = "areal") -> Path:
    """Locate a station's temporal-pattern increment file.

    Both kinds are bundled per region, so which one you want has to be said
    rather than inferred: ``data/tps/<region>`` holds the point patterns and
    ``data/tps/Areal_<region>`` the areal ones.

    Parameters
    ----------
    root:
        Root of the bundled demo data (see :func:`~pyfloodrisk.demo_paths`).
    station:
        Station id.
    kind:
        ``"areal"`` (default) or ``"point"``.  Areal is the default because
        the demo catchments are a few hundred km2, where a catchment-average
        pattern is the right one; point patterns exist for the short bursts
        ARR publishes no areal pattern for.

    Returns
    -------
    Path to the increment CSV file.
    """
    if kind not in ("point", "areal"):
        raise ValueError(f"kind must be 'point' or 'areal', got {kind!r}")
    region = station_tp_region(station)
    folder = region if kind == "point" else f"Areal_{region}"
    matches = list((root / "tps" / folder).glob("*_Increments.csv"))
    if not matches:
        raise FileNotFoundError(
            f"No {kind} temporal-pattern increment file for station "
            f"{station} (looked in {root / 'tps' / folder})")
    return matches[0]


def _read_increment_file(path: Path) -> pd.DataFrame:
    """Parse a temporal-pattern increment CSV into a tidy DataFrame.

    The bundled increment files have a ragged number of ``Inc*`` columns
    per row, so this reads them as plain text and pads each row to the
    longest one before handing off to pandas.

    Parameters
    ----------
    path:
        Path to the increment CSV file.

    Returns
    -------
    DataFrame with the fixed identifying columns (``EventID``,
    ``Duration``, ``TimeStep``, ``Region``, and ``AEP`` or ``Area``)
    plus one ``Inc{n}`` column per increment.
    """
    is_areal = path.name.startswith("Areal_")
    fixed_names = (
        ["EventID", "Duration", "TimeStep", "Region", "Area"]
        if is_areal
        else ["EventID", "Duration", "TimeStep", "Region", "AEP"]
    )

    with path.open() as fh:
        lines = fh.read().splitlines()

    if len(lines) < 2:
        raise ValueError(f"Increment file is empty: {path}")

    rows = [line.split(",") for line in lines[1:]]
    max_inc = max(len(r) for r in rows) - len(fixed_names)

    records = []
    for row in rows:
        row = [v.strip() for v in row]
        fixed = row[: len(fixed_names)]
        increments = row[len(fixed_names):]
        increments += [""] * (max_inc - len(increments))
        records.append(fixed + increments)

    inc_names = [f"Inc{i + 1}" for i in range(max_inc)]
    df = pd.DataFrame(records, columns=fixed_names + inc_names)

    df["EventID"] = pd.to_numeric(df["EventID"], errors="coerce").astype("Int64")
    df["Duration"] = pd.to_numeric(df["Duration"], errors="coerce").astype("Int64")
    df["TimeStep"] = pd.to_numeric(df["TimeStep"], errors="coerce").astype("Int64")
    if is_areal:
        df["Area"] = pd.to_numeric(df["Area"], errors="coerce")
    for col in inc_names:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df
