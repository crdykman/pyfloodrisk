"""Build design storms from bundled temporal-pattern increment files."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .demo_data import catchment_data, demo_paths, station_tp_region


def resample_increments(inc: np.ndarray, ts_min: float, dt_min: float) -> np.ndarray:
    """Re-express rainfall increments on a new timestep, conserving total mass.

    Works for both aggregation (dt > ts, e.g. 5-minute ARR patterns to hourly
    GR4H) and disaggregation, by linear interpolation of the cumulative mass
    curve.  The pattern duration is padded to a whole number of new timesteps.
    """
    inc = np.asarray(inc, dtype=float)
    total_min = inc.size * ts_min
    n_new = int(np.ceil(total_min / dt_min - 1e-9))
    t_src = np.arange(inc.size + 1) * ts_min
    cum_src = np.concatenate([[0.0], np.cumsum(inc)])
    t_new = np.minimum(np.arange(n_new + 1) * dt_min, total_min)
    cum_new = np.interp(t_new, t_src, cum_src)
    return np.diff(cum_new)

def build_design_storm(
    station: str = "117002A",
    duration_hours: int = 12,
    timestep_minutes: int | None = None,
    intensity_mm: float = 107.0,
    aep_band_index: int = 3,
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
        Increment length of the temporal patterns, in minutes.  Left as
        ``None`` it is read from the file's own ``TimeStep`` column, which
        is what you want: ARR coarsens the step as the storm lengthens (30
        min at 12 h, 60 min at 18-24 h, up to 180 min at 72-168 h), so no
        single value is right for every duration.  Pass one only to
        override a file whose ``TimeStep`` is wrong.
    intensity_mm:
        Total storm depth (mm) across the design duration.
    aep_band_index:
        1-based index into the unique AEP values found in the increment
        file.  **Ignored for areal patterns**, which are published per
        standard catchment area with no AEP dependence -- there the
        ensemble is chosen by the station's own catchment area.

    Returns
    -------
    dict with keys:

    * ``station``
    * ``intensity_mm``
    * ``temporal_patterns`` – hourly DataFrame (No, Date, Percent)
    * ``rainfall_matrix`` – wide DataFrame, one row per temporal pattern:
      ``No`` then one column per hour of the storm, in mm
    """
    root = demo_paths()["root"]
    increment_path = _pick_increment_file(root, station)
    tmp_tp = _read_increment_file(increment_path)

    duration_minutes = duration_hours * 60

    tmp_tp_dur = tmp_tp[tmp_tp["Duration"] == duration_minutes].copy()
    if tmp_tp_dur.empty:
        available = sorted(pd.to_numeric(tmp_tp["Duration"], errors="coerce")
                           .dropna().unique() / 60.0)
        raise ValueError(
            f"No temporal patterns for a {duration_hours} h storm in "
            f"{increment_path.name}; it covers {available} h. Areal patterns "
            "only go down to 12 h.")

    # ARR publishes a coarser increment as the storm lengthens, and the file
    # states which one it used -- guessing a constant here silently truncates
    # the pattern, or asks for increment columns that do not exist
    ts_min = float(timestep_minutes if timestep_minutes is not None
                   else tmp_tp_dur["TimeStep"].iloc[0])
    if not np.isfinite(ts_min) or ts_min <= 0:
        raise ValueError(
            f"{increment_path.name} gives a timestep of {ts_min} min at "
            f"{duration_hours} h; pass timestep_minutes to override it")
    step_rain = int(round(duration_minutes / ts_min))

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

    all_inc = [c for c in tmp_tp_dur.columns if c.startswith("Inc")]
    n_filled = int(tmp_tp_dur[all_inc].notna().any(axis=0).sum())
    if step_rain != n_filled:
        raise ValueError(
            f"a {duration_hours} h storm on a {ts_min:g} min step needs "
            f"{step_rain} increments, but {increment_path.name} holds "
            f"{n_filled} at that duration (its own TimeStep is "
            f"{float(tmp_tp_dur['TimeStep'].iloc[0]):g} min). Leave "
            "timestep_minutes as None to take the file's own step.")
    inc_cols = all_inc[:step_rain]
    tp_selected = tmp_tp_dur[tmp_tp_dur[key_col] == tp_sub][
        ["EventID", "Duration", "TimeStep", "Region", key_col] + inc_cols
    ].dropna(axis=1, how="all").reset_index(drop=True)
    tp_selected.insert(0, "No", range(1, len(tp_selected) + 1))

    # Re-express the increments on the hourly step the model runs at.  A
    # plain floor-to-the-hour and sum only works while an increment fits
    # inside an hour; above that it drops a whole multi-hour block's depth
    # into the block's first hour.  resample_increments interpolates the
    # cumulative mass curve instead, so it aggregates and disaggregates
    # alike and conserves the total.
    pct = tp_selected[inc_cols].to_numpy(float)
    hourly_pct = np.vstack([resample_increments(row, ts_min, 60.0)
                            for row in pct])

    origin = pd.Timestamp("2000-01-01 00:00:00", tz="UTC")
    hours = pd.date_range(origin, periods=hourly_pct.shape[1], freq="h")

    tp_hourly = pd.DataFrame({
        "No": np.repeat(tp_selected["No"].to_numpy(), hourly_pct.shape[1]),
        "Date": np.tile(hours, hourly_pct.shape[0]),
        "Percent": hourly_pct.ravel(),
    })

    # Build wide rainfall matrix (No, hour1, hour2, …)
    rain_matrix = hourly_pct / 100.0 * intensity_mm
    col_names = [str(c) for c in hours]
    rainfall_matrix = pd.DataFrame(rain_matrix, columns=col_names)
    rainfall_matrix.insert(0, "No", tp_selected["No"].tolist())

    return {
        "station": station,
        "intensity_mm": intensity_mm,
        "temporal_patterns": tp_hourly,
        "rainfall_matrix": rainfall_matrix,
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
