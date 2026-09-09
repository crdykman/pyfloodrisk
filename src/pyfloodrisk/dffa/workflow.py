"""
Bridges between the rest of ``pyfloodrisk`` and the derived-FFA machinery.

The Monte Carlo framework needs four inputs.  Three of them are things
``pyfloodrisk`` already has, and this module adapts them:

===========================  ==========================================
framework input              where it comes from
===========================  ==========================================
GR4H event engine            a calibrated parameter set, run through
                             :class:`~pyfloodrisk.dffa.engine.GR4HEventEngine`
initial state distribution   a continuous GR4H run over the station's
                             climate record (:func:`continuous_state_table`)
temporal patterns            the station's ARR Data Hub increments file
                             (:func:`station_patterns`)
design rainfall (IFD)        **not** bundled -- supply BoM depths, or
                             fit the record (:func:`station_ifd`)
===========================  ==========================================

:func:`run_dffa` wires all four together for a bundled demo station, which
is the shortest path to a working example; for real work assemble the four
inputs yourself and call :class:`~pyfloodrisk.dffa.mcs.DerivedFFA` directly,
so that each input is one you have checked.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..demo_data import catchment_data, demo_paths
from ..design_storm import _pick_increment_file
from ..gr4h.GR4H_model import GR4H
from .engine import GR4HEventEngine
from .ifd import DEFAULT_IFD_AEPS, IFDCurve, ifd_from_record
from .mcs import STANDARD_DURATIONS_H, DerivedFFA, MCSConfig
from .patterns import TemporalPatternLibrary
from .states import InitialStateSampler, PETClimatology
from .stratification import Stratification

__all__ = [
    "load_station_forcings",
    "continuous_state_table",
    "event_onset_states",
    "state_sampler_from_run",
    "pet_climatology",
    "station_patterns",
    "station_ifd",
    "run_dffa",
    "DEMO_PARAMETERS",
]

#: Plausible-but-uncalibrated GR4H parameters, so the demo runs without a
#: calibration step.  Calibrate before reading anything into the numbers.
DEMO_PARAMETERS: dict[str, dict[str, float]] = {
    "421026": {"ps0": 0.5, "rs0": 0.5, "x1": 350.0, "x2": -0.8,
               "x3": 60.0, "x4": 6.0},
    "418005": {"ps0": 0.5, "rs0": 0.5, "x1": 400.0, "x2": -1.0,
               "x3": 55.0, "x4": 9.0},
}


# --------------------------------------------------------------- forcings
def load_station_forcings(station: str = "421026") -> pd.DataFrame:
    """Load a bundled station's hourly climate record.

    Returns a DataFrame indexed by timestamp with columns ``prec``, ``pet``
    and (where present) ``qt``, i.e. the layout ``GR4H.run`` expects.  The
    two bundled files name their date column differently, which is why this
    is not just a ``read_csv``.
    """
    path = demo_paths()["climate"] / f"GR4H_climatedata_{station}_hr.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing demo climate file for station {station}.")
    df = pd.read_csv(path)
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], format="%m/%d/%Y %H:%M")
    return df.set_index(date_col).rename_axis("date")


# ------------------------------------------------------- state distribution
def continuous_state_table(
    forcings: pd.DataFrame,
    parameters: Mapping[str, float],
    area_km2: float,
    warmup_hours: int = 8760,
    thin: int = 1,
    wet_only: bool = False,
    min_depth_mm: float = 0.0,
) -> pd.DataFrame:
    """Run GR4H continuously and return its state at every timestep.

    This is the input with no ARR equivalent: the distribution the design
    events' antecedent conditions are drawn from.  One row per retained
    timestep, holding the production and routing stores in mm, the
    unit-hydrograph memory, and the date -- exactly what
    :class:`~pyfloodrisk.dffa.states.InitialStateSampler` consumes.

    Parameters
    ----------
    forcings :
        Hourly ``prec``/``pet`` (mm), datetime-indexed; see
        :func:`load_station_forcings`.
    parameters :
        GR4H parameters.  ``x4`` fixes the length of the UH memory, so the
        table is only valid for the parameter set that produced it.
    area_km2 :
        Catchment area (only affects the reported discharge).
    warmup_hours :
        Leading hours discarded, so the stores are not still relaxing from
        their arbitrary initial values.  One year by default.
    thin :
        Keep every ``thin``-th row.  The states of consecutive hours are
        nearly identical, so thinning costs almost no information and makes
        the sampler's donor pool cheaper to hold and to fit models to.
    wet_only :
        Keep only rows whose *following* timestep has rainfall.  Turns the
        table into a distribution of states at the onset of rain rather
        than at an arbitrary hour, which is closer to the conditioning
        implied by sampling the state at the start of a burst.
    min_depth_mm :
        Rainfall threshold used by ``wet_only``.

    Returns
    -------
    DataFrame with columns ``date``, ``prod_store``, ``rout_store``,
    ``q_cumecs``, ``prec_next``, ``uh1_0..``, ``uh2_0..``.

    Notes
    -----
    The state on row ``t`` is the state *after* hour ``t`` has been routed,
    so an event started from it begins at hour ``t + 1``: the same
    convention as :func:`~pyfloodrisk.extract_initial_states`.
    """
    model = GR4H(area=area_km2, params=dict(parameters))
    out = model.run_from_state(
        forcings["prec"].to_numpy(float),
        forcings["pet"].to_numpy(float),
        record_uh=True,
    )

    n = len(forcings)
    prec = forcings["prec"].to_numpy(float)
    prec_next = np.concatenate([prec[1:], [np.nan]])

    table = pd.DataFrame({
        "date": forcings.index,
        "prod_store": out["prod_store"],
        "rout_store": out["rout_store"],
        "q_cumecs": out["qt_cumecs"],
        "prec_next": prec_next,
    })
    for j in range(out["uh1"].shape[1]):
        table[f"uh1_{j}"] = out["uh1"][:, j]
    for j in range(out["uh2"].shape[1]):
        table[f"uh2_{j}"] = out["uh2"][:, j]

    keep = np.zeros(n, dtype=bool)
    keep[int(warmup_hours):] = True
    if thin > 1:
        thinned = np.zeros(n, dtype=bool)
        thinned[::int(thin)] = True
        keep &= thinned
    if wet_only:
        keep &= np.nan_to_num(prec_next, nan=-1.0) > float(min_depth_mm)
    table = table.loc[keep].reset_index(drop=True)
    if table.empty:
        raise ValueError("no states left after warm-up/thinning/wet-only filtering")
    return table


def event_onset_states(state_table: pd.DataFrame,
                       event_method: str = "maxima",
                       method_kwargs: Mapping[str, Any] | None = None,
                       pre_event: int = 24,
                       alpha: float = 0.925) -> pd.DataFrame:
    """Restrict a state table to the onset of each simulated flood event.

    Runs the package's event delineation
    (:func:`~pyfloodrisk.hydro_event_pipeline`) over the continuous run's
    own discharge and keeps the antecedent state ahead of each delineated
    event, exactly as :func:`~pyfloodrisk.extract_initial_states` does for
    the single-event design flood workflow.

    This is the other reading of "the distribution of antecedent states":
    not the state at an arbitrary hour, but the state at the start of the
    events the model actually produces.  It is a much smaller donor pool --
    one row per event rather than one per hour -- so it is the case where
    the ``smoothed`` or copula sampling methods earn their keep over
    ``bootstrap``.

    Parameters
    ----------
    state_table :
        Output of :func:`continuous_state_table`, **unthinned**: the
        baseflow filter and the event delineation both assume consecutive
        hourly values.
    event_method, method_kwargs, alpha :
        Passed to :func:`~pyfloodrisk.hydro_event_pipeline`.
    pre_event :
        Hours ahead of each peak to search for the onset; passed to
        :func:`~pyfloodrisk.extract_initial_states`.

    Returns
    -------
    The rows of ``state_table`` at the delineated event onsets, with an
    added ``event_id`` column.
    """
    from ..hydroevents import extract_initial_states, hydro_event_pipeline

    gaps = pd.Series(state_table["date"]).diff().dropna().unique()
    if len(gaps) > 1 or (len(gaps) == 1 and gaps[0] != pd.Timedelta(hours=1)):
        raise ValueError("event delineation needs an unthinned, hourly state "
                         "table; call continuous_state_table with thin=1")

    _, events = hydro_event_pipeline(
        state_table["q_cumecs"].to_numpy(float), event_method=event_method,
        method_kwargs=dict(method_kwargs) if method_kwargs else None,
        alpha=alpha)
    if events.empty:
        raise ValueError("event delineation found no events in the run")

    stores = np.vstack([state_table["prod_store"].to_numpy(float),
                        state_table["rout_store"].to_numpy(float)])
    onsets = extract_initial_states(stores, events, pre_event=pre_event)
    out = state_table.iloc[onsets["state_index"].to_numpy()].copy()
    out.insert(0, "event_id", onsets["event_id"].to_numpy())
    return out.reset_index(drop=True)


def state_sampler_from_run(state_table: pd.DataFrame,
                           parameters: Mapping[str, float],
                           method: str = "bootstrap",
                           **kwargs) -> InitialStateSampler:
    """Build an :class:`InitialStateSampler` over a continuous-run state table.

    Picks up the UH column names from the table and the store capacities
    from the parameter set, which is the part that is easy to get wrong: the
    UH memory has to be the length the engine expects, and ``x1``/``x3`` have
    to be the ones the states were generated with.
    """
    uh1_cols = sorted((c for c in state_table.columns if c.startswith("uh1_")),
                      key=lambda c: int(c.split("_")[1]))
    uh2_cols = sorted((c for c in state_table.columns if c.startswith("uh2_")),
                      key=lambda c: int(c.split("_")[1]))
    return InitialStateSampler(
        state_table, x1=float(parameters["x1"]), x3=float(parameters["x3"]),
        date_col="date", uh1_cols=uh1_cols or None, uh2_cols=uh2_cols or None,
        method=method, **kwargs)


def pet_climatology(forcings: pd.DataFrame, diurnal: str = "sine"
                    ) -> PETClimatology:
    """Monthly mean daily PET from a station's record."""
    pet = pd.Series(forcings["pet"].to_numpy(float), index=forcings.index)
    monthly = pet.groupby(pet.index.month).mean() * 24.0     # mm/h -> mm/d
    values = np.array([float(monthly.get(m, np.nan)) for m in range(1, 13)])
    if not np.isfinite(values).all():
        values = np.where(np.isfinite(values), values, np.nanmean(values))
    return PETClimatology(monthly_mm_per_day=values, diurnal=diurnal)


# ------------------------------------------------------------ rainfall inputs
def station_patterns(station: str = "421026",
                     increments_path: str | Path | None = None
                     ) -> TemporalPatternLibrary:
    """ARR temporal pattern ensembles for a bundled station.

    Reads the same Data Hub increments file
    :func:`~pyfloodrisk.build_design_storm` uses, but keeps the whole
    ensemble for every duration and AEP band instead of selecting one.
    """
    if increments_path is None:
        increments_path = _pick_increment_file(demo_paths()["root"], station)
    return TemporalPatternLibrary.from_arr_increments_csv(increments_path)


def station_ifd(station: str = "421026",
                durations_h: Sequence[float] = STANDARD_DURATIONS_H,
                aeps: Sequence[float] = DEFAULT_IFD_AEPS,
                arf=None, **fit_kwargs) -> IFDCurve:
    """Design rainfall curve fitted to a bundled station's rainfall record.

    A convenience for the demo only.  No design rainfalls are bundled with
    the package, and a fit to six years of one pluviograph is not one: read
    the warning on :func:`~pyfloodrisk.dffa.ifd.ifd_from_record` before
    using the result for anything but a demonstration.
    """
    forcings = load_station_forcings(station)
    table = ifd_from_record(forcings["prec"].to_numpy(float), durations_h,
                            dt_hours=1.0, aeps=aeps, **fit_kwargs)
    return IFDCurve(table, arf=arf)


# ------------------------------------------------------------------ workflow
def run_dffa(
    station: str = "421026",
    parameters: Mapping[str, float] | None = None,
    durations_h: Sequence[float] = (3, 6, 12, 24, 48, 72),
    ifd: IFDCurve | None = None,
    stratification: Stratification | None = None,
    state_method: str = "bootstrap",
    thin: int = 6,
    warmup_hours: int = 8760,
    seed: int = 20260909,
    progress: bool = True,
    **config_kwargs: Any,
) -> dict[str, Any]:
    """Derived flood frequency analysis for a bundled demo station.

    Runs the whole chain: continuous GR4H over the station's climate record,
    the state distribution from that run, temporal patterns from the
    station's increments file, design rainfalls fitted to the station's own
    record, then the stratified Monte Carlo.

    Parameters
    ----------
    station :
        Bundled station id.
    parameters :
        GR4H parameters.  Defaults to :data:`DEMO_PARAMETERS`, which is a
        plausible set rather than a calibrated one -- calibrate first
        (:func:`~pyfloodrisk.calibration`) and pass the result for anything
        you intend to look at twice.
    durations_h :
        Storm durations to envelope over.  Fewer than the ARR standard set,
        because the demo should finish in a minute or two.
    ifd :
        Design rainfall curve.  Defaults to :func:`station_ifd`, which is a
        fit to the station's own record and **not** a design IFD.
    stratification :
        Rainfall sampling scheme; defaults to 25 intervals x 60 simulations
        over 90% to 1 in 10^5 AEP.
    state_method :
        Initial-state sampling method; see
        :class:`~pyfloodrisk.dffa.states.InitialStateSampler`.
    thin, warmup_hours :
        Passed to :func:`continuous_state_table`.
    seed, progress :
        Passed to :class:`~pyfloodrisk.dffa.mcs.MCSConfig`.
    **config_kwargs :
        Further :class:`~pyfloodrisk.dffa.mcs.MCSConfig` fields, e.g.
        ``preburst=PreBurstSampler(...)`` or ``tail_multiple=2.0``.

    Returns
    -------
    dict with keys ``station``, ``parameters``, ``area_km2``,
    ``state_table``, ``states``, ``ifd``, ``patterns``, ``engine`` and
    ``results`` (a :class:`~pyfloodrisk.dffa.mcs.DFFAResults`).
    """
    area = catchment_data(station)
    params = dict(DEMO_PARAMETERS[station] if parameters is None else parameters)

    forcings = load_station_forcings(station)
    state_table = continuous_state_table(
        forcings, params, area, warmup_hours=warmup_hours, thin=thin)
    states = state_sampler_from_run(state_table, params, method=state_method)

    engine = GR4HEventEngine(params, area_km2=area, dt_hours=1.0)
    patterns = station_patterns(station)
    curve = station_ifd(station, durations_h=durations_h) if ifd is None else ifd
    strat = stratification or Stratification.uniform_in_z(
        aep_max=0.9, aep_min=1e-5, n_strata=25, n_per_stratum=60)

    config = MCSConfig(area_km2=area, durations_h=tuple(durations_h),
                       dt_hours=1.0, seed=seed, progress=progress,
                       **config_kwargs)
    results = DerivedFFA(curve, patterns, states, engine, config, strat,
                         pet=pet_climatology(forcings)).run()

    return {
        "station": station,
        "parameters": params,
        "area_km2": area,
        "state_table": state_table,
        "states": states,
        "ifd": curve,
        "patterns": patterns,
        "engine": engine,
        "results": results,
    }
