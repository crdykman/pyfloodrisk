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
design rainfall (IFD)        a Bureau IFD download as-issued, read by
                             :func:`~pyfloodrisk.dffa.ifd.ifd_table_from_bom_csv`.
                             One download is bundled per demo station, and
                             :func:`station_ifd` reads it and applies the
                             region's ARR 2019 areal reduction factor.
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

from ..demo_data import (AREAL_TP_FROM_H, DEMO_STATION, POINT_TP_DURATIONS_H,
                         catchment_data, demo_paths, load_station_forcings,
                         station_arf_region,
                         station_tp_region)
from ..design_storm import _pick_increment_file
from ..gr4h.GR4H_model import GR4H
from ..gr4h.state_table import continuous_state_table, uh_columns
from .engine import GR4HEventEngine
from .ifd import ARR2019ARF, IFDCurve, ifd_table_from_bom_csv
from .mcs import DerivedFFA, MCSConfig
from .patterns import TemporalPatternLibrary
from .states import InitialStateSampler, PETClimatology
from .stratification import Stratification

__all__ = [
    "load_station_forcings",
    "continuous_state_table",
    "state_sampler_from_run",
    "state_samplers_by_duration",
    "pet_climatology",
    "station_patterns",
    "station_ifd",
    "run_dffa",
    "DEMO_PARAMETERS",
]


DEMO_PARAMETERS: dict[str, dict[str, float]] = {
    "117002A": {"ps0": 0.5, "rs0": 0.5, "x1": 125.78, "x2": -7.94253,
                "x3": 31.1227, "x4": 21.237},
    "405214":  {"ps0": 0.5, "rs0": 0.5, "x1": 611.41, "x2": -8.12765,
                "x3": 111.418, "x4": 12.2814},
    "303203":  {"ps0": 0.5, "rs0": 0.5, "x1": 36.0292, "x2": -1.58247,
                "x3": 134.811, "x4": 17.4297},
}


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
    uh1_cols, uh2_cols = uh_columns(state_table)
    return InitialStateSampler(
        state_table, x1=float(parameters["x1"]), x3=float(parameters["x3"]),
        date_col="date", uh1_cols=uh1_cols or None, uh2_cols=uh2_cols or None,
        method=method, **kwargs)


def state_samplers_by_duration(
        state_table: pd.DataFrame,
        forcings: pd.DataFrame,
        parameters: Mapping[str, float],
        durations_h: Sequence[float],
        ey: int = 6,
        **kwargs) -> dict[float, InitialStateSampler]:
    """One state sampler per storm duration, conditioned on that duration.

    Each duration draws its antecedent state from the states the catchment
    was *actually* in immediately before its own large bursts: the largest
    ``ey * nyears`` bursts of that length are found and the state table row
    one timestep before each is kept.  This is a thin wrapper over
    :func:`~pyfloodrisk.hydroevents.extract_initial_states_per_duration`,
    which does the selection, and turns each pool into a sampler.

    Parameters
    ----------
    state_table :
        Output of :func:`continuous_state_table`, **unthinned**: the rows are
        matched to burst onsets by timestamp, and thinning drops most of
        them.
    forcings :
        The hourly record the state table was generated from; its ``prec``
        column selects the bursts.
    parameters :
        GR4H parameters, as for :func:`state_sampler_from_run`.
    durations_h :
        Storm durations to build samplers for.  Pass the same list you give
        :class:`~pyfloodrisk.dffa.mcs.MCSConfig`.
    ey :
        Bursts per year of record to condition on.  Default 6.
    **kwargs :
        Passed to :func:`state_sampler_from_run`, e.g. ``method="smoothed"``.

    Returns
    -------
    dict
        ``duration_h -> InitialStateSampler``, ready to hand to
        :class:`~pyfloodrisk.dffa.mcs.DerivedFFA` in place of a single
        sampler.

    Notes
    -----
    Each pool holds only ``ey * nyears`` states -- 66 for six bursts a year
    over eleven years -- and the Monte Carlo draws far more events than that
    from it.  With a pool this small, ``method="bootstrap"`` resamples a
    handful of distinct states many times over, and the smoothed or copula
    methods are worth comparing against.
    """
    from ..hydroevents import extract_initial_states_per_duration

    if "date" not in state_table:
        raise ValueError("state_table needs its 'date' column to match bursts")
    # bursts are selected over the span the state table covers, not the whole
    # record: the warm-up the table discards has no states to draw from, and
    # counting those years would thin the pool for every duration
    dates = pd.DatetimeIndex(state_table["date"])
    window = forcings.loc[dates.min():dates.max()]

    pools = extract_initial_states_per_duration(
        window, state_table, ey=ey, durs=tuple(durations_h))
    return {duration_h: state_sampler_from_run(pool, parameters, **kwargs)
            for duration_h, pool in pools.items()}


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
def station_patterns(station: str = DEMO_STATION,
                     increments_path: str | Path | None = None,
                     point_durations_h: Sequence[float] = POINT_TP_DURATIONS_H,
                     ) -> TemporalPatternLibrary:
    """ARR temporal pattern ensembles for a bundled station.

    Reads the station region's Data Hub increments files and keeps the whole
    ensemble for every duration, rather than selecting one as
    :func:`~pyfloodrisk.build_design_storm` does.

    Two files are combined, because ARR publishes no single set that spans
    the durations this framework needs
    (:func:`~pyfloodrisk.demo_data.station_tp_region` names the region):

    * **12 h and longer** come from the *areal* patterns, published per
      standard catchment area.  The station's own area picks the nearest
      standard one, and because areal patterns carry no AEP dependence that
      single ensemble serves every AEP band.
    * **shorter than 12 h**, down to ``min(point_durations_h)``, come from
      the *point* patterns, which do vary by AEP band.  ARR publishes no
      areal pattern below 12 h, so the alternative is not having those
      durations at all.

    Mixing the two is a compromise worth stating in a write-up: the short
    bursts are point patterns applied to a catchment-average depth, so their
    within-burst variability is that of a gauge rather than of a 255-357 km2
    catchment, and it is not damped the way the areal patterns' is.  The
    areal reduction factor still applies to the *depth* at every duration
    (see :func:`station_ifd`); it is only the *shape* that is a point shape.

    Parameters
    ----------
    station :
        Bundled station id.
    increments_path :
        Read this one file instead, point or areal, and use it for every
        duration it carries.  No combining is done.
    point_durations_h :
        Durations to take from the point patterns; must all be shorter than
        :data:`~pyfloodrisk.demo_data.AREAL_TP_FROM_H`.  Pass an empty
        sequence for areal patterns only.
    """
    root = demo_paths()["root"]
    if increments_path is not None:
        return TemporalPatternLibrary.from_arr_increments_csv(
            increments_path, area_km2=catchment_data(station))

    areal = TemporalPatternLibrary.from_arr_increments_csv(
        _pick_increment_file(root, station, kind="areal"),
        area_km2=catchment_data(station))
    if not len(point_durations_h):
        return areal

    too_long = [d for d in point_durations_h if d >= AREAL_TP_FROM_H]
    if too_long:
        raise ValueError(
            f"point_durations_h must be shorter than {AREAL_TP_FROM_H} h, "
            f"where the areal patterns take over; got {too_long}")
    point = TemporalPatternLibrary.from_arr_increments_csv(
        _pick_increment_file(root, station, kind="point"))
    return TemporalPatternLibrary.combine(
        point.subset(round(d * 60) for d in point_durations_h), areal)


def station_ifd(station: str = DEMO_STATION, arf=None) -> IFDCurve:
    """The bundled BoM design rainfall table for a demo station.

    Design rainfalls enter the framework one way only -- a CSV -- and this
    reads the Bureau IFD download bundled for ``station``
    (``data/ifd/<station>_ifds.csv``) with
    :func:`~pyfloodrisk.dffa.ifd.ifd_table_from_bom_csv`, which handles the
    download's metadata preamble.  Depths are tabulated from 63.2% to 1%
    AEP over 1-168 h bursts; anything rarer than 1% is extrapolated by
    :class:`~pyfloodrisk.dffa.ifd.IFDCurve`, so add ARR's rare design
    rainfalls as extra columns if you have them.

    The depths are *point* depths for the grid cell nearest the gauge, so
    they are reduced to a catchment average by the ARR 2019 areal reduction
    factor for the station's ARF region
    (:func:`~pyfloodrisk.demo_data.station_arf_region`) unless you pass your
    own ``arf``.  Pass :func:`~pyfloodrisk.dffa.ifd.unit_arf` to turn the
    reduction off and work in point depths.

    Parameters
    ----------
    station :
        Bundled station id.
    arf :
        Areal reduction factor callable.  Defaults to
        :class:`~pyfloodrisk.dffa.ifd.ARR2019ARF` for the station's region.
    """
    path = demo_paths()["ifd"] / f"{station}_ifds.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"no bundled IFD table for station {station}; supply your own "
            "depths with ifd_table_from_bom_csv()")
    if arf is None:
        arf = ARR2019ARF(region=station_arf_region(station))
    return IFDCurve(ifd_table_from_bom_csv(path), arf=arf)


# ------------------------------------------------------------------ workflow
def run_dffa(
    station: str = DEMO_STATION,
    parameters: Mapping[str, float] | None = None,
    durations_h: Sequence[float] = (6, 12, 24, 48, 72),
    ifd: IFDCurve | None = None,
    stratification: Stratification | None = None,
    state_method: str = "bootstrap",
    ey: int = 6,
    warmup_hours: int = 8760,
    seed: int = 20260909,
    progress: bool = True,
    **config_kwargs: Any,
) -> dict[str, Any]:
    """Derived flood frequency analysis for a bundled demo station.

    Runs the whole chain: continuous GR4H over the station's climate record,
    one state distribution per storm duration drawn from that run, temporal
    patterns from the station's increments file, the bundled demonstration
    design rainfall table, then the stratified Monte Carlo.

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
        Storm durations to envelope over.  Fewer than
        :data:`~pyfloodrisk.dffa.mcs.STANDARD_DURATIONS_H`, because the demo
        should finish in a minute or two.  The bundled areal temporal
        patterns start at 12 h, so a shorter duration has no ensemble and
        raises.
    ifd :
        Design rainfall curve.  Defaults to :func:`station_ifd`, the bundled
        demonstration table, which is **not** a design IFD -- pass your own
        (``IFDCurve(ifd_table_from_bom_csv(path))``) for anything you report.
    stratification :
        Rainfall sampling scheme; defaults to 25 intervals x 60 simulations
        over 90% to 1 in 10^5 AEP.
    state_method :
        Initial-state sampling method; see
        :class:`~pyfloodrisk.dffa.states.InitialStateSampler`.  Each
        duration's pool holds only ``ey`` states per year of record, so
        ``bootstrap`` resamples a small donor set many times over and the
        smoothed or copula methods are worth comparing against.
    ey :
        Bursts per year of record each duration's state pool is drawn
        from; see :func:`state_samplers_by_duration`.
    warmup_hours :
        Passed to :func:`continuous_state_table`.  The table is not
        thinned: pools are matched to burst onsets by timestamp, and
        thinning drops most of the rows they need.
    seed, progress :
        Passed to :class:`~pyfloodrisk.dffa.mcs.MCSConfig`.
    **config_kwargs :
        Further :class:`~pyfloodrisk.dffa.mcs.MCSConfig` fields

    Returns
    -------
    dict with keys ``station``, ``parameters``, ``area_km2``,
    ``state_table``, ``states`` (``duration_h -> InitialStateSampler``),
    ``ifd``, ``patterns``, ``engine`` and ``results``
    (a :class:`~pyfloodrisk.dffa.mcs.DFFAResults`).
    """
    area = catchment_data(station)
    params = dict(DEMO_PARAMETERS[station] if parameters is None else parameters)

    forcings = load_station_forcings(station)
    state_table = continuous_state_table(
        forcings, params, area, warmup_hours=warmup_hours)
    states = state_samplers_by_duration(state_table, forcings, params,
                                        durations_h, ey=ey, method=state_method)

    engine = GR4HEventEngine(params, area_km2=area, dt_hours=1.0)
    patterns = station_patterns(station)
    curve = station_ifd(station) if ifd is None else ifd
    strat = stratification or Stratification.uniform_in_z(
        aep_max=0.9, aep_min=2e-2, n_strata=50, n_per_stratum=100)

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
