"""
Derived flood frequency analysis with GR4H in event mode.

A Monte Carlo (joint probability) framework in the form used with event-based
models for design flood estimation under ARR, with the sampled initial loss
replaced by a jointly sampled GR4H **state vector** drawn from a
continuous simulation.
Rainfall depths come from an IFD curve, temporal patterns from the ARR
ensembles, and antecedent conditions from the state distribution of a
continuous run of the same calibrated model.  Sampling of the rainfall
probability domain is stratified, and flood quantiles are estimated
empirically by the total probability theorem (ARR Book 4, Section 4.3.3.3).

Modules
-------
``ifd``             design rainfall curve, areal reduction, record-based fit
``stratification``  intervals over the rainfall AEP domain, with weights
``patterns``        ARR temporal pattern ensembles, pre-burst rainfall
``states``          joint resampling of GR4H states, PET climatology
``engine``          event-model interface and the GR4H wrappers
``mcs``             the Monte Carlo driver and the results object
``tpt``             weighted exceedance / quantiles / bootstrap
``diagnostics``     figures
``workflow``        bridges to the rest of ``pyfloodrisk``

Shortest path, on a bundled demo station::

    from pyfloodrisk.dffa import run_dffa, diagnostics

    out = run_dffa(station="421026")
    print(out["results"].summary())
    diagnostics.plot_all(out["results"], "figures", ifd=out["ifd"])

With your own data, assemble the four inputs yourself::

    from pyfloodrisk.dffa import (DerivedFFA, GR4HEventEngine, IFDCurve,
                                  MCSConfig, STANDARD_DURATIONS_H,
                                  Stratification, TemporalPatternLibrary,
                                  continuous_state_table,
                                  state_sampler_from_run)

    params = {"x1": X1, "x2": X2, "x3": X3, "x4": X4}     # calibrated
    table  = continuous_state_table(forcings, params, AREA)
    states = state_sampler_from_run(table, params, method="bootstrap")
    engine = GR4HEventEngine(params, area_km2=AREA)

    res = DerivedFFA(IFDCurve(ifd_table, arf=my_arf),
                     TemporalPatternLibrary.from_arr_increments_csv(path),
                     states, engine,
                     MCSConfig(area_km2=AREA, durations_h=STANDARD_DURATIONS_H),
                     Stratification.uniform_in_z()).run()
"""

from .ifd import (ARR2019LongDurationARF, DEFAULT_IFD_AEPS, IFDCurve,
                  ifd_table_from_csv, unit_arf)
from .patterns import (DEFAULT_AEP_BANDS, PreBurstSampler, TemporalPatternLibrary,
                       resample_increments)
from .states import (UH_TOTAL, InitialStateSampler, PETClimatology,
                     have_pyvinecopulib)
from .stratification import Stratification
from .engine import EventModel, GR4HEventEngine, mm_per_step_to_cumecs
from .mcs import STANDARD_DURATIONS_H, DerivedFFA, DFFAResults, MCSConfig
from .workflow import (DEMO_PARAMETERS, continuous_state_table,
                       event_onset_states, load_station_forcings,
                       pet_climatology, run_dffa, state_sampler_from_run,
                       station_ifd, station_patterns)
from . import diagnostics, tpt

__all__ = [
    "IFDCurve", "unit_arf", "ARR2019LongDurationARF", "ifd_table_from_csv",
    "DEFAULT_IFD_AEPS",
    "TemporalPatternLibrary", "PreBurstSampler", "resample_increments",
    "DEFAULT_AEP_BANDS",
    "InitialStateSampler", "PETClimatology", "UH_TOTAL", "have_pyvinecopulib",
    "Stratification",
    "EventModel", "GR4HEventEngine", "mm_per_step_to_cumecs",
    "MCSConfig", "DerivedFFA", "DFFAResults", "STANDARD_DURATIONS_H",
    "continuous_state_table", "event_onset_states", "state_sampler_from_run",
    "load_station_forcings",
    "pet_climatology", "station_patterns", "station_ifd", "run_dffa",
    "DEMO_PARAMETERS",
    "diagnostics", "tpt",
]
