"""Flood Risk Assessment: GR4H-Based Design Flood Analysis."""

from . import dffa
from .calibration_robust import behavioural_posterior, calibration, robust_calibration
from .demo_data import (
    catchment_data,
    demo_paths,
    list_demo_stations,
    load_demo_station_data,
)
from .design_flood import run_demo_workflow, simulate_design_flood
from .design_storm import build_design_storm
from .dffa import (
    DerivedFFA,
    DFFAResults,
    GR4HEventEngine,
    IFDCurve,
    InitialStateSampler,
    MCSConfig,
    Stratification,
    TemporalPatternLibrary,
    continuous_state_table,
    run_dffa,
)
from .hydroevents import extract_initial_states, hydro_event_pipeline

__version__ = "0.2.0"
__all__ = [
    "DFFAResults",
    "DerivedFFA",
    "GR4HEventEngine",
    "IFDCurve",
    "InitialStateSampler",
    "MCSConfig",
    "Stratification",
    "TemporalPatternLibrary",
    "behavioural_posterior",
    "build_design_storm",
    "calibration",
    "catchment_data",
    "continuous_state_table",
    "demo_paths",
    "dffa",
    "extract_initial_states",
    "hydro_event_pipeline",
    "list_demo_stations",
    "load_demo_station_data",
    "robust_calibration",
    "run_demo_workflow",
    "run_dffa",
    "simulate_design_flood",
]
