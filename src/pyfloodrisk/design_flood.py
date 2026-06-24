"""Design flood simulation and demo workflow orchestration.

``simulate_design_flood`` and ``run_demo_workflow`` are stubs that mirror the
R package API.  Full simulation requires a Python GR4H engine.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .calibration_robust import behavioural_posterior, calibration, robust_calibration
from .demo_data import catchment_data, demo_paths
from .gr4h.GR4H_model import GR4H
from .hydroevents import extract_initial_states, hydro_event_pipeline
from .design_storm import build_design_storm


def simulate_design_flood(
    design_storm: dict[str, Any],
    initial_states: pd.DataFrame,
    parameters: dict[str, Any],
    area: float,
    pet_avg: float,
    horizon_padding: int = 48,
) -> np.ndarray:
    """Simulate a design flood ensemble.

    Runs GR4H forward over every combination of temporal pattern and
    initial antecedent state, then collects the resulting hydrographs.

    Parameters
    ----------
    design_storm:
        Object returned by :func:`build_design_storm`.
    initial_states:
        DataFrame returned by :func:`~pyfloodrisk.extract_initial_states`,
        with one row (``Prod``, ``Rout``) per antecedent state to simulate.
    parameters:
        GR4H model parameters (``x1``-``x4``); ``ps0``/``rs0`` are
        overwritten per initial state.
    area:
        Catchment area in km2.
    pet_avg:
        Constant potential evapotranspiration (mm) applied across the
        whole simulation horizon.
    horizon_padding:
        Additional dry hours appended after the design storm.

    Returns
    -------
    2-D array of simulated streamflow (time steps x scenarios), where
    scenarios are the cartesian product of temporal patterns and initial
    states.
    """
    # Setup time horizons and constants
    rain_long = design_storm["rainfall_long"]
    # Group and split into a list of DataFrames based on 'No' column
    rain_ifd_ls = [group for _, group in rain_long.groupby("No", sort=False)]

    step_rain = design_storm["rainfall_matrix"].shape[1] - 1
    total_steps = step_rain + horizon_padding

    # Generate simulation inputs across scenarios
    inputs_model_sim = []
    for df in rain_ifd_ls:
        # Initialize padded precipitation series with zeros
        precip_padded = np.zeros(total_steps)
        # Pad up to step_rain using the scenario's Rain length
        precip_padded[: len(df["Rain"])] = df["Rain"].values
        # Potential evapotranspiration
        pet_series = np.repeat(pet_avg, total_steps)
        forcings = pd.DataFrame({"prec": precip_padded, "pet": pet_series})
        inputs_model_sim.append(forcings)

    # Core simulation loop: every temporal pattern x every initial state
    sim_columns = []
    for forcings in inputs_model_sim:
        for i_s in range(len(initial_states)):
            # Setup initial store conditions
            parameters["ps0"] = initial_states["Prod"].iloc[i_s]
            parameters["rs0"] = initial_states["Rout"].iloc[i_s]
            # Create a model
            model = GR4H(area=area, params=parameters)
            # Execute GR4H simulation
            sim = model.run(forcings)
            # Collect simulation vector instead of progressively column-binding
            sim_columns.append(sim.qt)

    # Stack column list into a single 2D NumPy array matrix
    sim_matrix = np.column_stack(sim_columns)

    return sim_matrix


def run_demo_workflow(station: str = "421026", robust: bool = False) -> dict[str, Any]:
    """Run the bundled demo workflow end-to-end.

    Calibrates GR4H against the bundled climate data, optionally narrows
    the behavioural posterior with robust calibration against design
    storm events, then re-simulates each retained parameter set and runs
    a design flood ensemble for it.

    Parameters
    ----------
    station:
        Station id.  Defaults to the fully worked example ``421026``.
    robust:
        If True, narrow the behavioural posterior using
        :func:`~pyfloodrisk.robust_calibration` against the bundled
        1-in-2000-year, 12-hour design storm events before simulating.

    Returns
    -------
    dict with keys ``station``, ``calibration``, ``initial_states``,
    ``design_storm``, ``simulations``.
    """
    design_storm = build_design_storm(station=station)

    climate_dir = demo_paths()["climate"]
    data = climate_dir / f"GR4H_climatedata_{station}_hr.csv"
    area = catchment_data(station)
    results = calibration(area, data, 1000, eventsidx=None, save_output=False)
    posterior = behavioural_posterior(results, Cb=0, n=10)
    if robust:
        storms_dir = demo_paths()["storms"]
        tps = []
        for tpi in range(1, 11):
            tp = pd.read_csv(storms_dir / f"421026_1in2000_12hr_tp{tpi:02d}.csv")
            tps.append(tp)
        events = {"12hr": {"2000": tps}}
        posterior = robust_calibration(posterior, area, events, topn=5)

    forcings = pd.read_csv(data, index_col=[0], parse_dates=True, dayfirst=True)
    pot_evap_avg = np.nanmean(forcings["pet"])

    outputs = np.zeros((3, len(forcings), len(posterior)))
    simulations = []
    for i in range(len(posterior)):
        parameters = {
            "ps0": 1.0,  # Initial production storage as a fraction ps0=(ps/X1)
            "rs0": 0.5,  # Initial routing storage as a fraction pr0=(rs/X3)
            "x1": posterior[i][1],  # Maximum production capacity (mm)
            "x2": posterior[i][2],  # Water exchange coefficient (mm); positive
            # if gaining, negative if losing, or null
            "x3": posterior[i][3],  # Routing maximum capacity (mm)
            "x4": posterior[i][4],  # Unit hydrograph time base (hrs)
        }
        # Create a model
        model = GR4H(area=area, params=parameters)
        # Run the model
        sim = model.run(forcings)
        # Save outputs
        outputs[0, :, i] = sim.qt.values
        outputs[1, :, i] = sim.ps.values
        outputs[2, :, i] = sim.rs.values

        _, events_summary = hydro_event_pipeline(
            outputs[0, :, i],
            event_method="maxima",
            method_kwargs=None,
            alpha=0.925,
            passes=3,
            r=30,
        )
        initial_states = extract_initial_states(
            outputs[1:, :, i], events_summary, pre_event=24
        )

        simulation = simulate_design_flood(
            design_storm=design_storm,
            initial_states=initial_states,
            parameters=parameters,
            area=area,
            pet_avg=pot_evap_avg,
        )
        simulations.append(simulation)

    return {
        "station": station,
        "calibration": outputs[0, :, :],
        "initial_states": outputs[1:, :, :],
        "design_storm": design_storm,
        "simulations": simulations,
    }