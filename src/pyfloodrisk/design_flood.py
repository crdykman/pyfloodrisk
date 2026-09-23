"""Design flood simulation and demo workflow orchestration.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .calibration_robust import behavioural_posterior, calibration, robust_calibration
from .demo_data import catchment_data, demo_paths, load_station_forcings
from .gr4h.GR4H_model import GR4H
from .hydroevents import extract_initial_states_per_duration
from .design_storm import build_design_storm
from .gr4h.state_table import continuous_state_table, state_from_row, uh_columns

#: Leading hours of a continuous run discarded before states are drawn from
#: it, so the stores are not still relaxing from their arbitrary start.
WARMUP_HOURS = 8760


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
        One row per antecedent state to simulate, as
        :func:`~pyfloodrisk.extract_initial_states_per_duration` returns
        them: ``prod_store`` and ``rout_store`` in mm, plus the
        unit-hydrograph memory in ``uh1_0..``/``uh2_0..`` columns.  The
        event is hot-started from the whole vector, so it begins with the
        water the catchment already had in transit -- not from an empty
        unit hydrograph.
    parameters:
        GR4H model parameters (``x1``-``x4``).  ``x4`` fixes the length of
        the UH memory, so these must be the parameters the state table was
        generated with; ``ps0``/``rs0`` are not used, the state supplies
        both stores.
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
    # Setup time horizons and constants.  One row per temporal pattern, one
    # column per hour of the storm -- the loop wants exactly this.
    rain = design_storm["rainfall_matrix"].drop(columns="No").to_numpy(float)
    step_rain = rain.shape[1]
    total_steps = step_rain + horizon_padding

    # x4 sets how many timesteps of memory each unit hydrograph carries, so a
    # pool built under different parameters would silently be the wrong length
    uh1_cols, uh2_cols = uh_columns(initial_states)

    model = GR4H(area=area, params=parameters)
    if len(uh1_cols) != model.n_uh1 or len(uh2_cols) != model.n_uh2:
        raise ValueError(
            f"initial_states carries {len(uh1_cols)}/{len(uh2_cols)} UH columns "
            f"but x4={float(parameters['x4']):g} needs "
            f"{model.n_uh1}/{model.n_uh2}; the state table has to come from "
            "the same parameter set as this simulation")

    pet_series = np.repeat(pet_avg, total_steps)

    # Core simulation loop: every temporal pattern x every initial state
    sim_columns = []
    for pattern in rain:
        # the storm, then horizon_padding dry hours for the recession
        precip_padded = np.zeros(total_steps)
        precip_padded[:step_rain] = pattern
        for i_s in range(len(initial_states)):
            state = state_from_row(initial_states.iloc[i_s], uh1_cols, uh2_cols)
            out = model.run_from_state(precip_padded, pet_series, state=state)
            # Collect simulation vector instead of progressively column-binding
            sim_columns.append(out["qt_cumecs"])

    # Stack column list into a single 2D NumPy array matrix
    return np.column_stack(sim_columns)


def run_demo_workflow(station: str = "117002A", duration_hours: int = 12,
                      timestep_minutes: int | None = None,
                      robust: bool = False) -> dict[str, Any]:
    """Run the bundled demo workflow end-to-end.

    Calibrates GR4H against the bundled climate data, optionally narrows
    the behavioural posterior with robust calibration against design
    storm events, then re-simulates each retained parameter set and runs
    a design flood ensemble for it.

    ``duration_hours`` sets the storm duration *and* the bursts the
    antecedent states are drawn from, so the ensemble starts from the
    wetness the catchment was actually in before its own bursts of that
    length.  Pairing a storm of one duration with states conditioned on
    another is the thing this is meant to prevent, so the two are not
    settable apart.

    Parameters
    ----------
    station:
        Station id.  Defaults to the fully worked example ``117002A``.
    duration_hours:
        Storm duration, and the burst duration the antecedent states are
        conditioned on.  Default 12 h.  Areal temporal patterns are only
        published from 12 h up, so shorter durations are rejected by
        :func:`build_design_storm`.
    timestep_minutes:
        Increment length of the temporal patterns.  ``None`` takes the
        file's own step, which is what you want -- ARR coarsens it as the
        storm lengthens.  See :func:`build_design_storm`.
    robust:
        If True, narrow the behavioural posterior using
        :func:`~pyfloodrisk.robust_calibration` against the bundled
        1-in-2000-year design storm events of this duration before
        simulating.

    Returns
    -------
    dict with keys ``station``, ``calibration``, ``initial_states``,
    ``design_storm``, ``simulations``.
    """
    design_storm = build_design_storm(station=station,
                                      duration_hours=duration_hours,
                                      timestep_minutes=timestep_minutes)

    climate_dir = demo_paths()["climate"]
    data = climate_dir / f"GR4H_climatedata_{station}_hr.csv"
    area = catchment_data(station)
    results = calibration(area, data, 1000, eventsidx=None, save_output=False)
    posterior = behavioural_posterior(results, Cb=0, n=10)
    if robust:
        storms_dir = demo_paths()["storms"]
        paths = sorted(storms_dir.glob(f"*_1in2000_{duration_hours}hr_tp*.csv"))
        if not paths:
            raise FileNotFoundError(
                f"robust=True needs bundled {duration_hours} h design storm "
                f"events in {storms_dir}, "
                "and none are bundled for the current demo stations. Build them "
                "with build_design_storm() and pass them to robust_calibration() "
                "yourself, or run with robust=False.")
        tps = [pd.read_csv(p) for p in paths]
        events = {f"{duration_hours}hr": {"2000": tps}}
        posterior = robust_calibration(posterior, area, events, topn=5)

    forcings = load_station_forcings(station)
    pot_evap_avg = np.nanmean(forcings["pet"])

    outputs = np.zeros((len(forcings), len(posterior)))
    initial_states_all = []
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
        # One continuous run gives both the calibration series and the state
        # table: prod/rout in mm plus the unit-hydrograph memory, which a
        # plain ``run`` does not expose
        table = continuous_state_table(forcings, parameters, area,
                                       warmup_hours=0)
        outputs[:, i] = table["q_cumecs"].to_numpy(float)

        # bursts are selected over the span the states cover, so drop the
        # warm-up year in which the stores are still relaxing from their
        # arbitrary initial values
        states = table.iloc[WARMUP_HOURS:].reset_index(drop=True)
        window = forcings.iloc[WARMUP_HOURS:]

        initial_states = extract_initial_states_per_duration(
            window, states, ey=6, durs=(duration_hours,)
        )
        initial_states_all.append(initial_states)

        simulation = simulate_design_flood(
            design_storm=design_storm,
            initial_states=initial_states[duration_hours],
            parameters=parameters,
            area=area,
            pet_avg=pot_evap_avg,
        )
        simulations.append(simulation)

    return {
        "station": station,
        "calibration": outputs,
        "initial_states": initial_states_all,
        "design_storm": design_storm,
        "simulations": simulations,
    }