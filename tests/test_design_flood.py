"""Tests for the design flood ensemble.

The event is hot-started from the whole GR4H state vector, unit-hydrograph
memory included, so it begins with the water the catchment already had in
transit rather than from an empty unit hydrograph.
"""

import numpy as np
import pandas as pd
import pytest

from pyfloodrisk.design_flood import WARMUP_HOURS, simulate_design_flood
from pyfloodrisk.design_storm import build_design_storm
from pyfloodrisk.dffa import DEMO_PARAMETERS
from pyfloodrisk.dffa.workflow import continuous_state_table, load_station_forcings
from pyfloodrisk.demo_data import catchment_data
from pyfloodrisk.hydroevents import extract_initial_states_per_duration

STATION = "117002A"
DURATION_H = 12


@pytest.fixture(scope="module")
def params():
    p = dict(DEMO_PARAMETERS[STATION])
    p.setdefault("ps0", 1.0)
    p.setdefault("rs0", 0.5)
    return p


@pytest.fixture(scope="module")
def pool(params):
    forcings = load_station_forcings(STATION)
    table = continuous_state_table(forcings, params, catchment_data(STATION),
                                   warmup_hours=0)
    states = table.iloc[WARMUP_HOURS:].reset_index(drop=True)
    window = forcings.iloc[WARMUP_HOURS:]
    return extract_initial_states_per_duration(window, states, ey=6,
                                               durs=(DURATION_H,))[DURATION_H]


@pytest.fixture(scope="module")
def storm():
    return build_design_storm(station=STATION, duration_hours=DURATION_H)


def test_the_pool_carries_the_whole_state_vector(pool, params):
    """Stores in mm plus UH memory -- the two the hot start needs."""
    assert {"prod_store", "rout_store", "date"} <= set(pool.columns)
    uh1 = [c for c in pool.columns if c.startswith("uh1_")]
    uh2 = [c for c in pool.columns if c.startswith("uh2_")]
    assert len(uh1) == int(np.ceil(params["x4"]))
    assert len(uh2) == int(np.ceil(2 * params["x4"]))
    # mm, not fractions of x1: a fraction could never exceed 1
    assert pool["prod_store"].max() > 1.0


def test_simulation_shape_is_patterns_by_states(pool, storm, params):
    n_patterns = storm["rainfall_matrix"].shape[0]
    sim = simulate_design_flood(design_storm=storm, initial_states=pool,
                                parameters=params, area=catchment_data(STATION),
                                pet_avg=0.1)
    assert sim.shape[1] == n_patterns * len(pool)
    assert np.isfinite(sim).all()


def test_uh_memory_changes_the_hydrograph(pool, storm, params):
    """If it made no difference the hot start would not be worth doing.

    Zeroing the memory is what ``GR4H.run`` did implicitly, so this is the
    old behaviour against the new one on identical donors.
    """
    area = catchment_data(STATION)
    uh = [c for c in pool.columns if c.startswith(("uh1_", "uh2_"))]
    empty = pool.copy()
    empty[uh] = 0.0

    kw = dict(design_storm=storm, parameters=params, area=area, pet_avg=0.1)
    with_uh = simulate_design_flood(initial_states=pool, **kw)
    without = simulate_design_flood(initial_states=empty, **kw)

    assert with_uh.max() > without.max()
    # every scenario is at least as large: in-transit water is only ever added
    assert (with_uh.max(axis=0) >= without.max(axis=0) - 1e-9).all()


def test_a_pool_from_other_parameters_is_rejected(pool, storm, params):
    """x4 sets the memory length, so a mismatched pool must not be run."""
    other = dict(params)
    other["x4"] = params["x4"] * 2 + 3
    with pytest.raises(ValueError, match="UH columns"):
        simulate_design_flood(design_storm=storm, initial_states=pool,
                              parameters=other, area=catchment_data(STATION),
                              pet_avg=0.1)


# ------------------------------------------------------------- layering
def test_the_main_package_does_not_import_from_dffa():
    """dffa builds on the main package, never the other way round.

    Only the top-level ``__init__`` aggregates the derived-FFA namespace;
    a module-level dependency the other way would make the dependency
    graph cyclic in spirit and the subpackage impossible to lift out.
    """
    import pathlib
    import pyfloodrisk

    root = pathlib.Path(pyfloodrisk.__file__).parent
    offenders = []
    for path in root.glob("*.py"):
        if path.name == "__init__.py":
            continue
        text = path.read_text(encoding="utf-8")
        if "from .dffa" in text or "from pyfloodrisk.dffa" in text:
            offenders.append(path.name)
    assert offenders == []


def test_gr4h_is_a_leaf_package():
    """It may be imported by anything and import nothing of ours."""
    import pathlib
    import pyfloodrisk

    gr4h = pathlib.Path(pyfloodrisk.__file__).parent / "gr4h"
    offenders = []
    for path in gr4h.glob("*.py"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("from ..") or line.startswith("from pyfloodrisk"):
                offenders.append(f"{path.name}: {line}")
    assert offenders == []


def test_uh_columns_sorts_numerically_not_lexically():
    """uh1_10 sorts before uh1_9 as a string, scrambling the memory."""
    from pyfloodrisk.gr4h import uh_columns

    frame = pd.DataFrame(columns=["date", "prod_store"]
                         + [f"uh1_{j}" for j in range(12)]
                         + [f"uh2_{j}" for j in range(3)])
    uh1, uh2 = uh_columns(frame)
    assert uh1 == [f"uh1_{j}" for j in range(12)]
    assert uh2 == [f"uh2_{j}" for j in range(3)]


def test_state_from_row_matches_a_hand_built_state(pool):
    """The helper the sampler and the design flood now share."""
    from pyfloodrisk.gr4h import state_from_row, uh_columns

    uh1, uh2 = uh_columns(pool)
    row = pool.iloc[3]
    state = state_from_row(row, uh1, uh2)
    assert state["prod_store"] == pytest.approx(float(row["prod_store"]))
    assert state["rout_store"] == pytest.approx(float(row["rout_store"]))
    np.testing.assert_allclose(state["uh1"], row[uh1].to_numpy(float))
    np.testing.assert_allclose(state["uh2"], row[uh2].to_numpy(float))
    # resolved from the row when the columns are not supplied
    assert state_from_row(row).keys() == state.keys()
