"""
Tests for the bridges between ``pyfloodrisk`` and the derived-FFA framework.

These are the joins where an integration goes wrong quietly: a state table
whose unit-hydrograph memory is a different length from the one the engine
expects, a state convention off by one timestep, an increments file parsed
into the wrong AEP band.  Each of those produces plausible-looking numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyfloodrisk.dffa import (DEMO_PARAMETERS, GR4HEventEngine, IFDCurve,
                              Stratification, continuous_state_table,
                              event_onset_states, ifd_from_record,
                              load_station_forcings,
                              pet_climatology, run_dffa, state_sampler_from_run,
                              station_patterns)
from pyfloodrisk.dffa.ifd import _parse_aep, ifd_table_from_csv
from pyfloodrisk.gr4h.GR4H_model import GR4H

STATION = "421026"
PARAMS = DEMO_PARAMETERS[STATION]
AREA = 149.689


@pytest.fixture(scope="module")
def forcings():
    return load_station_forcings(STATION)


@pytest.fixture(scope="module")
def state_table(forcings):
    return continuous_state_table(forcings, PARAMS, AREA,
                                  warmup_hours=8760, thin=24)


# --------------------------------------------------------------- forcings
def test_station_forcings_are_hourly_and_complete(forcings):
    assert list(forcings.columns) == ["pet", "prec", "qt"]
    assert forcings.index.is_monotonic_increasing
    gaps = forcings.index.to_series().diff().dropna().unique()
    assert list(gaps) == [pd.Timedelta(hours=1)]
    assert forcings[["pet", "prec"]].notna().all().all()


def test_missing_station_raises():
    with pytest.raises(FileNotFoundError, match="999999"):
        load_station_forcings("999999")


# ------------------------------------------------------- state distribution
def test_state_table_has_the_columns_the_sampler_needs(state_table):
    model = GR4H(area=AREA, params=PARAMS)
    uh1 = [c for c in state_table.columns if c.startswith("uh1_")]
    uh2 = [c for c in state_table.columns if c.startswith("uh2_")]
    assert len(uh1) == model.n_uh1
    assert len(uh2) == model.n_uh2
    assert {"date", "prod_store", "rout_store"} <= set(state_table.columns)
    # stores are in mm, inside their capacities
    assert state_table["prod_store"].between(0, PARAMS["x1"]).all()
    assert (state_table["rout_store"] >= 0).all()


def test_state_table_respects_warmup_and_thinning(forcings):
    table = continuous_state_table(forcings, PARAMS, AREA,
                                   warmup_hours=100, thin=10)
    # thinning is applied on the full index, not restarted after the
    # warm-up cut, so the retained hours are the multiples of 10 past 100
    expected = [i for i in range(len(forcings)) if i >= 100 and i % 10 == 0]
    assert len(table) == len(expected)
    assert table["date"].iloc[0] == forcings.index[expected[0]]
    assert table["date"].iloc[-1] == forcings.index[expected[-1]]


def test_state_table_wet_only_keeps_states_before_rain(forcings):
    table = continuous_state_table(forcings, PARAMS, AREA, warmup_hours=100,
                                   wet_only=True, min_depth_mm=0.2)
    assert (table["prec_next"] > 0.2).all()
    assert 0 < len(table) < len(forcings)


def test_state_table_rows_resume_the_continuous_run(forcings, state_table):
    """The convention that makes the hot start meaningful.

    A row of the state table must be the state the continuous run was in at
    that hour, so that an event started from it is the continuation of the
    run.  Check it against the run itself.
    """
    model = GR4H(area=AREA, params=PARAMS)
    prec = forcings["prec"].to_numpy(float)
    pet = forcings["pet"].to_numpy(float)
    full = model.run_from_state(prec, pet)
    uh1 = [c for c in state_table.columns if c.startswith("uh1_")]
    uh2 = [c for c in state_table.columns if c.startswith("uh2_")]

    for row in (0, len(state_table) // 2, len(state_table) - 1):
        r = state_table.iloc[row]
        t = int(np.flatnonzero(forcings.index == r["date"])[0])
        if t >= len(prec) - 2:
            continue
        state = {"prod_store": r["prod_store"], "rout_store": r["rout_store"],
                 "uh1": r[uh1].to_numpy(float), "uh2": r[uh2].to_numpy(float)}
        resumed = model.run_from_state(prec[t + 1:], pet[t + 1:], state=state)
        assert np.allclose(resumed["qt"], full["qt"][t + 1:]), r["date"]


def test_state_table_rejects_over_filtering(forcings):
    with pytest.raises(ValueError, match="no states left"):
        continuous_state_table(forcings, PARAMS, AREA,
                               warmup_hours=len(forcings) + 1)


def test_event_onset_states_line_up_with_the_event_delineation(forcings):
    """States at the onset of the events the continuous run itself produced."""
    unthinned = continuous_state_table(forcings, PARAMS, AREA, warmup_hours=8760)
    onsets = event_onset_states(unthinned, event_method="maxima")
    assert 0 < len(onsets) < len(unthinned)
    assert list(onsets["event_id"]) == list(range(1, len(onsets) + 1))
    assert set(onsets.columns) - {"event_id"} == set(unthinned.columns)
    # every onset row is a row of the state table, unchanged
    merged = onsets.merge(unthinned, on=list(unthinned.columns), how="inner")
    assert len(merged) == len(onsets)
    # the delineated onsets are, on average, wetter than an arbitrary hour
    assert onsets["prod_store"].mean() > unthinned["prod_store"].mean()


def test_event_onset_states_reject_a_thinned_table(state_table):
    with pytest.raises(ValueError, match="unthinned"):
        event_onset_states(state_table)


def test_sampler_states_fit_the_engine(state_table):
    """The sampled state must be the shape the engine expects, unpadded."""
    sampler = state_sampler_from_run(state_table, PARAMS)
    engine = GR4HEventEngine(PARAMS, area_km2=AREA)
    sampled = sampler.sample(20, np.random.default_rng(0))
    for state in sampler.to_state_dicts(sampled):
        assert state["uh1"].size == engine.n_uh1
        assert state["uh2"].size == engine.n_uh2
        assert 0 <= state["prod_store"] <= PARAMS["x1"]
        assert state["rout_store"] >= 0
        assert isinstance(state["date"], pd.Timestamp)


def test_sampler_uh_columns_are_in_ordinate_order():
    """``uh1_10`` must not sort before ``uh1_2``."""
    table = pd.DataFrame({"date": pd.date_range("2000-01-01", periods=5, freq="h"),
                          "prod_store": np.linspace(10, 50, 5),
                          "rout_store": np.linspace(1, 5, 5)})
    for j in range(12):
        table[f"uh1_{j}"] = float(j)
        table[f"uh2_{j}"] = float(j)
    sampler = state_sampler_from_run(table, {"x1": 100.0, "x3": 10.0})
    assert list(sampler.uh1_cols) == [f"uh1_{j}" for j in range(12)]
    assert list(sampler.uh2_cols) == [f"uh2_{j}" for j in range(12)]


def test_pet_climatology_is_a_seasonal_cycle(forcings):
    pet = pet_climatology(forcings)
    assert pet.monthly_mm_per_day.size == 12
    assert np.isfinite(pet.monthly_mm_per_day).all()
    assert (pet.monthly_mm_per_day > 0).all()
    # southern hemisphere: January must be wetter in PET terms than June
    assert pet.monthly_mm_per_day[0] > pet.monthly_mm_per_day[5]
    # and the series conserves the monthly mean over a day
    series = pet.series(24, 1.0, month=1)
    assert series.sum() == pytest.approx(pet.monthly_mm_per_day[0], rel=1e-6)


# ------------------------------------------------------------ rainfall inputs
def test_station_patterns_parse_into_bands_and_sum_to_one():
    """ARR increments are percentages of the burst; check one duration by hand."""
    lib = station_patterns(STATION)
    bands = {band for _, band in lib.patterns}
    assert bands == {"frequent", "intermediate", "rare"}
    assert 360 in lib.durations_min                     # 6 h
    entry = lib.patterns[(360, "rare")]
    assert np.allclose(entry["increments"].sum(axis=1), 100.0, atol=0.6)
    # aggregated to the model timestep the fractions still sum to one
    frac, _, band, _ = lib.sample(6.0, 0.005, 1.0, np.random.default_rng(0))
    assert band == "rare"
    assert frac.size == 6 and np.isclose(frac.sum(), 1.0)


def test_record_based_ifd_is_monotone_and_ordered():
    rng = np.random.default_rng(0)
    n = 8 * 8766
    prec = np.where(rng.random(n) < 0.1, rng.gamma(1.5, 2.0, n), 0.0)
    table = ifd_from_record(prec, [1, 6, 24], dt_hours=1.0)
    assert list(table.index) == [1.0, 6.0, 24.0]
    # columns come back in the order asked for, i.e. AEP descending, so
    # depth grows across them; and it grows down the durations too
    assert list(table.columns) == [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
    assert np.all(np.diff(table.to_numpy(), axis=1) > 0)
    assert np.all(np.diff(table.to_numpy(), axis=0) > 0)
    IFDCurve(table)                                        # accepted downstream


def test_record_based_ifd_rejects_a_short_record():
    with pytest.raises(ValueError, match="years long"):
        ifd_from_record(np.ones(100), [1], dt_hours=1.0)


def test_record_based_ifd_rejects_a_sub_timestep_duration():
    prec = np.ones(3 * 8766)
    with pytest.raises(ValueError, match="shorter than"):
        ifd_from_record(prec, [0.25], dt_hours=1.0)


def test_aep_labels_are_parsed_from_the_usual_notations():
    assert _parse_aep("0.01") == pytest.approx(0.01)
    assert _parse_aep("1%") == pytest.approx(0.01)
    assert _parse_aep("1 in 100") == pytest.approx(0.01)
    assert _parse_aep("100y") == pytest.approx(0.01)
    assert _parse_aep(50) == pytest.approx(0.5)          # bare percentage


def test_ifd_table_from_csv_roundtrips(tmp_path):
    path = tmp_path / "ifd.csv"
    path.write_text("duration_min,50%,10%,1%\n60,22.1,36.0,54.8\n"
                    "360,41.0,63.0,95.0\n")
    table = ifd_table_from_csv(path, duration_col="duration_min",
                               duration_units="minutes")
    assert list(table.index) == [1.0, 6.0]
    assert list(table.columns) == [0.5, 0.1, 0.01]
    assert table.loc[1.0, 0.01] == pytest.approx(54.8)
    IFDCurve(table)


# ------------------------------------------------------------------ workflow
def test_run_dffa_end_to_end():
    """The whole chain, on the smallest experiment that still means anything."""
    out = run_dffa(station=STATION, durations_h=(6, 24),
                   stratification=Stratification.uniform_in_z(
                       aep_max=0.9, aep_min=1e-4, n_strata=6, n_per_stratum=8,
                       n_per_end=8),
                   thin=48, progress=False)
    res = out["results"]
    assert set(res.durations) == {6.0, 24.0}
    assert len(res.events) == 2 * 8 * 8            # 6 strata + 2 open ends
    assert (res.events["q_peak"] > 0).all()
    assert np.isclose(res.events.groupby("duration_h")["weight"].sum(), 1.0).all()

    aeps = [0.1, 0.01]
    env = res.envelope(aeps)
    assert np.all(np.diff(env["q_peak"].to_numpy()) > 0)   # rarer is bigger
    assert set(env["critical_duration_h"]) <= {6.0, 24.0}
    # the state distribution the events were started from is the run's own
    assert out["state_table"]["prod_store"].max() <= out["parameters"]["x1"]
    assert isinstance(out["engine"], GR4HEventEngine)
