"""Antecedent states: where an event's initial condition is taken from."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyfloodrisk.events import hydro_event_pipeline
from pyfloodrisk.states import extract_initial_states, initial_state_indices

HOURS_PER_YEAR = 365.25 * 24.0
def _events(peaks, volumes=None, spacing=100):
    """An events table of the shape delineation returns."""
    peaks = np.asarray(peaks, float)
    volumes = peaks * 10.0 if volumes is None else np.asarray(volumes, float)
    starts = np.arange(len(peaks)) * spacing
    return pd.DataFrame({
        "start": starts,
        "end": starts + 50,
        "max_index": starts + 10,
        "max": peaks,
        "sum": volumes,
    })


def _synthetic_hydrograph(n_years=2.0, spacing=240, seed=0):
    """Hourly flow with one well-separated triangular event per ``spacing`` h."""
    rng = np.random.default_rng(seed)
    n = int(round(n_years * HOURS_PER_YEAR))
    q = np.full(n, 1.0)
    onsets = np.arange(spacing, n - 120, spacing)
    heights = rng.uniform(2.0, 60.0, onsets.size)
    for t0, h in zip(onsets, heights):
        rise = np.linspace(0.0, h, 7)[1:]
        recession = h * np.exp(-np.arange(1, 41) / 12.0)
        pulse = np.concatenate([rise, recession])
        q[t0:t0 + pulse.size] += pulse
    return q, onsets.size


# ------------------------------------------------- initial_state_indices
def _ramp_stores(n=60, trough=12):
    """Stores that decline to ``trough`` then rise: the onset is the trough."""
    prod = np.concatenate([np.linspace(10, 5, trough),
                           np.linspace(5, 20, n - trough)])
    return np.vstack([prod, prod.copy()])


@pytest.mark.parametrize("peak", [5, 11, 20, 40, 55])
def test_onset_is_searched_inside_the_clamped_window(peak):
    """An event peaking near the record start must not land before its window.

    The window is clamped at 0; the offset into it has to be measured from
    the same place, or early events collapse onto index 0.
    """
    states = _ramp_stores()
    events = pd.DataFrame({"max_index": [peak]})
    idx = initial_state_indices(states[0], states[1], events, pre_event=24)[0]

    start_win = max(0, peak - 24 + 1)
    declining = np.diff(states[0, start_win:peak + 1]) < 0
    expected = start_win + (np.where(declining)[0][-1] if declining.any() else 0)
    assert idx == expected
    assert start_win <= idx <= peak


def test_extract_initial_states_reads_the_stores_at_those_indices():
    """The wrapper adds lookup and nothing else."""
    states = _ramp_stores()
    events = pd.DataFrame({"max_index": [20, 40, 55]})
    idx = initial_state_indices(states[0], states[1], events)
    out = extract_initial_states(states, events)

    assert list(out["state_index"]) == list(idx)
    assert list(out["event_id"]) == [1, 2, 3]
    np.testing.assert_allclose(out["Prod"], states[0, idx])
    np.testing.assert_allclose(out["Rout"], states[1, idx])


def test_onset_search_is_invariant_to_store_units():
    """Only the sign of the differences is used, so mm and fractions agree."""
    states = _ramp_stores()
    events = pd.DataFrame({"max_index": [20, 40, 55]})
    x1, x3 = 36.0292, 134.811
    np.testing.assert_array_equal(
        initial_state_indices(states[0], states[1], events),
        initial_state_indices(states[0] * x1, states[1] * x3, events))


def test_extract_initial_states_on_no_events():
    out = extract_initial_states(_ramp_stores(),
                                 pd.DataFrame({"max_index": []}))
    assert out.empty
    assert list(out.columns) == ["event_id", "state_index", "Prod", "Rout"]


# --------------------------------------------------- onset="start"
def test_onset_start_takes_the_delineated_start_verbatim():
    """The replacement for the old extract_initial_states_basic."""
    states = _ramp_stores()
    events = pd.DataFrame({"start": [3, 18, 44], "max_index": [9, 25, 51]})

    idx = initial_state_indices(states[0], states[1], events, onset="start")
    np.testing.assert_array_equal(idx, [3, 18, 44])

    out = extract_initial_states(states, events, onset="start")
    assert list(out["state_index"]) == [3, 18, 44]
    assert list(out["event_id"]) == [1, 2, 3]
    np.testing.assert_allclose(out["Prod"], states[0, [3, 18, 44]])
    # what extract_initial_states_basic used to build, index and all
    assert list(out.index) == [0, 1, 2]


def test_onset_start_ignores_pre_event_and_the_stores():
    """No window search happens, so neither input can move the answer."""
    states = _ramp_stores()
    events = pd.DataFrame({"start": [3, 18, 44], "max_index": [9, 25, 51]})
    base = initial_state_indices(states[0], states[1], events, onset="start")
    np.testing.assert_array_equal(
        base, initial_state_indices(states[0], states[1], events,
                                    onset="start", pre_event=999))
    np.testing.assert_array_equal(
        base, initial_state_indices(np.zeros(states.shape[1]),
                                    np.zeros(states.shape[1]),
                                    events, onset="start"))


def test_the_two_onset_modes_disagree():
    """Otherwise the parameter would not be worth having."""
    states = _ramp_stores()
    events = pd.DataFrame({"start": [3, 18, 44], "max_index": [9, 25, 51]})
    assert not np.array_equal(
        initial_state_indices(states[0], states[1], events, onset="start"),
        initial_state_indices(states[0], states[1], events, onset="search"))


def test_onset_errors():
    states = _ramp_stores()
    events = pd.DataFrame({"start": [3, 18], "max_index": [9, 25]})

    with pytest.raises(ValueError, match="'search' or 'start'"):
        initial_state_indices(states[0], states[1], events, onset="begin")
    with pytest.raises(ValueError, match="'start' column"):
        initial_state_indices(states[0], states[1],
                              events.drop(columns="start"), onset="start")
    with pytest.raises(ValueError, match="'max_index' column"):
        initial_state_indices(states[0], states[1],
                              events.drop(columns="max_index"))
    with pytest.raises(ValueError, match="same length"):
        initial_state_indices(states[0], states[1, :-1], events)


@pytest.mark.parametrize("bad", [-2, 10_000])
def test_onset_start_rejects_indices_outside_the_run(bad):
    """numpy would wrap a negative index silently; say so instead."""
    states = _ramp_stores()
    events = pd.DataFrame({"start": [3, bad]})
    with pytest.raises(ValueError, match="outside the"):
        initial_state_indices(states[0], states[1], events, onset="start")


def test_onset_start_on_real_delineated_events():
    """End to end: the pipeline's own 'start' column drives it."""
    q, _ = _synthetic_hydrograph(n_years=2.0)
    _, events = hydro_event_pipeline(q, ey=6)
    states = np.vstack([np.linspace(0.2, 0.8, q.size),
                        np.linspace(0.8, 0.2, q.size)])
    out = extract_initial_states(states, events, onset="start")
    assert len(out) == len(events)
    np.testing.assert_array_equal(out["state_index"],
                                  np.asarray(events["start"], int))


# ------------------------------------------------- the ensemble form
def _synthetic_forcings(years=3, seed=0, name=None):
    """Hourly prec/pet with enough wet spells to delineate bursts from."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2000-01-01", periods=years * 8760, freq="h")
    prec = rng.gamma(0.25, 1.4, idx.size) * (rng.random(idx.size) < 0.18)
    df = pd.DataFrame({"prec": prec, "pet": 0.14}, index=idx)
    if name:
        df.attrs["name"] = name
    return df


_PARAMS = {"ps0": 0.5, "rs0": 0.5, "x1": 120.0, "x2": -2.0,
           "x3": 90.0, "x4": 8.0}


def test_one_member_ensemble_matches_the_single_record_form():
    """The ensemble adds pooling, and nothing else."""
    from pyfloodrisk.gr4h import continuous_state_table
    from pyfloodrisk.states import (extract_initial_states_ensemble,
                                    extract_initial_states_per_duration)

    fc = _synthetic_forcings(name="one")
    table = continuous_state_table(fc, _PARAMS, 50.0, warmup_hours=8760)
    direct = extract_initial_states_per_duration(
        fc.iloc[8760:], table, ey=6, durs=(12, 24))
    ens = extract_initial_states_ensemble(
        [fc], _PARAMS, 50.0, ey=6, durs=(12, 24), progress=False)

    assert set(ens) == set(direct)
    for d in direct:
        pd.testing.assert_frame_equal(
            ens[d].drop(columns="realisation"), direct[d])


def test_pools_concatenate_across_realisations():
    """Each duration's donors come from every member, tagged by source."""
    from pyfloodrisk.states import extract_initial_states_ensemble

    members = [_synthetic_forcings(seed=s, name=f"r{s}") for s in range(3)]
    pools = extract_initial_states_ensemble(
        members, _PARAMS, 50.0, ey=6, durs=(12, 24), progress=False)

    for d, pool in pools.items():
        assert list(pool["realisation"].unique()) == ["r0", "r1", "r2"]
        assert pool["realisation"].value_counts().nunique() == 1  # even split
        assert pool.index.equals(pd.RangeIndex(len(pool)))
        # different rainfall gives different donors, not three copies
        by_member = [g["prod_store"].to_numpy()
                     for _, g in pool.groupby("realisation", sort=True)]
        assert not np.allclose(by_member[0], by_member[1])


def test_ensemble_carries_the_whole_state_vector():
    """A pooled donor must still be able to hot-start the engine."""
    from pyfloodrisk.states import extract_initial_states_ensemble

    pools = extract_initial_states_ensemble(
        [_synthetic_forcings(seed=s) for s in range(2)], _PARAMS, 50.0,
        ey=6, durs=(24,), progress=False)
    pool = pools[24.0]
    assert {"realisation", "date", "prod_store", "rout_store"} <= set(pool.columns)
    assert len([c for c in pool.columns if c.startswith("uh1_")]) == 8   # x4=8
    assert len([c for c in pool.columns if c.startswith("uh2_")]) == 16


def test_ensemble_reads_realisations_from_disk(tmp_path):
    """Paths and frames must give the same pools; the name comes from the file."""
    from pyfloodrisk.states import extract_initial_states_ensemble

    members, paths = [], []
    for s in range(2):
        fc = _synthetic_forcings(seed=s)
        p = tmp_path / f"sim_{s:03d}.csv"
        fc.rename_axis("Date").to_csv(p)
        members.append(fc)
        paths.append(p)

    from_disk = extract_initial_states_ensemble(
        paths, _PARAMS, 50.0, ey=6, durs=(24,), progress=False)[24.0]
    in_memory = extract_initial_states_ensemble(
        members, _PARAMS, 50.0, ey=6, durs=(24,), progress=False)[24.0]

    assert list(from_disk["realisation"].unique()) == ["sim_000", "sim_001"]
    np.testing.assert_allclose(from_disk["prod_store"], in_memory["prod_store"])
