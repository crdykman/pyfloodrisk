"""Event delineation, and the exceedance-per-year trim applied after it."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyfloodrisk.hydroevents import (extract_initial_states,
                                     initial_state_indices,
                                     hydro_event_pipeline,
                                     threshold_events_by_ey)

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


# ------------------------------------------------------------- the EY rule
def test_ey_keeps_that_many_events_per_year_of_record():
    n = int(round(2.0 * HOURS_PER_YEAR))          # exactly two years, hourly
    events = _events(np.arange(100, 0, -1))       # 100 events, descending peak
    kept = threshold_events_by_ey(events, n, ey=6.0)
    assert len(kept) == 12                        # 6 per year x 2 years


def test_ey_rate_scales_with_record_length():
    events = _events(np.arange(200, 0, -1))
    for years in (1.0, 3.0, 7.0):
        n = int(round(years * HOURS_PER_YEAR))
        kept = threshold_events_by_ey(events, n, ey=6.0)
        assert len(kept) == round(6 * years)


def test_ey_respects_the_timestep():
    """A daily series of the same length is 24x the record, so 24x the events."""
    events = _events(np.arange(300, 0, -1))
    n = int(round(2.0 * HOURS_PER_YEAR))
    hourly = threshold_events_by_ey(events, n, ey=6.0, dt_hours=1.0)
    daily = threshold_events_by_ey(events, n, ey=6.0, dt_hours=24.0)
    assert len(hourly) == 12
    assert len(daily) == 12 * 24


def test_peak_and_volume_select_different_events():
    """A tall narrow event and a low broad one rank differently."""
    events = _events(peaks=[100.0, 10.0], volumes=[200.0, 9000.0])
    n = int(round(HOURS_PER_YEAR / 6.0))          # two months -> keep 1
    by_peak = threshold_events_by_ey(events, n, ey=6.0, rank_by="peak")
    by_volume = threshold_events_by_ey(events, n, ey=6.0, rank_by="volume")
    assert len(by_peak) == len(by_volume) == 1
    assert by_peak["max"].iloc[0] == 100.0        # the tall narrow one
    assert by_volume["sum"].iloc[0] == 9000.0     # the low broad one


def test_none_keeps_every_event():
    events = _events(np.arange(100, 0, -1))
    n = int(round(2.0 * HOURS_PER_YEAR))
    assert len(threshold_events_by_ey(events, n, ey=None)) == len(events)


def test_kept_events_are_the_largest_ones():
    events = _events([5.0, 50.0, 1.0, 30.0, 2.0])
    n = int(round(HOURS_PER_YEAR / 3.0))          # four months -> keep 2
    kept = threshold_events_by_ey(events, n, ey=6.0)
    assert sorted(kept["max"]) == [30.0, 50.0]


def test_kept_events_come_back_in_order_with_a_reset_index():
    """extract_initial_states looks rows up by label, so this matters."""
    events = _events([5.0, 50.0, 1.0, 30.0, 2.0])
    n = int(round(HOURS_PER_YEAR / 3.0))
    kept = threshold_events_by_ey(events, n, ey=6.0)
    assert list(kept.index) == list(range(len(kept)))
    assert kept["start"].is_monotonic_increasing


def test_fewer_events_than_asked_for_is_not_an_error():
    events = _events([5.0, 50.0])
    n = int(round(10.0 * HOURS_PER_YEAR))         # room for 60, only 2 exist
    assert len(threshold_events_by_ey(events, n, ey=6.0)) == 2


def test_a_short_record_still_keeps_one_event():
    events = _events([5.0, 50.0, 1.0])
    n = 24                                        # one day
    assert len(threshold_events_by_ey(events, n, ey=6.0)) == 1


def test_an_empty_event_table_survives():
    empty = pd.DataFrame(columns=["start", "end", "max_index", "max", "sum"])
    assert threshold_events_by_ey(empty, 1000, ey=6.0).empty


def test_bad_arguments_are_rejected():
    events = _events([1.0, 2.0])
    with pytest.raises(ValueError, match="rank_by"):
        threshold_events_by_ey(events, 1000, rank_by="discharge")
    with pytest.raises(ValueError, match="ey must be positive"):
        threshold_events_by_ey(events, 1000, ey=0)


# --------------------------------------------------------- through the pipeline
def test_pipeline_keeps_every_event_by_default():
    """Trimming is opt-in: the default returns every delineated rise."""
    q, _ = _synthetic_hydrograph(n_years=2.0)
    _, default = hydro_event_pipeline(q)
    _, explicit_none = hydro_event_pipeline(q, ey=None)
    assert len(default) == len(explicit_none) > 12


def test_pipeline_applies_ey_when_asked():
    q, _ = _synthetic_hydrograph(n_years=2.0)
    _, events = hydro_event_pipeline(q, ey=6)
    _, unthresholded = hydro_event_pipeline(q, ey=None)
    # the record holds far more rises than 6 EY admits
    assert len(unthresholded) > 12
    assert len(events) == 12
    # and the ones kept are the largest of them
    assert events["max"].min() >= unthresholded["max"].nlargest(12).min()


def test_pipeline_ranks_by_volume_when_asked():
    q, _ = _synthetic_hydrograph(n_years=2.0)
    by_peak = hydro_event_pipeline(q, ey=6, rank_by="peak")[1]
    by_volume = hydro_event_pipeline(q, ey=6, rank_by="volume")[1]
    assert len(by_peak) == len(by_volume) == 12
    assert by_volume["sum"].sum() >= by_peak["sum"].sum()
    assert by_peak["max"].sum() >= by_volume["max"].sum()


def test_pipeline_can_return_the_event_timestep_indices():
    """``idx=True`` gives the form ``calibration(eventsidx=...)`` wants."""
    q, _ = _synthetic_hydrograph(n_years=2.0)
    _, events, eventsidx = hydro_event_pipeline(q, ey=6, idx=True)
    assert eventsidx.dtype.kind == "i"
    # every retained event contributes its whole start..end span, once
    expected = sum(int(e) - int(s) + 1
                   for s, e in zip(events["start"], events["end"]))
    assert eventsidx.size == expected
    assert eventsidx.min() >= 0 and eventsidx.max() < q.size


def test_pipeline_events_feed_extract_initial_states():
    """The reset index is what makes this work; without it it raises."""
    q, _ = _synthetic_hydrograph(n_years=2.0)
    _, events = hydro_event_pipeline(q, ey=6)
    states = np.vstack([np.linspace(0.2, 0.8, q.size),
                        np.linspace(0.3, 0.7, q.size)])
    onsets = extract_initial_states(states, events, pre_event=24)
    assert len(onsets) == len(events)
    assert onsets["state_index"].between(0, q.size - 1).all()


def test_pipeline_accepts_pot_delineation_too():
    q, _ = _synthetic_hydrograph(n_years=2.0)
    _, events = hydro_event_pipeline(
        q, event_method="POT", ey=6,
        method_kwargs={"threshold": 0.5, "min_diff": 24})
    assert len(events) == 12
    assert events["start"].is_monotonic_increasing


# ------------------------------------------------- events WRT rainfall
def test_events_wrt_rainfall_moves_starts_earlier():
    """The point of the function: begin at the rain, not at the rise."""
    from pyfloodrisk.hydroevents import events_WRT_rainfall
    from pyfloodrisk.demo_data import demo_paths
    path = demo_paths()["climate"] / "GR4H_climatedata_117002A_hr.csv"
    q = pd.read_csv(path)["qt"].to_numpy(float)
    _, events = hydro_event_pipeline(q, ey=None)
    out, _ = events_WRT_rainfall(path, events, ey=6)
    # every retained row starts no later than the runoff event it matched
    matched = events.loc[out["index"], "start"].to_numpy()
    assert (out["start"].to_numpy() <= matched).all()


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
