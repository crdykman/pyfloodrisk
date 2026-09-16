"""Event delineation, and the exceedance-per-year trim applied after it."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyfloodrisk.hydroevents import (extract_initial_states,
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
