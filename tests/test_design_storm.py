"""Tests for design storm construction."""

import numpy as np
import pandas as pd
import pytest

from pyfloodrisk.design_storm import build_design_storm


def test_build_design_storm_default_returns_expected_keys():
    result = build_design_storm()
    assert set(result.keys()) == {
        "station",
        "intensity_mm",
        "temporal_patterns",
        "rainfall_matrix",
    }


def test_build_design_storm_is_burst_only():
    """No pre-burst column: the matrix is the burst and nothing else.

    Antecedent wetness is carried by the initial state, not by rain
    prepended to the storm.
    """
    m = build_design_storm()["rainfall_matrix"]
    assert "pre" not in m.columns
    assert list(m.columns[:1]) == ["No"]
    # every remaining column is an hour of the burst
    assert all(c != "pre" for c in m.columns[1:])


def test_build_design_storm_total_depth_is_the_intensity():
    """With no pre-burst depth added, each pattern sums to intensity_mm.

    The patterns are percentages of the burst depth, so this is the check
    that nothing is added to or held back from the total.
    """
    result = build_design_storm(intensity_mm=107.0)
    m = result["rainfall_matrix"]
    totals = m.drop(columns="No").sum(axis=1)
    assert totals.min() == pytest.approx(107.0, rel=1e-6)
    assert totals.max() == pytest.approx(107.0, rel=1e-6)


def test_build_design_storm_intensity_scales():
    result_100 = build_design_storm(intensity_mm=100.0)
    result_200 = build_design_storm(intensity_mm=200.0)
    rain_cols = [c for c in result_100["rainfall_matrix"].columns if c != "No"]
    total_100 = result_100["rainfall_matrix"][rain_cols].values.sum()
    total_200 = result_200["rainfall_matrix"][rain_cols].values.sum()
    assert total_200 == pytest.approx(total_100 * 2, rel=1e-6)


def test_build_design_storm_ignores_aep_band_for_areal_patterns():
    """Areal patterns have no AEP dependence, so the band index does nothing.

    They are published per standard catchment area instead, and the ensemble
    is chosen by the station's own area.
    """
    default = build_design_storm()
    other = build_design_storm(aep_band_index=9999)
    pd.testing.assert_frame_equal(default["rainfall_matrix"],
                                  other["rainfall_matrix"])


def test_build_design_storm_rejects_a_duration_areal_patterns_lack():
    with pytest.raises(ValueError, match="only go down to 12 h"):
        build_design_storm(duration_hours=6)


# ------------------------------------------------ the file's own timestep
ALL_DURATIONS_H = [12, 18, 24, 36, 48, 72, 96, 120, 144, 168]


@pytest.mark.parametrize("duration_hours", ALL_DURATIONS_H)
def test_every_bundled_duration_builds(duration_hours):
    """ARR coarsens the increment as the storm lengthens.

    The timestep is read from the file's own ``TimeStep`` column, so every
    duration the increments file covers is reachable -- not just the one
    that happens to be on a 30-minute step.
    """
    m = build_design_storm(duration_hours=duration_hours)["rainfall_matrix"]
    assert m.shape[1] - 1 == duration_hours      # one column per hour


@pytest.mark.parametrize("duration_hours", ALL_DURATIONS_H)
def test_resampling_to_hourly_conserves_the_depth(duration_hours):
    """Disaggregating a 3 h increment must not create or destroy rain."""
    result = build_design_storm(duration_hours=duration_hours,
                                intensity_mm=107.0)
    totals = result["rainfall_matrix"].drop(columns="No").sum(axis=1)
    assert totals.min() == pytest.approx(107.0, rel=1e-9)
    assert totals.max() == pytest.approx(107.0, rel=1e-9)


def test_twelve_hour_storm_is_unchanged_by_reading_the_timestep():
    """The one duration that worked before must still give the same numbers."""
    from_file = build_design_storm(duration_hours=12)
    explicit = build_design_storm(duration_hours=12, timestep_minutes=30)
    pd.testing.assert_frame_equal(from_file["rainfall_matrix"],
                                  explicit["rainfall_matrix"])


def test_a_wrong_timestep_override_is_rejected_not_silently_truncated():
    """Overriding 168 h to 30 min once gave a 28 h storm with no complaint."""
    with pytest.raises(ValueError, match="needs 336 increments"):
        build_design_storm(duration_hours=168, timestep_minutes=30)


def test_temporal_patterns_are_hourly_and_match_the_matrix():
    result = build_design_storm(duration_hours=72)
    m, tp = result["rainfall_matrix"], result["temporal_patterns"]
    assert len(tp) == m.shape[0] * (m.shape[1] - 1)
    assert set(tp.columns) == {"No", "Date", "Percent"}
    gaps = tp.groupby("No")["Date"].diff().dropna().unique()
    assert list(gaps) == [pd.Timedelta(hours=1)]


@pytest.mark.parametrize("duration_hours", [12, 24, 72])
def test_the_matrix_is_ordered_by_hour_across_its_columns(duration_hours):
    """One column per hour of the storm, in order, with no index column.

    ``simulate_design_flood`` slices whole rows straight into the
    precipitation series, so a scrambled column order would reorder the
    storm silently rather than raising.
    """
    m = build_design_storm(duration_hours=duration_hours)["rainfall_matrix"]
    assert list(m.columns[:1]) == ["No"]
    assert m.shape[1] - 1 == duration_hours
    hours = pd.to_datetime(list(m.columns[1:]))
    assert list(hours.diff()[1:]) == [pd.Timedelta(hours=1)] * (duration_hours - 1)


@pytest.mark.parametrize("duration_hours", [12, 24, 72])
def test_matrix_rows_are_what_the_simulation_consumes(duration_hours):
    """The wide form is the only rainfall the design flood needs.

    ``rainfall_long`` used to carry the same numbers in melted form, and
    the simulation undid the melt with a groupby to get back to this.
    """
    result = build_design_storm(duration_hours=duration_hours)
    rows = result["rainfall_matrix"].drop(columns="No").to_numpy(float)
    assert rows.shape == (len(result["rainfall_matrix"]), duration_hours)
    np.testing.assert_allclose(rows.sum(axis=1), 107.0, rtol=1e-9)
