"""Tests for design storm construction."""

import pandas as pd
import pytest

from pyfloodrisk.design_storm import build_design_storm


def test_build_design_storm_default_returns_expected_keys():
    result = build_design_storm()
    assert set(result.keys()) == {
        "station",
        "intensity_mm",
        "preburst_depth",
        "temporal_patterns",
        "rainfall_matrix",
        "rainfall_long",
    }


def test_build_design_storm_preburst_depth():
    result = build_design_storm(preburst_index=6, preburst_factor=1.0)
    assert result["preburst_depth"] == pytest.approx(13.4)


def test_build_design_storm_intensity_scales():
    result_100 = build_design_storm(intensity_mm=100.0)
    result_200 = build_design_storm(intensity_mm=200.0)
    # All non-pre-burst rainfall should scale proportionally
    rain_cols = [c for c in result_100["rainfall_matrix"].columns if c not in ("No", "pre")]
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


def test_build_design_storm_invalid_preburst_index_raises():
    with pytest.raises(ValueError, match="preburst index"):
        build_design_storm(preburst_index=9999)
