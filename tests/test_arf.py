"""ARR 2019 areal reduction factors.

The ARF multiplies every design depth, so an error here is a silent scaling
of the whole flood frequency curve.  These tests pin the equations against
values computed from the reference implementation, check the internal
consistency ARR itself provides, and check the branch structure -- the joins
between the short, long and interpolated bands are where a port goes wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyfloodrisk.dffa.ifd import (ARF_REGIONS, ARR2019ARF, IFDCurve,
                                  _arf_long, _arf_short, unit_arf)

REGION = "East Coast North"


# ------------------------------------------------------------- known values
#: (region, area_km2, duration_h, aep) -> ARF, from the reference
#: implementation this was ported from.
KNOWN = {
    ("East Coast North", 255.2, 24.0, 0.01): 0.928,
    ("East Coast North", 255.2, 72.0, 0.01): 0.959,
    ("East Coast North", 255.2, 12.0, 0.01): 0.886,
    ("East Coast North", 255.2, 18.0, 0.01): 0.907,
    ("Southern Temperate", 357.4, 24.0, 0.01): 0.928,
    ("Southern Temperate", 357.4, 72.0, 0.01): 0.949,
    ("Southern Temperate", 357.4, 12.0, 0.5): 0.906,
    ("Tasmania", 5.0, 48.0, 0.05): 0.985,
    ("Inland Arid", 20000.0, 168.0, 0.0005): 0.777,
    ("SE Coast", 100.0, 24.0, 0.1): 0.967,
}


@pytest.mark.parametrize("key,expected", sorted(KNOWN.items()))
def test_arf_matches_reference_values(key, expected):
    region, area, duration_h, aep = key
    assert ARR2019ARF(region=region)(area, duration_h, aep) == pytest.approx(
        expected, abs=5e-4)


# --------------------------------------------------------------- behaviour
@pytest.mark.parametrize("region", sorted(ARF_REGIONS))
def test_arf_is_a_reduction_for_every_region(region):
    """An ARF is a reduction: in (0, 1], and never an amplification."""
    arf = ARR2019ARF(region=region)
    for area in (2.0, 10.0, 255.2, 1000.0, 20000.0):
        for duration_h in (12.0, 18.0, 24.0, 48.0, 168.0):
            if duration_h <= 12.0 and area > 1000.0:
                continue                # outside ARR's stated applicability
            for aep in (0.5, 0.01, 0.0005):
                value = arf(area, duration_h, aep)
                assert 0.0 < value <= 1.0, (region, area, duration_h, aep)


@pytest.mark.parametrize("region", sorted(ARF_REGIONS))
def test_bigger_catchments_are_reduced_more(region):
    """The whole point of an ARF: rain is less uniform over a larger area."""
    arf = ARR2019ARF(region=region, round_to=None)
    for duration_h in (24.0, 72.0):
        values = [arf(a, duration_h, 0.01) for a in (10, 100, 1000, 10000)]
        assert np.all(np.diff(values) < 0), (region, duration_h, values)


@pytest.mark.parametrize("region", sorted(ARF_REGIONS))
def test_longer_bursts_are_reduced_less(region):
    """A long burst is closer to uniform over the catchment than a short one."""
    arf = ARR2019ARF(region=region, round_to=None)
    values = [arf(255.2, d, 0.01) for d in (24.0, 48.0, 96.0, 168.0)]
    assert np.all(np.diff(values) > 0), (region, values)


def test_a_small_catchment_gets_no_reduction():
    assert ARR2019ARF(region=REGION)(0.5, 24.0, 0.01) == 1.0


def test_the_small_area_scaling_meets_the_equation_at_10_km2():
    """The 1-10 km2 scaling must be continuous with the equation at 10 km2."""
    arf = ARR2019ARF(region=REGION, round_to=None)
    just_under = arf(9.999, 24.0, 0.01)
    at_ten = arf(10.0, 24.0, 0.01)
    assert just_under == pytest.approx(at_ten, abs=1e-3)


def test_the_interpolation_band_joins_both_equations():
    """12-24 h interpolates; it must meet short at 12 h and long at 24 h."""
    arf = ARR2019ARF(region=REGION, round_to=None)
    area, aep = 255.2, 0.01
    assert arf(area, 12.0, aep) == pytest.approx(
        _arf_short(area, 720.0, aep), abs=1e-9)
    assert arf(area, 24.0, aep) == pytest.approx(
        _arf_long(area, 1440.0, aep, ARF_REGIONS[REGION]), abs=1e-9)
    # and the band is monotone between the two ends
    band = [arf(area, d, aep) for d in (12.0, 15.0, 18.0, 21.0, 24.0)]
    assert np.all(np.diff(band) > 0)


def test_aep_is_clamped_not_rejected():
    """A derived-FFA run samples far outside the range ARR states.

    Clamping keeps the ARF flat outside the fitted range; rejecting would
    fail the whole Monte Carlo at the first rare stratum.
    """
    arf = ARR2019ARF(region=REGION, round_to=None)
    assert arf(255.2, 24.0, 1e-5) == arf(255.2, 24.0, 0.0005)
    assert arf(255.2, 24.0, 0.9) == arf(255.2, 24.0, 0.5)


# ------------------------------------------------------------- input checks
def test_unknown_region_is_rejected():
    with pytest.raises(ValueError, match="unknown ARF region"):
        ARR2019ARF(region="Narnia")


def test_custom_coefficients_must_be_complete():
    with pytest.raises(ValueError, match="missing ARF coefficients"):
        ARR2019ARF(region="anything", coeffs={"a": 0.1, "b": 0.2})


@pytest.mark.parametrize("area", [-1.0, 30001.0])
def test_area_outside_the_applicable_range_is_rejected(area):
    with pytest.raises(ValueError, match="area must be"):
        ARR2019ARF(region=REGION)(area, 24.0, 0.01)


@pytest.mark.parametrize("duration_h", [0.0, -3.0, 169.0])
def test_duration_outside_the_applicable_range_is_rejected(duration_h):
    with pytest.raises(ValueError, match="duration must be"):
        ARR2019ARF(region=REGION)(255.2, duration_h, 0.01)


def test_short_bursts_on_large_catchments_are_rejected():
    """ARR states the short-duration equation stops at 1000 km2."""
    with pytest.raises(ValueError, match="does not apply to catchments"):
        ARR2019ARF(region=REGION)(1500.0, 6.0, 0.01)
    # but the same catchment is fine on a long burst
    assert 0 < ARR2019ARF(region=REGION)(1500.0, 24.0, 0.01) < 1


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_inputs_are_rejected(bad):
    """The reference implementation's NaN guards never fired (`x == nan`)."""
    with pytest.raises(ValueError, match="finite"):
        ARR2019ARF(region=REGION)(bad, 24.0, 0.01)


# ------------------------------------------------------------- integration
def test_arf_reduces_the_design_depths_an_ifd_curve_returns():
    table = __import__("pandas").DataFrame(
        {0.5: [40.0, 60.0], 0.01: [90.0, 130.0]}, index=[24.0, 72.0])
    point = IFDCurve(table, arf=unit_arf)
    reduced = IFDCurve(table, arf=ARR2019ARF(region=REGION))
    for duration_h in (24.0, 72.0):
        p = float(point.depth(duration_h, 0.01, area_km2=255.2))
        r = float(reduced.depth(duration_h, 0.01, area_km2=255.2))
        assert 0 < r < p
    # and asking for no area at all leaves the point depth alone
    assert float(reduced.depth(24.0, 0.01)) == pytest.approx(90.0)
