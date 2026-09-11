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

from scipy.stats import norm

from pyfloodrisk.dffa.ifd import (ARF_REGIONS, ARR2019ARF, IFDCurve,
                                  ParabolicRareExtension,
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


# ------------------------------------------- depth interpolation domain
def _demo_table(station="117002A"):
    from pyfloodrisk.dffa.ifd import ifd_table_from_bom_csv
    from pyfloodrisk.demo_data import demo_paths
    return ifd_table_from_bom_csv(demo_paths()["ifd"] / f"{station}_ifds.csv")


@pytest.mark.parametrize("station", ["117002A", "405214"])
def test_arithmetic_and_log_agree_inside_the_tabulated_range(station):
    """ARR 4.2.1 prefers the arithmetic-normal domain; inside the table it
    barely matters, which is the point worth pinning."""
    t = _demo_table(station)
    log = IFDCurve(t, depth_interp="log")
    ari = IFDCurve(t, depth_interp="arithmetic")
    for duration_h in (24.0, 72.0):
        for aep in (0.3, 0.15, 0.03, 0.007, 0.001):     # all inside the table
            a = float(log.point_depth(duration_h, aep))
            b = float(ari.point_depth(duration_h, aep))
            assert abs(a / b - 1) < 0.01, (station, duration_h, aep, a, b)


def test_both_domains_reproduce_the_tabulated_depths_exactly():
    t = _demo_table()
    for mode in ("log", "arithmetic"):
        c = IFDCurve(t, depth_interp=mode)
        for aep in t.columns:
            got = float(c.point_depth(24.0, aep))
            assert got == pytest.approx(float(t.loc[24.0, aep]), rel=1e-9), mode


def test_arithmetic_extrapolates_below_log():
    """Past the table the two diverge: log grows exponentially in z."""
    t = _demo_table()
    log = IFDCurve(t, depth_interp="log")
    ari = IFDCurve(t, depth_interp="arithmetic")
    rarest = min(t.columns)
    for aep in (rarest / 5, rarest / 50):
        assert float(log.point_depth(24.0, aep)) > float(ari.point_depth(24.0, aep))


def test_depth_interp_is_validated():
    with pytest.raises(ValueError, match="depth_interp must be"):
        IFDCurve(_demo_table(), depth_interp="loglog")


# ------------------------------------------------ parabolic rare extension
def _anchor(table, factor=2.0):
    """An extreme anchor a given multiple of the rarest tabulated depth."""
    rarest = min(table.columns)
    return {float(d): float(table.loc[d, rarest]) * factor for d in table.index}


def test_parabolic_extension_hits_its_anchor():
    t = _demo_table()
    ext = ParabolicRareExtension(depths_mm=_anchor(t), aep=1e-7)
    c = IFDCurve(t, rare_extension=ext)
    for duration_h in (24.0, 72.0):
        want = _anchor(t)[duration_h]
        assert float(c.point_depth(duration_h, 1e-7)) == pytest.approx(want, rel=1e-6)


def test_parabolic_extension_is_tangent_at_the_join():
    """No kink where the extension takes over: value and slope both match."""
    t = _demo_table()
    c = IFDCurve(t, rare_extension=ParabolicRareExtension(_anchor(t), aep=1e-7))
    rarest = min(t.columns)
    # continuous in value at the rarest tabulated AEP
    assert float(c.point_depth(24.0, rarest)) == pytest.approx(
        float(t.loc[24.0, rarest]), rel=1e-9)
    # and in slope: one-sided gradients in (z, log depth) agree
    z0 = norm.ppf(1 - rarest)
    h = 1e-4
    def logd(z):
        return np.log(float(c.point_depth(24.0, 1 - norm.cdf(z))))
    inside = (logd(z0) - logd(z0 - h)) / h
    outside = (logd(z0 + h) - logd(z0)) / h
    assert inside == pytest.approx(outside, rel=5e-3)


def test_parabolic_extension_is_monotone_and_below_the_log_tail():
    """It curves toward the anchor, so it must sit under the straight tail."""
    t = _demo_table()
    plain = IFDCurve(t)
    ext = IFDCurve(t, rare_extension=ParabolicRareExtension(_anchor(t), aep=1e-7))
    aeps = np.array([4e-4, 1e-4, 1e-5, 1e-6, 1e-7])
    d = np.array([float(ext.point_depth(24.0, a)) for a in aeps])
    assert np.all(np.diff(d) > 0)                      # rarer is deeper
    p = np.array([float(plain.point_depth(24.0, a)) for a in aeps])
    assert np.all(d[1:] < p[1:])


def test_parabolic_extension_works_in_the_arithmetic_domain_too():
    t = _demo_table()
    c = IFDCurve(t, depth_interp="arithmetic",
                 rare_extension=ParabolicRareExtension(_anchor(t), aep=1e-7))
    assert float(c.point_depth(24.0, 1e-7)) == pytest.approx(
        _anchor(t)[24.0], rel=1e-6)


def test_rare_extension_rejects_a_too_frequent_anchor():
    t = _demo_table()
    with pytest.raises(ValueError, match="must be rarer than"):
        IFDCurve(t, rare_extension=ParabolicRareExtension(_anchor(t), aep=0.01))


def test_rare_extension_rejects_an_anchor_below_the_table():
    t = _demo_table()
    with pytest.raises(ValueError, match="must exceed the rarest tabulated"):
        IFDCurve(t, rare_extension=ParabolicRareExtension(
            _anchor(t, factor=0.5), aep=1e-7))


def test_rare_extension_rejects_an_anchor_that_turns_the_curve_over():
    """Too small an anchor gap and the parabola peaks before reaching it."""
    t = _demo_table()
    with pytest.raises(ValueError, match="turns over"):
        IFDCurve(t, rare_extension=ParabolicRareExtension(
            _anchor(t, factor=1.001), aep=1e-9))


def test_rare_extension_validates_its_own_inputs():
    with pytest.raises(ValueError, match="at least one anchor"):
        ParabolicRareExtension({}, aep=1e-7)
    with pytest.raises(ValueError, match="anchor depths must be positive"):
        ParabolicRareExtension({24.0: -1.0}, aep=1e-7)
    with pytest.raises(ValueError, match="anchor aep must lie"):
        ParabolicRareExtension({24.0: 900.0}, aep=1.5)
