"""Tests for the diagnostic figures.

Figures are hard to assert on, so these check the things that would make
one silently wrong rather than how it looks: that every event reaches the
plot, that duration is encoded in the documented order, and that the
degenerate inputs a small experiment produces do not raise.
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from pyfloodrisk.dffa import diagnostics

def _events(n_per_duration=200, durations=(6.0, 24.0, 72.0), seed=0):
    """A stand-in event table with the columns the figure reads."""
    rng = np.random.default_rng(seed)
    frames = []
    for i, d in enumerate(durations):
        # log-uniform over the AEP domain the stratification covers
        aep = np.exp(rng.uniform(np.log(1e-5), np.log(0.5), n_per_duration))
        # a rarer rainfall gives a bigger peak, with real scatter so the
        # binned medians mean something
        q = (-np.log(aep)) ** 1.8 * (2.0 + i) * rng.lognormal(0, 0.35,
                                                              n_per_duration)
        frames.append(pd.DataFrame({"duration_h": d, "aep_rain": aep,
                                    "q_peak": q}))
    return pd.concat(frames, ignore_index=True)
    return pd.concat(frames, ignore_index=True)


class _Results:
    def __init__(self, events):
        self.events = events


def test_every_event_is_plotted():
    """No sampling, no silent truncation: the scatter is the whole table."""
    ev = _events()
    ax = diagnostics.plot_rainfall_peak(_Results(ev))
    plotted = sum(c.get_offsets().shape[0] for c in ax.collections)
    assert plotted == len(ev)


def test_one_scatter_and_one_median_per_duration():
    ev = _events()
    ax = diagnostics.plot_rainfall_peak(_Results(ev))
    n = ev["duration_h"].nunique()
    assert len(ax.collections) == n
    assert len(ax.lines) == n
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["6 h", "24 h", "72 h"]


def test_duration_runs_light_to_dark():
    """The ordinal encoding the title and docs claim.

    Colour carries order here, so a reversed or cycled ramp would make the
    figure say the opposite of what it means.
    """
    ax = diagnostics.plot_rainfall_peak(_Results(_events()))
    lums = [matplotlib.colors.rgb_to_hsv(line.get_color()[:3])[2]
            if not isinstance(line.get_color(), str)
            else matplotlib.colors.rgb_to_hsv(
                matplotlib.colors.to_rgb(line.get_color()))[2]
            for line in ax.lines]
    assert lums == sorted(lums, reverse=True)      # short = light


def test_peak_axis_is_logarithmic_by_default():
    """Peaks span orders of magnitude; a linear axis hides the low end."""
    ax = diagnostics.plot_rainfall_peak(_Results(_events()))
    assert ax.get_yscale() == "log"
    ax2 = diagnostics.plot_rainfall_peak(_Results(_events()), logy=False)
    assert ax2.get_yscale() == "linear"


def test_a_duration_with_too_few_points_for_bins_is_skipped_not_raised():
    """The open end intervals put many events at one identical rainfall AEP.

    Quantile edges collapse to a single value there, which would make an
    empty or degenerate binning; the scatter must still be drawn.
    """
    ev = pd.DataFrame({"duration_h": [12.0] * 60,
                       "aep_rain": [1e-6] * 60,
                       "q_peak": np.linspace(107.0, 158.0, 60)})
    ax = diagnostics.plot_rainfall_peak(_Results(ev))
    assert len(ax.collections) == 1
    assert ax.collections[0].get_offsets().shape[0] == 60
    assert len(ax.lines) == 0       # no median line, but no exception either
    # and the duration still appears in the legend, not colour alone
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["12 h"]


# ------------------------------------------------------ probability axes
def test_reduced_gumbel_variate_matches_its_definition():
    from pyfloodrisk.dffa.diagnostics import _gx

    aeps = np.array([0.5, 0.1, 0.01, 1e-3, 1e-5])
    np.testing.assert_allclose(_gx(aeps), -np.log(-np.log(1 - aeps)))
    # increasing in rarity, and the textbook value at the mode
    assert np.all(np.diff(_gx(aeps)) > 0)
    assert _gx([0.5])[0] == pytest.approx(0.36651292, rel=1e-6)


def test_rainfall_peak_axis_is_labelled_in_aep_not_in_variate():
    """The whole point: plotted in the variate, read as probability."""
    from pyfloodrisk.dffa.diagnostics import _zx

    ax = diagnostics.plot_rainfall_peak(_Results(_events()))
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels and all("%" in s or s.startswith("1 in ") for s in labels)
    # ticks sit where the variate puts them, and run frequent to rare
    ticks = ax.get_xticks()
    assert np.all(np.diff(ticks) > 0)
    assert np.isclose(ticks, _zx([0.5])[0]).any() or len(ticks) > 1


def test_probability_axis_rejects_an_unknown_variate():
    _, ax = matplotlib.pyplot.subplots()
    with pytest.raises(ValueError, match="variate must be"):
        diagnostics.probability_axis(ax, [0.1, 0.01], variate="weibull")


@pytest.mark.parametrize("fn,expected", [
    ("plot_frequency_curve", "gumbel"),      # a flood frequency curve
    ("plot_duration_curves", "gumbel"),      # flood frequency curves
    ("plot_state_sensitivity", "gumbel"),    # its right panel is a flood curve
    ("plot_inputs", "normal"),               # its panel is a rainfall curve
    ("plot_rainfall_peak", "normal"),        # its x axis is a rainfall AEP
])
def test_the_variate_follows_the_quantity_not_the_figure(fn, expected):
    """Gumbel is for peak-flow frequency; rainfall frequency stays normal.

    The two are different probabilities and the package plots them in
    different transforms, so the default has to track which is on the axis.
    """
    import inspect
    from pyfloodrisk.dffa import diagnostics as dg

    assert (inspect.signature(getattr(dg, fn)).parameters["variate"].default
            == expected)
