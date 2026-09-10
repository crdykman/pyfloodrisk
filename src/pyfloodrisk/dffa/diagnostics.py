"""
Diagnostic figures.

Design conventions used here (they are deliberate, not decoration):

* **Duration and stratum are ordinal**, so they are encoded with a single-hue
  light-to-dark blue ramp rather than a categorical palette -- eleven cycled
  hues would be unreadable and would imply the durations are unrelated
  categories.  The enveloped curve, being a different kind of thing, gets the
  contrasting categorical slot (orange).
* **Design AEPs compared against each other are categorical** and use the first
  three slots of a colour-vision-deficiency-validated categorical order.
* No dual axes anywhere; wide-range axes are transformed explicitly (probability
  axis in the standard normal variate, discharge in log space when asked).
* Grid and axes are recessive; labels sit in text ink, never in the series
  colour, so identity is never carried by colour alone.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from .tpt import weighted_quantile

__all__ = [
    "PALETTE", "apply_style", "probability_axis",
    "plot_frequency_curve", "plot_duration_curves", "plot_stratum_contributions",
    "plot_inputs", "plot_state_sensitivity", "plot_convergence",
    "plot_hydrographs", "plot_curve_comparison", "plot_all",
]

#: Validated categorical order (first three slots are all-pairs safe).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

#: Single-hue ordinal ramp (blue steps 250 -> 700), for ordered categories.
ORDINAL_BLUE = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
                "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

#: Continuous sequential ramp (blue steps 100 -> 700), for magnitude.
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                 "#256abf", "#184f95", "#0d366b"])

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

_STD_AEPS = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001,
             0.0005, 0.0002, 0.0001]


def apply_style():
    """Recessive, print-safe rcParams."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.size": 9,
        "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
        "axes.titlecolor": INK, "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "legend.frameon": False, "lines.linewidth": 2.0,
        "figure.dpi": 130,
    })


def ordinal_colours(n: int) -> list[str]:
    cmap = LinearSegmentedColormap.from_list("ord", ORDINAL_BLUE)
    return [mpl.colors.to_hex(cmap(x)) for x in np.linspace(0, 1, max(n, 2))][:n]


def probability_axis(ax, aeps: Sequence[float] = None,
                     label="Annual exceedance probability", max_ticks: int = 7):
    """Label an x axis whose coordinate is the standard normal variate of AEP.

    All candidate AEPs become minor ticks; at most ``max_ticks`` are labelled,
    spread evenly in the plotted coordinate, so the rare end does not turn into
    a stack of overlapping text.
    """
    aeps = sorted(set(_STD_AEPS if aeps is None else np.asarray(aeps).ravel().tolist()),
                  reverse=True)
    z = _zx(aeps)
    ax.set_xticks(z, minor=True)
    # thin greedily in the *plotted* coordinate: the AEP list is far from
    # uniform in z, so decimating by list position still collides at the tail
    span = z[-1] - z[0]
    min_sep = span / max(max_ticks - 1, 1) if span > 0 else np.inf
    keep = [0]
    for i in range(1, len(z) - 1):
        if z[i] - z[keep[-1]] >= min_sep:
            keep.append(i)
    if len(z) > 1 and z[-1] - z[keep[-1]] >= 0.6 * min_sep:
        keep.append(len(z) - 1)
    ax.set_xticks(z[keep])
    ax.set_xticklabels([_fmt_aep(aeps[i]) for i in keep])
    ax.set_xlabel(label)
    return ax


def _fmt_aep(a: float) -> str:
    if a >= 0.01:
        return f"{a*100:.3g}%"
    return f"1 in {1/a:.0f}"


def _zx(aeps):
    return norm.ppf(1 - np.asarray(aeps, float))


# --------------------------------------------------------------------- figures
def plot_frequency_curve(results, aeps=None, observed=None, ax=None,
                         n_boot=400, level=0.90, logy=True, title=None):
    """Enveloped derived frequency curve with Monte Carlo confidence limits.

    ``observed`` may be a ``DataFrame`` with columns ``aep`` and ``q_peak``
    (e.g. an at-site LP3 fit or plotting positions of the gauged annual maxima)
    for comparison.
    """
    apply_style()
    aeps = np.asarray(aeps if aeps is not None
                      else results.valid_aeps(_STD_AEPS), float)
    ci = results.confidence(aeps, n_boot=n_boot, level=level)
    ax = ax or plt.subplots(figsize=(6.4, 4.4))[1]
    x = _zx(aeps)

    ax.fill_between(x, ci["lower"], ci["upper"], color=PALETTE[0], alpha=0.16,
                    linewidth=0, zorder=1)
    ax.plot(x, ci["q_peak"], color=PALETTE[0], zorder=3,
            label="Derived (GR4H Monte Carlo, duration envelope)")
    if observed is not None:
        obs = pd.DataFrame(observed)
        ax.plot(_zx(obs["aep"]), obs["q_peak"], marker="o", markersize=5,
                linestyle="none", color=PALETTE[1], markeredgecolor=SURFACE,
                markeredgewidth=1.2, zorder=4, label="Observed / at-site FFA")

    if logy:
        ax.set_yscale("log")
    ax.set_ylabel("Peak discharge (m$^3$ s$^{-1}$)")
    probability_axis(ax, aeps)
    ax.set_title(title or "Derived flood frequency curve")
    ax.legend(loc="upper left")
    # direct label of the design point, selectively
    if 0.01 in set(np.round(aeps, 10)):
        q01 = float(ci.loc[0.01, "q_peak"])
        ax.annotate(f"1% AEP  {q01:,.0f} m$^3$ s$^{{-1}}$",
                    xy=(_zx([0.01])[0], q01), xytext=(6, -14),
                    textcoords="offset points", color=INK2, fontsize=8.5)
    ax.figure.tight_layout()
    return ax


def plot_duration_curves(results, aeps=None, ax=None, logy=True):
    """One curve per storm duration, with the envelope over the top."""
    apply_style()
    aeps = np.asarray(aeps if aeps is not None
                      else results.valid_aeps(_STD_AEPS), float)
    q = results.quantiles(aeps)
    env = results.envelope(aeps)
    ax = ax or plt.subplots(figsize=(6.8, 4.6))[1]
    x = _zx(aeps)
    cols = ordinal_colours(q.shape[1])
    ax.plot(x, env["q_peak"], color=PALETTE[1], linewidth=2.6, zorder=3,
            label="Envelope")
    for c, d in zip(cols, q.columns):
        ax.plot(x, q[d], color=c, linewidth=1.4, zorder=2, label=f"{d:g} h")
    if logy:
        ax.set_yscale("log")
    ax.set_ylabel("Peak discharge (m$^3$ s$^{-1}$)")
    probability_axis(ax, aeps)
    ax.set_title("Quantiles by storm duration (blue, light to dark = short to long)")
    ax.legend(loc="upper left", ncols=2, fontsize=8, columnspacing=1.2,
              handlelength=1.6)
    ax.margins(x=0.03)
    ax.figure.tight_layout()
    return ax


def plot_stratum_contributions(results, aeps=(0.1, 0.01, 0.001), fig=None):
    """Which rainfall strata generate the exceedance probability of each quantile.

    Left: the per-stratum share for the middle AEP (noisy by nature -- it is a
    weighted count of a few hundred events per stratum).  Right: the cumulative
    share, which is what you read the answer off.  The annotated point is the
    rainfall AEP at which half the flood's exceedance probability has
    accumulated -- for a catchment where antecedent conditions matter this sits
    at a *more frequent* rainfall than the flood AEP, and by how much is the
    substantive result.

    If a curve reaches 1 only at the last stratum, or starts above 0 at the
    first, the stratification does not bracket the probability range of interest
    and should be extended.
    """
    apply_style()
    aeps = list(aeps)
    contrib = results.stratum_contributions(aeps)
    mid_aep = results.stratification.interval_mid_aep
    fig = fig or plt.figure(figsize=(9.2, 3.9))
    ax1, ax2 = fig.subplots(1, 2)

    mid = aeps[len(aeps) // 2]
    ax1.bar(np.arange(len(mid_aep)), contrib.loc[mid].to_numpy(), width=0.8,
            color=PALETTE[0], edgecolor=SURFACE, linewidth=0.8)
    ax1.set_xlabel("Rainfall interval (frequent to rare)")
    ax1.set_ylabel("Share of exceedance probability")
    ax1.set_title(f"Per-stratum share, {_fmt_aep(mid)} AEP flood")
    ax1.grid(axis="x", visible=False)

    for i, a in enumerate(aeps):
        c = np.cumsum(contrib.loc[a].to_numpy())
        ax2.plot(mid_aep, c, color=PALETTE[i % 3], linewidth=1.8,
                 label=f"{_fmt_aep(a)} AEP flood")
        half = np.interp(0.5, c, mid_aep) if c[-1] > 0.5 else np.nan
        if np.isfinite(half):
            ax2.plot([half], [0.5], marker="o", markersize=5, color=PALETTE[i % 3],
                     markeredgecolor=SURFACE, markeredgewidth=1.2)
            ax2.annotate(f"{_fmt_aep(half)} rain", xy=(half, 0.5),
                         xytext=(5, -12 - 11 * i), textcoords="offset points",
                         color=INK2, fontsize=8)
    ax2.set_xscale("log")
    ax2.invert_xaxis()
    ax2.axhline(0.5, color=MUTED, linewidth=0.8, zorder=0)
    ax2.set_xlabel("Rainfall AEP of the stratum")
    ax2.set_ylabel("Cumulative share")
    ax2.set_title("Cumulative contribution")
    ax2.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def plot_inputs(results, ifd=None, states=None, fig=None):
    """Check that the sampled inputs are what you think they are."""
    apply_style()
    ev = results.events
    fig = fig or plt.figure(figsize=(9.2, 6.4))
    axs = fig.subplots(2, 2)

    # 1. sampled depths against the fitted IFD curve, critical-ish duration
    d0 = results.durations[len(results.durations) // 2]
    s = ev[ev["duration_h"] == d0]
    ax = axs[0, 0]
    ax.plot(_zx(s["aep_rain"]), s["depth_mm"], ".", color=PALETTE[0],
            markersize=2.5, alpha=0.5, label="sampled events")
    if ifd is not None:
        aa = np.logspace(np.log10(results.stratification.edges[-1]),
                         np.log10(results.stratification.edges[0]), 200)
        ax.plot(_zx(aa), np.atleast_1d(
            ifd.depth(d0, aa, results.config.area_km2 if results.config.apply_arf else None)),
            color=PALETTE[1], linewidth=1.8, label="fitted IFD x ARF")
    probability_axis(ax, [a for a in _STD_AEPS
                          if results.stratification.edges[-1] <= a
                          <= results.stratification.edges[0]], max_ticks=5)
    ax.set_ylabel("Burst depth (mm)")
    ax.set_title(f"Rainfall sampling, {d0:g} h")
    ax.legend(loc="upper left")

    # 2. temporal pattern usage
    ax = axs[0, 1]
    counts = s["pattern_index"].value_counts().sort_index()
    ax.bar(counts.index + 0.0, counts.to_numpy(), width=0.72,
           color=PALETTE[0], edgecolor=SURFACE, linewidth=1.0)
    ax.set_xlabel("Ensemble member")
    ax.set_ylabel("Times sampled")
    ax.set_title("Temporal pattern usage")
    ax.grid(axis="x", visible=False)

    # 3. initial states, as a fraction of capacity so the two are comparable
    ax = axs[1, 0]
    if states is not None:
        p_ = ev["prod_store"] / states.x1
        r_ = ev["rout_store"] / states.x3
        xlab = "Store level / capacity (-)"
    else:
        p_, r_ = ev["prod_store"], ev["rout_store"]
        xlab = "Store level (mm)"
    bins = np.linspace(0, max(p_.max(), r_.max()), 41)
    ax.hist(p_, bins=bins, color=PALETTE[0], density=True, histtype="stepfilled",
            alpha=0.25, linewidth=0)
    ax.hist(r_, bins=bins, color=PALETTE[1], density=True, histtype="stepfilled",
            alpha=0.25, linewidth=0)
    ax.hist(p_, bins=bins, color=PALETTE[0], density=True, histtype="step",
            linewidth=1.8, label="production store")
    ax.hist(r_, bins=bins, color=PALETTE[1], density=True, histtype="step",
            linewidth=1.8, label="routing store")
    ax.set_xlabel(xlab)
    ax.set_ylabel("Density")
    ax.set_title("Sampled initial states")
    ax.legend(loc="upper right")

    # 4. pre-burst (or joint state scatter if pre-burst is off)
    ax = axs[1, 1]
    if ev["preburst_depth_mm"].max() > 0:
        ax.hist(ev["preburst_ratio"], bins=40, color=PALETTE[2], alpha=0.9)
        ax.set_xlabel("Pre-burst depth / burst depth")
        ax.set_ylabel("Count")
        ax.set_title("Sampled pre-burst ratio")
    else:
        ax.plot(ev["prod_store"], ev["rout_store"], ".", markersize=1.8,
                alpha=0.35, color=PALETTE[0])
        ax.set_xlabel("Production store (mm)")
        ax.set_ylabel("Routing store (mm)")
        ax.set_title("Joint initial states (no pre-burst sampled)")
    fig.tight_layout()
    return fig


def plot_state_sensitivity(results, aep=0.01, n_bins=4, fig=None):
    """How much of the flood quantile is set by the antecedent state.

    Left: peaks against production store level, coloured by rainfall AEP.
    Right: the frequency curve recomputed within quantile bands of the initial
    production store -- i.e. the curve you would get if antecedent wetness were
    fixed rather than sampled.  The spread between bands is the part of the
    design estimate that the continuous-simulation state distribution is
    controlling.
    """
    apply_style()
    env = results.envelope([aep])
    d = float(env["critical_duration_h"].iloc[0])
    s = results.subset(d).copy()
    fig = fig or plt.figure(figsize=(9.2, 3.9))
    ax1, ax2 = fig.subplots(1, 2)

    z = _zx(s["aep_rain"])
    sc = ax1.scatter(s["prod_store"], s["q_peak"], c=z, s=5, cmap=SEQ_BLUE,
                     linewidths=0, alpha=0.7)
    cb = fig.colorbar(sc, ax=ax1)
    ticks = [a for a in _STD_AEPS if a >= results.stratification.edges[-1]][:6]
    cb.set_ticks(_zx(ticks))
    cb.set_ticklabels([_fmt_aep(a) for a in ticks])
    cb.set_label("Rainfall AEP", color=INK2)
    ax1.set_yscale("log")
    ax1.set_xlabel("Initial production store (mm)")
    ax1.set_ylabel("Peak discharge (m$^3$ s$^{-1}$)")
    ax1.set_title(f"Peaks vs antecedent wetness, {d:g} h")

    qs = np.quantile(s["prod_store"], np.linspace(0, 1, n_bins + 1))
    aeps = np.asarray(results.valid_aeps(_STD_AEPS), float)
    cols = ordinal_colours(n_bins)
    for i in range(n_bins):
        m = (s["prod_store"] >= qs[i]) & (s["prod_store"] <= qs[i + 1])
        sub = s[m]
        # conditioning on the state changes the weights: within each stratum the
        # retained events must still carry the stratum's full probability mass
        n_full = s.groupby("stratum")["q_peak"].size()
        n_sub = sub.groupby("stratum")["q_peak"].size()
        w = (sub["weight"] * sub["stratum"].map(n_full)
             / sub["stratum"].map(n_sub)).to_numpy()
        ax2.plot(_zx(aeps), np.atleast_1d(weighted_quantile(
                     sub["q_peak"].to_numpy(), w, aeps)),
                 color=cols[i], linewidth=1.7,
                 label=f"store {100*i/n_bins:.0f}-{100*(i+1)/n_bins:.0f}th pct")
    ax2.set_yscale("log")
    probability_axis(ax2, aeps)
    ax2.set_ylabel("Peak discharge (m$^3$ s$^{-1}$)")
    ax2.set_title("Curve conditional on antecedent wetness")
    ax2.legend(loc="upper left", fontsize=7.5)
    fig.tight_layout()
    return fig


def plot_convergence(results, aeps=(0.01, 0.001), n_rep=5, ax=None):
    """Is the number of simulations per stratum enough?

    Line is the mean over independent subsamples, band is their range: read the
    point where the band narrows to a width you can live with.
    """
    apply_style()
    conv = results.convergence(list(aeps), n_rep=n_rep)
    ax = ax or plt.subplots(figsize=(6.2, 3.9))[1]
    for i, a in enumerate(aeps):
        c = conv[conv["aep"] == a]
        ref = float(c[c["fraction"] == c["fraction"].max()]["q_peak"].mean())
        g = c.groupby("n_per_stratum")["q_peak"]
        n = np.asarray(list(g.groups.keys()), float)
        pct = lambda v: 100 * (v / ref - 1)
        ax.fill_between(n, pct(g.min().to_numpy()), pct(g.max().to_numpy()),
                        color=PALETTE[i % 3], alpha=0.15, linewidth=0)
        ax.plot(n, pct(g.mean().to_numpy()), color=PALETTE[i % 3], marker="o",
                markersize=4, label=f"{_fmt_aep(a)} AEP")
    ax.axhline(0, color=MUTED, linewidth=0.8, zorder=0)
    ax.set_xlabel("Simulations per stratum")
    ax.set_ylabel("Departure from full sample (%)")
    ax.set_title("Convergence of the enveloped quantiles")
    ax.legend(loc="upper right")
    ax.figure.tight_layout()
    return ax


def plot_hydrographs(results, duration_h=None, n=None, ax=None):
    """One simulated event per target rainfall AEP, frequent to rare.

    The events are those retained by :attr:`MCSConfig.hydrograph_aeps`, so
    the figure spans the design range rather than whichever events happened
    to be simulated first.  Each line is labelled with the *sampled*
    rainfall AEP, which is the one nearest the target rather than the
    target itself.

    Note these are single realisations: the state and the temporal pattern
    were drawn at random for each, so one line being above another is not
    evidence about the design flood -- read that off the frequency curve.
    """
    apply_style()
    duration_h = duration_h or results.durations[len(results.durations) // 2]
    kept = list(results.hydrographs.get(float(duration_h), []))
    if n is not None:
        kept = kept[:n]
    if not kept:
        raise ValueError("no hydrographs retained; set MCSConfig.store_hydrographs > 0")
    ax = ax or plt.subplots(figsize=(6.8, 4.0))[1]
    dt = results.config.dt_hours
    cols = ordinal_colours(len(kept))
    for c, rec in zip(cols, kept):
        q = rec["q"]
        ax.plot(np.arange(q.size) * dt, q, color=c, linewidth=1.3,
                label=f"{rec['aep_rain']:.3g} AEP  (1 in {1 / rec['aep_rain']:.0f} y)")
    ax.set_xlabel("Time from start of event (h)")
    ax.set_ylabel("Discharge (m$^3$ s$^{-1}$)")
    ax.set_title(f"Simulated hydrographs by rainfall AEP, {duration_h:g} h storms")
    ax.legend(title="rainfall AEP", fontsize="small", frameon=False)
    ax.figure.tight_layout()
    return ax


def plot_curve_comparison(runs, aeps=None, ax=None, logy=True,
                          title="Effect of the state-sampling scheme",
                          ratio_to=None):
    """Overlay enveloped curves from several runs -- e.g. state-sampling methods.

    ``runs`` is a mapping ``{label: DFFAResults}``.  Pass ``ratio_to=<label>``
    to plot each curve as a ratio to that one, which is the more readable form
    when the curves are close: the question is usually "how much does this
    choice move the design estimate", not "what is the estimate".

    At most four labels -- categorical colour is a fixed-order palette, not a
    cycled one, and a fifth series would have to fold into "other".
    """
    apply_style()
    labels = list(runs)
    if len(labels) > 4:
        raise ValueError("at most four runs; facet instead of adding colours")
    if aeps is None:
        common = None
        for r in runs.values():
            v = set(r.valid_aeps(_STD_AEPS))
            common = v if common is None else (common & v)
        aeps = sorted(common, reverse=True)
    aeps = np.asarray(aeps, float)
    ax = ax or plt.subplots(figsize=(6.4, 4.2))[1]
    x = _zx(aeps)

    base = (np.asarray(runs[ratio_to].envelope(aeps)["q_peak"], float)
            if ratio_to is not None else None)
    for i, lab in enumerate(labels):
        q = np.asarray(runs[lab].envelope(aeps)["q_peak"], float)
        y = q / base if base is not None else q
        ax.plot(x, y, color=PALETTE[i % 3 if len(labels) <= 3 else i],
                linewidth=2.0 if lab != ratio_to else 1.4,
                linestyle="-" if lab != ratio_to else "--", label=lab)
    if base is not None:
        ax.axhline(1.0, color=MUTED, linewidth=0.8, zorder=0)
        ax.set_ylabel(f"Peak discharge / {ratio_to} (-)")
    else:
        if logy:
            ax.set_yscale("log")
        ax.set_ylabel("Peak discharge (m$^3$ s$^{-1}$)")
    probability_axis(ax, aeps)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8.5)
    ax.figure.tight_layout()
    return ax


def plot_all(results, outdir=".", ifd=None, states=None, observed=None,
             prefix="dffa"):
    """Write the whole diagnostic set as PNGs; returns the file list."""
    import os
    os.makedirs(outdir, exist_ok=True)
    files = []

    def save(obj, name):
        fig = obj.figure if hasattr(obj, "figure") else obj
        p = os.path.join(outdir, f"{prefix}_{name}.png")
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        files.append(p)

    save(plot_frequency_curve(results, observed=observed), "frequency_curve")
    save(plot_duration_curves(results), "duration_curves")
    save(plot_stratum_contributions(results), "stratum_contributions")
    save(plot_inputs(results, ifd=ifd, states=states), "inputs")
    save(plot_state_sensitivity(results), "state_sensitivity")
    save(plot_convergence(results), "convergence")
    try:
        save(plot_hydrographs(results), "hydrographs")
    except ValueError:
        pass
    return files
