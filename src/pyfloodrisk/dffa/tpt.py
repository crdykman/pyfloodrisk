"""
Empirical flood quantiles by the total probability theorem.

Each simulated event *i* has a peak discharge ``q_i``, a probability weight
``w_i`` and belongs to a rainfall interval.  ARR Book 4, Section 4.3.3.3 writes
the estimator interval-wise,

    P(Q > q) = Σ_i p[R_i] · P(Q > q | R_i)

with ``p[R_i]`` the width of interval *i* in the probability domain and
``P(Q > q | R_i)`` the proportion ``n/N`` of that interval's simulations whose
peak exceeds ``q``.  Two estimators are provided:

:func:`interval_exceedance_curve` / :func:`interval_quantile`
    The ARR form, interval by interval, **including the end-interval
    treatment**: at the open ends the conditional exceedance probability is
    replaced by a geometric mean, because the conditional probability varies
    highly non-linearly across an unbounded interval.  For the frequent end it
    is the geometric mean of ``c`` and ``first_factor·c`` (ARR suggests 0.1),
    i.e. ``c·sqrt(first_factor)``; for the rare end, the geometric mean of
    ``c`` and 1, i.e. ``sqrt(c)`` — the assumption being that as rainfall
    becomes arbitrarily rare, exceedance of any given threshold becomes
    certain.  This is the default.

:func:`weighted_quantile` / :func:`weighted_exceedance`
    The pooled weighted empirical exceedance function, which is what the
    interval-wise form collapses to when every interval is closed and treated
    identically.  Simpler, exact inside the sampled range, and truncated
    outside it.  Used by the analytical unit test and available via
    ``estimator="truncated"``.

Conventions worth being explicit about:

* **Plotting positions.**  The conditional exceedance probability inside an
  interval is a plotting-position estimate rather than the raw ``n/N``:

      P(Q > q | R_k) = (n_k(q) - a) / (N_k + 1 - 2a)

  with **Cunnane's a = 0.4** by default (approximately quantile-unbiased
  across the distributions used in flood frequency work; Cunnane, 1978).
  ``a = 0.5`` recovers Hazen, ``a = 0`` the Weibull form ``n/(N+1)``,
  ``a = 0.44`` Gringorten.  It is applied *within* each interval, and is
  forced to zero where no simulation in the interval exceeds ``q``, so it
  never introduces a probability floor in the far tail.
* The pooled estimator uses the weight-generalised form of the same family,

      P_i = W (C_i - a w_i) / (W + (1 - 2a) w_i)

  with ``C_i`` the cumulative weight and ``W`` the total.  For equal weights
  this reduces exactly to ``(i - a)/(n + 1 - 2a)``.
* Inversion is linear in ``log P`` against ``q``.
"""

from __future__ import annotations

import numpy as np

from .stratification import FREQUENT_OPEN, INTERIOR, RARE_OPEN

__all__ = [
    "PLOTTING_POSITIONS",
    "DEFAULT_PLOTTING_POSITION",
    "plotting_position_a",
    "weighted_exceedance",
    "weighted_quantile",
    "exceedance_probability",
    "interval_exceedance_curve",
    "interval_quantile",
    "interval_exceedance_probability",
    "stratum_contribution",
    "bootstrap_quantiles",
]

ARR_FIRST_FACTOR = 0.1

#: Plotting-position parameter ``a`` in ``(i - a) / (n + 1 - 2a)``.
PLOTTING_POSITIONS = {
    "cunnane": 0.4,       # approximately quantile-unbiased (Cunnane, 1978)
    "hazen": 0.5,
    "weibull": 0.0,       # i/(n+1), unbiased exceedance probability
    "gringorten": 0.44,   # Gumbel/EV1
    "blom": 0.375,        # normal
    "beard": 0.31,
    "apl": 0.35,          # APL / Landwehr
}
DEFAULT_PLOTTING_POSITION = "cunnane"


def plotting_position_a(spec) -> float:
    """Resolve a plotting-position name (or a bare ``a``) to the value of ``a``."""
    if isinstance(spec, str):
        try:
            return PLOTTING_POSITIONS[spec.lower()]
        except KeyError:
            raise ValueError(
                f"unknown plotting position '{spec}'; "
                f"choose from {sorted(PLOTTING_POSITIONS)} or pass a float") from None
    a = float(spec)
    if not 0.0 <= a <= 0.5:
        raise ValueError("plotting-position a must lie in [0, 0.5]")
    return a


# ------------------------------------------------------- pooled estimator
def weighted_exceedance(q, w, plotting_position=DEFAULT_PLOTTING_POSITION):
    """Sorted peaks (descending) and their annual exceedance probabilities.

    Weight-generalised plotting position:
    ``P_i = W (C_i - a w_i) / (W + (1 - 2a) w_i)``, which reduces to
    ``(i - a)/(n + 1 - 2a)`` for equal weights.
    """
    q = np.asarray(q, dtype=float)
    w = np.asarray(w, dtype=float)
    if q.shape != w.shape:
        raise ValueError("q and w must have the same shape")
    a = plotting_position_a(plotting_position)
    order = np.argsort(-q)
    qs, ws = q[order], w[order]
    W = ws.sum()
    C = np.cumsum(ws)
    return qs, W * (C - a * ws) / (W + (1.0 - 2.0 * a) * ws)


def weighted_quantile(q, w, aep, plotting_position=DEFAULT_PLOTTING_POSITION):
    """Discharge with the given annual exceedance probability (pooled form).

    Values outside the simulated range come back as ``nan`` rather than being
    extrapolated -- if you hit that, extend the stratification.
    """
    qs, p = weighted_exceedance(q, w, plotting_position)
    aep = np.atleast_1d(np.asarray(aep, dtype=float))
    out = np.interp(np.log(aep), np.log(p), qs, left=np.nan, right=np.nan)
    return out if out.size > 1 else float(out[0])


def exceedance_probability(q, w, discharge,
                           plotting_position=DEFAULT_PLOTTING_POSITION):
    """Annual exceedance probability of a nominated discharge (pooled form)."""
    qs, p = weighted_exceedance(q, w, plotting_position)
    discharge = np.atleast_1d(np.asarray(discharge, dtype=float))
    out = np.interp(discharge, qs[::-1], p[::-1], left=p[-1], right=p[0])
    return out if out.size > 1 else float(out[0])


# --------------------------------------------- ARR interval-wise estimator
def _interval_conditional(qs_sorted, grid, kind, a, first_factor):
    """P(Q > grid | interval), with the ARR end-interval geometric means."""
    n = qs_sorted.size
    n_gt = n - np.searchsorted(qs_sorted, grid, side="right")
    c = (n_gt - a) / (n + 1.0 - 2.0 * a)
    c = np.where(n_gt == 0, 0.0, np.clip(c, 0.0, 1.0))
    if kind == FREQUENT_OPEN:
        # geometric mean of c and first_factor * c
        return c * np.sqrt(first_factor)
    if kind == RARE_OPEN:
        # geometric mean of c and 1
        return np.sqrt(c)
    return c


def interval_exceedance_curve(q, weight, interval, kind=None, grid=None,
                              plotting_position=DEFAULT_PLOTTING_POSITION,
                              first_factor: float = ARR_FIRST_FACTOR):
    """``(grid, P, contributions)`` by the interval-wise total probability theorem.

    ``contributions`` is a ``(n_intervals, n_grid)`` array of each interval's
    additive contribution to ``P``, which is what the stratum-contribution
    diagnostic reads.
    """
    q = np.asarray(q, float)
    weight = np.asarray(weight, float)
    interval = np.asarray(interval, int)
    kind = (np.zeros_like(interval) if kind is None else np.asarray(kind, int))
    grid = np.unique(q) if grid is None else np.asarray(grid, float)
    a = plotting_position_a(plotting_position)

    ids = np.unique(interval)
    contrib = np.zeros((ids.size, grid.size))
    for row, k in enumerate(ids):
        m = interval == k
        kinds_here = np.unique(kind[m])
        if kinds_here.size != 1:
            raise ValueError(f"interval {k} mixes kinds {kinds_here}")
        c = _interval_conditional(np.sort(q[m]), grid, int(kinds_here[0]),
                                  a, first_factor)
        contrib[row] = float(weight[m].sum()) * c
    return grid, contrib.sum(axis=0), contrib


def _monotone(grid, P):
    """Keep the strictly decreasing part of the curve, for inversion."""
    ok = P > 0
    g, p = grid[ok], P[ok]
    if g.size == 0:
        return g, p
    keep = np.concatenate([[True], np.diff(p) < 0])
    return g[keep], p[keep]


def interval_quantile(q, weight, interval, kind=None, aeps=None,
                      plotting_position=DEFAULT_PLOTTING_POSITION,
                      first_factor: float = ARR_FIRST_FACTOR):
    """Flood quantiles from the interval-wise estimator."""
    grid, P, _ = interval_exceedance_curve(q, weight, interval, kind,
                                           plotting_position=plotting_position,
                                           first_factor=first_factor)
    g, p = _monotone(grid, P)
    aeps = np.atleast_1d(np.asarray(aeps, float))
    if g.size < 2:
        return np.full(aeps.size, np.nan)
    out = np.interp(np.log(aeps), np.log(p)[::-1], g[::-1],
                    left=np.nan, right=np.nan)
    return out


def interval_exceedance_probability(q, weight, interval, kind=None,
                                    discharge=None, plotting_position=DEFAULT_PLOTTING_POSITION,
                                    first_factor: float = ARR_FIRST_FACTOR):
    """Annual exceedance probability of a discharge, interval-wise form."""
    grid, P, _ = interval_exceedance_curve(q, weight, interval, kind,
                                           plotting_position=plotting_position,
                                           first_factor=first_factor)
    g, p = _monotone(grid, P)
    discharge = np.atleast_1d(np.asarray(discharge, float))
    out = np.interp(discharge, g, p, left=p[0], right=p[-1])
    return out if out.size > 1 else float(out[0])


def stratum_contribution(q, weight, interval, discharge, kind=None,
                         plotting_position=DEFAULT_PLOTTING_POSITION,
                         first_factor: float = ARR_FIRST_FACTOR):
    """Fractional contribution of each rainfall interval to ``P(Q > discharge)``.

    The classic diagnostic for checking that the stratification brackets the
    probability range of interest: if most of the exceedance probability at
    your design AEP comes from an *open end* interval, the domain is truncated
    in the wrong place and the end-interval approximation is doing work it
    should not be.
    """
    grid = np.atleast_1d(np.asarray(discharge, float))
    _, P, contrib = interval_exceedance_curve(q, weight, interval, kind,
                                              grid=grid,
                                              plotting_position=plotting_position,
                                              first_factor=first_factor)
    total = P[0]
    out = contrib[:, 0]
    return out / total if total > 0 else out


def bootstrap_quantiles(q, weight, interval, aeps, kind=None, n_boot=500,
                        rng=None, estimator="arr",
                        plotting_position=DEFAULT_PLOTTING_POSITION,
                        first_factor: float = ARR_FIRST_FACTOR):
    """Monte Carlo sampling uncertainty of the quantiles, by within-interval
    bootstrap.

    Events are resampled with replacement *within* each interval, preserving
    the interval weights, so this isolates the sampling variability of the
    simulation itself.  It says nothing about uncertainty in the IFD, the
    temporal pattern sample, the initial state distribution or the model
    parameters -- those need their own outer loop.

    Returns an array of shape ``(n_boot, len(aeps))``.
    """
    rng = np.random.default_rng() if rng is None else rng
    q = np.asarray(q, float)
    weight = np.asarray(weight, float)
    interval = np.asarray(interval, int)
    kind = (np.zeros_like(interval) if kind is None else np.asarray(kind, int))
    aeps = np.atleast_1d(np.asarray(aeps, float))
    groups = [np.flatnonzero(interval == k) for k in np.unique(interval)]
    groups = [g for g in groups if g.size]

    out = np.empty((n_boot, aeps.size))
    for b in range(n_boot):
        pick = np.concatenate([rng.choice(g, size=g.size, replace=True)
                               for g in groups])
        if estimator == "arr":
            out[b] = interval_quantile(q[pick], weight[pick], interval[pick],
                                       kind[pick], aeps, plotting_position,
                                       first_factor)
        else:
            out[b] = np.atleast_1d(weighted_quantile(q[pick], weight[pick], aeps,
                                                     plotting_position))
    return out
