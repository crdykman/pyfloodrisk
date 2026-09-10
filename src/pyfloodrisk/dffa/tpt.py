"""
Empirical flood quantiles by the total probability theorem.

Each simulated event *i* has a peak discharge ``q_i``, a probability weight
``w_i`` and belongs to a rainfall interval.  ARR Book 4, Section 4.3.3.3 writes
the estimator interval-wise,

    P(Q > q) = Σ_i p[R_i] · P(Q > q | R_i)

with ``p[R_i]`` the width of interval *i* in the probability domain and
``P(Q > q | R_i)`` the proportion ``n/N`` of that interval's simulations whose
peak exceeds ``q``.

This is the only estimator here: the events come from *stratified* sampling of
the rainfall probability domain, so they are combined interval by interval
through the total probability theorem, never pooled and ranked as though they
were one sample.  Pooling would discard the stratification that gives each
event its weight, and it cannot represent the open end intervals at all.

:func:`interval_exceedance_curve` / :func:`interval_quantile` implement the ARR
form **including the end-interval treatment**: at the open ends the conditional
exceedance probability is replaced by a geometric mean, because it varies
highly non-linearly across an unbounded interval.  For the frequent end it is
the geometric mean of ``c`` and ``first_factor·c`` (ARR suggests 0.1), i.e.
``c·sqrt(first_factor)``; for the rare end, the geometric mean of ``c`` and 1,
i.e. ``sqrt(c)`` -- the assumption being that as rainfall becomes arbitrarily
rare, exceedance of any given threshold becomes certain.

Conventions worth being explicit about:

* **The conditional exceedance probability is the raw proportion**

      P(Q > q | R_k) = n_k(q) / N_k

  of that interval's ``N_k`` simulations whose peak exceeds ``q``.  There is no
  plotting position: a plotting position assigns probabilities to *ranked
  observations* of unknown distribution, whereas here the interval is sampled
  by design, ``N_k`` is chosen, and every event in it carries the same weight,
  so ``n/N`` is already unbiased for the quantity wanted.  It is zero where
  nothing in the interval exceeds ``q`` and one where everything does, so the
  curve neither floors nor caps artificially.
* Inversion is linear in ``log P`` against ``q``.
"""

from __future__ import annotations

import numpy as np

from .stratification import FREQUENT_OPEN, INTERIOR, RARE_OPEN

__all__ = [
    "interval_exceedance_curve",
    "interval_quantile",
    "interval_exceedance_probability",
    "stratum_contribution",
    "bootstrap_quantiles",
]

ARR_FIRST_FACTOR = 0.1


# --------------------------------------------- ARR interval-wise estimator
def _interval_conditional(qs_sorted, grid, kind, first_factor):
    """P(Q > grid | interval), with the ARR end-interval geometric means.

    The conditional probability is the raw proportion ``n/N``: of the ``N``
    events simulated inside this interval, the ``n`` whose peak exceeds the
    threshold.  No plotting position is applied.  Within an interval every
    event carries the same weight and the interval is sampled by design
    rather than observed, so ``n/N`` is the quantity ARR Book 4, Section
    4.3.3.3 asks for, and it is already unbiased for it.
    """
    n = qs_sorted.size
    n_gt = n - np.searchsorted(qs_sorted, grid, side="right")
    c = n_gt / float(n)
    if kind == FREQUENT_OPEN:
        # geometric mean of c and first_factor * c
        return c * np.sqrt(first_factor)
    if kind == RARE_OPEN:
        # geometric mean of c and 1
        return np.sqrt(c)
    return c


def interval_exceedance_curve(q, weight, interval, kind=None, grid=None,
                              first_factor: float = ARR_FIRST_FACTOR):
    """``(grid, P, contributions)`` by the interval-wise total probability theorem.

    The conditional exceedance inside each interval is the raw ``n/N``; there
    is no plotting position anywhere in this module.

    ``contributions`` is a ``(n_intervals, n_grid)`` array of each interval's
    additive contribution to ``P``, which is what the stratum-contribution
    diagnostic reads.
    """
    q = np.asarray(q, float)
    weight = np.asarray(weight, float)
    interval = np.asarray(interval, int)
    kind = (np.zeros_like(interval) if kind is None else np.asarray(kind, int))
    grid = np.unique(q) if grid is None else np.asarray(grid, float)

    ids = np.unique(interval)
    contrib = np.zeros((ids.size, grid.size))
    for row, k in enumerate(ids):
        m = interval == k
        kinds_here = np.unique(kind[m])
        if kinds_here.size != 1:
            raise ValueError(f"interval {k} mixes kinds {kinds_here}")
        c = _interval_conditional(np.sort(q[m]), grid, int(kinds_here[0]),
                                  first_factor)
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
                      first_factor: float = ARR_FIRST_FACTOR):
    """Flood quantiles from the interval-wise estimator."""
    grid, P, _ = interval_exceedance_curve(q, weight, interval, kind,
                                           first_factor=first_factor)
    g, p = _monotone(grid, P)
    aeps = np.atleast_1d(np.asarray(aeps, float))
    if g.size < 2:
        return np.full(aeps.size, np.nan)
    out = np.interp(np.log(aeps), np.log(p)[::-1], g[::-1],
                    left=np.nan, right=np.nan)
    return out


def interval_exceedance_probability(q, weight, interval, kind=None,
                                    discharge=None,
                                    first_factor: float = ARR_FIRST_FACTOR):
    """Annual exceedance probability of a discharge, interval-wise form."""
    grid, P, _ = interval_exceedance_curve(q, weight, interval, kind,
                                           first_factor=first_factor)
    g, p = _monotone(grid, P)
    discharge = np.atleast_1d(np.asarray(discharge, float))
    out = np.interp(discharge, g, p, left=p[0], right=p[-1])
    return out if out.size > 1 else float(out[0])


def stratum_contribution(q, weight, interval, discharge, kind=None,
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
                                              first_factor=first_factor)
    total = P[0]
    out = contrib[:, 0]
    return out / total if total > 0 else out


def bootstrap_quantiles(q, weight, interval, aeps, kind=None, n_boot=500,
                        rng=None, first_factor: float = ARR_FIRST_FACTOR):
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
        out[b] = interval_quantile(q[pick], weight[pick], interval[pick],
                                   kind[pick], aeps, first_factor=first_factor)
    return out
