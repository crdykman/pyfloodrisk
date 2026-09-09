"""
Stratified sampling of the rainfall probability domain.

The rainfall frequency curve is divided into intervals; a fixed number of events
is simulated within each, and each simulated event carries the probability
weight ``mass_k / n_k``.  The flood frequency curve is recovered by the total
probability theorem (:mod:`pyfloodrisk.dffa.tpt`).  This is what makes it possible to
estimate a 1 in 10^5 AEP flood from ~10^4 simulations.

Defaults follow ARR Book 4, Chapter 4, Section 4.3.3.3: intervals uniformly
spaced in the standardised normal probability domain, ~50 of them, with 50-200
simulations in each.

End intervals
-------------
ARR treats the two end intervals differently from the intermediate ones, so that
the whole probability domain is accounted for rather than truncated:

* the **frequent** end interval covers everything more frequent than
  ``edges[0]``, with probability mass ``1 - edges[0]`` (the non-exceedance
  probability of its upper-bound rainfall);
* the **rare** end interval covers everything rarer than ``edges[-1]``, with
  probability mass ``edges[-1]`` (the exceedance probability of its lower-bound
  rainfall);
* in both, the rainfall is held at the interval's internal bound rather than
  sampled within the (unbounded) interval, and the conditional exceedance
  probability is adjusted by a geometric mean — see
  :func:`pyfloodrisk.dffa.tpt.interval_exceedance_curve`.

With ``open_ends=True`` (the default) the interval masses therefore sum to
exactly 1. Set ``open_ends=False`` for the plain truncated scheme, where the
masses sum to ``edges[0] - edges[-1]`` and the curve is only defined inside
that range.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

__all__ = ["Stratification", "INTERIOR", "FREQUENT_OPEN", "RARE_OPEN"]

#: Interval-kind codes carried by each simulated event.
INTERIOR = 0
FREQUENT_OPEN = 1
RARE_OPEN = 2


@dataclass
class Stratification:
    """Intervals over the AEP domain of rainfall.

    Parameters
    ----------
    edges
        Internal interval boundaries as AEPs, **strictly decreasing**, length
        ``M+1`` for ``M`` intermediate intervals.
    n_per_stratum
        Simulations in each intermediate interval (scalar or length-``M``).
    within
        ``"systematic"`` (default) places the intermediate-interval samples at
        equal-probability mid-points — a Latin-hypercube-style layout that
        removes within-interval sampling noise in the rainfall dimension.
        ``"random"`` draws uniformly within the interval, which is ARR's
        description and what you want if you are quantifying Monte Carlo
        sampling error by repeating the whole experiment with new seeds.
    open_ends
        Include the two open-ended end intervals of ARR Section 4.3.3.3.
    n_per_end
        Simulations in each end interval (default: the mean of
        ``n_per_stratum``).  Rainfall is fixed at the bound, so these
        simulations sample only the other stochastic inputs.
    """

    edges: np.ndarray
    n_per_stratum: np.ndarray
    within: str = "systematic"
    open_ends: bool = True
    n_per_end: int | None = None

    def __post_init__(self):
        self.edges = np.asarray(self.edges, dtype=float)
        if np.any(np.diff(self.edges) >= 0):
            raise ValueError("edges must be strictly decreasing AEPs")
        if np.any(self.edges <= 0) or self.edges[0] > 1.0:
            raise ValueError("edges must lie in (0, 1]")
        self.n_per_stratum = np.broadcast_to(
            np.asarray(self.n_per_stratum, dtype=int), (self.n_intermediate,)).copy()
        if self.n_per_end is None:
            self.n_per_end = int(round(self.n_per_stratum.mean()))
        if self.within not in ("systematic", "random"):
            raise ValueError("within must be 'systematic' or 'random'")

    # ----------------------------------------------------------- properties
    @property
    def n_intermediate(self) -> int:
        return len(self.edges) - 1

    @property
    def n_intervals(self) -> int:
        return self.n_intermediate + (2 if self.open_ends else 0)

    #: kept for backwards compatibility -- number of intermediate intervals
    @property
    def n_strata(self) -> int:
        return self.n_intermediate

    @property
    def mass(self) -> np.ndarray:
        """Probability mass of every interval, ordered frequent to rare."""
        inner = self.edges[:-1] - self.edges[1:]
        if not self.open_ends:
            return inner
        return np.concatenate([[1.0 - self.edges[0]], inner, [self.edges[-1]]])

    @property
    def covered_mass(self) -> float:
        return float(self.mass.sum())

    @property
    def kinds(self) -> np.ndarray:
        """Interval-kind code for every interval, in the same order."""
        inner = np.full(self.n_intermediate, INTERIOR)
        if not self.open_ends:
            return inner
        return np.concatenate([[FREQUENT_OPEN], inner, [RARE_OPEN]])

    @property
    def interval_mid_aep(self) -> np.ndarray:
        """Representative AEP of each interval (geometric centre; bound at the
        open ends, where the rainfall really is held at the bound)."""
        mids = np.sqrt(self.edges[:-1] * self.edges[1:])
        if not self.open_ends:
            return mids
        return np.concatenate([[self.edges[0]], mids, [self.edges[-1]]])

    @property
    def n_per_interval(self) -> np.ndarray:
        if not self.open_ends:
            return self.n_per_stratum.copy()
        return np.concatenate([[self.n_per_end], self.n_per_stratum, [self.n_per_end]])

    @property
    def n_events(self) -> int:
        return int(self.n_per_interval.sum())

    # -------------------------------------------------------- constructors
    @classmethod
    def uniform_in_z(cls, aep_max: float = 0.5, aep_min: float = 1e-6,
                     n_strata: int = 50, n_per_stratum: int = 200,
                     within: str = "systematic", open_ends: bool = True,
                     n_per_end: int | None = None) -> "Stratification":
        """Intervals equally spaced in the standard normal variate (ARR default)."""
        z = np.linspace(norm.ppf(1 - aep_max), norm.ppf(1 - aep_min), n_strata + 1)
        return cls(1.0 - norm.cdf(z), n_per_stratum, within, open_ends, n_per_end)

    @classmethod
    def uniform_in_log_aep(cls, aep_max: float = 0.5, aep_min: float = 1e-6,
                           n_strata: int = 50, n_per_stratum: int = 200,
                           within: str = "systematic", open_ends: bool = True,
                           n_per_end: int | None = None) -> "Stratification":
        """Intervals equally spaced in log(AEP) -- heavier emphasis on the tail."""
        edges = np.exp(np.linspace(np.log(aep_max), np.log(aep_min), n_strata + 1))
        return cls(edges, n_per_stratum, within, open_ends, n_per_end)

    # ------------------------------------------------------------- sampling
    def sample(self, rng: np.random.Generator | None = None):
        """Return ``(aep, weight, interval, kind)`` for every event.

        Intervals are numbered frequent to rare: with ``open_ends=True``,
        interval 0 is the frequent open interval, ``1 .. M`` the intermediate
        ones and ``M+1`` the rare open interval.  ``weight`` is the annual
        probability attributed to the event (``mass / n``), so the weights sum
        to :attr:`covered_mass`.
        """
        rng = np.random.default_rng() if rng is None else rng
        aeps, weights, idx, kinds = [], [], [], []
        mass, kind_of, n_of = self.mass, self.kinds, self.n_per_interval

        for j in range(self.n_intervals):
            n = int(n_of[j])
            k = int(kind_of[j])
            if k == FREQUENT_OPEN:
                a = np.full(n, self.edges[0])          # rainfall fixed at the bound
            elif k == RARE_OPEN:
                a = np.full(n, self.edges[-1])
            else:
                i = j - 1 if self.open_ends else j     # index into edges
                hi, lo = self.edges[i], self.edges[i + 1]
                if self.within == "systematic":
                    u = rng.permutation((np.arange(n) + 0.5) / n)
                else:
                    u = rng.random(n)
                a = lo + u * (hi - lo)
            aeps.append(a)
            weights.append(np.full(n, mass[j] / n))
            idx.append(np.full(n, j, dtype=int))
            kinds.append(np.full(n, k, dtype=int))
        return (np.concatenate(aeps), np.concatenate(weights),
                np.concatenate(idx), np.concatenate(kinds))
