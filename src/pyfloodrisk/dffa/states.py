"""
Sampling GR4H initial states from a continuous simulation.

This is the part of the framework that replaces the sampled initial loss of a
conventional event-based Monte Carlo implementation.  The input is a table of
model states written out at every timestep (or every storm onset) of a long
continuous GR4H run: production store, routing store, and optionally the
unit-hydrograph memory and exponential store.

Modelled variables
------------------
The non-bootstrap methods do not model every column.  They model a small set of
**state variables** -- the production store, the routing store, the exponential
store if present, and (if the UH columns are given) the *total* water in transit
through the unit hydrographs, ``uh_total`` in mm.  ``uh_total`` is the
physically meaningful scalar; the *shape* of the UH memory is a deterministic
consequence of the last few timesteps of rainfall, so it is taken from a donor
row and rescaled to the sampled total rather than being sampled ordinate by
ordinate (which would produce states GR4H could never reach).

Methods
-------
``"bootstrap"`` (default)
    Draw whole rows with replacement.  Preserves every dependence in the
    continuous run exactly, including the UH memory.  With a long enough
    continuous record this is the right choice.
``"smoothed"``
    Smoothed bootstrap: draw a row, then jitter the state variables in a
    transformed (unbounded) space.  Fills the gaps between discrete states in a
    short or heavily conditioned donor pool without inventing dependence
    structure.  Two details make it behave: the jitter kernel takes its
    covariance from the donor pool rather than being isotropic (an isotropic
    kernel dilutes the cross-correlation between the stores), and the donor is
    shrunk toward the pool mean by ``1/sqrt(1 + h**2)`` before jittering, which
    cancels the variance inflation kernel smoothing would otherwise cause.  The
    sampled covariance in the transformed space is then preserved exactly to
    second order.
``"empirical_copula"``
    Nonparametric **vine copula** (``pyvinecopulib``, TLL local-likelihood pair
    copulas) fitted to the pseudo-observations, with kernel or empirical
    marginals.  The dependence structure is estimated from the data rather than
    assumed, so unlike a Gaussian copula it can represent asymmetric and
    tail dependence -- which matters here, because the design flood comes from
    the joint upper tail of the state distribution.
``"independent_kde"``
    Each state variable sampled independently from its own boundary-corrected
    one-dimensional kernel density (``pyvinecopulib.Kde1d``, which handles the
    bounded support of the stores and the point mass at zero in ``uh_total``).
    This deliberately destroys the cross-correlation between the stores, which
    is exactly what makes it useful: the difference between this and
    ``"bootstrap"`` isolates how much the *joint* structure of the antecedent
    state contributes to the design flood, as opposed to the marginals alone.

Conditioning
------------
``sample(..., month=...)`` restricts the donor pool to a calendar month (or set
of months) so seasonality in antecedent wetness is preserved.
``sample(..., pool=...)`` accepts an arbitrary boolean mask for conditioning on
anything else -- an antecedent rainfall class, a climate index state, a burst
magnitude class if you want to admit dependence between rainfall and antecedent
conditions.  Fitted models (copulas, KDEs) are cached per pool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

__all__ = ["InitialStateSampler", "PETClimatology", "UH_TOTAL", "have_pyvinecopulib"]

#: Name of the derived state variable holding total UH memory (mm).
UH_TOTAL = "uh_total"

_METHODS = ("bootstrap", "smoothed", "empirical_copula", "independent_kde")


def have_pyvinecopulib() -> bool:
    try:
        import pyvinecopulib  # noqa: F401
        return True
    except ImportError:
        return False


def _require_pv():
    try:
        import pyvinecopulib as pv
    except ImportError as exc:   # pragma: no cover
        raise ImportError(
            "this state-sampling method needs pyvinecopulib "
            "(pip install pyvinecopulib)") from exc
    return pv


def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _expit(x):
    return 1.0 / (1.0 + np.exp(-x))


def _pseudo_obs(X: np.ndarray) -> np.ndarray:
    """Rank-based pseudo-observations in (0, 1)."""
    n = X.shape[0]
    ranks = np.argsort(np.argsort(X, axis=0), axis=0) + 1
    return ranks / (n + 1.0)


@dataclass
class InitialStateSampler:
    """Sampling of GR4H initial states from a continuous-run state table.

    Parameters
    ----------
    states
        ``DataFrame`` of states from the continuous run.
    x1, x3
        Production and routing store capacities (mm) of the calibrated model.
        Used for the bounded transforms and for reporting store fractions;
        pass the same values you give the engine.
    prod_col, rout_col, exp_col
        Column names of the store levels in mm.
    uh1_cols, uh2_cols
        Column names of the unit-hydrograph memory, in order.  Optional; if
        absent the UH memory is initialised to zero, which is acceptable only
        if the event has enough pre-burst rainfall or dry lead-in for the
        memory to be irrelevant.
    date_col
        Optional datetime column, used for seasonal conditioning and for
        selecting climatological PET.
    method
        One of ``bootstrap``, ``smoothed``, ``empirical_copula``,
        ``independent_kde``.
    bandwidth
        Kernel bandwidth multiplier (``smoothed`` jitter, and the ``Kde1d``
        multiplier for the KDE-based methods).  1.0 is the automatic choice.
    marginals
        ``"kde"`` (default) or ``"empirical"`` -- how the copula methods invert
        each margin.
    uh_profile
        How the UH memory shape is set for the non-bootstrap methods:
        ``"scaled_donor"`` (default) takes the nearest donor's profile and
        rescales it to the sampled ``uh_total``; ``"donor"`` copies the donor
        profile unchanged; ``"mean"`` uses the pool-mean normalised profile,
        which keeps ``independent_kde`` fully independent of the stores.
    model_uh_total
        Include ``uh_total`` among the modelled state variables (default True
        when UH columns are supplied).
    """

    states: pd.DataFrame
    x1: float
    x3: float
    prod_col: str = "prod_store"
    rout_col: str = "rout_store"
    exp_col: str | None = None
    uh1_cols: Sequence[str] | None = None
    uh2_cols: Sequence[str] | None = None
    date_col: str | None = None
    method: str = "bootstrap"
    bandwidth: float = 1.0
    marginals: str = "kde"
    uh_profile: str = "scaled_donor"
    model_uh_total: bool = True

    def __post_init__(self):
        df = self.states.reset_index(drop=True).copy()
        for c in (self.prod_col, self.rout_col):
            if c not in df:
                raise ValueError(f"state table has no column '{c}'")
        if self.method not in _METHODS:
            raise ValueError(f"method must be one of {_METHODS}")
        if self.marginals not in ("kde", "empirical"):
            raise ValueError("marginals must be 'kde' or 'empirical'")
        if self.uh_profile not in ("scaled_donor", "donor", "mean"):
            raise ValueError("uh_profile must be 'scaled_donor', 'donor' or 'mean'")

        self._uh_cols = list(self.uh1_cols or []) + list(self.uh2_cols or [])
        if self._uh_cols:
            df[UH_TOTAL] = df[self._uh_cols].to_numpy(float).sum(axis=1)

        vars_ = [self.prod_col, self.rout_col]
        if self.exp_col:
            vars_.append(self.exp_col)
        if self._uh_cols and self.model_uh_total:
            vars_.append(UH_TOTAL)
        self.state_vars = vars_

        self.states = df
        self._months = (pd.to_datetime(df[self.date_col]).dt.month.to_numpy()
                        if self.date_col else None)
        self._X = df[self.state_vars].to_numpy(float)
        self._Z = self._transform(self._X)
        self._Zstd = self._Z.std(axis=0, ddof=1)
        self._Zstd[self._Zstd == 0] = 1.0
        self._Zmean = self._Z.mean(axis=0)
        # normalised mean UH profile, for uh_profile="mean"
        if self._uh_cols:
            prof = df[self._uh_cols].to_numpy(float)
            tot = prof.sum(axis=1, keepdims=True)
            ok = tot[:, 0] > 0
            self._mean_profile = (prof[ok] / tot[ok]).mean(axis=0) if ok.any() \
                else np.zeros(len(self._uh_cols))
        self._fit_cache: dict = {}

    # -------------------------------------------------------- bounds/transform
    def _bounds(self):
        """(xmin, xmax, kde_type) per modelled variable."""
        out = []
        for c in self.state_vars:
            if c == self.prod_col:
                out.append((0.0, float(self.x1), "continuous"))
            elif c == UH_TOTAL:
                zi = float((self.states[c].to_numpy() <= 0).mean()) > 0.01
                out.append((0.0, None, "zero_inflated" if zi else "continuous"))
            else:
                out.append((0.0, None, "continuous"))
        return out

    def _transform(self, X: np.ndarray) -> np.ndarray:
        """Modelled variables -> unbounded coordinates (for the jitter kernel)."""
        Z = np.empty_like(X)
        for j, c in enumerate(self.state_vars):
            if c == self.prod_col:
                Z[:, j] = _logit(X[:, j] / self.x1)
            elif c == self.rout_col:
                Z[:, j] = np.log(np.maximum(X[:, j], 1e-6) / self.x3)
            else:
                Z[:, j] = np.log1p(np.maximum(X[:, j], 0.0))
        return Z

    def _inverse(self, Z: np.ndarray) -> np.ndarray:
        X = np.empty_like(Z)
        for j, c in enumerate(self.state_vars):
            if c == self.prod_col:
                X[:, j] = _expit(Z[:, j]) * self.x1
            elif c == self.rout_col:
                X[:, j] = np.exp(Z[:, j]) * self.x3
            else:
                X[:, j] = np.expm1(Z[:, j])
        return X

    # ------------------------------------------------------------------ fits
    def _pool(self, month=None, pool=None) -> np.ndarray:
        idx = np.arange(len(self.states))
        if pool is not None:
            idx = idx[np.asarray(pool, bool)]
        if month is not None:
            if self._months is None:
                raise ValueError("seasonal conditioning needs date_col")
            idx = idx[np.isin(self._months[idx], np.atleast_1d(month))]
        if idx.size == 0:
            raise ValueError("empty donor pool for the requested conditioning")
        return idx

    def _pool_key(self, idx: np.ndarray):
        return (int(idx.size), int(idx[0]), int(idx[-1]), int(idx.sum()))

    def _kde_marginals(self, idx: np.ndarray):
        """Boundary-corrected 1-D KDEs, one per modelled variable."""
        pv = _require_pv()
        key = ("kde", self._pool_key(idx))
        if key in self._fit_cache:
            return self._fit_cache[key]
        fits = []
        for j, (lo, hi, typ) in enumerate(self._bounds()):
            kw = dict(type=typ, multiplier=float(self.bandwidth))
            if lo is not None:
                kw["xmin"] = float(lo)
            if hi is not None:
                kw["xmax"] = float(hi)
            k = pv.Kde1d(**kw)
            k.fit(np.ascontiguousarray(self._X[idx, j]))
            fits.append(k)
        self._fit_cache[key] = fits
        return fits

    def _vine(self, idx: np.ndarray):
        """Nonparametric (TLL) vine copula on the pseudo-observations."""
        pv = _require_pv()
        key = ("vine", self._pool_key(idx))
        if key in self._fit_cache:
            return self._fit_cache[key]
        if len(self.state_vars) < 2:
            raise ValueError("a copula needs at least two modelled variables")
        u = _pseudo_obs(self._X[idx])
        controls = pv.FitControlsVinecop(family_set=pv.nonparametric,
                                         nonparametric_mult=float(self.bandwidth))
        vc = pv.Vinecop.from_data(np.asfortranarray(u), controls=controls)
        self._fit_cache[key] = vc
        return vc

    def _invert_marginals(self, U: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """Uniform (0,1) columns -> state variables, through the margins."""
        if self.marginals == "kde":
            fits = self._kde_marginals(idx)
            return np.column_stack([f.quantile(np.ascontiguousarray(U[:, j]))
                                    for j, f in enumerate(fits)])
        return np.column_stack([np.quantile(self._X[idx, j], U[:, j])
                                for j in range(U.shape[1])])

    # ----------------------------------------------------------- sampling
    def sample(self, n: int, rng: np.random.Generator,
               month=None, pool=None) -> pd.DataFrame:
        """Return ``n`` sampled state rows (columns of the input table)."""
        idx = self._pool(month, pool)
        donors = rng.choice(idx, size=n, replace=True)
        out = self.states.iloc[donors].reset_index(drop=True).copy()
        out["donor_index"] = donors

        if self.method == "bootstrap":
            return out

        d = len(self.state_vars)
        if self.method == "smoothed":
            Zp = self._Z[idx]
            h = self.bandwidth * idx.size ** (-1.0 / (d + 4))
            zbar = Zp.mean(axis=0)
            S = np.atleast_2d(np.cov(Zp, rowvar=False))
            L = np.linalg.cholesky(S + 1e-12 * np.eye(d))
            jitter = h * (rng.normal(size=(n, d)) @ L.T)
            shrunk = zbar + (self._Z[donors] - zbar) / np.sqrt(1.0 + h ** 2)
            Xnew = self._inverse(shrunk + jitter)
        elif self.method == "independent_kde":
            fits = self._kde_marginals(idx)
            U = rng.random((n, d))          # independent across variables
            Xnew = np.column_stack([f.quantile(np.ascontiguousarray(U[:, j]))
                                    for j, f in enumerate(fits)])
        else:  # empirical_copula
            vc = self._vine(idx)
            seeds = [int(s) for s in rng.integers(1, 2 ** 31 - 1, size=4)]
            U = np.asarray(vc.simulate(int(n), seeds=seeds))
            Xnew = self._invert_marginals(U, idx)

        Xnew = self._clip(Xnew)
        for j, c in enumerate(self.state_vars):
            out[c] = Xnew[:, j]

        if self._uh_cols:
            out = self._set_uh(out, Xnew, idx, donors)
        return out

    def _clip(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float).copy()
        for j, c in enumerate(self.state_vars):
            if c == self.prod_col:
                X[:, j] = np.clip(X[:, j], 0.0, self.x1)
            else:
                X[:, j] = np.maximum(X[:, j], 0.0)
        return X

    def _set_uh(self, out: pd.DataFrame, Xnew: np.ndarray,
                idx: np.ndarray, donors: np.ndarray) -> pd.DataFrame:
        """Give each sampled state a physically reachable UH memory profile."""
        n = len(out)
        if self.uh_profile == "mean":
            shape = np.tile(self._mean_profile, (n, 1))
            src = donors
        else:
            # nearest donor in standardised transformed state space
            A = (self._Z[idx] - self._Zmean) / self._Zstd
            B = (self._transform(Xnew) - self._Zmean) / self._Zstd
            nn = np.array([np.argmin(((A - b) ** 2).sum(axis=1)) for b in B])
            src = idx[nn]
            shape = self.states[self._uh_cols].to_numpy(float)[src]
            out["donor_index"] = src
            if self.uh_profile == "donor":
                for j, c in enumerate(self._uh_cols):
                    out[c] = shape[:, j]
                out[UH_TOTAL] = shape.sum(axis=1)
                return out
            tot = shape.sum(axis=1, keepdims=True)
            shape = np.where(tot > 0, shape / np.maximum(tot, 1e-12),
                             self._mean_profile[None, :])

        if UH_TOTAL in self.state_vars:
            target = out[UH_TOTAL].to_numpy(float)[:, None]
        else:
            target = self.states[UH_TOTAL].to_numpy(float)[src][:, None]
        prof = shape * target
        for j, c in enumerate(self._uh_cols):
            out[c] = prof[:, j]
        out[UH_TOTAL] = prof.sum(axis=1)
        return out

    # ------------------------------------------------------------ helpers
    def to_state_dicts(self, sampled: pd.DataFrame) -> list[dict]:
        """Convert sampled rows to the dicts the engine expects."""
        recs = []
        for _, r in sampled.iterrows():
            s = {"prod_store": float(r[self.prod_col]),
                 "rout_store": float(r[self.rout_col])}
            if self.exp_col:
                s["exp_store"] = float(r[self.exp_col])
            if self.uh1_cols:
                s["uh1"] = r[list(self.uh1_cols)].to_numpy(float)
            if self.uh2_cols:
                s["uh2"] = r[list(self.uh2_cols)].to_numpy(float)
            if self.date_col:
                s["date"] = pd.Timestamp(r[self.date_col])
            recs.append(s)
        return recs

    def summary(self) -> pd.DataFrame:
        """Marginal quantiles of the donor states, as a sanity check."""
        q = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
        out = self.states[self.state_vars].quantile(q)
        out[self.prod_col + "_frac"] = out[self.prod_col] / self.x1
        out[self.rout_col + "_frac"] = out[self.rout_col] / self.x3
        return out

    def dependence(self, sampled: pd.DataFrame | None = None) -> pd.DataFrame:
        """Kendall's tau between modelled variables, input vs sampled.

        The check to run before trusting any non-bootstrap method: it says
        whether the dependence you meant to preserve (or destroy) actually was.
        """
        from scipy.stats import kendalltau
        rows = []
        v = self.state_vars
        for a in range(len(v)):
            for b in range(a + 1, len(v)):
                t_in = kendalltau(self.states[v[a]], self.states[v[b]]).statistic
                rec = {"pair": f"{v[a]} / {v[b]}", "tau_input": t_in}
                if sampled is not None:
                    rec["tau_sampled"] = kendalltau(sampled[v[a]],
                                                    sampled[v[b]]).statistic
                rows.append(rec)
        return pd.DataFrame(rows).set_index("pair")


@dataclass
class PETClimatology:
    """Climatological potential evapotranspiration for the event window.

    GR4H needs a PET series.  Over a burst of a few hours to a few days PET is
    a second-order control on the flood peak, so a mean seasonal cycle is
    adequate; what matters is that it is not zero when the pre-burst period is
    long.

    Parameters
    ----------
    monthly_mm_per_day
        Length-12 array of mean daily PET (mm/d) by calendar month.
    diurnal
        ``"uniform"`` or ``"sine"`` distribution within the day.
    """

    monthly_mm_per_day: np.ndarray
    diurnal: str = "sine"

    def __post_init__(self):
        self.monthly_mm_per_day = np.asarray(self.monthly_mm_per_day, float)
        if self.monthly_mm_per_day.size != 12:
            raise ValueError("monthly_mm_per_day must have 12 values")

    def series(self, n_steps: int, dt_hours: float, month: int = 1,
               start_hour: int = 0) -> np.ndarray:
        daily = self.monthly_mm_per_day[int(month) - 1]
        per_step = daily * dt_hours / 24.0
        if self.diurnal == "uniform":
            return np.full(n_steps, per_step)
        h = (start_hour + np.arange(n_steps) * dt_hours) % 24.0
        w = np.clip(np.sin(np.pi * (h - 6.0) / 12.0), 0.0, None)
        cyc = np.clip(np.sin(np.pi * (np.arange(0, 24, dt_hours) - 6.0) / 12.0),
                      0.0, None)
        scale = (24.0 / dt_hours) / cyc.sum() if cyc.sum() > 0 else 1.0
        return per_step * w * scale
