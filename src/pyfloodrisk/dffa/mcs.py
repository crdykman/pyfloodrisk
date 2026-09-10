"""
Monte Carlo driver: derived flood frequency analysis with GR4H in event mode.

Structure of one simulated event
--------------------------------
1. A storm duration is fixed (the outer loop runs every duration and the
   quantiles are enveloped, as in ARR design practice).
2. A rainfall AEP is drawn by stratified sampling of the rainfall frequency
   domain; the event carries the probability weight ``mass_k / n_k``.
3. The point burst depth comes from the fitted IFD curve and is reduced to a
   catchment-average depth by the ARF.
4. A temporal pattern is drawn at random from the ARR ensemble for that duration
   and AEP band, and aggregated to the model timestep.
5. Optional pre-burst rainfall is prepended.
6. An initial GR4H state vector is drawn jointly from the distribution derived
   from continuous simulation.
7. GR4H is run over the burst plus a recession tail; the peak discharge is
   retained.

The flood frequency curve is then the weighted empirical exceedance function of
the simulated peaks (total probability theorem), and the design quantile at each
AEP is the envelope over durations.

What this framework does and does not assume
--------------------------------------------
* Rainfall depth, temporal pattern and initial state are sampled
  **independently**.  The pattern ensembles are conditioned on the AEP band, so
  there is a coarse depth-pattern dependence; there is none between depth and
  antecedent state unless you supply a conditioning mask.  Testing that
  independence assumption is exactly what a continuous-simulation state
  distribution lets you do -- see ``InitialStateSampler.sample(pool=...)``.
* Design rainfalls are bursts, so the derived curve inherits the embedded-burst
  problem.  Pre-burst rainfall is the mechanism provided for dealing with it.
* Enveloping across durations is consistent with ARR practice but introduces a
  small positive bias, because the maximum of several noisy estimates is biased
  high.  :meth:`DFFAResults.pooled_quantiles` gives the unbiased alternative if
  you are prepared to assume probabilities for each duration.
* Baseflow is *not* added separately: GR4H's routing store produces it, which is
  one of the substantive differences from a conventional event-based
  implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .ifd import IFDCurve
from .patterns import PreBurstSampler, TemporalPatternLibrary
from .states import InitialStateSampler, PETClimatology
from .stratification import FREQUENT_OPEN, RARE_OPEN, Stratification
from .tpt import (ARR_FIRST_FACTOR, DEFAULT_PLOTTING_POSITION, bootstrap_quantiles, exceedance_probability,
                  interval_exceedance_curve, interval_exceedance_probability,
                  interval_quantile, stratum_contribution, weighted_exceedance,
                  weighted_quantile)

__all__ = ["MCSConfig", "DerivedFFA", "DFFAResults", "STANDARD_DURATIONS_H"]

#: Design storm durations (hours) to envelope over -- the durations the
#: bundled ARR temporal patterns cover.  From 12 h up these are *areal*
#: patterns; 6 and 9 h are *point* patterns, because ARR publishes no areal
#: pattern shorter than 12 h.  The point files go down to 10 minutes, so
#: extend the short end if a burst that short is meaningful on your
#: catchment.
STANDARD_DURATIONS_H = (6, 9, 12, 18, 24, 36, 48, 72, 96, 120, 144, 168)


@dataclass
class MCSConfig:
    """Run configuration."""

    area_km2: float
    durations_h: Sequence[float] = STANDARD_DURATIONS_H
    dt_hours: float = 1.0
    #: recession tail appended after the burst, in hours; if None, uses
    #: ``tail_multiple * duration`` bounded below by ``tail_min_hours``
    tail_hours: float | None = None
    tail_multiple: float = 1.5
    tail_min_hours: float = 24.0
    #: dry lead-in before the (pre-)burst, lets the UH memory drain if you did
    #: not sample it
    lead_hours: float = 0.0
    apply_arf: bool = True
    preburst: PreBurstSampler | None = None
    seed: int = 20260909
    chunk_size: int = 2000
    #: whether to retain hydrographs for plotting.  Retention is by target
    #: AEP (see :attr:`hydrograph_aeps`), so this is really an on/off switch
    #: with a cap: 0 keeps none, any positive number caps how many of the
    #: targets are kept.
    store_hydrographs: int = 20
    #: rainfall AEPs to retain a hydrograph nearest to, one each.  The event
    #: whose *rainfall* AEP is closest to each target -- closest in log AEP,
    #: so the rare targets are not swamped by the frequent ones -- is kept.
    #: Retaining by target rather than by position is what makes the stored
    #: events design-scale: sampling order runs from frequent to rare, so
    #: taking the first N would keep only the most frequent events.
    hydrograph_aeps: Sequence[float] = (0.5, 0.2, 0.1, 0.05, 0.02, 0.01)
    progress: bool = True


class DerivedFFA:
    """Assemble and run the Monte Carlo experiment."""

    def __init__(self, ifd: IFDCurve, patterns: TemporalPatternLibrary,
                 states: InitialStateSampler, engine, config: MCSConfig,
                 stratification: Stratification | None = None,
                 pet: PETClimatology | None = None,
                 state_pool_fn=None):
        self.ifd = ifd
        self.patterns = patterns
        self.states = states
        self.engine = engine
        self.cfg = config
        self.strat = stratification or Stratification.uniform_in_z()
        self.pet = pet
        #: optional callable ``f(aep, duration_h) -> boolean mask`` giving the
        #: donor pool for the state sampler, i.e. dependence between rainfall
        #: magnitude and antecedent conditions
        self.state_pool_fn = state_pool_fn
        if abs(engine.dt_hours - config.dt_hours) > 1e-9:
            raise ValueError("engine and config timesteps differ")

    # ------------------------------------------------------------ assembly
    def _tail_steps(self, duration_h: float) -> int:
        tail = (self.cfg.tail_hours if self.cfg.tail_hours is not None
                else max(self.cfg.tail_min_hours, self.cfg.tail_multiple * duration_h))
        return int(round(tail / self.cfg.dt_hours))

    def _build_event(self, depth_mm, duration_h, aep, rng):
        """Rainfall series (mm per timestep) plus pattern metadata."""
        dt = self.cfg.dt_hours
        frac, pat_id, band, pat_idx = self.patterns.sample(duration_h, aep, dt, rng)
        burst = frac * depth_mm

        pre = np.zeros(0)
        ratio = 0.0
        if self.cfg.preburst is not None:
            pre, ratio = self.cfg.preburst.series(depth_mm, duration_h, aep, dt, rng)
            gap = int(round(self.cfg.preburst.gap_hours / dt))
            pre = np.concatenate([pre, np.zeros(gap)])

        lead = np.zeros(int(round(self.cfg.lead_hours / dt)))
        tail = np.zeros(self._tail_steps(duration_h))
        rain = np.concatenate([lead, pre, burst, tail])
        return rain, dict(pattern_id=pat_id, pattern_band=band,
                          pattern_index=pat_idx, preburst_ratio=ratio,
                          preburst_depth_mm=float(pre.sum()),
                          n_steps=rain.size,
                          burst_start_step=lead.size + pre.size)

    def _pet_series(self, n_steps, month):
        if self.pet is None:
            return np.zeros(n_steps)
        return self.pet.series(n_steps, self.cfg.dt_hours, month=month)

    # ----------------------------------------------------------------- run
    def run_duration(self, duration_h: float, rng: np.random.Generator) -> pd.DataFrame:
        cfg = self.cfg
        aep, weight, stratum, kind = self.strat.sample(rng)
        n = aep.size

        d_point = np.atleast_1d(self.ifd.point_depth(duration_h, aep))
        if cfg.apply_arf:
            arf = np.array([float(self.ifd.arf(cfg.area_km2, duration_h, a)) for a in aep])
        else:
            arf = np.ones(n)
        depth = d_point * arf

        # initial states, optionally conditioned on the sampled rainfall
        if self.state_pool_fn is None:
            sampled = self.states.sample(n, rng)
        else:
            parts = [self.states.sample(1, rng, pool=self.state_pool_fn(a, duration_h))
                     for a in aep]
            sampled = pd.concat(parts, ignore_index=True)
        state_dicts = self.states.to_state_dicts(sampled)

        # hydrograph retention by target AEP: the running best match for each
        # target, as target -> (distance in log AEP, record)
        targets = ([float(t) for t in cfg.hydrograph_aeps]
                   if cfg.store_hydrographs else [])
        log_targets = np.log10(targets) if targets else np.empty(0)
        best: dict[float, tuple] = {}

        rows = []
        for start in range(0, n, cfg.chunk_size):
            sl = slice(start, min(start + cfg.chunk_size, n))
            batch, meta = [], []
            for i in range(sl.start, sl.stop):
                rain, m = self._build_event(depth[i], duration_h, aep[i], rng)
                st = state_dicts[i]
                month = st["date"].month if "date" in st else 1
                pet = self._pet_series(rain.size, month)
                batch.append((rain, pet, st))
                m["month"] = month
                meta.append(m)
            qs = self.engine.run_batch(batch)
            for j, q in enumerate(qs):
                i = sl.start + j
                m = meta[j]
                ipk = int(np.argmax(q))
                rows.append(dict(
                    duration_h=duration_h, stratum=int(stratum[i]),
                    kind=int(kind[i]),
                    aep_rain=float(aep[i]), weight=float(weight[i]),
                    depth_point_mm=float(d_point[i]), arf=float(arf[i]),
                    depth_mm=float(depth[i]),
                    q_peak=float(q[ipk]),
                    t_peak_h=ipk * cfg.dt_hours,
                    volume_m3=float(q.sum() * cfg.dt_hours * 3600.0),
                    prod_store=state_dicts[i]["prod_store"],
                    rout_store=state_dicts[i]["rout_store"],
                    donor_index=int(sampled["donor_index"].iloc[i])
                    if "donor_index" in sampled else -1,
                    **{k: m[k] for k in ("pattern_id", "pattern_band", "pattern_index",
                                         "preburst_ratio", "preburst_depth_mm",
                                         "burst_start_step", "month")}))
                if targets:
                    la = np.log10(max(float(aep[i]), 1e-15))
                    for t, lt in zip(targets, log_targets):
                        dist = abs(la - lt)
                        if t not in best or dist < best[t][0]:
                            best[t] = (dist, {
                                "target_aep": t,
                                "aep_rain": float(aep[i]),
                                "index": i,
                                "duration_h": duration_h,
                                "depth_mm": float(depth[i]),
                                "q_peak": float(q[ipk]),
                                "rain": batch[j][0],
                                "q": q,
                            })

        # rarest target last, so a legend reads frequent -> rare
        kept = [rec for _, rec in
                sorted(best.values(), key=lambda dr: -dr[1]["target_aep"])]
        if cfg.store_hydrographs:
            kept = kept[:int(cfg.store_hydrographs)]
        #: hydrographs retained from the most recent call, for plotting
        self.last_hydrographs = kept
        return pd.DataFrame(rows)

    def run(self) -> "DFFAResults":
        rng = np.random.default_rng(self.cfg.seed)
        frames, hydro = [], {}
        for d in self.cfg.durations_h:
            if self.cfg.progress:
                print(f"  duration {d:g} h: {self.strat.n_events} events", flush=True)
            frames.append(self.run_duration(float(d), rng))
            hydro[float(d)] = self.last_hydrographs
        events = pd.concat(frames, ignore_index=True)
        return DFFAResults(events=events, config=self.cfg,
                           stratification=self.strat, hydrographs=hydro)


@dataclass
class DFFAResults:
    """Simulated events and the estimators built on them.

    ``estimator`` selects how the total probability theorem is applied:

    ``"arr"`` (default)
        Interval-wise, with the ARR Book 4 Section 4.3.3.3 end-interval
        closure -- the open frequent and rare intervals carry the remaining
        probability mass and their conditional exceedance probabilities are
        replaced by geometric means.  Interval masses sum to 1.
    ``"truncated"``
        Pooled weighted empirical exceedance function over the sampled range
        only.  Simpler; the curve is undefined outside
        ``[edges[-1], edges[0]]``.

    ``plotting_position`` is Cunnane's ``a = 0.4`` by default; see
    :mod:`pyfloodrisk.dffa.tpt` for the family and the weighted generalisation.
    """

    events: pd.DataFrame
    config: MCSConfig
    stratification: Stratification
    #: ``duration_h -> [record, ...]``, one record per
    #: :attr:`MCSConfig.hydrograph_aeps` target, frequent first.  Each record
    #: is a dict with ``target_aep``, ``aep_rain`` (the sampled AEP actually
    #: nearest that target), ``index``, ``duration_h``, ``depth_mm``,
    #: ``q_peak``, ``rain`` and ``q``.
    hydrographs: Mapping[float, list] = field(default_factory=dict)
    estimator: str = "arr"
    plotting_position: str | float = DEFAULT_PLOTTING_POSITION
    first_factor: float = ARR_FIRST_FACTOR

    # --------------------------------------------------------------- basics
    @property
    def durations(self) -> list[float]:
        return sorted(self.events["duration_h"].unique())

    def subset(self, duration_h: float) -> pd.DataFrame:
        return self.events[self.events["duration_h"] == duration_h]

    def _arrays(self, duration_h: float):
        s = self.subset(duration_h)
        kind = (s["kind"].to_numpy() if "kind" in s
                else np.zeros(len(s), dtype=int))
        return (s["q_peak"].to_numpy(), s["weight"].to_numpy(),
                s["stratum"].to_numpy(), kind)

    def exceedance_curve(self, duration_h: float):
        """``(discharge, annual exceedance probability)`` for one duration."""
        q, w, iv, kd = self._arrays(duration_h)
        if self.estimator == "arr":
            grid, P, _ = interval_exceedance_curve(
                q, w, iv, kd, plotting_position=self.plotting_position,
                first_factor=self.first_factor)
            return grid, P
        return weighted_exceedance(q, w)

    # ------------------------------------------------------------ quantiles
    def _quantile(self, duration_h: float, aeps) -> np.ndarray:
        q, w, iv, kd = self._arrays(duration_h)
        if self.estimator == "arr":
            return interval_quantile(q, w, iv, kd, aeps, self.plotting_position,
                                     self.first_factor)
        return np.atleast_1d(weighted_quantile(q, w, aeps))

    def quantiles(self, aeps: Sequence[float]) -> pd.DataFrame:
        """Quantiles (m3/s) for every duration; columns are durations."""
        aeps = np.atleast_1d(np.asarray(aeps, float))
        out = {d: self._quantile(d, aeps) for d in self.durations}
        return pd.DataFrame(out, index=pd.Index(aeps, name="aep"))

    def valid_aeps(self, candidates: Sequence[float], margin: float = 0.9
                   ) -> list[float]:
        """Candidate AEPs the simulation can actually resolve.

        Taken from the range of exceedance probabilities the estimator
        actually spans, which with the ARR end-interval closure is wider than
        ``[edges[-1], edges[0]]`` but is *not* accurate right out to AEP = 1:
        the frequent open interval holds its rainfall at ``edges[0]``, so the
        frequent end of the curve is a boundary approximation.  If you need
        50%-20% AEP quantiles, raise ``aep_max`` and supply design rainfalls
        that reach there (ARR's very frequent design rainfalls go to 12 EY).
        """
        lo, hi = np.inf, 0.0
        for d in self.durations:
            _, P = self.exceedance_curve(d)
            P = P[P > 0]
            if P.size:
                lo = min(lo, float(P.min()))
                hi = max(hi, float(P.max()))
        return [float(a) for a in candidates if lo / margin < a < margin * hi]

    def envelope(self, aeps: Sequence[float]) -> pd.DataFrame:
        """Enveloped design quantiles and the critical duration at each AEP."""
        q = self.quantiles(aeps)
        vals = q.to_numpy(float)
        durs = np.asarray(q.columns, float)
        ok = np.isfinite(vals).any(axis=1)
        qmax = np.full(vals.shape[0], np.nan)
        crit = np.full(vals.shape[0], np.nan)
        if ok.any():
            qmax[ok] = np.nanmax(vals[ok], axis=1)
            crit[ok] = durs[np.nanargmax(vals[ok], axis=1)]
        return pd.DataFrame({"q_peak": qmax,
                             "critical_duration_h": crit,
                             "aep_1_in_y": 1.0 / np.asarray(aeps, float)},
                            index=q.index)

    def pooled_quantiles(self, aeps, duration_weights: Mapping[float, float]
                         ) -> pd.Series:
        """Quantiles with duration treated as a random variable.

        Unbiased alternative to enveloping, given a probability mass function
        over durations (must sum to 1).  Durations are pooled by scaling each
        duration's interval weights, so the ARR end-interval treatment is
        preserved: interval ids are offset per duration so that each
        (duration, interval) pair stays its own interval.
        """
        w = np.array([duration_weights[d] for d in self.durations], float)
        if not np.isclose(w.sum(), 1.0):
            raise ValueError("duration_weights must sum to 1")
        qq, ww, ii, kk = [], [], [], []
        offset = 0
        for d, wd in zip(self.durations, w):
            q, wt, iv, kd = self._arrays(d)
            qq.append(q)
            ww.append(wt * wd)
            ii.append(iv + offset)
            kk.append(kd)
            offset += int(iv.max()) + 1
        aeps = np.atleast_1d(np.asarray(aeps, float))
        if self.estimator == "arr":
            vals = interval_quantile(np.concatenate(qq), np.concatenate(ww),
                                     np.concatenate(ii), np.concatenate(kk),
                                     aeps, self.plotting_position, self.first_factor)
        else:
            vals = np.atleast_1d(weighted_quantile(
                np.concatenate(qq), np.concatenate(ww), aeps))
        return pd.Series(vals, index=pd.Index(aeps, name="aep"), name="q_peak")

    def exceedance_of(self, discharge: float, duration_h: float) -> float:
        q, w, iv, kd = self._arrays(duration_h)
        if self.estimator == "arr":
            return interval_exceedance_probability(
                q, w, iv, kd, discharge, self.plotting_position, self.first_factor)
        return exceedance_probability(q, w, discharge)

    # --------------------------------------------------------- uncertainty
    def confidence(self, aeps, n_boot: int = 500, level: float = 0.90,
                   duration_h: float | None = None, seed: int = 7) -> pd.DataFrame:
        """Within-interval bootstrap confidence limits on the quantiles.

        With ``duration_h=None`` the bootstrap is applied to the critical
        duration at each AEP, which is the quantity actually reported.
        """
        aeps = np.atleast_1d(np.asarray(aeps, float))
        rng = np.random.default_rng(seed)
        lo_p, hi_p = (1 - level) / 2, 1 - (1 - level) / 2

        def boot(d, targets):
            q, w, iv, kd = self._arrays(d)
            return bootstrap_quantiles(q, w, iv, targets, kd, n_boot, rng,
                                       self.estimator, self.plotting_position,
                                       self.first_factor)

        if duration_h is not None:
            b = boot(duration_h, aeps)
            lo, hi = np.nanquantile(b, [lo_p, hi_p], axis=0)
        else:
            crit = self.envelope(aeps)["critical_duration_h"].to_numpy()
            lo, hi = np.empty(aeps.size), np.empty(aeps.size)
            for j, (a, d) in enumerate(zip(aeps, crit)):
                if not np.isfinite(d):
                    lo[j] = hi[j] = np.nan
                    continue
                b = boot(d, [a])
                lo[j], hi[j] = np.nanquantile(b[:, 0], [lo_p, hi_p])
        env = self.envelope(aeps)
        return pd.DataFrame({"q_peak": env["q_peak"].to_numpy(),
                             "lower": lo, "upper": hi},
                            index=pd.Index(aeps, name="aep"))

    def stratum_contributions(self, aeps: Sequence[float],
                              duration_h: float | None = None) -> pd.DataFrame:
        """Fraction of ``P(Q > q_aep)`` contributed by each rainfall interval."""
        aeps = np.atleast_1d(np.asarray(aeps, float))
        env = self.envelope(aeps)
        rows = {}
        for a in aeps:
            d = (duration_h if duration_h is not None
                 else env.loc[a, "critical_duration_h"])
            q, w, iv, kd = self._arrays(d)
            qa = (float(env.loc[a, "q_peak"]) if duration_h is None
                  else float(np.atleast_1d(self._quantile(d, [a]))[0]))
            rows[a] = stratum_contribution(q, w, iv, qa, kd, self.plotting_position,
                                           self.first_factor)
        out = pd.DataFrame(rows).T
        out.index.name = "aep"
        out.columns.name = "interval"
        return out

    def end_interval_contributions(self, aeps: Sequence[float]) -> pd.DataFrame:
        """Share of each quantile's exceedance probability from the open ends.

        The ARR end-interval treatment is premised on the boundary intervals
        being "distant from the probability region of interest".  That premise
        is worth checking here rather than assuming: when antecedent wetness is
        itself a sampled variable, a frequent burst on a saturated catchment
        can produce a large flood, so the frequent open interval -- whose
        rainfall is pinned at ``edges[0]`` and whose conditional exceedance is
        damped by the arbitrary ``sqrt(first_factor)`` -- can end up doing real
        work on the answer.  If these shares are not small, move the boundary
        (raise ``aep_max``, lower ``aep_min``) rather than arguing about the
        value of ``first_factor``.
        """
        aeps = np.atleast_1d(np.asarray(aeps, float))
        env = self.envelope(aeps)
        rows = {}
        for a in aeps:
            d = env.loc[a, "critical_duration_h"]
            if not np.isfinite(d):
                rows[a] = dict(frequent_open=np.nan, rare_open=np.nan)
                continue
            q, w, iv, kd = self._arrays(d)
            c = stratum_contribution(q, w, iv, float(env.loc[a, "q_peak"]), kd,
                                     self.plotting_position, self.first_factor)
            ids = np.unique(iv)
            kind_of = {int(k): int(kd[iv == k][0]) for k in ids}
            rows[a] = dict(
                frequent_open=float(sum(c[j] for j, k in enumerate(ids)
                                        if kind_of[int(k)] == FREQUENT_OPEN)),
                rare_open=float(sum(c[j] for j, k in enumerate(ids)
                                    if kind_of[int(k)] == RARE_OPEN)))
        out = pd.DataFrame(rows).T
        out.index.name = "aep"
        return out

    def convergence(self, aeps: Sequence[float],
                    fractions=(0.1, 0.2, 0.4, 0.6, 0.8, 1.0),
                    n_rep: int = 5, seed: int = 11) -> pd.DataFrame:
        """Enveloped quantiles recomputed from a growing subsample per stratum.

        ``n_rep`` independent subsamples are drawn at each size, so the spread
        at a given number of simulations per stratum is itself informative: it
        is roughly the sampling error you would incur by stopping there.
        """
        rng = np.random.default_rng(seed)
        aeps = np.atleast_1d(np.asarray(aeps, float))
        recs = []
        for f in fractions:
            reps = 1 if f >= 1.0 else n_rep
            for rep in range(reps):
                keep = []
                for _, g in self.events.groupby(["duration_h", "stratum"], sort=False):
                    n_k = len(g)
                    m = max(2, int(round(f * n_k)))
                    g = g.iloc[rng.permutation(n_k)[:m]].copy()
                    # a stratum keeps its probability mass, over fewer events
                    g["weight"] = g["weight"] * (n_k / m)
                    keep.append(g)
                sub = pd.concat(keep, ignore_index=True)
                env = DFFAResults(sub, self.config, self.stratification, {},
                                  self.estimator, self.plotting_position,
                                  self.first_factor).envelope(aeps)
                n_eff = int(round(f * self.stratification.n_per_interval.mean()))
                for a in aeps:
                    recs.append(dict(fraction=f, rep=rep, n_per_stratum=n_eff,
                                     aep=a, q_peak=float(env.loc[a, "q_peak"])))
        return pd.DataFrame(recs)

    # ------------------------------------------------------------- exports
    def summary(self, aeps=(0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001,
                            0.0005, 0.0002)) -> pd.DataFrame:
        aeps = self.valid_aeps(aeps)
        env = self.envelope(aeps)
        ci = self.confidence(aeps)
        env["lower_90"] = ci["lower"]
        env["upper_90"] = ci["upper"]
        if self.estimator == "arr" and self.stratification.open_ends:
            ends = self.end_interval_contributions(aeps)
            env["end_share"] = ends.sum(axis=1)
        return env

    def to_csv(self, path: str):
        self.events.to_csv(path, index=False)
