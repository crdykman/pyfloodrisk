"""
Tests for the parts of the derived-FFA framework that can be checked
against something known.

The important one is ``test_tpt_recovers_analytical_quantiles``: it replaces
the hydrological model with the identity map so that the derived frequency
curve of "peaks" must reproduce the rainfall frequency curve exactly.  If
the stratification, the weights or the total probability theorem estimator
are wrong, that test fails.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from pyfloodrisk.dffa import (DerivedFFA, GR4HEventEngine, IFDCurve,
                              InitialStateSampler, MCSConfig,
                              Stratification, TemporalPatternLibrary,
                              have_pyvinecopulib, resample_increments, tpt)

PARAMS = {"x1": 300.0, "x2": 0.0, "x3": 40.0, "x4": 3.0}


# ------------------------------------------------------------------ IFD curve
def test_ifd_roundtrip():
    aeps = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
    depths = [40, 55, 66, 78, 95, 110]
    ifd = IFDCurve(pd.DataFrame({24.0: depths}, index=aeps).T)
    for a, d in zip(aeps, depths):
        assert np.isclose(float(ifd.point_depth(24.0, a)), d, rtol=1e-6)
        assert np.isclose(float(ifd.aep_of_depth(24.0, d)), a, rtol=2e-2)


def test_ifd_monotone_and_extrapolates():
    aeps = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
    ifd = IFDCurve(pd.DataFrame({6.0: [20, 28, 34, 40, 49, 57],
                                 24.0: [40, 55, 66, 78, 95, 110]}, index=aeps).T)
    a = np.logspace(0, -6, 60)
    d = np.atleast_1d(ifd.point_depth(24.0, a))
    assert np.all(np.diff(d) > 0)              # depth increases as AEP decreases
    assert float(ifd.point_depth(12.0, 0.01)) > float(ifd.point_depth(6.0, 0.01))


# -------------------------------------------------------------- stratification
def test_truncated_weights_sum_to_covered_mass():
    s = Stratification.uniform_in_z(aep_max=0.5, aep_min=1e-6, n_strata=25,
                                    n_per_stratum=40, open_ends=False)
    aep, w, k, kind = s.sample(np.random.default_rng(0))
    assert aep.size == 25 * 40
    assert np.isclose(w.sum(), s.covered_mass, rtol=1e-12)
    assert np.isclose(s.covered_mass, 0.5 - 1e-6)
    assert np.all(kind == 0)
    # every sample lies inside its own interval
    assert np.all(aep <= s.edges[k]) and np.all(aep >= s.edges[k + 1])


def test_arr_open_ends_close_the_probability_domain():
    """ARR Book 4, Section 4.3.3.3: the interval masses must sum to 1."""
    s = Stratification.uniform_in_z(aep_max=0.5, aep_min=1e-6, n_strata=25,
                                    n_per_stratum=40, n_per_end=40)
    aep, w, k, kind = s.sample(np.random.default_rng(0))
    assert np.isclose(s.covered_mass, 1.0, rtol=1e-12)
    assert np.isclose(w.sum(), 1.0, rtol=1e-12)
    assert s.n_intervals == 27 and aep.size == 27 * 40
    # end intervals: rainfall held at the internal bound, mass from the tails
    freq, rare = kind == 1, kind == 2
    assert np.allclose(aep[freq], 0.5) and np.allclose(aep[rare], 1e-6)
    assert np.isclose(w[freq].sum(), 0.5)        # 1 - non-exceedance of bound
    assert np.isclose(w[rare].sum(), 1e-6)       # exceedance of bound
    # ordered frequent to rare, contiguous ids
    assert np.array_equal(np.unique(k), np.arange(27))
    assert k[freq].max() == 0 and k[rare].min() == 26


def test_tpt_recovers_analytical_quantiles():
    """Identity model: the derived curve must equal the rainfall curve."""
    s = Stratification.uniform_in_z(aep_max=0.5, aep_min=1e-7, n_strata=60,
                                    n_per_stratum=100, open_ends=False)
    aep, w, k, kind = s.sample(np.random.default_rng(1))
    # "discharge" = a strictly increasing function of the rainfall variate
    q = 10.0 * norm.ppf(1 - aep) + 100.0
    for target in (0.1, 0.01, 0.001, 1e-4, 1e-5):
        got = tpt.weighted_quantile(q, w, target)
        want = 10.0 * norm.ppf(1 - target) + 100.0
        assert abs(got - want) < 0.02 * abs(want - 100.0) + 0.05, (target, got, want)


def test_exceedance_and_quantile_are_inverses():
    rng = np.random.default_rng(3)
    s = Stratification.uniform_in_z(n_strata=20, n_per_stratum=50,
                                    open_ends=False)
    aep, w, k, kind = s.sample(rng)
    q = np.exp(norm.ppf(1 - aep))
    for target in (0.05, 0.01, 0.002):
        qq = tpt.weighted_quantile(q, w, target)
        assert np.isclose(tpt.exceedance_probability(q, w, qq), target, rtol=0.05)


def test_stratum_contributions_sum_to_one():
    s = Stratification.uniform_in_z(n_strata=15, n_per_stratum=40)
    aep, w, k, kind = s.sample(np.random.default_rng(4))
    q = norm.ppf(1 - aep) + 10.0
    qa = float(tpt.interval_quantile(q, w, k, kind, [0.01])[0])
    c = tpt.stratum_contribution(q, w, k, qa, kind)
    assert np.isclose(c.sum(), 1.0)
    # the design quantile must not be driven by an open end interval
    assert np.argmax(c) not in (0, len(c) - 1)


def test_arr_estimator_recovers_analytical_quantiles():
    """Identity model again, but through the interval-wise ARR estimator.

    Inside the intermediate intervals this must be near-exact: the rare open
    interval contributes its full mass (geometric mean of 1 and 1) and the
    frequent open interval contributes nothing above its bound.
    """
    s = Stratification.uniform_in_z(aep_max=0.5, aep_min=1e-7, n_strata=60,
                                    n_per_stratum=100, n_per_end=100)
    aep, w, k, kind = s.sample(np.random.default_rng(1))
    q = 10.0 * norm.ppf(1 - aep) + 100.0
    for target in (0.1, 0.01, 0.001, 1e-4, 1e-5):
        got = float(tpt.interval_quantile(q, w, k, kind, [target])[0])
        want = 10.0 * norm.ppf(1 - target) + 100.0
        assert abs(got - want) < 0.03 * abs(want - 100.0) + 0.05, (target, got, want)


def test_arr_end_interval_geometric_means():
    """The two geometric-mean rules, checked against the closed form.

    Frequent open interval: P(Q>q|R_1) -> geometric mean of c and 0.1c, i.e.
    c*sqrt(0.1).  Rare open interval: geometric mean of c and 1, i.e. sqrt(c).
    """
    n = 50
    q = np.full(2 * n, 10.0)
    w = np.concatenate([np.full(n, 0.5 / n), np.full(n, 1.0 / n)])
    iv = np.concatenate([np.zeros(n, int), np.ones(n, int)])
    kind = np.concatenate([np.full(n, 1), np.full(n, 2)])   # frequent, rare
    grid = np.array([1.0])                                  # every event exceeds
    for pp in ("cunnane", "hazen", "weibull"):
        a = tpt.plotting_position_a(pp)
        c = (n - a) / (n + 1 - 2 * a)                       # all n exceed
        _, P, contrib = tpt.interval_exceedance_curve(q, w, iv, kind, grid=grid,
                                                      plotting_position=pp)
        assert np.isclose(contrib[0, 0], 0.5 * c * np.sqrt(0.1)), pp
        assert np.isclose(contrib[1, 0], 1.0 * np.sqrt(c)), pp
        assert np.isclose(P[0], 0.5 * c * np.sqrt(0.1) + np.sqrt(c)), pp


def test_plotting_positions():
    """Equal weights must reduce to the textbook (i - a)/(n + 1 - 2a)."""
    n = 40
    q = np.arange(n, dtype=float)
    w = np.full(n, 1.0 / n)
    i = np.arange(1, n + 1)
    for name, a in tpt.PLOTTING_POSITIONS.items():
        qs, p = tpt.weighted_exceedance(q, w, name)
        assert np.allclose(p, (i - a) / (n + 1 - 2 * a)), name
        assert np.all(np.diff(qs) < 0)
    assert tpt.plotting_position_a("cunnane") == 0.4
    assert tpt.DEFAULT_PLOTTING_POSITION == "cunnane"
    with pytest.raises(ValueError):
        tpt.plotting_position_a("gumbel")
    with pytest.raises(ValueError):
        tpt.plotting_position_a(0.8)


def test_unequal_weight_plotting_position_stays_monotone():
    rng = np.random.default_rng(0)
    q = rng.random(200)
    w = rng.random(200)
    w /= w.sum() / 0.5                       # total mass 0.5
    for pp in ("cunnane", "hazen"):
        qs, p = tpt.weighted_exceedance(q, w, pp)
        assert np.all(np.diff(p) > 0)
        assert 0 < p[0] and p[-1] < 0.5


# -------------------------------------------------------- temporal patterns
def test_resample_conserves_mass():
    inc = np.array([1.0, 3.0, 6.0, 2.0, 0.5, 0.5])       # 6 x 5 min
    for dt in (5.0, 10.0, 15.0, 60.0, 2.5):
        out = resample_increments(inc, 5.0, dt)
        assert np.isclose(out.sum(), inc.sum())
    assert resample_increments(inc, 5.0, 60.0).size == 1
    assert resample_increments(inc, 5.0, 10.0).size == 3


def test_pattern_sampling_uses_the_right_band():
    inc = {(60, "frequent"): (np.eye(10) * 100, 6.0),
           (60, "rare"): (np.ones((10, 10)) * 10, 6.0)}
    lib = TemporalPatternLibrary.from_arrays(inc)
    rng = np.random.default_rng(0)
    f, _, band, _ = lib.sample(1.0, 0.5, 1.0, rng)
    # a 1 h storm aggregated to an hourly timestep is a single increment
    assert band == "frequent" and f.size == 1 and np.isclose(f[0], 1.0)
    f, _, band, _ = lib.sample(1.0, 0.001, 0.1, rng)
    assert band == "rare" and f.size == 10 and np.allclose(f, 0.1)


# ------------------------------------------------------------ state sampling
def _state_table(n=5000, x1=300.0, x3=50.0, seed=0):
    rng = np.random.default_rng(seed)
    z1 = np.clip(rng.beta(2, 2, n), 0.01, 0.99)
    z2 = np.clip(0.6 * z1 + 0.3 * rng.random(n), 0.01, 0.99)
    df = pd.DataFrame({"date": pd.date_range("1990-01-01", periods=n, freq="6h"),
                       "prod_store": z1 * x1, "rout_store": z2 * x3})
    prof = np.array([4.0, 3.0, 2.0, 1.0])
    mag = z1 ** 2 * (0.5 + rng.random(n))          # magnitude, not deterministic
    for j in range(4):
        df[f"uh1_{j}"] = mag * prof[j]
    return df


@pytest.mark.parametrize("method", ["bootstrap", "smoothed", "empirical_copula"])
def test_state_sampler_preserves_bounds_and_dependence(method):
    if "copula" in method and not have_pyvinecopulib():
        pytest.skip("pyvinecopulib not installed")
    df = _state_table()
    s = InitialStateSampler(df, x1=300.0, x3=50.0, date_col="date",
                            uh1_cols=[f"uh1_{j}" for j in range(4)], method=method)
    out = s.sample(4000, np.random.default_rng(2))
    assert out["prod_store"].between(0, 300).all()
    assert (out["rout_store"] >= 0).all()
    tau = s.dependence(out)
    for pair, row in tau.iterrows():
        assert abs(row["tau_input"] - row["tau_sampled"]) < 0.08, (method, pair, row)


def test_independent_kde_destroys_dependence_but_keeps_marginals():
    """The point of the independent option: same margins, no cross-correlation."""
    if not have_pyvinecopulib():
        pytest.skip("pyvinecopulib not installed")
    df = _state_table(n=8000)
    s = InitialStateSampler(df, x1=300.0, x3=50.0, date_col="date",
                            uh1_cols=[f"uh1_{j}" for j in range(4)],
                            method="independent_kde", uh_profile="mean")
    out = s.sample(8000, np.random.default_rng(3))
    tau = s.dependence(out)
    assert (tau["tau_input"].abs() > 0.3).any()          # input is dependent
    assert (tau["tau_sampled"].abs() < 0.06).all()       # output is not
    # marginals preserved to within KDE smoothing
    for c in ("prod_store", "rout_store"):
        for p in (0.1, 0.5, 0.9):
            assert np.isclose(np.quantile(out[c], p), np.quantile(df[c], p),
                              rtol=0.12), (c, p)
    assert out["prod_store"].between(0, 300).all()


def test_uh_memory_is_rescaled_not_resampled_ordinate_wise():
    """Sampled UH profiles must stay on the reachable manifold."""
    if not have_pyvinecopulib():
        pytest.skip("pyvinecopulib not installed")
    df = _state_table(n=4000)
    cols = [f"uh1_{j}" for j in range(4)]
    s = InitialStateSampler(df, x1=300.0, x3=50.0, uh1_cols=cols,
                            method="empirical_copula", uh_profile="scaled_donor")
    out = s.sample(2000, np.random.default_rng(4))
    assert (out[cols].to_numpy() >= 0).all()
    # the total matches the sampled uh_total, and the profile keeps its shape
    assert np.allclose(out[cols].sum(axis=1), out["uh_total"], atol=1e-9)
    shape_in = df[cols].to_numpy() / df[cols].to_numpy().sum(axis=1, keepdims=True)
    shape_out = (out[cols].to_numpy()
                 / np.maximum(out[cols].to_numpy().sum(axis=1, keepdims=True), 1e-12))
    assert np.allclose(shape_in.mean(axis=0), shape_out.mean(axis=0), atol=0.02)


def test_seasonal_conditioning():
    df = _state_table()
    s = InitialStateSampler(df, x1=300.0, x3=50.0, date_col="date")
    out = s.sample(500, np.random.default_rng(5), month=[6, 7, 8])
    assert set(pd.to_datetime(out["date"]).dt.month.unique()) <= {6, 7, 8}


# -------------------------------------------------------------- event engine
def test_engine_accepts_a_mapping_or_a_sequence():
    a = GR4HEventEngine(PARAMS, area_km2=100.0)
    b = GR4HEventEngine([PARAMS["x1"], PARAMS["x2"], PARAMS["x3"], PARAMS["x4"]],
                        area_km2=100.0)
    assert a.params == b.params
    assert a.state_lengths() == (3, 6) == b.state_lengths()
    # calibration parameters carry ps0/rs0 as well; the state supersedes them
    assert GR4HEventEngine(dict(PARAMS, ps0=0.9, rs0=0.1),
                           area_km2=100.0).params == a.params
    with pytest.raises(ValueError, match="missing"):
        GR4HEventEngine({"x1": 1.0}, area_km2=1.0)
    with pytest.raises(ValueError, match="four parameters"):
        GR4HEventEngine([1.0, 2.0, 3.0], area_km2=1.0)


def test_engine_respects_mass_balance():
    """Runoff cannot exceed rainfall plus the water already in store."""
    eng = GR4HEventEngine(PARAMS, area_km2=100.0)
    rain = np.concatenate([np.full(6, 20.0), np.zeros(400)])
    pet = np.zeros(rain.size)
    st = eng.initial_state(prod_frac=0.6, rout_frac=0.5)
    q = eng.run_event(rain, pet, st)
    depth_out = q.sum() / (100.0 / 3.6)                       # mm, dt = 1 h
    storage0 = st["prod_store"] + st["rout_store"]
    assert 0.0 < depth_out <= rain.sum() + storage0
    assert q.argmax() < 40                                    # peak during the burst


def test_engine_unit_conversion():
    """1 mm/h of runoff from 100 km2 is 27.8 m3/s."""
    from pyfloodrisk.dffa import mm_per_step_to_cumecs
    assert np.isclose(mm_per_step_to_cumecs(1.0, 100.0, 1.0), 100.0 / 3.6)
    assert np.isclose(mm_per_step_to_cumecs(1.0, 100.0, 0.5), 100.0 / 1.8)


def test_wetter_states_give_bigger_peaks():
    """The premise of the whole framework, on the engine it now runs on."""
    eng = GR4HEventEngine(PARAMS, area_km2=100.0)
    rain = np.concatenate([np.full(12, 8.0), np.zeros(100)])
    pet = np.zeros(rain.size)
    peaks = [eng.run_event(rain, pet, eng.initial_state(f, f)).max()
             for f in (0.1, 0.4, 0.7, 0.95)]
    assert np.all(np.diff(peaks) > 0)


def test_engine_warns_at_a_non_hourly_timestep():
    with pytest.warns(UserWarning, match="hourly model"):
        GR4HEventEngine(PARAMS, area_km2=100.0, dt_hours=0.25)


# ------------------------------------------------- hydrograph retention
def _retention_run(store=20, aeps=(0.5, 0.2, 0.1, 0.05, 0.02, 0.01)):
    """A small run whose only purpose is the retained hydrographs."""
    ifd = IFDCurve(pd.DataFrame(
        {0.5: [40.0, 60.0], 0.05: [70.0, 100.0], 0.005: [110.0, 150.0]},
        index=[12.0, 24.0]))
    tp = TemporalPatternLibrary.from_arrays(
        {(720, b): (np.tile(100.0 / 12, (3, 12)), 60.0)
         for b in ("frequent", "intermediate", "rare")})
    table = pd.DataFrame({"date": pd.date_range("2000-01-01", periods=200, freq="h"),
                          "prod_store": np.linspace(20, 300, 200),
                          "rout_store": np.linspace(5, 50, 200)})
    states = InitialStateSampler(table, x1=350.0, x3=60.0, date_col="date")
    engine = GR4HEventEngine({"x1": 350.0, "x2": -0.8, "x3": 60.0, "x4": 6.0},
                             area_km2=100.0)
    cfg = MCSConfig(area_km2=100.0, durations_h=[12.0], dt_hours=1.0,
                    store_hydrographs=store, hydrograph_aeps=aeps,
                    progress=False, seed=7)
    strat = Stratification.uniform_in_z(aep_max=0.9, aep_min=1e-4,
                                        n_strata=12, n_per_stratum=20,
                                        n_per_end=20)
    return DerivedFFA(ifd, tp, states, engine, cfg, strat).run()


def test_hydrographs_are_retained_one_per_target_aep():
    res = _retention_run()
    kept = res.hydrographs[12.0]
    assert [r["target_aep"] for r in kept] == [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
    # frequent first, so a legend reads in order
    assert [r["aep_rain"] for r in kept] == sorted(
        (r["aep_rain"] for r in kept), reverse=True)


def test_each_retained_hydrograph_is_the_nearest_event_to_its_target():
    """The whole point: retention must not depend on simulation order."""
    res = _retention_run()
    events = res.events[res.events["duration_h"] == 12.0].reset_index(drop=True)
    all_aeps = events["aep_rain"].to_numpy(float)
    for rec in res.hydrographs[12.0]:
        nearest = np.abs(np.log10(all_aeps) - np.log10(rec["target_aep"])).min()
        got = abs(np.log10(rec["aep_rain"]) - np.log10(rec["target_aep"]))
        assert got == pytest.approx(nearest)
        # and the record describes the event it points at
        row = events.iloc[rec["index"]]
        assert rec["aep_rain"] == pytest.approx(row["aep_rain"])
        assert rec["q_peak"] == pytest.approx(row["q_peak"])


def test_retention_reaches_rare_events_not_just_the_first_simulated():
    """The old rule kept every 37th event, so it never left the frequent end."""
    res = _retention_run()
    kept = res.hydrographs[12.0]
    assert min(r["aep_rain"] for r in kept) <= 0.02
    # rarer rainfall, deeper burst
    depths = [r["depth_mm"] for r in kept]
    assert depths == sorted(depths)


def test_store_hydrographs_zero_retains_none():
    assert _retention_run(store=0).hydrographs[12.0] == []


def test_store_hydrographs_caps_the_targets_kept():
    kept = _retention_run(store=2).hydrographs[12.0]
    assert [r["target_aep"] for r in kept] == [0.5, 0.2]
