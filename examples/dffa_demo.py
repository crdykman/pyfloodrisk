"""
Derived flood frequency analysis, end to end on a bundled demo station.

Every input comes from the package: the climate record drives the
continuous GR4H run that the antecedent states are drawn from, the
temporal patterns come from the station's ARR Data Hub increments file,
and the design rainfalls come from the bundled *demonstration* IFD table
-- which is the one input you must replace with real BoM IFD depths
(``ifd_table_from_csv``) before reading anything into the numbers (see
``docs/dffa.md``).

Run:  python examples/dffa_demo.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pyfloodrisk.dffa import (DEMO_PARAMETERS, GR4HEventEngine, MCSConfig,
                              PreBurstSampler, Stratification, DerivedFFA,
                              continuous_state_table, diagnostics,
                              have_pyvinecopulib, load_station_forcings,
                              pet_climatology, state_sampler_from_run,
                              station_ifd, station_patterns)
from pyfloodrisk.demo_data import catchment_data

STATION = "421026"
DURATIONS_H = [3, 6, 12, 24, 48, 72]
AEPS = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002]


def main():
    area = catchment_data(STATION)
    params = DEMO_PARAMETERS[STATION]          # calibrate() for real work
    forcings = load_station_forcings(STATION)

    # 1. the state distribution: a continuous GR4H run over the record
    state_table = continuous_state_table(forcings, params, area,
                                         warmup_hours=8760, thin=6)
    states = state_sampler_from_run(state_table, params, method="bootstrap")

    # 2. rainfall: patterns from the station's ARR file, IFD from the bundled
    #    demonstration table (replace with BoM depths via ifd_table_from_csv)
    patterns = station_patterns(STATION)
    ifd = station_ifd(STATION)

    # 3. the event model: the same GR4H, hot-started from a sampled state
    engine = GR4HEventEngine(params, area_km2=area, dt_hours=1.0)

    # A modest experiment so the demo runs in a couple of minutes.  The ARR
    # implementation uses 50 strata x 200 simulations for each duration.
    # aep_max=0.9 rather than the ARR default 0.5: see the end-interval
    # discussion in docs/dffa.md
    strat = Stratification.uniform_in_z(aep_max=0.9, aep_min=1e-5,
                                        n_strata=30, n_per_stratum=60)
    cfg = MCSConfig(area_km2=area, durations_h=DURATIONS_H, dt_hours=1.0,
                    tail_multiple=2.0, tail_min_hours=36.0,
                    preburst=PreBurstSampler(fixed_ratio=0.0),   # off here
                    store_hydrographs=24, seed=20260909)

    print(f"{STATION}: {area:.0f} km2, {len(state_table)} donor states, "
          f"{len(DURATIONS_H)} durations x {strat.n_events} events "
          f"= {len(DURATIONS_H) * strat.n_events} GR4H event runs")
    res = DerivedFFA(ifd, patterns, states, engine, cfg, strat,
                     pet=pet_climatology(forcings)).run()

    print("\nDerived flood frequency curve (duration envelope):")
    print(res.summary().round(1).to_string())

    res.to_csv("dffa_events.csv")
    files = diagnostics.plot_all(res, outdir="figures", ifd=ifd, states=states)
    print("\nFigures written:", *files, sep="\n  ")

    compare_state_methods(ifd, patterns, state_table, params, engine, forcings)
    return res


def compare_state_methods(ifd, patterns, state_table, params, engine, forcings,
                          duration_h=12.0):
    """How much of the design estimate comes from the *joint* state structure?

    Same rainfall sampling, same seed, one duration, several ways of
    sampling the initial state.  ``independent_kde`` keeps every marginal
    and throws away the cross-correlation, so the gap between it and
    ``bootstrap`` is the part of the design flood that depends on the
    stores being wet *together* -- which is the question a reviewer will
    ask about the state distribution.
    """
    strat = Stratification.uniform_in_z(aep_max=0.9, aep_min=1e-5,
                                        n_strata=25, n_per_stratum=40,
                                        n_per_end=40)
    cfg = MCSConfig(area_km2=engine.area_km2, durations_h=[duration_h],
                    dt_hours=1.0, tail_multiple=2.0, tail_min_hours=36.0,
                    store_hydrographs=0, progress=False, seed=20260909)
    pet = pet_climatology(forcings)

    methods = ["bootstrap", "smoothed", "empirical_copula", "independent_kde"]
    if not have_pyvinecopulib():
        methods = ["bootstrap", "smoothed"]
        print("\npyvinecopulib not installed - skipping the copula/KDE methods")

    runs, taus = {}, {}
    for method in methods:
        states = state_sampler_from_run(
            state_table, params, method=method,
            uh_profile="mean" if method == "independent_kde" else "scaled_donor")
        runs[method] = DerivedFFA(ifd, patterns, states, engine, cfg, strat,
                                  pet=pet).run()
        sampled = states.sample(20000, np.random.default_rng(0))
        taus[method] = states.dependence(sampled)["tau_sampled"]

    table = pd.DataFrame({m: r.envelope(AEPS)["q_peak"] for m, r in runs.items()})
    print(f"\nState-sampling comparison ({duration_h:g} h storms, "
          f"{strat.n_events} events each):")
    print(table.round(1).to_string())
    print("\nratio to bootstrap:")
    print(table.div(table["bootstrap"], axis=0).round(3).to_string())
    print("\nKendall tau between state variables as sampled:")
    print(pd.DataFrame(taus).round(3).to_string())

    ax = diagnostics.plot_curve_comparison(runs, AEPS, ratio_to="bootstrap")
    ax.figure.savefig("figures/dffa_state_methods.png", bbox_inches="tight")
    print("\n  figures/dffa_state_methods.png")
    return runs


if __name__ == "__main__":
    main()
