"""Initial-state distributions from a disaggregated rainfall ensemble.

The DFFA is unchanged by this.  It still draws a state per duration from a
pool of states the catchment was actually in immediately before its own
bursts.  All that changes is how much record those pools are drawn from:
100 pyraingen realisations of 30 hourly years, against the ten years the
bundled record holds.

The realisations share one daily parent record and one PET series, so they
are not 100 independent climates -- they are 100 ways of distributing the
same rain within its days.  That matters more at some durations than
others, which is why this prints distinct burst dates alongside the row
count: at long durations the ensemble largely repeats the same events, and
the row count on its own would flatter the pool.

Run:  python examples/ensemble_states.py
"""

from __future__ import annotations

from pathlib import Path

from pyfloodrisk.dffa import DEMO_PARAMETERS
from pyfloodrisk.demo_data import catchment_data
from pyfloodrisk.states import extract_initial_states_ensemble

STATION = "303203"
SIMS = Path(r"C:\Users\z3460382\OneDrive - UNSW\postdoc\flood_risk_software"
            r"\examples\303203_pyraingen\subdaily_sims")
OUT = Path("ensemble_states")
DURATIONS_H = (6, 9, 12, 24, 36, 48, 72, 96)
EY = 6


def main():
    params = DEMO_PARAMETERS[STATION]          # the calibrated set
    paths = sorted(SIMS.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"no realisations under {SIMS}")
    print(f"{STATION}: {len(paths)} realisations x {len(DURATIONS_H)} durations, "
          f"{EY} bursts per year of each")

    pools = extract_initial_states_ensemble(
        paths, params, catchment_data(STATION), ey=EY, durs=DURATIONS_H)

    OUT.mkdir(exist_ok=True)
    print(f"\n{'duration':>9s} {'donors':>8s} {'distinct':>9s} {'per real.':>10s}"
          f" {'median prod':>12s} {'median uh':>10s}")
    for d, pool in pools.items():
        pool.to_csv(OUT / f"states_{d:g}h.csv.gz", index=False)
        uh = [c for c in pool.columns if c.startswith(("uh1_", "uh2_"))]
        # distinct *dates*, not rows: the realisations share a daily parent,
        # so the same long-duration burst recurs in most of them
        print(f"{d:8.0f}h {len(pool):8d} {pool['date'].nunique():9d} "
              f"{len(pool) // len(paths):10d} "
              f"{pool['prod_store'].median():12.2f} "
              f"{pool[uh].sum(axis=1).median():10.2f}")
    print(f"\nwritten to {OUT.resolve()}")
    return pools


if __name__ == "__main__":
    main()
