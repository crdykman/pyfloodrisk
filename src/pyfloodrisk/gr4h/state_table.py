"""The GR4H state table: a continuous run's state at every timestep.

This is the model's own output format, not a derived-FFA concept -- a
single design event and a full Monte Carlo experiment both hot-start from
it -- so it lives beside the model rather than in either consumer.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from .GR4H_model import GR4H

__all__ = ["continuous_state_table", "uh_columns", "state_from_row"]


def uh_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    """The unit-hydrograph memory columns of a state table, in order.

    Sorted by their numeric suffix rather than lexically: as strings
    ``uh1_10`` sorts before ``uh1_9``, which would silently scramble the
    memory an event is hot-started from.

    Returns
    -------
    ``(uh1_cols, uh2_cols)``, either of which may be empty if the frame
    carries no memory.
    """
    def _sorted(prefix: str) -> list[str]:
        return sorted((c for c in frame.columns if c.startswith(prefix)),
                      key=lambda c: int(c.split("_")[1]))
    return _sorted("uh1_"), _sorted("uh2_")


def state_from_row(row, uh1_cols=None, uh2_cols=None,
                   prod_col="prod_store", rout_col="rout_store") -> dict:
    """One state-table row as the dict :meth:`GR4H.run_from_state` takes.

    Parameters
    ----------
    row :
        A row of a state table, as ``frame.iloc[i]``.
    uh1_cols, uh2_cols :
        Memory column names.  Pass them when converting many rows, so the
        columns are resolved once rather than per row; ``None`` resolves
        them from the row itself.
    prod_col, rout_col :
        Column names of the two stores, for a table that does not use the
        default names.

    Returns
    -------
    dict with ``prod_store`` and ``rout_store`` in mm, plus ``uh1``/``uh2``
    where the row carries them.
    """
    if uh1_cols is None or uh2_cols is None:
        cols = pd.DataFrame(columns=list(row.index))
        auto1, auto2 = uh_columns(cols)
        uh1_cols = auto1 if uh1_cols is None else uh1_cols
        uh2_cols = auto2 if uh2_cols is None else uh2_cols

    state = {"prod_store": float(row[prod_col]),
             "rout_store": float(row[rout_col])}
    if len(uh1_cols):
        state["uh1"] = row[list(uh1_cols)].to_numpy(float)
    if len(uh2_cols):
        state["uh2"] = row[list(uh2_cols)].to_numpy(float)
    return state


def continuous_state_table(
    forcings: pd.DataFrame,
    parameters: Mapping[str, float],
    area_km2: float,
    warmup_hours: int = 8760,
    thin: int = 1,
) -> pd.DataFrame:
    """Run GR4H continuously and return its state at every timestep.

    This is the input with no ARR equivalent: the distribution the design
    events' antecedent conditions are drawn from.  One row per retained
    timestep, holding the production and routing stores in mm, the
    unit-hydrograph memory, and the date -- exactly what
    :class:`~pyfloodrisk.dffa.states.InitialStateSampler` consumes.

    Parameters
    ----------
    forcings :
        Hourly ``prec``/``pet`` (mm), datetime-indexed; see
        :func:`~pyfloodrisk.demo_data.load_station_forcings`.
    parameters :
        GR4H parameters.  ``x4`` fixes the length of the UH memory, so the
        table is only valid for the parameter set that produced it.
    area_km2 :
        Catchment area (only affects the reported discharge).
    warmup_hours :
        Leading hours discarded, so the stores are not still relaxing from
        their arbitrary initial values.  One year by default.
    thin :
        Keep every ``thin``-th row.  The states of consecutive hours are
        nearly identical, so thinning costs almost no information and makes
        the sampler's donor pool cheaper to hold and to fit models to.

    Returns
    -------
    DataFrame with columns ``date``, ``prod_store``, ``rout_store``,
    ``q_cumecs``, ``uh1_0..``, ``uh2_0..``.

    Notes
    -----
    The state on row ``t`` is the state *after* hour ``t`` has been routed,
    so an event started from it begins at hour ``t + 1``: the same
    convention as :func:`~pyfloodrisk.extract_initial_states`.
    """
    model = GR4H(area=area_km2, params=dict(parameters))
    out = model.run_from_state(
        forcings["prec"].to_numpy(float),
        forcings["pet"].to_numpy(float),
        record_uh=True,
    )

    n = len(forcings)

    table = pd.DataFrame({
        "date": forcings.index,
        "prod_store": out["prod_store"],
        "rout_store": out["rout_store"],
        "q_cumecs": out["qt_cumecs"],
    })
    # built in one concat rather than column by column: a long x4 means a
    # long unit hydrograph, and inserting 120-odd columns one at a time
    # fragments the frame badly
    uh = {f"uh1_{j}": out["uh1"][:, j] for j in range(out["uh1"].shape[1])}
    uh.update({f"uh2_{j}": out["uh2"][:, j]
               for j in range(out["uh2"].shape[1])})
    table = pd.concat([table, pd.DataFrame(uh, index=table.index)], axis=1)

    keep = np.zeros(n, dtype=bool)
    keep[int(warmup_hours):] = True
    if thin > 1:
        thinned = np.zeros(n, dtype=bool)
        thinned[::int(thin)] = True
        keep &= thinned
    table = table.loc[keep].reset_index(drop=True)
    if table.empty:
        raise ValueError("no states left after warm-up and thinning")
    return table

