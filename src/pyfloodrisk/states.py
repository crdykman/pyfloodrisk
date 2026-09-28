"""Baseflow separation and hydrologic event delineation.

Python port of the R ``hydroEvents`` package: baseflow separation via the
Lyne-Hollick recursive digital filter, event delineation by either a
peaks-over-threshold (POT) or local-maxima method, and extraction of
antecedent model states at each delineated event for design flood
simulation.

Delineation finds every rise in the series, most of which are not floods, so
:func:`hydro_event_pipeline` can trim the list to an exceedance-per-year rate
-- ``ey=6`` keeps the six largest events per year of record, ranked on peak
flow or on event volume (:func:`threshold_events_by_ey`).  That trimming is
opt-in: the default ``ey=None`` returns every rise.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .events import rainfall_events
from .gr4h.state_table import continuous_state_table


def initial_state_indices(prod, rout, events_summary, onset="search",
                          pre_event=24):
    """Positions of the antecedent state ahead of each delineated event.

    Two ways of placing the onset, per ``onset``:

    ``"search"``
        Look back ``pre_event`` timesteps from the event's peak for the point
        where both stores stop declining.  Only the *sign* of consecutive
        differences is used, so ``prod`` and ``rout`` may be storages in mm
        or fractions of ``x1``/``x3`` -- the result is the same either way.
    ``"start"``
        Take the event's delineated start, unmodified.  The stores are not
        consulted at all, so this is the faster and more literal reading: the
        state where the hydrograph began to rise.

    Parameters
    ----------
    prod, rout : array_like
        Production and routing storage over the run, one value per timestep.
    events_summary : DataFrame
        Output of event delineation (:func:`event_POT`/:func:`event_maxima`).
        Needs a ``max_index`` column for ``onset="search"``, a ``start``
        column for ``onset="start"``.
    onset : {"search", "start"}, optional
        How to place the onset.  Default ``"search"``.
    pre_event : int, optional
        Number of pre-event timesteps to inspect.  Default 24.  Ignored when
        ``onset="start"``.

    Returns
    -------
    ndarray of int
        One position per event, in the order the events are listed.

    Raises
    ------
    ValueError
        If ``onset`` is not one of the two, if the column it needs is
        missing, or if ``onset="start"`` and an event starts outside the
        timesteps covered by ``prod``/``rout``.
    """
    prod = np.asarray(prod, dtype=float)
    rout = np.asarray(rout, dtype=float)
    if prod.size != rout.size:
        raise ValueError("prod and rout must be the same length")
    if onset not in ("search", "start"):
        raise ValueError(f"onset must be 'search' or 'start', got {onset!r}")

    col = "max_index" if onset == "search" else "start"
    if col not in events_summary:
        raise ValueError(f"onset={onset!r} needs a {col!r} column")

    if onset == "start":
        # taken as given, so an index off the end is the caller's error, not
        # something to clamp away: numpy would wrap a negative one silently
        out = np.asarray(events_summary[col], int)
        if out.size and (out.min() < 0 or out.max() >= prod.size):
            raise ValueError(
                f"event starts run from {out.min()} to {out.max()}, outside "
                f"the {prod.size} timesteps of stores given")
        return out

    out = np.empty(len(events_summary), dtype=int)
    for i, max_idx in enumerate(np.asarray(events_summary[col], int)):
        start_win = max(0, max_idx - pre_event + 1)
        end_win = max_idx + 1
        prod_match = _safe_tail_match(np.diff(prod[start_win:end_win]) < 0)
        rout_match = _safe_tail_match(np.diff(rout[start_win:end_win]) < 0)
        # offset from the *clamped* window start: for an event peaking within
        # pre_event of the record start, the unclamped base is negative and
        # lands the onset before the window it was searched in
        idx = start_win + min(prod_match, rout_match)
        out[i] = max(0, min(idx, prod.size - 1))
    return out


def extract_initial_states(states, events_summary, onset="search",
                           pre_event=24):
    """Extract antecedent production/routing states ahead of each event.

    A lookup wrapper over :func:`initial_state_indices`: that function finds
    the onset of each event, this one reads the two stores off it.

    Parameters
    ----------
    states : ndarray, shape (2, n_timesteps)
        Production storage (row 0) and routing storage (row 1).
    events_summary : DataFrame
        Output of event delineation (:func:`event_POT`/:func:`event_maxima`).
    onset : {"search", "start"}, optional
        How to place the onset; see :func:`initial_state_indices`.  Default
        ``"search"``.
    pre_event : int, optional
        Number of pre-event hours to inspect. Default 24.  Ignored when
        ``onset="start"``.

    Returns
    -------
    DataFrame with columns ``event_id``, ``state_index``, ``Prod``,
    ``Rout`` -- one row per event.

    Notes
    -----
    ``Prod`` and ``Rout`` come back in **whatever units went in**.
    ``GR4H.run`` gives fractions of ``x1``/``x3`` (columns ``ps``, ``rs``),
    which is what :func:`~pyfloodrisk.design_flood.simulate_design_flood`
    wants for ``ps0``/``rs0``; ``GR4H.run_from_state`` gives mm.  Pass the
    one the consumer expects.
    """
    if len(events_summary) == 0:
        return pd.DataFrame(columns=["event_id", "state_index", "Prod", "Rout"])

    idx = initial_state_indices(states[0], states[1], events_summary,
                                onset=onset, pre_event=pre_event)
    return pd.DataFrame({
        "event_id": np.arange(1, idx.size + 1),
        "state_index": idx,
        "Prod": states[0, idx],
        "Rout": states[1, idx],
    })


def _safe_tail_match(boolean_array):
    """Return the index of the last True value in boolean_array, or 0."""
    matches = np.where(boolean_array)[0]
    if len(matches) > 0:
        return matches[-1]
    return 0


def extract_initial_states_per_duration(
        forcings, states, ey=6,
        durs=(6, 9, 12, 18, 24, 36, 48, 72, 96, 120, 144, 168)):
    """Antecedent states immediately before the largest burst of each duration.

    One distribution of initial states *per storm duration*, rather than one
    pooled over the whole record: for each duration the largest
    ``ey * nyears`` rainfall bursts are found (:func:`rainfall_events`) and
    the state one timestep before each is kept.  An event of a given length
    is then started from the wetness the catchment's own storms of that
    length actually found it in.

    Parameters
    ----------
    forcings : DataFrame or path
        Hourly forcing record, datetime-indexed with a ``prec`` column in mm,
        or a CSV of one (index in the first column, dates read day-first).
    states : DataFrame
        State table from a continuous run over that record, one row per
        timestep, with a ``date`` column of timestamps.  Rows are matched to
        burst onsets **by timestamp**, so the table must be unthinned; every
        other column is carried through untouched, which is how a full GR4H
        state vector -- stores *and* unit-hydrograph memory -- survives into
        the pools.
    ey : int, optional
        Bursts to keep per year of record.  Default 6.
    durs : sequence of int, optional
        Burst durations in hours.

    Returns
    -------
    dict
        ``duration_h`` (float) ``-> DataFrame``, each the rows of ``states``
        preceding that duration's bursts, chronological, index reset.

    Raises
    ------
    ValueError
        If ``forcings`` spans less than a year, if ``states`` has no ``date``
        column, or if any burst onset is missing from ``states`` -- which
        means the table is thinned, or does not cover the record.

    Notes
    -----
    Bursts are selected over the span of ``forcings``, so trim it to the span
    the state table covers before calling.  A continuous run usually discards
    a warm-up year, and counting those years towards ``ey * nyears`` thins
    every duration's pool.

    Each pool holds only ``ey * nyears`` states -- 66 for six bursts a year
    over eleven years -- while the Monte Carlo draws far more events than
    that from it; see
    :func:`~pyfloodrisk.dffa.workflow.state_samplers_by_duration`, which
    wraps this function for the derived-FFA path.
    """
    if not isinstance(forcings, pd.DataFrame):
        forcings = pd.read_csv(forcings, index_col=0, parse_dates=True,
                               dayfirst=True)
    if "date" not in states:
        raise ValueError("states needs its 'date' column to match burst onsets")

    span_days = (forcings.index[-1] - forcings.index[0]).days
    nyears = int(span_days / 365)
    if nyears < 1:
        raise ValueError("need at least a year of record to select bursts "
                         f"over; got {span_days} days")

    dates = pd.DatetimeIndex(states["date"])
    out: dict[float, pd.DataFrame] = {}
    for dur in durs:
        pos = rainfall_events(forcings, int(dur), nyears, ey=ey)
        onset = forcings.index[np.maximum(pos - 1, 0)]   # state *before* it
        keep = dates.isin(onset)
        if int(keep.sum()) < len(onset):
            raise ValueError(
                f"only {int(keep.sum())} of {len(onset)} burst onsets at "
                f"{dur} h are present in the state table; it must be "
                "unthinned and cover the span of 'forcings'")
        out[float(dur)] = states.loc[keep].reset_index(drop=True)
    return out


def extract_initial_states_ensemble(
        forcings, parameters, area_km2, ey=6,
        durs=(6, 9, 12, 18, 24, 36, 48, 72, 96, 120, 144, 168),
        warmup_hours=8760, progress=True):
    """Antecedent states pooled across an ensemble of forcing realisations.

    The ensemble form of :func:`extract_initial_states_per_duration`.  Each
    realisation is run through GR4H in turn, its bursts are found in its own
    rainfall, and the states before them are kept; the pools are then
    concatenated, so each duration's donors come from every realisation.
    Nothing about the conditioning changes -- only how many donors it yields.

    Realisations are processed one at a time and only the donor rows are
    retained.  A 30-year hourly state table is around 116 MB, so a hundred of
    them at once is not something to hold.

    Parameters
    ----------
    forcings : sequence of DataFrame or path
        The realisations, each hourly with ``prec`` and ``pet`` and its own
        datetime index.  They may share that index: bursts are found per
        realisation, so identical timestamps never collide between them.
    parameters : Mapping
        GR4H parameters.  ``x4`` fixes the UH memory length, so every pool is
        only valid for this set.
    area_km2 : float
        Catchment area; affects only the reported discharge.
    ey : int, optional
        Bursts kept per year of each realisation.  Default 6.
    durs : sequence of int, optional
        Burst durations in hours.
    warmup_hours : int, optional
        Leading hours discarded from each realisation before its states are
        drawn.  One year by default.
    progress : bool, optional
        Print a counter as realisations are processed.

    Returns
    -------
    dict
        ``duration_h -> DataFrame``, each with a ``realisation`` column naming
        the source it came from.

    Notes
    -----
    A disaggregated ensemble shares its parent daily record, so the donors it
    adds are not all independent.  On the 303203 pyraingen set the hourly
    maxima differ by a factor of 2.5 between realisations while the 72-hour
    maxima agree to 0.1%: the short-duration pools gain real diversity, the
    long-duration ones mostly repeat the same events.  Count distinct burst
    dates per duration before reading the row count as a sample size.
    """
    forcings = list(forcings)
    pools: dict[float, list] = {float(d): [] for d in durs}
    for i, item in enumerate(forcings, 1):
        if isinstance(item, pd.DataFrame):
            fc, name = item, item.attrs.get("name") or str(i)
        else:
            fc = pd.read_csv(item, index_col=0, parse_dates=True)
            name = Path(str(item)).stem
        table = continuous_state_table(fc, parameters, area_km2,
                                       warmup_hours=warmup_hours)
        got = extract_initial_states_per_duration(
            fc.iloc[int(warmup_hours):], table, ey=ey, durs=tuple(durs))
        del table                       # before the next one is built
        for d, pool in got.items():
            pool.insert(0, "realisation", name)
            pools[d].append(pool)
        if progress:
            print(f"\r  realisation {i}/{len(forcings)}", end="", flush=True)
    if progress:
        print()
    return {d: pd.concat(v, ignore_index=True) for d, v in pools.items()}
