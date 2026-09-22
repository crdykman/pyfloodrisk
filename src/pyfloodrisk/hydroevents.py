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

import numpy as np
import pandas as pd
from numba import njit


@njit(cache=True)
def _filter_pass(qf, bf, start, stop, step, alpha):
    """One forward/backward pass of the recurrence, filling qf in place.

    Iterates i = start, start+step, ... up to (not including) stop.
    qf[i] = alpha*qf[i-step] + ((1+alpha)/2)*(bf[i] - bf[i-step])
    """
    c = (1.0 + alpha) / 2.0
    for i in range(start, stop, step):
        qf[i] = alpha * qf[i - step] + c * (bf[i] - bf[i - step])


def baseflow_b(q, alpha=0.98, passes=3, r=30):
    """Lyne-Hollick recursive digital baseflow filter (port of R's baseflowB).

    Parameters
    ----------
    q : array_like
        Streamflow time series (1-D). Must have len(q) > r.
    alpha : float
        Filter parameter (recession constant), default 0.98.
    passes : int
        Number of filter passes (forward/backward/forward/...), default 3.
        Note: the algorithm assumes passes >= 2.
    r : int
        Number of points used for reflective padding at each end, default 30.

    Returns
    -------
    dict with keys:
        'bf'  : np.ndarray, separated baseflow (same length as q)
        'bfi' : np.ndarray, baseflow index (bf / q)
    """
    q = np.asarray(q, dtype=float)
    n_i = len(q)

    # --- Reflective padding -------------------------------------------------
    # R: q.c = c(q[(r+1):2], q, q[(n.i-1):(n.i-r)])
    # Left  reflection: 1-based indices r+1..2 (descending) -> 0-based r..1
    # Right reflection: 1-based indices n_i-1..n_i-r        -> 0-based n_i-2..n_i-r-1
    left_idx = np.arange(r, 0, -1)                  # r, r-1, ..., 1
    right_idx = np.arange(n_i - 2, n_i - r - 2, -1)  # n_i-2, ..., n_i-r-1
    q_c = np.concatenate([q[left_idx], q, q[right_idx]])
    n = len(q_c)

    # --- Per-pass loop parameters -------------------------------------------
    # R: srt = rep(c(1,n), ...); end = rep(c(n,1), ...); add = rep(c(1,-1), ...)
    srt = np.tile([0, n - 1], passes)[:passes]   # start index of each pass
    end = np.tile([n - 1, 0], passes)[:passes]   # end index of each pass
    add = np.tile([1, -1], passes)[:passes]      # direction of each pass

    # --- Initialise (qf is allocated ONCE and carries state across passes) --
    qf = np.zeros(n)
    qf[0] = q_c[0]            # R: qf[1] = q.c[1]
    bf = q_c.copy()

    # --- Run filter -----------------------------------------------------------
    for j in range(passes):
        s, e, a = int(srt[j]), int(end[j]), int(add[j])
        # Recurrence (numba-accelerated): i from s+a to e inclusive, step a.
        _filter_pass(qf, bf, s + a, e + a, a, alpha)
        mask = qf > 0
        bf[mask] = bf[mask] - qf[mask]
        qf[e] = bf[e]        # seed the boundary for the next pass

    # --- Strip padding and return ------------------------------------------
    bf = bf[r:-r]            # R: head(tail(bf, -r), -r)
    bfi = bf / q
    return {"bf": bf, "bfi": bfi}


def event_POT(data, threshold=0, min_diff=1):
    """Delineate events by peaks-over-threshold (POT).

    Python equivalent of ``hydroEvents::eventPOT(out.style = 'summary')``.
    Uses 0-based Python indexing but returns all R summary metrics.

    Parameters
    ----------
    data : array_like
        Time series to delineate (typically a quickflow series).
    threshold : float, optional
        Values above this threshold are considered part of an event.
    min_diff : int, optional
        Events separated by fewer than this many time steps are merged.

    Returns
    -------
    DataFrame with columns ``start``, ``end``, ``max_index``, ``max``,
    ``sum``, one row per delineated event.
    """
    df = pd.DataFrame({"val": np.array(data)})

    df["above"] = df["val"] > threshold
    df["block"] = (df["above"] != df["above"].shift()).cumsum()

    events_df = df[df["above"]].copy()
    if events_df.empty:
        return pd.DataFrame(columns=["start", "end", "max_index", "max", "sum"])

    summary = events_df.groupby("block").apply(
        lambda x: pd.Series({"start": x.index.min(), "end": x.index.max()})
    ).reset_index(drop=True)

    if min_diff > 1 and len(summary) > 1:
        merged = []
        curr_start, curr_end = summary.loc[0, "start"], summary.loc[0, "end"]
        for i in range(1, len(summary)):
            next_start = summary.loc[i, "start"]
            next_end = summary.loc[i, "end"]

            if (next_start - curr_end) <= min_diff:
                curr_end = next_end
            else:
                merged.append({"start": curr_start, "end": curr_end})
                curr_start, curr_end = next_start, next_end
        merged.append({"start": curr_start, "end": curr_end})
        summary = pd.DataFrame(merged)

    output_metrics = []
    for _, row in summary.iterrows():
        start_idx = int(row["start"])
        end_idx = int(row["end"])
        event_slice = df["val"].iloc[start_idx:end_idx + 1]

        output_metrics.append({
            "start": start_idx,
            "end": end_idx,
            "max_index": event_slice.idxmax(),
            "max": event_slice.max(),
            "sum": event_slice.sum(),
        })

    return pd.DataFrame(output_metrics)


def event_maxima(data, delta_y=200, delta_x=1, threshold=-1):
    """Delineate events by local maxima with valley-drop merging.

    Python version of ``hydroEvents::eventMaxima`` bypassing SciPy. Uses
    native directional searches to mirror the R plateau logic precisely.

    Parameters
    ----------
    data : array_like
        Time series to delineate (typically a quickflow series).
    delta_y : float, optional
        Minimum drop (if positive) or fraction of peak height (if
        negative) required in the trough between two peaks for them to
        be treated as separate events; otherwise they are merged.
    delta_x : int, optional
        Half-width (in time steps) of the local window used to test
        whether a point is a peak.
    threshold : float, optional
        Minimum peak value for an event to be retained in the output.

    Returns
    -------
    DataFrame with columns ``start``, ``end``, ``max_index``, ``max``,
    ``sum``, one row per delineated event.
    """
    y = np.array(data, dtype=float)
    n = len(y)

    if n == 0:
        return pd.DataFrame(columns=["start", "end", "max_index", "max", "sum"])

    # 1. Point-by-point peak detection
    peaks = []
    for i in range(n):
        left_bound = max(0, i - delta_x)
        right_bound = min(n, i + delta_x + 1)
        is_peak = True

        for j in range(left_bound, i):
            if y[i] < y[j]:
                is_peak = False
                break
        if is_peak:
            for j in range(i + 1, right_bound):
                if y[i] <= y[j]:
                    is_peak = False
                    break
        if is_peak:
            if 0 < i < n - 1:
                peaks.append(i)
            elif i == 0 and n > 1 and y[0] > y[1]:
                peaks.append(i)
            elif i == n - 1 and n > 1 and y[n - 1] > y[n - 2]:
                peaks.append(i)

    if not peaks:
        return pd.DataFrame(columns=["start", "end", "max_index", "max", "sum"])

    # 2. Delineate boundaries surrounding peaks
    raw_events = []
    for peak in peaks:
        start = peak
        while start > 0 and y[start - 1] <= y[start]:
            start -= 1
        end = peak
        while end < n - 1 and y[end + 1] <= y[end]:
            end += 1

        peak_val = y[peak]
        drop_limit = peak_val * abs(delta_y) if delta_y < 0 else delta_y

        raw_events.append({
            "start": start,
            "end": end,
            "max_index": peak,
            "max": peak_val,
            "drop_limit": drop_limit,
        })

    # 3. Inter-peak valley merging logic
    merged_events = []
    if raw_events:
        curr = raw_events[0]
        for next_ev in raw_events[1:]:
            if next_ev["start"] <= curr["end"]:
                trough_val = y[int(curr["end"])]
                drop_curr = curr["max"] - trough_val
                drop_next = next_ev["max"] - trough_val
                if drop_curr <= curr["drop_limit"] or drop_next <= next_ev["drop_limit"]:
                    curr["end"] = max(curr["end"], next_ev["end"])
                    if next_ev["max"] > curr["max"]:
                        curr["max"] = next_ev["max"]
                        curr["max_index"] = next_ev["max_index"]
                else:
                    merged_events.append(curr)
                    curr = next_ev
            else:
                merged_events.append(curr)
                curr = next_ev
        merged_events.append(curr)

    # 4. Generate final summary dataframe
    final_output = []
    for ev in merged_events:
        start_idx = int(ev["start"])
        end_idx = int(ev["end"])
        if ev["max"] >= threshold:
            final_output.append({
                "start": start_idx,
                "end": end_idx,
                "max_index": int(ev["max_index"]),
                "max": ev["max"],
                "sum": np.sum(y[start_idx:end_idx + 1]),
            })

    return pd.DataFrame(final_output)


#: Mean hours in a year, for converting a record length to years.
_HOURS_PER_YEAR = 365.25 * 24.0

#: Event metric each ``rank_by`` option ranks on.
_RANK_COLUMNS = {"peak": "max", "volume": "sum"}


def threshold_events_by_ey(
    events_summary, n_timesteps, ey=6, rank_by="peak", dt_hours=1.0
):
    """Keep the largest events, ``ey`` of them per year of record on average.

    Delineation returns every rise it can find, most of which are far too
    small to be floods.  This trims the list to an exceedance-per-year (EY)
    rate: ``ey=6`` keeps the six largest events per year of record, which is
    the usual starting point for a partial duration series.

    Parameters
    ----------
    events_summary : DataFrame
        Delineated events, as returned by :func:`event_POT` or
        :func:`event_maxima`.
    n_timesteps : int
        Length of the series the events were delineated from; with
        ``dt_hours`` this gives the record length in years.
    ey : int or None, optional
        Events to keep per year of record. ``None`` keeps every event.
        Default 6.
    rank_by : {"peak", "volume"}, optional
        Rank events on peak flow (the ``max`` column) or event volume (the
        ``sum`` column). Because the series is evenly spaced, ``sum`` is
        proportional to volume, so no unit conversion is needed to rank on
        it. Default ``"peak"``.
    dt_hours : float, optional
        Timestep of the series, in hours. Default 1.0.

    Returns
    -------
    DataFrame
        The retained events in chronological order, with a reset index.
        Ranking is by the chosen metric, ties broken by earlier ``start``.
        Returns fewer rows than asked for if delineation found fewer.
    """
    if rank_by not in _RANK_COLUMNS:
        raise ValueError(
            f"rank_by must be one of {sorted(_RANK_COLUMNS)}, got {rank_by!r}"
        )
    if ey is None:
        return events_summary.reset_index(drop=True)
    if ey <= 0:
        raise ValueError("ey must be positive, or None to keep every event")
    if events_summary.empty:
        return events_summary.reset_index(drop=True)

    n_years = float(n_timesteps) * float(dt_hours) / _HOURS_PER_YEAR
    n_keep = max(1, int(round(ey * n_years)))
    if n_keep >= len(events_summary):
        return events_summary.reset_index(drop=True)

    column = _RANK_COLUMNS[rank_by]
    kept = events_summary.sort_values(
        [column, "start"], ascending=[False, True]
    ).head(n_keep)
    return kept.sort_values("start").reset_index(drop=True)


def hydro_event_pipeline(
    q_array, event_method="maxima", method_kwargs=None, alpha=0.925, passes=3, r=30,
    ey=None, rank_by="peak", dt_hours=1.0, idx=False,
):
    """Run baseflow separation followed by event delineation on quickflow.

    Parameters
    ----------
    q_array : array_like
        Streamflow time series.
    event_method : str, optional
        Delineation method: ``"POT"`` or ``"maxima"`` (default).
    method_kwargs : dict, optional
        Keyword arguments passed to the chosen delineation function
        (:func:`event_POT` or :func:`event_maxima`).
    alpha, passes, r :
        Passed through to :func:`baseflow_b`.
    ey : float or None, optional
        Keep only the largest events, ``ey`` of them per year of record on
        average -- ``ey=6`` keeps the six largest per year, the usual
        starting point for a partial duration series.  **Default ``None``,
        which keeps every delineated event**, most of which are too small to
        be floods.  See :func:`threshold_events_by_ey`.
    rank_by : {"peak", "volume"}, optional
        Whether "largest" means largest peak flow or largest event volume.
        Only consulted when ``ey`` is given.  Default ``"peak"``.
    dt_hours : float, optional
        Timestep of ``q_array`` in hours, used to convert its length to
        years.  Only consulted when ``ey`` is given.  Default 1.0 (hourly).
    idx : bool, optional
        Also return the timestep indices spanned by the retained events, as
        one flat array -- the form ``calibration(eventsidx=...)`` wants.
        Only available when ``ey`` is given, since it is built from the
        retained events.

    Returns
    -------
    df_processed : DataFrame
        Columns ``baseflow`` and ``quickflow`` (same length as q_array).
    events_summary : DataFrame
        The retained events in chronological order, with a reset index;
        see :func:`event_POT`/:func:`event_maxima` for the columns.
    eventsidx : ndarray
        Only when ``idx`` and ``ey`` are both given: the concatenated
        ``start..end`` indices of every retained event.
    """
    if method_kwargs is None:
        method_kwargs = {}

    # Step 1: Baseflow separation
    bf_results = baseflow_b(q_array, alpha=alpha, passes=passes, r=r)

    df_processed = pd.DataFrame()
    df_processed["baseflow"] = bf_results["bf"]
    df_processed["quickflow"] = q_array - bf_results["bf"]

    # Step 2: Delineate events using chosen method on quickflow channel
    if event_method.lower() == "pot":
        threshold = method_kwargs.get("threshold", 0.0)
        min_diff = method_kwargs.get("min_diff", 1)
        events_summary = event_POT(
            df_processed["quickflow"], threshold=threshold, min_diff=min_diff
        )
    elif event_method.lower() == "maxima":
        delta_y = method_kwargs.get("delta_y", -0.75)
        delta_x = method_kwargs.get("delta_x", 24)
        threshold = method_kwargs.get("threshold", 0)
        events_summary = event_maxima(
            df_processed["quickflow"],
            delta_y=delta_y,
            delta_x=delta_x,
            threshold=threshold,
        )
    else:
        raise ValueError("Invalid event_method selection. Use 'POT' or 'maxima'.")

    if ey is not None:
        # Step 3: Keep only the largest events, on average `ey` per year.  The
        # index is reset here because extract_initial_states looks its rows up
        # by label.
        events_summary = threshold_events_by_ey(
            events_summary, len(df_processed), ey=ey, rank_by=rank_by,
            dt_hours=dt_hours,
        )

        if idx:
            eventsidx = np.array([]).astype(int)
            for i in range(len(events_summary)):
                eventsidx = np.append(eventsidx,
                    np.arange(
                    events_summary["start"].iloc[i],
                    events_summary["end"].iloc[i]+1
                    )
                )

            return df_processed, events_summary, eventsidx

    return df_processed, events_summary


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


def events_WRT_rainfall(rdata, events_summary, ey=6):
    """Re-define event boundaries with respect to the rainfall that caused them.

    :func:`hydro_event_pipeline` delineates events from the flow series
    alone, so an event starts where the hydrograph rises.  For calibrating
    against events, what matters is the rainfall that produced the rise: the
    event should start when the rain started, not when the catchment
    responded.  This selects the largest 3-day rainfall totals, matches each
    to the runoff event it produced, and walks the start back to the onset of
    the rainfall.

    The procedure, in order:

    1. Total the record into 3-day rainfall blocks and keep the
       ``ey * nyears`` largest, as the rainfall events of interest.
    2. For each, take the first delineated runoff event beginning strictly
       after the rainfall block starts.
       A rainfall block with no runoff event to its right -- one falling after
       the last delineated event -- is dropped, so fewer than ``ey * nyears``
       events may come back.
    3. If that runoff event begins more than 3 days after the rainfall, treat
       the pairing as failed and fall back to the rainfall block's own bounds
       (start, start + 3 days).
    4. For the successfully paired events, walk the start back day by day
       while the preceding day had more than 1 mm of rain, up to 10 days, so
       the event begins on the first wet day rather than at the rise.

    Parameters
    ----------
    rdata : str or path
        CSV of the *hourly* forcing record: a datetime index in the first
        column and a ``prec`` column in mm.  Dates are read day-first.  This
        must be the same record ``events_summary`` was delineated from, since
        that table indexes into it by position.
    events_summary : DataFrame
        Delineated events, as returned by :func:`hydro_event_pipeline`, whose
        ``start`` and ``end`` are integer positions into ``rdata``.
    ey : int, optional
        Rainfall events to keep per year, as for
        :func:`threshold_events_by_ey`.  Default 6.

    Returns
    -------
    events_summary : DataFrame
        The matched runoff events, one per retained rainfall event, with
        ``start`` and ``end`` replaced by the rainfall-referenced bounds.  The
        positional index of the runoff event each row was matched to is kept
        as an ``index`` column.
    eventsidx : ndarray
        The timestep positions those events span, sorted and de-duplicated --
        the form ``calibration(eventsidx=...)`` wants.

    Notes
    -----
    * Walking starts back can make consecutive events overlap, so
      ``eventsidx`` is de-duplicated and is shorter than
      ``sum(end - start + 1)``: by 6-25% on the three bundled records.
    * ``nyears`` is the record length in whole years, floored, so an 11.01
      year record keeps ``6 * 11`` events.
    * Dates in ``rdata`` are read day-first, matching the bundled records.  An
      ISO-formatted record is left as strings by ``dayfirst=True`` rather than
      raising, and the resampling below then fails on a non-datetime index.
    """
    df = pd.read_csv(rdata, index_col=0, parse_dates=True, dayfirst=True)

    # Identify 3 day rainfall maxima
    nyears = int((df.index[-1] - df.index[0]).days / 365)
    prec3d = df['prec'].resample('3D', label='left').sum()
    idxmaxs = prec3d.nlargest(ey*nyears).index.sort_values()
    
    # Identify runoff events starting after beginning of rainfall events
    start_times = df.iloc[events_summary.start].index #flow
    end_times = df.iloc[events_summary.end].index # flow
    targets = idxmaxs # rainfall
    idx = np.searchsorted(start_times, targets, side='right')   # first index strictly after

    # a rainfall maximum past the last delineated runoff event has nothing to
    # pair with; drop it
    keep = idx < len(start_times)                  
    targets, idx = targets[keep], idx[keep]

    # Ensure runoff begins within 3 days of start of rainfall event
    # if not set t_rise and t_ends to 3 day rainfall event bounds
    t_rises = start_times[idx].normalize()
    t_ends = end_times[idx]
    bool_mask = t_rises > targets + pd.Timedelta(days=3)
    t_rises = np.where(bool_mask, targets, t_rises)
    t_ends = np.where(bool_mask, targets + pd.Timedelta(days=3), t_ends)

    # the start of the event was defined as the first day of rainfall > 1 mm 
    # occurring prior to the rise in streamflow that occurred either during or 
    # before the three -day rainfall event
    p_thresh = 1  # mm
    max_lookback = pd.Timedelta(days=10)
    prec1d = df['prec'].resample('D').sum()  # convert hourly to daily rainfall
    t_starts = []
    for i, t_rise in enumerate(t_rises):
        t = t_rise
        if not bool_mask[i]:
            while t > t_rise - max_lookback and prec1d.loc[(t - pd.Timedelta(days=1)).strftime('%Y-%m-%d')] > p_thresh:
                t = t - pd.Timedelta(days=1)
        t_starts.append(t)
    t_starts = pd.to_datetime(t_starts)

    eventsidx = np.array([]).astype(int)
    t_startsidx = []
    t_endsidx = []
    for i in range(len(t_starts)):
        t_startid = df.index.get_loc(t_starts[i])
        t_endid = df.index.get_loc(t_ends[i])
        t_startsidx.append(t_startid)
        t_endsidx.append(t_endid)
        eventsidx = np.append(
            eventsidx, np.arange(t_startid, t_endid+1)
        )

    events_summary = events_summary.iloc[idx]
    events_summary['start'] = t_startsidx
    events_summary['end'] = t_endsidx
    events_summary.reset_index(inplace=True)

    return events_summary, np.unique(eventsidx)


def rainfall_events(df, dur, nyears, ey=6):
    """Positions of the largest ``dur``-hour rainfall bursts in the record.

    The rainfall analogue of :func:`threshold_events_by_ey`: the record is
    totalled into non-overlapping blocks of ``dur`` hours and the largest
    ``ey * nyears`` of them are kept, so ``ey=6`` gives six bursts per year
    on average.

    Parameters
    ----------
    df : DataFrame
        Hourly forcing record, datetime-indexed, with a ``prec`` column in mm.
    dur : int
        Burst duration in hours.
    nyears : int
        Record length in years, used only to scale ``ey``.
    ey : int, optional
        Bursts to keep per year.  Default 6.

    Returns
    -------
    ndarray
        Integer positions into ``df`` of each retained burst's **first**
        timestep, in chronological order.

    Notes
    -----
    Blocks are aligned to the resampling origin rather than to the wettest
    window, so a burst straddling a block boundary is split between two
    blocks and may be missed.  For a record shorter than ``ey * nyears``
    blocks, fewer positions come back than asked for.
    """
    prec = df['prec'].resample(f'{dur}h', label='left').sum()
    locmax = prec.nlargest(ey*nyears).index.sort_values()
    idxmax = df.index.get_indexer(locmax)
    return idxmax


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
