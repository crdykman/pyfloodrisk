"""Baseflow separation and hydrologic event delineation.

Python port of the R ``hydroEvents`` package: baseflow separation via the
Lyne-Hollick recursive digital filter, event delineation by either a
peaks-over-threshold (POT) or local-maxima method, and extraction of
antecedent model states at each delineated event for design flood
simulation.
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


def hydro_event_pipeline(
    q_array, event_method="maxima", method_kwargs=None, alpha=0.925, passes=3, r=30
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

    Returns
    -------
    df_processed : DataFrame
        Columns ``baseflow`` and ``quickflow`` (same length as q_array).
    events_summary : DataFrame
        Delineated events; see :func:`event_POT`/:func:`event_maxima`.
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

    return df_processed, events_summary


def extract_initial_states(states, events_summary, pre_event=24):
    """Extract antecedent production/routing states ahead of each event.

    For each delineated event, searches the ``pre_event`` hours leading up
    to its peak for the point where both states stop declining, and takes
    that point's storage values as the event's initial conditions.

    Parameters
    ----------
    states : ndarray, shape (2, n_timesteps)
        Production storage fraction (row 0) and routing storage fraction
        (row 1), as returned by ``GR4H.run`` (columns ``ps``, ``rs``).
    events_summary : DataFrame
        Output of event delineation (:func:`event_POT`/:func:`event_maxima`),
        must contain a ``max_index`` column.
    pre_event : int, optional
        Number of pre-event hours to inspect. Default 24.

    Returns
    -------
    DataFrame with columns ``event_id``, ``state_index``, ``Prod``,
    ``Rout`` -- one row per event.
    """
    qobs_max = events_summary["max_index"]
    nevents = len(events_summary)

    if nevents == 0:
        return pd.DataFrame(columns=["event_id", "state_index", "Prod", "Rout"])

    state_idx_list = []

    # Analyze pre-event windows using 0-based sliding slices
    for i in range(nevents):
        max_idx = qobs_max[i]

        # Define window boundaries
        start_win = max(0, max_idx - pre_event + 1)
        end_win = max_idx + 1

        prod_slice = states[0, start_win:end_win]
        rout_slice = states[1, start_win:end_win]

        # Compute differences
        diff_prod = np.diff(prod_slice) < 0
        diff_rout = np.diff(rout_slice) < 0

        # Apply tail matching logic
        prod_match_idx = _safe_tail_match(diff_prod)
        rout_match_idx = _safe_tail_match(diff_rout)

        # Compute the concrete state index step
        # (Using 0-based logic, min index change aligns with the step forward
        # from window start)
        calculated_idx = (max_idx - pre_event + 1) + min(prod_match_idx, rout_match_idx)

        # Constrain boundary limits safely
        calculated_idx = max(0, min(calculated_idx, states.shape[1] - 1))
        state_idx_list.append(int(calculated_idx))

    # Generate final output DataFrame
    state_idx_arr = np.array(state_idx_list)
    output_df = pd.DataFrame({
        "event_id": np.arange(1, nevents + 1),
        "state_index": state_idx_arr,
        "Prod": states[0, state_idx_arr],
        "Rout": states[1, state_idx_arr],
    })
    return output_df


def _safe_tail_match(boolean_array):
    """Return the index of the last True value in boolean_array, or 0."""
    matches = np.where(boolean_array)[0]
    if len(matches) > 0:
        return matches[-1]
    return 0