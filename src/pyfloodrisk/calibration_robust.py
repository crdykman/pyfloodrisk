"""DREAM-based calibration and robust (multi-event) re-calibration of GR4H."""

import numpy as np
import pandas as pd
import seaborn as sns
import spotpy

from .gr4h.GR4H_model import GR4H
from .gr4h.GR4H_calibrate import spot_setup


def calibration(area, data, nsamples, eventsidx=None, save_output=True):
    """Calibrate GR4H against observed streamflow using DREAM.

    Parameters
    ----------
    area : float
        Catchment area in km2.
    data : str or Path
        Path to a climate CSV with ``prec``, ``pet``, and ``qt`` columns.
    nsamples : int
        Maximum number of DREAM function evaluations (repetitions).
    eventsidx : array_like, optional
        Indices to restrict the objective function to specific events.
        If None, all data after the warm-up period is used.
    save_output : bool, optional
        If True (default), save the DREAM chain history to
        ``dream_calibration_results.npy``.

    Returns
    -------
    Structured array of DREAM chain results (parameters, likelihood,
    and chain id per sample), as returned by ``spotpy``'s
    ``sampler.getdata()``.
    """
    spotsetup = spot_setup(
        data,
        area,
        obj_func=spotpy.objectivefunctions.nashsutcliffe,
        ps0=1.0,
        rs0=0.5,
        warmup=365,
        events=eventsidx,
    )

    sampler = spotpy.algorithms.dream(
        spotsetup,
        dbname="DREAM_GR4H",
        dbformat="ram",
        save_threshold=0.0,
        save_sim=False,
    )

    # DREAM sampling settings
    rep = nsamples  # maximum number of function evaluations allowed
    nChains = max(nsamples // 1000, 10)
    nCr = 3
    delta = 3
    c = 0.1
    eps = 10e-6
    convergence_limit = 1.01
    runs_after_convergence = 100
    acceptance_test_option = 1

    sampler.sample(
        rep,
        nChains,
        nCr,
        delta,
        c,
        eps,
        convergence_limit,
        runs_after_convergence,
        acceptance_test_option,
    )
    results = sampler.getdata()
    if save_output:
        np.save("dream_calibration_results.npy", results)

    return results


def behavioural_posterior(results, Cb=0.6, n=100):
    """Select the behavioural posterior from a DREAM calibration run.

    Parameters
    ----------
    results : structured ndarray
        Output of :func:`calibration`.
    Cb : float, optional
        Minimum likelihood (``like1``) threshold for a sample to be
        considered behavioural. Default 0.6.
    n : int, optional
        Number of top-likelihood samples to consider before applying the
        ``Cb`` threshold. Default 100.

    Returns
    -------
    Structured ndarray subset of ``results``, sorted by descending
    likelihood and filtered to ``like1 >= Cb``.
    """
    topn = np.sort(results, order="like1")[-n:][::-1]
    bpost = topn[topn["like1"] >= Cb]
    return bpost


def robust_calibration(
    posterior,
    area,
    events,
    ps=[0.5, 0.6, 0.7, 0.8, 0.9],
    rs=[0.5, 0.6, 0.7, 0.8, 0.9],
    topn=20,
    save_output=False,
):
    """Re-rank a behavioural posterior by robustness against design events.

    For every posterior parameter set and every combination of initial
    production/routing state (``ps``/``rs``), simulates each design event
    and scores parameter sets by how consistently they reproduce a
    plausible peak-flow distribution (estimated via KDE) across events of
    the same frequency.

    Parameters
    ----------
    posterior : structured ndarray
        Behavioural posterior from :func:`behavioural_posterior`.
    area : float
        Catchment area in km2.
    events : dict
        Nested mapping ``{duration: {frequency: [forcings, ...]}}``, where
        each ``forcings`` is a DataFrame accepted by ``GR4H.run``.
    ps, rs : list of float, optional
        Candidate initial production/routing storage fractions to test.
    topn : int, optional
        Number of most robust parameter sets to keep. Default 20.
    save_output : bool, optional
        If True, save the retained parameter sets to
        ``robust_models_{duration}hr.npy``.

    Returns
    -------
    Structured ndarray subset of ``posterior`` containing the ``topn``
    most robust parameter sets for the (first) duration in ``events``.
    """
    durs = events.keys()
    for d in durs:
        df_likelihood = pd.DataFrame()
        frqs = events[d].keys()
        for f in frqs:
            sims_event_max = pd.DataFrame()
            nevents = len(events[d][f])
            nsimspfd = nevents * len(ps) * len(rs)
            for j in range(len(posterior)):
                for m in ps:
                    for n in rs:
                        parameters = {
                            "ps0": m,  # Initial production storage (ps/X1)
                            "rs0": n,  # Initial routing storage (rs/X3)
                            "x1": posterior[j][1],  # Max production capacity (mm)
                            "x2": posterior[j][2],  # Water exchange coeff. (mm)
                            "x3": posterior[j][3],  # Routing max capacity (mm)
                            "x4": posterior[j][4],  # Unit hydrograph base (hrs)
                        }
                        # Create a model
                        model = GR4H(area=area, params=parameters)
                        for i in range(nevents):
                            # Run model
                            sim = model.run(events[d][f][i])
                            col = f"Param-{i}_ps0-{m}_rs0-{n}_event-{i}_1in{f}_dur-{d}"
                            sims_event_max[col] = [sim.qt.max()]

            # Robustness: score peak flows against a KDE of their distribution
            x, y = (
                sns.kdeplot(sims_event_max.values[0], cut=10, clip=(0, None))
                .get_lines()[0]
                .get_data()
            )
            likelihood = np.zeros(len(posterior))
            for i in range(len(posterior)):
                window = sims_event_max.values[0, i * nsimspfd:(i + 1) * nsimspfd]
                likelihood[i] = np.sum(np.interp(window, x, np.log10(y)))
            df_likelihood[f"Likelihood_1in{f}"] = likelihood

        df_likelihood["Likelihood_total"] = df_likelihood.sum(axis=1)
        robust_models_idx = df_likelihood.nlargest(topn, "Likelihood_total").index
        robust_models = posterior[robust_models_idx]
        if save_output:
            np.save(f"robust_models_{d}hr.npy", robust_models)

        return robust_models