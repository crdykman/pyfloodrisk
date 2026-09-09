"""Tests for the explicit-state GR4H entry points.

The point of these is the hot start: an event simulated from a state taken
out of a continuous run must continue that run exactly, because that
equivalence is what lets the design event inherit its antecedent conditions
from continuous simulation instead of from a sampled initial loss.
"""

import numpy as np
import pandas as pd
import pytest

from pyfloodrisk.gr4h.GR4H_model import GR4H, _gr4h, _gr4h_from_state


PARAMS = {"ps0": 0.4, "rs0": 0.6, "x1": 350.0, "x2": -0.8, "x3": 60.0, "x4": 6.3}


def _forcing(n=600, seed=0):
    rng = np.random.default_rng(seed)
    prec = np.where(rng.random(n) < 0.2, rng.gamma(2.0, 2.0, n), 0.0)
    pet = np.full(n, 0.08)
    return prec, pet


def test_cold_start_matches_the_fraction_based_kernel():
    """``_gr4h`` is ``_gr4h_from_state`` with empty memory: same numbers."""
    prec, pet = _forcing()
    x1, x2, x3, x4 = (PARAMS[k] for k in ("x1", "x2", "x3", "x4"))
    qt, qd, qb, gwe, ps, rs = _gr4h(prec, pet, x1, x2, x3, x4,
                                    PARAMS["ps0"], PARAMS["rs0"])
    hot = _gr4h_from_state(prec, pet, x1, x2, x3, x4,
                           PARAMS["ps0"] * x1, PARAMS["rs0"] * x3,
                           np.zeros(int(np.ceil(x4))),
                           np.zeros(int(np.ceil(2 * x4))), False)
    assert np.allclose(qt, hot[0], rtol=1e-6, atol=1e-7)
    # the stores are reported as fractions by one and in mm by the other
    assert np.array_equal(ps, (hot[4] / x1).astype(np.float32))
    assert np.array_equal(rs, (hot[5] / x3).astype(np.float32))


def test_uh_memory_lengths_follow_x4():
    model = GR4H(area=100.0, params=PARAMS)
    assert model.n_uh1 == int(np.ceil(PARAMS["x4"]))
    assert model.n_uh2 == int(np.ceil(2 * PARAMS["x4"]))
    state = model.initial_state()
    assert state["uh1"].size == model.n_uh1
    assert state["uh2"].size == model.n_uh2
    assert state["prod_store"] == pytest.approx(PARAMS["ps0"] * PARAMS["x1"])


def test_recorded_state_resumes_the_run_exactly():
    """Row ``t`` of the recorded state, fed back in, reproduces ``t+1`` on."""
    prec, pet = _forcing()
    model = GR4H(area=320.0, params=PARAMS)
    full = model.run_from_state(prec, pet, record_uh=True)

    for t in (0, 137, 250, len(prec) - 2):
        state = {"prod_store": full["prod_store"][t],
                 "rout_store": full["rout_store"][t],
                 "uh1": full["uh1"][t], "uh2": full["uh2"][t]}
        resumed = model.run_from_state(prec[t + 1:], pet[t + 1:], state=state)
        assert np.array_equal(resumed["qt"], full["qt"][t + 1:]), t


def test_final_state_resumes_the_run_exactly():
    prec, pet = _forcing()
    model = GR4H(area=320.0, params=PARAMS)
    full = model.run_from_state(prec, pet)
    first = model.run_from_state(prec[:400], pet[:400])
    second = model.run_from_state(prec[400:], pet[400:], state=first["final_state"])
    assert np.array_equal(np.concatenate([first["qt"], second["qt"]]), full["qt"])


def test_uh_memory_is_padded_and_truncated_to_the_model_lengths():
    """A state whose UH arrays are the wrong length must not raise."""
    prec, pet = _forcing(n=120)
    model = GR4H(area=100.0, params=PARAMS)
    base = model.initial_state()
    short = dict(base, uh1=np.zeros(2), uh2=np.zeros(3))
    long = dict(base, uh1=np.zeros(50), uh2=np.zeros(50))
    q_ref = model.run_from_state(prec, pet, state=base)["qt"]
    assert np.array_equal(model.run_from_state(prec, pet, state=short)["qt"], q_ref)
    assert np.array_equal(model.run_from_state(prec, pet, state=long)["qt"], q_ref)


def test_uh_memory_enters_the_event_on_the_models_own_convention():
    """Memory left over from the continuous run must show up as discharge.

    With no rainfall and no groundwater exchange the direct flow is exactly
    ``0.1 x`` the UH2 memory as it shifts out, which pins down the two
    things about the convention that matter for hot-starting: the split
    happens at the point of use (so the arrays hold unsplit routed depth),
    and ordinate 0 has already been released by the step the state was
    recorded at, so it is dropped rather than released twice.
    """
    model = GR4H(area=100.0, params=dict(PARAMS, x2=0.0))
    dry = np.zeros(4 * model.n_uh2)
    uh1 = np.arange(1.0, model.n_uh1 + 1.0)
    uh2 = np.arange(1.0, model.n_uh2 + 1.0)
    out = model.run_from_state(dry, dry, state={
        "prod_store": 0.0, "rout_store": 0.0, "uh1": uh1, "uh2": uh2})

    expected_qd = np.zeros(dry.size)
    expected_qd[:model.n_uh2 - 1] = 0.1 * uh2[1:]
    assert np.allclose(out["qd"], expected_qd)
    # the other 90% enters the routing store, which drains it as a fourth
    # power law -- slowly -- so check the store's balance rather than its
    # outflow: everything that went in is either out or still in there
    assert out["qb"].sum() + out["rout_store"][-1] == pytest.approx(
        0.9 * uh1[1:].sum())


def test_run_from_state_rejects_mismatched_forcing_lengths():
    model = GR4H(area=100.0, params=PARAMS)
    with pytest.raises(ValueError, match="same length"):
        model.run_from_state(np.zeros(10), np.zeros(11))


def test_run_is_unchanged_by_the_refactor():
    """``GR4H.run`` still returns its documented columns and units."""
    prec, pet = _forcing(n=200)
    model = GR4H(area=100.0, params=PARAMS)
    out = model.run(pd.DataFrame({"prec": prec, "pet": pet}))
    assert list(out.columns) == ["qt", "qt_mm", "qd", "qb", "pet", "prec",
                                "gwe", "ps", "rs"]
    # qt is qt_mm converted for a 100 km2 catchment at an hourly step
    assert np.allclose(out["qt"], out["qt_mm"] * 100.0 / 3.6)
    assert out["ps"].between(0, 1).all()
