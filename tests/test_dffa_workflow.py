"""
Tests for the bridges between ``pyfloodrisk`` and the derived-FFA framework.

These are the joins where an integration goes wrong quietly: a state table
whose unit-hydrograph memory is a different length from the one the engine
expects, a state convention off by one timestep, an increments file parsed
into the wrong AEP band.  Each of those produces plausible-looking numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyfloodrisk.dffa import (DEMO_PARAMETERS, GR4HEventEngine, IFDCurve,
                              Stratification, continuous_state_table,
                              event_onset_states, load_station_forcings,
                              pet_climatology, run_dffa, state_sampler_from_run,
                              station_ifd, station_patterns)
from pyfloodrisk.dffa.ifd import (ARF_REGIONS, ARR2019ARF, _parse_aep,
                                  ifd_table_from_bom_csv, unit_arf)
from pyfloodrisk.demo_data import (AREAL_TP_DURATIONS_H, DEMO_ARF_REGIONS,
                                   POINT_TP_DURATIONS_H, TP_DURATIONS_H,
                                   catchment_data, station_arf_region,
                                   demo_paths, station_tp_region,
                                   list_demo_stations)
from pyfloodrisk.gr4h.GR4H_model import GR4H

STATION = "117002A"
PARAMS = DEMO_PARAMETERS[STATION]
AREA = catchment_data(STATION)


@pytest.fixture(scope="module")
def forcings():
    return load_station_forcings(STATION)


@pytest.fixture(scope="module")
def state_table(forcings):
    return continuous_state_table(forcings, PARAMS, AREA,
                                  warmup_hours=8760, thin=24)


# --------------------------------------------------------------- forcings
def test_station_forcings_are_hourly_and_complete(forcings):
    # matched by name, not order: the bundled files disagree on order
    assert {"pet", "prec", "qt"} <= set(forcings.columns)
    assert forcings.index.is_monotonic_increasing
    gaps = forcings.index.to_series().diff().dropna().unique()
    assert list(gaps) == [pd.Timedelta(hours=1)]
    assert forcings[["pet", "prec"]].notna().all().all()


def test_missing_station_raises():
    with pytest.raises(FileNotFoundError, match="999999"):
        load_station_forcings("999999")


# ------------------------------------------------------- state distribution
def test_state_table_has_the_columns_the_sampler_needs(state_table):
    model = GR4H(area=AREA, params=PARAMS)
    uh1 = [c for c in state_table.columns if c.startswith("uh1_")]
    uh2 = [c for c in state_table.columns if c.startswith("uh2_")]
    assert len(uh1) == model.n_uh1
    assert len(uh2) == model.n_uh2
    assert {"date", "prod_store", "rout_store"} <= set(state_table.columns)
    # stores are in mm, inside their capacities
    assert state_table["prod_store"].between(0, PARAMS["x1"]).all()
    assert (state_table["rout_store"] >= 0).all()


def test_state_table_respects_warmup_and_thinning(forcings):
    table = continuous_state_table(forcings, PARAMS, AREA,
                                   warmup_hours=100, thin=10)
    # thinning is applied on the full index, not restarted after the
    # warm-up cut, so the retained hours are the multiples of 10 past 100
    expected = [i for i in range(len(forcings)) if i >= 100 and i % 10 == 0]
    assert len(table) == len(expected)
    assert table["date"].iloc[0] == forcings.index[expected[0]]
    assert table["date"].iloc[-1] == forcings.index[expected[-1]]


def test_state_table_wet_only_keeps_states_before_rain(forcings):
    table = continuous_state_table(forcings, PARAMS, AREA, warmup_hours=100,
                                   wet_only=True, min_depth_mm=0.2)
    assert (table["prec_next"] > 0.2).all()
    assert 0 < len(table) < len(forcings)


def test_state_table_rows_resume_the_continuous_run(forcings, state_table):
    """The convention that makes the hot start meaningful.

    A row of the state table must be the state the continuous run was in at
    that hour, so that an event started from it is the continuation of the
    run.  Check it against the run itself.
    """
    model = GR4H(area=AREA, params=PARAMS)
    prec = forcings["prec"].to_numpy(float)
    pet = forcings["pet"].to_numpy(float)
    full = model.run_from_state(prec, pet)
    uh1 = [c for c in state_table.columns if c.startswith("uh1_")]
    uh2 = [c for c in state_table.columns if c.startswith("uh2_")]

    for row in (0, len(state_table) // 2, len(state_table) - 1):
        r = state_table.iloc[row]
        t = int(np.flatnonzero(forcings.index == r["date"])[0])
        if t >= len(prec) - 2:
            continue
        state = {"prod_store": r["prod_store"], "rout_store": r["rout_store"],
                 "uh1": r[uh1].to_numpy(float), "uh2": r[uh2].to_numpy(float)}
        resumed = model.run_from_state(prec[t + 1:], pet[t + 1:], state=state)
        assert np.allclose(resumed["qt"], full["qt"][t + 1:]), r["date"]


def test_state_table_rejects_over_filtering(forcings):
    with pytest.raises(ValueError, match="no states left"):
        continuous_state_table(forcings, PARAMS, AREA,
                               warmup_hours=len(forcings) + 1)


def test_event_onset_states_line_up_with_the_event_delineation(forcings):
    """States at the onset of the events the continuous run itself produced."""
    unthinned = continuous_state_table(forcings, PARAMS, AREA, warmup_hours=8760)
    onsets = event_onset_states(unthinned, event_method="maxima")
    assert 0 < len(onsets) < len(unthinned)
    assert list(onsets["event_id"]) == list(range(1, len(onsets) + 1))
    assert set(onsets.columns) - {"event_id"} == set(unthinned.columns)
    # every onset row is a row of the state table, unchanged
    merged = onsets.merge(unthinned, on=list(unthinned.columns), how="inner")
    assert len(merged) == len(onsets)
    # the delineated onsets are, on average, wetter than an arbitrary hour
    assert onsets["prod_store"].mean() > unthinned["prod_store"].mean()


def test_event_onset_states_reject_a_thinned_table(state_table):
    with pytest.raises(ValueError, match="unthinned"):
        event_onset_states(state_table)


def test_sampler_states_fit_the_engine(state_table):
    """The sampled state must be the shape the engine expects, unpadded."""
    sampler = state_sampler_from_run(state_table, PARAMS)
    engine = GR4HEventEngine(PARAMS, area_km2=AREA)
    sampled = sampler.sample(20, np.random.default_rng(0))
    for state in sampler.to_state_dicts(sampled):
        assert state["uh1"].size == engine.n_uh1
        assert state["uh2"].size == engine.n_uh2
        assert 0 <= state["prod_store"] <= PARAMS["x1"]
        assert state["rout_store"] >= 0
        assert isinstance(state["date"], pd.Timestamp)


def test_sampler_uh_columns_are_in_ordinate_order():
    """``uh1_10`` must not sort before ``uh1_2``."""
    table = pd.DataFrame({"date": pd.date_range("2000-01-01", periods=5, freq="h"),
                          "prod_store": np.linspace(10, 50, 5),
                          "rout_store": np.linspace(1, 5, 5)})
    for j in range(12):
        table[f"uh1_{j}"] = float(j)
        table[f"uh2_{j}"] = float(j)
    sampler = state_sampler_from_run(table, {"x1": 100.0, "x3": 10.0})
    assert list(sampler.uh1_cols) == [f"uh1_{j}" for j in range(12)]
    assert list(sampler.uh2_cols) == [f"uh2_{j}" for j in range(12)]


def test_pet_climatology_is_a_seasonal_cycle(forcings):
    pet = pet_climatology(forcings)
    assert pet.monthly_mm_per_day.size == 12
    assert np.isfinite(pet.monthly_mm_per_day).all()
    assert (pet.monthly_mm_per_day > 0).all()
    # southern hemisphere: January must be wetter in PET terms than June
    assert pet.monthly_mm_per_day[0] > pet.monthly_mm_per_day[5]
    # and the series conserves the monthly mean over a day
    series = pet.series(24, 1.0, month=1)
    assert series.sum() == pytest.approx(pet.monthly_mm_per_day[0], rel=1e-6)


# ------------------------------------------------------------ rainfall inputs
@pytest.mark.parametrize("station", list_demo_stations())
def test_station_patterns_parse_and_sum_to_one(station):
    """ARR increments are percentages of the burst; check one duration by hand.

    The bundled patterns are areal, so every increment row is a full storm
    whose increments sum to 100%.  Reading these files naively drops all but
    the first increment, which this would catch.
    """
    lib = station_patterns(station)
    assert 1440 in lib.durations_min                    # 24 h
    entry = lib.patterns[(1440, "rare")]
    assert np.allclose(entry["increments"].sum(axis=1), 100.0, atol=0.6)
    # the increments must span the burst: the file's timestep varies with
    # duration (30 min at 12 h, up to 180 min at 72 h and beyond), so this
    # is what catches increments being dropped on read
    for (dur_min, _), e in lib.patterns.items():
        assert e["increments"].shape[1] * e["timestep_min"] == dur_min
    # aggregated to the model timestep the fractions still sum to one
    frac, _, band, _ = lib.sample(24.0, 0.005, 1.0, np.random.default_rng(0))
    assert band == "rare"
    assert frac.size == 24 and np.isclose(frac.sum(), 1.0)


@pytest.mark.parametrize("station", list_demo_stations())
def test_patterns_span_point_and_areal_durations(station):
    """Point patterns below 12 h, areal from 12 h up, in one library."""
    lib = station_patterns(station)
    assert [d // 60 for d in lib.durations_min] == list(TP_DURATIONS_H)
    assert min(lib.durations_min) == 360                 # 6 h, from the point file


@pytest.mark.parametrize("station", list_demo_stations())
def test_short_durations_come_from_the_point_patterns(station):
    """Below 12 h the ensembles must vary by AEP band, as point patterns do.

    If the areal file were used here every band would share one ensemble,
    which is the tell that the wrong file was read.
    """
    lib = station_patterns(station)
    for duration_h in POINT_TP_DURATIONS_H:
        entries = [lib.patterns[(duration_h * 60, band)]
                   for band in ("frequent", "intermediate", "rare")]
        assert not np.array_equal(entries[0]["increments"],
                                  entries[-1]["increments"]), duration_h


def test_point_durations_must_be_shorter_than_the_areal_crossover():
    with pytest.raises(ValueError, match="must be shorter than"):
        station_patterns(STATION, point_durations_h=(6, 24))


def test_areal_only_library_is_still_available():
    lib = station_patterns(STATION, point_durations_h=())
    assert [d // 60 for d in lib.durations_min] == list(AREAL_TP_DURATIONS_H)


def test_areal_patterns_pick_the_nearest_standard_area():
    """Areal ensembles are published per area; the catchment picks one."""
    assert station_patterns("117002A").areal_area_km2 == 200.0   # 255.2 km2
    assert station_patterns("405214").areal_area_km2 == 500.0    # 357.4 km2


def test_areal_patterns_serve_every_aep_band():
    """Areal patterns carry no AEP dependence, so one ensemble serves all."""
    lib = station_patterns(STATION)
    assert {band for _, band in lib.patterns} == {"frequent", "intermediate",
                                                 "rare"}
    frequent = lib.patterns[(1440, "frequent")]["increments"]
    rare = lib.patterns[(1440, "rare")]["increments"]
    assert np.array_equal(frequent, rare)


def test_areal_patterns_need_an_area():
    from pyfloodrisk.dffa.patterns import TemporalPatternLibrary
    region = station_tp_region(STATION)
    path = demo_paths()["tps"] / f"Areal_{region}" / f"Areal_{region}_Increments.csv"
    with pytest.raises(ValueError, match="areal increments file"):
        TemporalPatternLibrary.from_arr_increments_csv(path)


def test_point_patterns_do_not_need_an_area():
    """A point file keys on AEP band, so no area is required to read it."""
    from pyfloodrisk.dffa.patterns import TemporalPatternLibrary
    region = station_tp_region(STATION)
    path = demo_paths()["tps"] / region / f"{region}_Increments.csv"
    lib = TemporalPatternLibrary.from_arr_increments_csv(path)
    assert lib.areal_area_km2 is None
    assert 360 in lib.durations_min                      # 6 h
    assert {band for _, band in lib.patterns} == {"frequent", "intermediate",
                                                  "rare"}


@pytest.mark.parametrize("station", list_demo_stations())
def test_bundled_bom_ifd_table_is_well_formed(station):
    """Every demo station ships a BoM IFD download the framework can use.

    Design rainfalls enter one way only -- a CSV -- so these files are the
    demo's single point of failure.
    """
    table = ifd_table_from_bom_csv(demo_paths()["ifd"] / f"{station}_ifds.csv")
    # frequent through to the ARR Rare depths, 1 in 200 to 1 in 2000
    assert list(table.columns) == [0.632, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01,
                                   0.005, 0.002, 0.001, 0.0005]
    assert list(table.index[:3]) == [1.0, 1.5, 2.0]     # hours, from minutes
    assert table.index[-1] == 168.0
    values = table.to_numpy(float)
    assert np.isfinite(values).all()             # the metadata preamble skipped
    assert (values > 0).all()
    # depth grows with rarity across the row, and does not fall with burst
    # length down it -- not strictly, because BoM rounds to three significant
    # figures, which ties 405214's 144 h and 168 h depths at 5% and 1% AEP
    assert np.all(np.diff(values, axis=1) > 0)
    assert np.all(np.diff(values, axis=0) >= 0)
    IFDCurve(table)                                     # accepted downstream


def test_bom_ifd_depths_match_the_file():
    """Spot-check against the raw download, so a mis-parse cannot pass."""
    table = ifd_table_from_bom_csv(demo_paths()["ifd"] / "117002A_ifds.csv")
    assert table.loc[1.0, 0.632] == pytest.approx(40.4)
    assert table.loc[24.0, 0.01] == pytest.approx(486.0)
    assert table.loc[168.0, 0.5] == pytest.approx(287.0)
    # the Rare columns, whose headers are "1 in 200" rather than a percentage
    assert table.loc[1.0, 0.005] == pytest.approx(123.0)
    assert table.loc[24.0, 0.0005] == pytest.approx(745.0)


def test_bom_reader_rejects_a_tidy_table(tmp_path):
    path = tmp_path / "tidy.csv"
    path.write_text("duration_h,0.5,0.01\n1,22.1,54.8\n")
    with pytest.raises(ValueError, match="does not look like a BoM IFD"):
        ifd_table_from_bom_csv(path)


def test_station_ifd_reads_the_bundled_table():
    curve = station_ifd(STATION)
    assert isinstance(curve, IFDCurve)
    # rarer burst, deeper burst -- at a duration the table does not carry
    assert curve.depth(4.0, 0.01) > curve.depth(4.0, 0.5) > 0


@pytest.mark.parametrize("station", list_demo_stations())
def test_station_ifd_applies_the_regional_arf_by_default(station):
    """The bundled depths are point depths; a catchment average is smaller.

    Without this the demo would silently run on unreduced point rainfall
    over a 255-357 km2 catchment.
    """
    area = catchment_data(station)
    reduced = station_ifd(station)
    point = station_ifd(station, arf=unit_arf)
    assert isinstance(reduced.arf, ARR2019ARF)
    assert reduced.arf.region == station_arf_region(station)
    for duration_h in (12.0, 24.0, 72.0):
        r = float(reduced.depth(duration_h, 0.01, area_km2=area))
        p = float(point.depth(duration_h, 0.01, area_km2=area))
        assert 0.5 * p < r < p, (station, duration_h)
    # asking for no area at all leaves the point depth untouched
    assert float(reduced.depth(24.0, 0.01)) == pytest.approx(
        float(point.depth(24.0, 0.01)))


def test_every_demo_station_has_a_valid_arf_region():
    assert set(DEMO_ARF_REGIONS) == set(list_demo_stations())
    for station, region in DEMO_ARF_REGIONS.items():
        assert region in ARF_REGIONS, (station, region)


def test_station_ifd_rejects_an_unknown_station():
    with pytest.raises(FileNotFoundError, match="no bundled IFD table"):
        station_ifd("not_a_station")


def test_aep_labels_are_parsed_from_the_usual_notations():
    assert _parse_aep("0.01") == pytest.approx(0.01)
    assert _parse_aep("1%") == pytest.approx(0.01)
    assert _parse_aep("1 in 100") == pytest.approx(0.01)
    assert _parse_aep("100y") == pytest.approx(0.01)
    assert _parse_aep(50) == pytest.approx(0.5)          # bare percentage


def test_bom_reader_accepts_a_hand_built_table(tmp_path):
    """A table you assembled yourself, in the download's own layout.

    This is the only loader, so it has to serve both a Bureau download and a
    table someone typed out; all it needs is the header row and the numeric
    'Duration in min' column.
    """
    path = tmp_path / "ifd.csv"
    path.write_text("some preamble line\n\n"
                    ",,Annual Exceedance Probability (AEP)\n"
                    "Duration,Duration in min,50%,10%,1 in 100\n"
                    "1 hour,60,22.1,36.0,54.8\n"
                    "6 hour,360,41.0,63.0,95.0\n")
    table = ifd_table_from_bom_csv(path)
    assert list(table.index) == [1.0, 6.0]               # minutes -> hours
    assert list(table.columns) == [0.5, 0.1, 0.01]       # %, and "1 in 100"
    assert table.loc[1.0, 0.01] == pytest.approx(54.8)
    IFDCurve(table)


def test_only_the_bom_reader_is_exposed():
    """Design rainfalls enter one way: the download as issued."""
    from pyfloodrisk.dffa import ifd as ifd_module
    assert not hasattr(ifd_module, "ifd_table_from_csv")
    assert "ifd_table_from_csv" not in ifd_module.__all__


# ------------------------------------------------------------------ workflow
def test_run_dffa_end_to_end():
    """The whole chain, on the smallest experiment that still means anything."""
    out = run_dffa(station=STATION, durations_h=(12, 24),
                   stratification=Stratification.uniform_in_z(
                       aep_max=0.9, aep_min=1e-4, n_strata=6, n_per_stratum=8,
                       n_per_end=8),
                   thin=48, progress=False)
    res = out["results"]
    assert set(res.durations) == {12.0, 24.0}
    assert len(res.events) == 2 * 8 * 8            # 6 strata + 2 open ends
    assert (res.events["q_peak"] > 0).all()
    assert np.isclose(res.events.groupby("duration_h")["weight"].sum(), 1.0).all()

    aeps = [0.1, 0.01]
    env = res.envelope(aeps)
    assert np.all(np.diff(env["q_peak"].to_numpy()) > 0)   # rarer is bigger
    assert set(env["critical_duration_h"]) <= {12.0, 24.0}
    # the state distribution the events were started from is the run's own
    assert out["state_table"]["prod_store"].max() <= out["parameters"]["x1"]
    assert isinstance(out["engine"], GR4HEventEngine)
