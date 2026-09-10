"""Tests for demo data helpers."""

import pandas as pd
import pytest

from pyfloodrisk.demo_data import (
    DEMO_TP_REGIONS,
    catchment_data,
    demo_paths,
    list_demo_stations,
    load_demo_station_data,
    station_tp_region,
)

STATION = "117002A"


def test_demo_paths_returns_existing_root():
    paths = demo_paths()
    assert paths["root"].exists()
    assert paths["climate"].exists()
    assert paths["ifd"].exists()
    assert paths["tps"].exists()


def test_list_demo_stations_returns_known_stations():
    stations = list_demo_stations()
    assert stations == ["117002A", "405214"]


@pytest.mark.parametrize("station", ["117002A", "405214"])
def test_every_demo_station_has_its_four_inputs(station):
    """Climate, IFD and temporal patterns must all be present for a station."""
    paths = demo_paths()
    assert (paths["climate"] / f"GR4H_climatedata_{station}_hr.csv").exists()
    assert (paths["ifd"] / f"{station}_ifds.csv").exists()
    region = station_tp_region(station)
    # both kinds: point for bursts under 12 h, areal from 12 h up
    assert (paths["tps"] / region / f"{region}_Increments.csv").exists()
    assert (paths["tps"] / f"Areal_{region}"
            / f"Areal_{region}_Increments.csv").exists()
    assert catchment_data(station) > 0


def test_station_tp_region_maps_each_station():
    assert station_tp_region("117002A") == "WT"
    assert station_tp_region("405214") == "MB"
    assert set(DEMO_TP_REGIONS) == set(list_demo_stations())


def test_station_tp_region_rejects_an_unknown_station():
    with pytest.raises(KeyError, match="no bundled temporal-pattern region"):
        station_tp_region("999999")


def test_load_demo_station_data_returns_dataframe():
    df = load_demo_station_data(STATION)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["Station", "Date", "PET", "PREC", "Q"]
    assert not df.empty
    assert (df["Station"] == STATION).all()


@pytest.mark.parametrize("station", ["117002A", "405214"])
def test_load_demo_station_data_maps_columns_by_name(station):
    """The bundled files disagree on column order, so position is not enough.

    Rainfall and evaporation are both non-negative, so a swap would not be
    caught by a range check -- this compares against the file's own headers.
    """
    raw = pd.read_csv(
        demo_paths()["climate"] / f"GR4H_climatedata_{station}_hr.csv")
    df = load_demo_station_data(station)
    assert df["PREC"].sum() == pytest.approx(raw["prec"].sum())
    assert df["PET"].sum() == pytest.approx(raw["pet"].sum())
    assert df["Q"].sum() == pytest.approx(raw["qt"].sum())


@pytest.mark.parametrize("station", ["117002A", "405214"])
def test_dates_parse_to_a_gapless_hourly_record(station):
    """One file is ISO, the other day-first; both must come out hourly.

    A day-first date read month-first silently reorders the record, so this
    checks the parsed index is strictly increasing and hourly throughout.
    """
    df = load_demo_station_data(station)
    gaps = df["Date"].diff().dropna().unique()
    assert list(gaps) == [pd.Timedelta(hours=1)]


def test_load_demo_station_data_date_is_utc():
    df = load_demo_station_data(STATION)
    assert df["Date"].dt.tz is not None


def test_load_demo_station_data_missing_station_raises():
    with pytest.raises(FileNotFoundError, match="999999"):
        load_demo_station_data("999999")
