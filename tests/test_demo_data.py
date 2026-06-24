"""Tests for demo data helpers."""

import pandas as pd
import pytest

from pyfloodrisk.demo_data import (
    demo_paths,
    list_demo_stations,
    load_demo_station_data,
)


def test_demo_paths_returns_existing_root():
    paths = demo_paths()
    assert paths["root"].exists()
    assert paths["climate"].exists()
    assert paths["storms"].exists()


def test_list_demo_stations_returns_known_stations():
    stations = list_demo_stations()
    assert isinstance(stations, list)
    assert len(stations) > 0
    assert "421026" in stations


def test_load_demo_station_data_returns_dataframe():
    df = load_demo_station_data("421026")
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["Station", "Date", "PET", "PREC", "Q"]
    assert not df.empty
    assert (df["Station"] == "421026").all()


def test_load_demo_station_data_date_is_utc():
    df = load_demo_station_data("421026")
    assert df["Date"].dt.tz is not None


def test_load_demo_station_data_missing_station_raises():
    with pytest.raises(FileNotFoundError, match="999999"):
        load_demo_station_data("999999")
