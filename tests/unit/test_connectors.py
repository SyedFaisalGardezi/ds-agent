from __future__ import annotations

import pandas as pd

from tests.conftest import MockConnector


def test_mock_connector_connect(mock_connector: MockConnector) -> None:
    result = mock_connector.connect()
    assert result.status == "ok"


def test_mock_connector_list_schemas(mock_connector: MockConnector) -> None:
    result = mock_connector.list_schemas()
    assert result.status == "ok"
    assert "mock_table" in result.data


def test_mock_connector_query_returns_df(mock_connector: MockConnector) -> None:
    result = mock_connector.query("SELECT * FROM mock_table")
    assert result.status == "ok"
    assert isinstance(result.data, pd.DataFrame)
    assert len(result.data) == 1000


def test_mock_connector_estimate_cost_zero(mock_connector: MockConnector) -> None:
    result = mock_connector.estimate_cost("SELECT 1")
    assert result.status == "ok"
    assert result.data == 0.0
