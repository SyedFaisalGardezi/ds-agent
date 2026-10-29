from __future__ import annotations

import os
import tempfile

# Keep test sessions out of the user's real session store (~/.ds-agent). The
# store reads this at import time, so it must be set before any agent import.
os.environ["DSAGENT_STATE_DIR"] = tempfile.mkdtemp(prefix="ds-agent-tests-")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from agent.connectors.base import DataConnector  # noqa: E402
from agent.core.types import ToolResult  # noqa: E402


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 1000
    return pd.DataFrame({
        "id": range(n),
        "feature_numeric_1": rng.normal(0, 1, n),
        "feature_numeric_2": rng.exponential(2, n),
        "feature_numeric_3": np.where(rng.random(n) < 0.1, np.nan, rng.uniform(-10, 10, n)),
        "feature_categorical": rng.choice(["A", "B", "C", "D"], n),
        "feature_datetime": pd.date_range("2023-01-01", periods=n, freq="h"),
        "target_binary": rng.integers(0, 2, n),
        "target_regression": rng.normal(50, 15, n),
    })


class MockConnector(DataConnector):
    name = "mock"

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def connect(self) -> ToolResult:
        return ToolResult(status="ok", data=True, explanation="Mock connected.", math_trace="")

    def list_schemas(self) -> ToolResult:
        return ToolResult(status="ok", data=["mock_table"], explanation="Mock schema.", math_trace="")

    def describe_table(self, table: str) -> ToolResult:
        cols = {col: str(dtype) for col, dtype in self._df.dtypes.items()}
        return ToolResult(status="ok", data=cols, explanation="Mock describe.", math_trace="")

    def query(self, query: str, params: dict | None = None) -> ToolResult:
        return ToolResult(status="ok", data=self._df.copy(), explanation="Mock query.", math_trace="")

    def estimate_cost(self, query: str) -> ToolResult:
        return ToolResult(status="ok", data=0.0, explanation="Mock cost $0.00.", math_trace="")

    def detect_pii(self, table: str) -> ToolResult:
        return ToolResult(status="ok", data={}, explanation="No PII detected in mock.", math_trace="")


@pytest.fixture
def mock_connector(synthetic_df: pd.DataFrame) -> MockConnector:
    return MockConnector(synthetic_df)
