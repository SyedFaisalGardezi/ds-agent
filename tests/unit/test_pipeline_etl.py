"""Unit tests for the ETL pipeline stages: Extractor, Transformers, Loader,
ProvenanceStore, and end-to-end PipelineOrchestrator composition.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from agent.core.types import ToolResult
from agent.pipeline.base import PipelineOrchestrator
from agent.pipeline.etl.extractor import Extractor
from agent.pipeline.etl.loader import Loader
from agent.pipeline.etl.provenance import ProvenanceStore
from agent.pipeline.etl.transformer import (
    FFTFeatures,
    IQRClipper,
    LagFeature,
    MedianImputer,
    PolynomialFeatures,
    RollingStats,
    TargetEncoder,
    TransformerStage,
    WaveletFeatures,
    ZScoreClipper,
)

# --------------------------------------------------------------------------- #
# Fake connectors for Extractor tests
# --------------------------------------------------------------------------- #


class _RecordsConnector:
    def __init__(self, records):
        self._records = records
        self.last_query = None
        self.last_params = None

    def query(self, sql, params=None):
        self.last_query = sql
        self.last_params = params
        return ToolResult(
            status="ok",
            data={"records": self._records, "rowcount": len(self._records)},
            explanation="", math_trace="",
        )


class _BodyListConnector:
    def __init__(self, body):
        self._body = body

    def query(self, sql, params=None):
        return ToolResult(status="ok", data={"body": self._body},
                          explanation="", math_trace="")


class _ErrorConnector:
    def query(self, sql, params=None):
        return ToolResult(status="error", data=None, explanation="boom",
                          math_trace="", error="RuntimeError: boom")


# --------------------------------------------------------------------------- #
# Extractor
# --------------------------------------------------------------------------- #


def test_extractor_records():
    c = _RecordsConnector([{"id": 1, "x": 1.0}, {"id": 2, "x": 2.0}])
    r = Extractor(c, "SELECT *").run(pd.DataFrame(), "run1")
    assert r.status == "ok"
    assert list(r.data.columns) == ["id", "x"]
    assert len(r.data) == 2


def test_extractor_passes_params():
    c = _RecordsConnector([{"id": 1}])
    Extractor(c, "SELECT * WHERE id = ?", params={"id": 1}).run(pd.DataFrame(), "r")
    assert c.last_params == {"id": 1}


def test_extractor_rest_body_list():
    c = _BodyListConnector([{"a": 1}, {"a": 2}])
    r = Extractor(c, "/x").run(pd.DataFrame(), "r")
    assert r.status == "ok"
    assert r.data["a"].tolist() == [1, 2]


def test_extractor_rest_body_dict_with_items():
    c = _BodyListConnector({"items": [{"k": "v"}], "meta": 1})
    r = Extractor(c, "/x").run(pd.DataFrame(), "r")
    assert r.status == "ok"
    assert r.data["k"].tolist() == ["v"]


def test_extractor_rest_body_dict_scalar():
    c = _BodyListConnector({"ok": True})
    r = Extractor(c, "/x").run(pd.DataFrame(), "r")
    assert r.status == "ok"
    assert bool(r.data.iloc[0]["ok"]) is True


def test_extractor_error_passthrough():
    r = Extractor(_ErrorConnector(), "SELECT 1").run(pd.DataFrame(), "r")
    assert r.status == "error"
    assert "boom" in r.error


def test_extractor_empty():
    r = Extractor(_RecordsConnector([]), "SELECT 1").run(pd.DataFrame(), "r")
    assert r.status == "ok"
    assert len(r.data) == 0


# --------------------------------------------------------------------------- #
# MedianImputer
# --------------------------------------------------------------------------- #


def test_median_imputer_fills_nan():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, 2.0, 4.0]})
    r = MedianImputer().fit_transform(df)
    assert r.status == "ok"
    assert r.data["a"].isna().sum() == 0
    assert r.data["b"].isna().sum() == 0
    assert r.data["a"].iloc[1] == 2.0  # median of [1,3]


def test_median_imputer_not_fitted():
    r = MedianImputer().transform(pd.DataFrame({"a": [1.0]}))
    assert r.status == "error"
    assert r.error == "NotFittedError"


# --------------------------------------------------------------------------- #
# IQRClipper
# --------------------------------------------------------------------------- #


def test_iqr_clipper_clips_outliers():
    df = pd.DataFrame({"a": list(range(10)) + [1000]})
    r = IQRClipper().fit_transform(df)
    assert r.status == "ok"
    assert r.data["a"].max() < 1000


# --------------------------------------------------------------------------- #
# ZScoreClipper
# --------------------------------------------------------------------------- #


def test_zscore_clipper_clips_outliers():
    df = pd.DataFrame({"a": [0.0] * 50 + [100.0]})
    r = ZScoreClipper(k=2.0).fit_transform(df)
    assert r.status == "ok"
    assert r.data["a"].max() < 100.0


def test_zscore_clipper_zero_std_skipped():
    df = pd.DataFrame({"a": [1.0, 1.0, 1.0]})
    r = ZScoreClipper().fit_transform(df)
    assert r.status == "ok"
    assert r.data["a"].tolist() == [1.0, 1.0, 1.0]


# --------------------------------------------------------------------------- #
# TargetEncoder
# --------------------------------------------------------------------------- #


def test_target_encoder_smoothing():
    df = pd.DataFrame({
        "cat": ["a", "a", "b", "b", "c", "c"],
        "y":   [1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
    })
    enc = TargetEncoder(target="y", columns=["cat"], smoothing=0.0).fit(df)
    out = enc.transform(df).data
    assert out.loc[0, "cat"] == pytest.approx(1.0)
    assert out.loc[2, "cat"] == pytest.approx(0.0)
    assert out.loc[4, "cat"] == pytest.approx(0.5)


def test_target_encoder_unseen_maps_to_global():
    df = pd.DataFrame({"cat": ["a", "a", "b", "b"], "y": [1.0, 1.0, 0.0, 0.0]})
    enc = TargetEncoder(target="y", columns=["cat"], smoothing=0.0).fit(df)
    new = pd.DataFrame({"cat": ["c"], "y": [0.0]})
    out = enc.transform(new).data
    assert out.loc[0, "cat"] == pytest.approx(0.5)  # global mean


def test_target_encoder_missing_target_raises():
    with pytest.raises(KeyError):
        TargetEncoder(target="missing").fit(pd.DataFrame({"a": [1]}))


# --------------------------------------------------------------------------- #
# LagFeature
# --------------------------------------------------------------------------- #


def test_lag_feature_adds_columns():
    df = pd.DataFrame({"x": [1, 2, 3, 4]})
    r = LagFeature(lags=[1, 2], columns=["x"]).fit_transform(df)
    assert r.status == "ok"
    assert r.data["x_lag1"].tolist()[1:] == [1, 2, 3]
    assert pd.isna(r.data["x_lag1"].iloc[0])
    assert "x_lag2" in r.data.columns


def test_lag_feature_rejects_nonpositive():
    with pytest.raises(ValueError):
        LagFeature(lags=[0], columns=["x"])


# --------------------------------------------------------------------------- #
# RollingStats
# --------------------------------------------------------------------------- #


def test_rolling_stats_mean_std():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    r = RollingStats(window=2, columns=["x"], stats=["mean"]).fit_transform(df)
    assert r.status == "ok"
    assert r.data["x_roll2_mean"].iloc[1] == pytest.approx(1.5)


def test_rolling_stats_rejects_bad_stat():
    with pytest.raises(ValueError):
        RollingStats(window=2, columns=["x"], stats=["kurt"])


def test_rolling_stats_rejects_bad_window():
    with pytest.raises(ValueError):
        RollingStats(window=0, columns=["x"])


# --------------------------------------------------------------------------- #
# PolynomialFeatures
# --------------------------------------------------------------------------- #


def test_polynomial_features_degree2():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [2, 3, 4]})
    r = PolynomialFeatures(degree=2).fit_transform(df)
    assert r.status == "ok"
    assert "poly(a·a)" in r.data.columns
    assert "poly(a·b)" in r.data.columns
    assert r.data["poly(a·b)"].tolist() == [2.0, 6.0, 12.0]


def test_polynomial_features_interaction_only():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    r = PolynomialFeatures(degree=2, interaction_only=True).fit_transform(df)
    assert "poly(a·b)" in r.data.columns
    assert "poly(a·a)" not in r.data.columns


def test_polynomial_features_bias():
    df = pd.DataFrame({"a": [1, 2]})
    r = PolynomialFeatures(degree=2, include_bias=True).fit_transform(df)
    assert (r.data["poly_bias"] == 1.0).all()


def test_polynomial_features_rejects_degree1():
    with pytest.raises(ValueError):
        PolynomialFeatures(degree=1)


# --------------------------------------------------------------------------- #
# FFTFeatures
# --------------------------------------------------------------------------- #


def test_fft_features_finds_dominant_freq():
    n = 128
    t = np.arange(n)
    x = np.sin(2 * np.pi * 8 * t / n)  # 8 cycles over N samples → freq 8/N
    df = pd.DataFrame({"x": x})
    r = FFTFeatures(columns=["x"], top_k=1, sample_rate=1.0).fit_transform(df)
    assert r.status == "ok"
    peak = r.data["fft(x)_peak1_freq"].iloc[0]
    assert peak == pytest.approx(8 / n, abs=1e-6)
    assert r.data["fft(x)_energy"].iloc[0] > 0


def test_fft_features_rejects_bad_params():
    with pytest.raises(ValueError):
        FFTFeatures(columns=["x"], top_k=0)
    with pytest.raises(ValueError):
        FFTFeatures(columns=["x"], sample_rate=0)


# --------------------------------------------------------------------------- #
# WaveletFeatures
# --------------------------------------------------------------------------- #


def test_wavelet_features_energy_positive():
    df = pd.DataFrame({"x": np.arange(16, dtype=float)})
    r = WaveletFeatures(columns=["x"], levels=3).fit_transform(df)
    assert r.status == "ok"
    for lvl in (1, 2, 3):
        col = f"wavelet(x)_L{lvl}_energy"
        assert col in r.data.columns
        assert (r.data[col] > 0).all()


def test_wavelet_features_odd_length_reflects():
    df = pd.DataFrame({"x": np.arange(7, dtype=float)})
    r = WaveletFeatures(columns=["x"], levels=2).fit_transform(df)
    assert r.status == "ok"


def test_wavelet_features_rejects_levels_zero():
    with pytest.raises(ValueError):
        WaveletFeatures(columns=["x"], levels=0)


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #


def test_loader_parquet(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3]})
    path = tmp_path / "out.parquet"
    r = Loader(path).run(df, "run1")
    assert r.status == "ok"
    assert path.exists()
    round_trip = pd.read_parquet(path)
    assert round_trip["a"].tolist() == [1, 2, 3]


def test_loader_csv(tmp_path):
    df = pd.DataFrame({"a": [1, 2]})
    path = tmp_path / "out.csv"
    r = Loader(path).run(df, "r")
    assert r.status == "ok"
    assert "a" in path.read_text()


def test_loader_jsonl(tmp_path):
    df = pd.DataFrame({"a": [1, 2]})
    path = tmp_path / "out.jsonl"
    r = Loader(path).run(df, "r")
    assert r.status == "ok"
    lines = [json.loads(l) for l in path.read_text().splitlines()]
    assert lines == [{"a": 1}, {"a": 2}]


def test_loader_explicit_format(tmp_path):
    df = pd.DataFrame({"a": [1]})
    path = tmp_path / "noext"
    r = Loader(path, format="json").run(df, "r")
    assert r.status == "ok"
    assert json.loads(path.read_text()) == [{"a": 1}]


def test_loader_rejects_unknown_format(tmp_path):
    with pytest.raises(ValueError):
        Loader(tmp_path / "x.xml")


def test_loader_creates_parent_dirs(tmp_path):
    path = tmp_path / "nested" / "deep" / "out.csv"
    r = Loader(path).run(pd.DataFrame({"a": [1]}), "r")
    assert r.status == "ok"
    assert path.exists()


# --------------------------------------------------------------------------- #
# ProvenanceStore
# --------------------------------------------------------------------------- #


def test_provenance_roundtrip(tmp_path):
    store = ProvenanceStore(db_path=tmp_path / "prov.db")
    store.log(
        run_id="r1", stage="transform", transformer="MedianImputer",
        params={"k": 1}, input_shape=(3, 2), output_shape=(3, 2),
        null_rate_in=0.1, null_rate_out=0.0, math_trace="test",
    )
    rows = store.get_run("r1")
    assert len(rows) == 1
    assert rows[0]["transformer"] == "MedianImputer"
    assert json.loads(rows[0]["params"]) == {"k": 1}
    assert rows[0]["null_rate_out"] == 0.0


def test_provenance_multiple_entries(tmp_path):
    store = ProvenanceStore(db_path=tmp_path / "prov.db")
    for i in range(3):
        store.log(run_id="r2", stage=f"s{i}", transformer="T",
                  params={}, input_shape=(1, 1), output_shape=(1, 1),
                  null_rate_in=0.0, null_rate_out=0.0)
    assert len(store.get_run("r2")) == 3


# --------------------------------------------------------------------------- #
# End-to-end orchestration
# --------------------------------------------------------------------------- #


def test_end_to_end_extract_transform_load(tmp_path, monkeypatch):
    # Run pipeline under tmp checkpoint root.
    monkeypatch.chdir(tmp_path)
    records = [{"x": float(i) if i != 2 else None} for i in range(5)]
    connector = _RecordsConnector(records)
    out_path = tmp_path / "out.parquet"
    orch = PipelineOrchestrator(stages=[
        Extractor(connector, "SELECT x FROM t"),
        TransformerStage(MedianImputer()),
        TransformerStage(LagFeature(lags=[1], columns=["x"])),
        Loader(out_path),
    ])
    results = orch.run(pd.DataFrame())
    assert [r.status for r in results] == ["ok", "ok", "ok", "ok"]
    final = results[-1].tool_result.data
    assert "x_lag1" in final.columns
    assert final["x"].isna().sum() == 0
    assert out_path.exists()


def test_orchestrator_stops_on_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    orch = PipelineOrchestrator(stages=[
        Extractor(_ErrorConnector(), "SELECT 1"),
        TransformerStage(MedianImputer()),
    ])
    results = orch.run(pd.DataFrame())
    assert len(results) == 1
    assert results[0].status == "error"
