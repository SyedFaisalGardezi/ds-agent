"""Unit tests for the EDA phase: DataProfiler, StatisticsAnalyser,
QualityScorer, Visualiser.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agent.pipeline.eda.profiler import DataProfiler
from agent.pipeline.eda.quality_scorer import QualityScorer
from agent.pipeline.eda.statistics import StatisticsAnalyser
from agent.pipeline.eda.visualiser import Visualiser

# --------------------------------------------------------------------------- #
# DataProfiler
# --------------------------------------------------------------------------- #


def _mixed_df():
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "x": [1.0, 2.0, np.nan, 4.0, 5.0],
        "cat": ["a", "a", "b", "b", "c"],
    })


def test_profiler_reports_nulls_and_cardinality():
    r = DataProfiler().run(_mixed_df(), "r")
    assert r.status == "ok"
    p = r.data["profile"].set_index("column")
    assert p.loc["x", "null_count"] == 1
    assert p.loc["cat", "n_unique"] == 3


def test_profiler_reports_overall_metrics():
    df = _mixed_df()
    r = DataProfiler().run(df, "r")
    assert r.data["n_rows"] == 5
    assert r.data["n_cols"] == 3
    assert r.data["overall_null_rate"] > 0
    assert r.data["duplicate_rows"] == 0


def test_profiler_numeric_stats():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    p = DataProfiler().run(df, "r").data["profile"].iloc[0]
    assert p["min"] == 1.0 and p["max"] == 5.0
    assert p["mean"] == 3.0


def test_profiler_categorical_top():
    df = pd.DataFrame({"c": ["a", "a", "b"]})
    p = DataProfiler().run(df, "r").data["profile"].iloc[0]
    assert p["top"] == "a"
    assert p["top_freq"] == 2


def test_profiler_empty_frame():
    r = DataProfiler().run(pd.DataFrame(columns=["a"]), "r")
    assert r.status == "warning"


def test_profiler_detects_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [1, 1, 2]})
    r = DataProfiler().run(df, "r")
    assert r.data["duplicate_rows"] == 1


# --------------------------------------------------------------------------- #
# StatisticsAnalyser
# --------------------------------------------------------------------------- #


def test_mutual_information_regression():
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=n)
    df = pd.DataFrame({"x": x, "noise": rng.normal(size=n), "y": 2 * x + rng.normal(scale=0.1, size=n)})
    r = StatisticsAnalyser().mutual_information(df, target="y")
    assert r.status == "ok"
    assert r.data["top"] == "x"


def test_mutual_information_missing_target():
    r = StatisticsAnalyser().mutual_information(pd.DataFrame({"a": [1]}), "y")
    assert r.status == "error"


def test_spearman_matrix_shape():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 4, 6, 8]})
    r = StatisticsAnalyser().spearman_correlation(df)
    assert r.status == "ok"
    assert r.data["matrix"].loc["a", "b"] == pytest.approx(1.0)


def test_spearman_requires_two_numeric():
    r = StatisticsAnalyser().spearman_correlation(pd.DataFrame({"a": [1, 2]}))
    assert r.status == "error"


def test_cramers_v_perfect_association():
    df = pd.DataFrame({"a": ["x", "x", "y", "y"], "b": ["1", "1", "2", "2"]})
    r = StatisticsAnalyser().cramers_v(df, "a", "b")
    assert r.status == "ok"
    assert r.data["v"] == pytest.approx(1.0, abs=1e-6)


def test_cramers_v_missing_column():
    r = StatisticsAnalyser().cramers_v(pd.DataFrame({"a": [1]}), "a", "z")
    assert r.status == "error"


def test_normality_test_gaussian():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"x": rng.normal(size=100)})
    r = StatisticsAnalyser().normality_test(df)
    assert r.status == "ok"
    assert r.data["tests"]["x"]["normal"] is True


def test_normality_test_skewed():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.exponential(size=500)})
    r = StatisticsAnalyser().normality_test(df)
    assert r.data["tests"]["x"]["normal"] is False


def test_adf_stationary():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(size=200))
    r = StatisticsAnalyser().adf_test(s)
    assert r.status == "ok"
    assert r.data["stationary"] is True


def test_adf_random_walk():
    rng = np.random.default_rng(0)
    s = pd.Series(np.cumsum(rng.normal(size=200)))
    r = StatisticsAnalyser().adf_test(s)
    assert r.status == "ok"
    assert r.data["stationary"] is False


def test_adf_too_short():
    r = StatisticsAnalyser().adf_test(pd.Series([1, 2, 3]))
    assert r.status == "error"


def test_kpss_stationary():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(size=200))
    r = StatisticsAnalyser().kpss_test(s)
    assert r.status == "ok"
    assert "stationary" in r.data


def test_stl_decompose_recovers_seasonality():
    t = np.arange(120)
    y = np.sin(2 * np.pi * t / 12) + 0.01 * t
    r = StatisticsAnalyser().stl_decompose(pd.Series(y), period=12)
    assert r.status == "ok"
    assert r.data["strength_seasonal"] > 0.5


def test_stl_decompose_bad_period():
    r = StatisticsAnalyser().stl_decompose(pd.Series([1, 2, 3, 4]), period=1)
    assert r.status == "error"


def test_stl_decompose_too_short():
    r = StatisticsAnalyser().stl_decompose(pd.Series([1, 2, 3]), period=4)
    assert r.status == "error"


def test_acf_pacf_lag_zero_is_one():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(size=200))
    r = StatisticsAnalyser().acf_pacf(s, lags=10)
    assert r.status == "ok"
    assert r.data["acf"][0] == pytest.approx(1.0)
    assert len(r.data["acf"]) == 11


def test_acf_pacf_too_short():
    r = StatisticsAnalyser().acf_pacf(pd.Series([1, 2, 3]), lags=10)
    assert r.status == "error"


# --------------------------------------------------------------------------- #
# QualityScorer
# --------------------------------------------------------------------------- #


def test_quality_scorer_perfect():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    r = QualityScorer().score(df)
    assert r.status == "ok"
    assert r.data["score"] > 0.6
    assert r.data["dimensions"]["completeness"] == 1.0
    assert r.data["dimensions"]["uniqueness"] == 1.0


def test_quality_scorer_penalises_nans():
    df = pd.DataFrame({"a": [1.0, np.nan, np.nan], "b": [np.nan, np.nan, 3.0]})
    r = QualityScorer().score(df)
    assert r.data["dimensions"]["completeness"] < 0.5


def test_quality_scorer_penalises_duplicates():
    df = pd.DataFrame({"a": [1, 1, 1], "b": [1, 1, 1]})
    r = QualityScorer().score(df)
    assert r.data["dimensions"]["uniqueness"] < 0.5


def test_quality_scorer_validity_flags_inf():
    df = pd.DataFrame({"a": [1.0, np.inf, 2.0]})
    r = QualityScorer().score(df)
    assert r.data["dimensions"]["validity"] < 1.0


def test_quality_scorer_baseline_matches_self():
    df = pd.DataFrame({"x": np.arange(50, dtype=float)})
    r = QualityScorer().score(df, baseline=df)
    assert r.data["dimensions"]["distribution"] == pytest.approx(1.0)


def test_quality_scorer_baseline_shifted_distribution():
    df = pd.DataFrame({"x": np.arange(100, dtype=float)})
    shifted = pd.DataFrame({"x": np.arange(100, dtype=float) + 1000})
    r = QualityScorer().score(df, baseline=shifted)
    assert r.data["dimensions"]["distribution"] < 0.5


def test_quality_scorer_warning_threshold():
    df = pd.DataFrame({"a": [np.nan] * 10})
    r = QualityScorer(warn_threshold=0.9).score(df)
    assert r.status == "warning"


def test_quality_scorer_rejects_bad_weights():
    with pytest.raises(ValueError):
        QualityScorer(weights={"completeness": 1.0})  # missing dims
    with pytest.raises(ValueError):
        QualityScorer(weights={k: 0 for k in
                               ["completeness", "validity", "uniqueness",
                                "timeliness", "distribution"]})


# --------------------------------------------------------------------------- #
# Visualiser
# --------------------------------------------------------------------------- #


def test_visualiser_distribution_plots(tmp_path):
    v = Visualiser(output_dir=tmp_path)
    df = pd.DataFrame({"a": np.arange(100, dtype=float), "b": np.arange(100, dtype=float) ** 0.5})
    r = v.distribution_plots(df, "run1")
    assert r.status == "ok"
    assert len(r.data["paths"]) == 2
    for p in r.data["paths"]:
        assert (tmp_path / "run1").exists()
        assert p.endswith(".png")


def test_visualiser_distribution_no_numeric(tmp_path):
    v = Visualiser(output_dir=tmp_path)
    r = v.distribution_plots(pd.DataFrame({"c": ["a", "b"]}), "run1")
    assert r.status == "warning"


def test_visualiser_correlation_heatmap(tmp_path):
    v = Visualiser(output_dir=tmp_path)
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 6.0]})
    r = v.correlation_heatmap(df, "run1")
    assert r.status == "ok"
    assert r.data["path"].endswith("correlation_heatmap.png")


def test_visualiser_correlation_too_few(tmp_path):
    v = Visualiser(output_dir=tmp_path)
    r = v.correlation_heatmap(pd.DataFrame({"a": [1, 2]}), "run1")
    assert r.status == "warning"


def test_visualiser_time_series(tmp_path):
    v = Visualiser(output_dir=tmp_path)
    s = pd.Series(np.arange(50, dtype=float), name="y")
    r = v.time_series_plot(s, "run1")
    assert r.status == "ok"
    assert r.data["path"].endswith(".png")


def test_visualiser_time_series_empty(tmp_path):
    v = Visualiser(output_dir=tmp_path)
    r = v.time_series_plot(pd.Series([], dtype=float), "run1")
    assert r.status == "warning"


# --------------------------------------------------------------------------- #
# StatisticsAnalyser — additional gap coverage (kpss, spearman direct, acf, stl)
# --------------------------------------------------------------------------- #


def test_statistics_kpss_stationary(monkeypatch):
    import numpy as np
    rng = np.random.default_rng(42)
    series = pd.Series(rng.normal(size=50))
    r = StatisticsAnalyser().kpss_test(series)
    assert r.status == "ok"
    assert "stat" in r.data
    assert "p_value" in r.data
    assert "stationary" in r.data


def test_statistics_kpss_too_short():
    r = StatisticsAnalyser().kpss_test(pd.Series([1.0, 2.0, 3.0]))
    assert r.status == "error"
    assert "10" in r.explanation


def test_statistics_spearman_direct():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [5, 4, 3, 2, 1]})
    r = StatisticsAnalyser().spearman_correlation(df)
    assert r.status == "ok"
    assert "matrix" in r.data


def test_statistics_spearman_single_column():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
    r = StatisticsAnalyser().spearman_correlation(df)
    assert r.status == "error"


def test_statistics_acf_pacf_basic():
    import numpy as np
    rng = np.random.default_rng(0)
    series = pd.Series(rng.normal(size=80))
    r = StatisticsAnalyser().acf_pacf(series, lags=10)
    assert r.status == "ok"
    assert "acf" in r.data
    assert "pacf" in r.data


def test_statistics_acf_pacf_too_short():
    r = StatisticsAnalyser().acf_pacf(pd.Series([1.0, 2.0, 3.0]), lags=10)
    assert r.status == "error"


def test_statistics_stl_valid():
    import numpy as np
    t = np.arange(60)
    series = pd.Series(np.sin(2 * np.pi * t / 12) + 0.1 * np.random.default_rng(0).normal(size=60))
    r = StatisticsAnalyser().stl_decompose(series, period=12)
    assert r.status == "ok"
    assert "strength_trend" in r.data
    assert "strength_seasonal" in r.data


def test_statistics_stl_period_too_small():
    series = pd.Series(list(range(30)))
    r = StatisticsAnalyser().stl_decompose(series, period=1)
    assert r.status == "error"


def test_statistics_stl_not_enough_obs():
    series = pd.Series(list(range(10)))
    r = StatisticsAnalyser().stl_decompose(series, period=12)
    assert r.status == "error"
