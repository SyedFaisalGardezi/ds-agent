"""Unit tests for FeatureGenerator, FeatureSelector, TimeSeriesFeatureEngineer."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agent.pipeline.feature_engineering.generator import FeatureGenerator
from agent.pipeline.feature_engineering.selector import FeatureSelector
from agent.pipeline.feature_engineering.time_series import TimeSeriesFeatureEngineer

# --------------------------------------------------------------------------- #
# FeatureGenerator
# --------------------------------------------------------------------------- #


def test_generator_unary_transforms():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    r = FeatureGenerator(max_pairs=0).generate(df)
    assert r.status == "ok"
    cols = r.data["frame"].columns
    assert "sq(a)" in cols and "log1p(a)" in cols and "sqrt(a)" in cols


def test_generator_pairwise_ratios():
    df = pd.DataFrame({"a": [2.0, 4.0, 6.0], "b": [1.0, 2.0, 3.0]})
    r = FeatureGenerator(max_pairs=5).generate(df)
    f = r.data["frame"]
    assert "ratio(a/b)" in f.columns
    assert "diff(a-b)" in f.columns
    assert (f["ratio(a/b)"] == 2.0).all()


def test_generator_skips_log_when_negatives():
    df = pd.DataFrame({"a": [-1.0, 1.0, 2.0]})
    r = FeatureGenerator(max_pairs=0).generate(df)
    assert "log1p(a)" not in r.data["frame"].columns
    assert "sq(a)" in r.data["frame"].columns


def test_generator_datetime_decomposes():
    df = pd.DataFrame({
        "ts": pd.to_datetime(["2020-01-02 03:04", "2020-06-15 18:30"]),
        "v": [1.0, 2.0],
    })
    r = FeatureGenerator(max_pairs=0).generate(df)
    f = r.data["frame"]
    assert f["dt_year(ts)"].tolist() == [2020, 2020]
    assert f["dt_month(ts)"].tolist() == [1, 6]
    assert f["dt_hour(ts)"].tolist() == [3, 18]


def test_generator_low_cardinality_onehot():
    df = pd.DataFrame({"c": ["x", "y", "x"], "v": [1, 2, 3]})
    r = FeatureGenerator(max_pairs=0, max_onehot_cardinality=5).generate(df)
    cols = r.data["frame"].columns
    assert any(c.startswith("oh(c)") for c in cols)


def test_generator_skips_high_cardinality():
    df = pd.DataFrame({"c": [f"v{i}" for i in range(50)], "v": range(50)})
    r = FeatureGenerator(max_pairs=0, max_onehot_cardinality=10).generate(df)
    assert not any(c.startswith("oh(c)") for c in r.data["frame"].columns)


def test_generator_excludes_target():
    df = pd.DataFrame({"a": [1.0, 2.0], "y": [1.0, 2.0]})
    r = FeatureGenerator(max_pairs=5).generate(df, target="y")
    assert "sq(y)" not in r.data["frame"].columns


def test_generator_empty_frame():
    r = FeatureGenerator().generate(pd.DataFrame())
    assert r.status == "warning"


def test_generator_rejects_bad_params():
    with pytest.raises(ValueError):
        FeatureGenerator(max_pairs=-1)
    with pytest.raises(ValueError):
        FeatureGenerator(max_onehot_cardinality=1)


# --------------------------------------------------------------------------- #
# FeatureSelector
# --------------------------------------------------------------------------- #


def _synthetic_regression(n: int = 120):
    rng = np.random.default_rng(0)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    noise = rng.normal(size=n)
    noise2 = rng.normal(size=n)
    y = 3 * x1 - 2 * x2 + 0.05 * rng.normal(size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "noise": noise,
                         "noise2": noise2, "y": y})


def _synthetic_classif(n: int = 120):
    rng = np.random.default_rng(0)
    x1 = rng.normal(size=n)
    noise = rng.normal(size=n)
    y = (x1 + 0.1 * rng.normal(size=n) > 0).astype(int)
    return pd.DataFrame({"x1": x1, "noise": noise, "y": y})


def test_rfe_selects_signal_features():
    df = _synthetic_regression()
    r = FeatureSelector().rfe(df, target="y", n_features=2)
    assert r.status == "ok"
    assert set(r.data["selected"]) == {"x1", "x2"}


def test_rfe_classif():
    df = _synthetic_classif()
    r = FeatureSelector().rfe(df, target="y", n_features=1)
    assert r.status == "ok"
    assert r.data["is_classif"] is True
    assert r.data["selected"] == ["x1"]


def test_rfe_bad_n_features():
    df = _synthetic_regression()
    r = FeatureSelector().rfe(df, target="y", n_features=0)
    assert r.status == "error"


def test_rfe_missing_target():
    r = FeatureSelector().rfe(pd.DataFrame({"a": [1, 2]}), target="y", n_features=1)
    assert r.status == "error"


def test_boruta_confirms_signal():
    df = _synthetic_regression()
    r = FeatureSelector().boruta(df, target="y", n_iter=10)
    assert r.status == "ok"
    assert "x1" in r.data["confirmed"]
    assert "x2" in r.data["confirmed"]


def test_boruta_rejects_noise():
    df = _synthetic_regression()
    r = FeatureSelector().boruta(df, target="y", n_iter=10)
    # noise columns should usually not be confirmed
    assert "noise" not in r.data["confirmed"] or "noise2" not in r.data["confirmed"]


def test_boruta_rejects_short_n_iter():
    df = _synthetic_regression()
    r = FeatureSelector().boruta(df, target="y", n_iter=2)
    assert r.status == "error"


def test_shap_selection_permutation_fallback():
    df = _synthetic_regression()
    r = FeatureSelector().shap_selection(df, target="y", top_k=2,
                                         method="permutation")
    assert r.status == "ok"
    assert r.data["method"] == "permutation"
    assert set(r.data["selected"]) == {"x1", "x2"}


def test_shap_selection_auto_returns_some_method():
    df = _synthetic_regression()
    r = FeatureSelector().shap_selection(df, target="y", top_k=2, method="auto")
    assert r.status == "ok"
    assert r.data["method"] in {"shap", "permutation"}


def test_shap_selection_missing_target():
    r = FeatureSelector().shap_selection(pd.DataFrame({"a": [1, 2]}), target="y")
    assert r.status == "error"


# --------------------------------------------------------------------------- #
# TimeSeriesFeatureEngineer
# --------------------------------------------------------------------------- #


def test_fourier_features_shape_and_values():
    s = pd.Series(np.arange(24.0))
    r = TimeSeriesFeatureEngineer().fourier_features(s, periods=[12, 24],
                                                      n_harmonics=2)
    assert r.status == "ok"
    f = r.data["frame"]
    # 2 periods × 2 harmonics × 2 (sin+cos) = 8 columns
    assert f.shape == (24, 8)
    assert "fourier_p12_h1_sin" in f.columns
    # sin(2π · 0 / 12) = 0
    assert f["fourier_p12_h1_sin"].iloc[0] == pytest.approx(0.0)


def test_fourier_features_rejects_bad_params():
    s = pd.Series([1.0, 2.0])
    assert TimeSeriesFeatureEngineer().fourier_features(s, [], 1).status == "error"
    assert TimeSeriesFeatureEngineer().fourier_features(s, [1], 1).status == "error"
    assert TimeSeriesFeatureEngineer().fourier_features(s, [4], 0).status == "error"


def test_fourier_features_empty_series():
    r = TimeSeriesFeatureEngineer().fourier_features(
        pd.Series([], dtype=float), [4], 1,
    )
    assert r.status == "error"


def test_seismic_features_dominant_frequency():
    fs = 100.0
    n = 1000
    t = np.arange(n) / fs
    x = np.sin(2 * np.pi * 5.0 * t)  # 5 Hz pure tone
    df = pd.DataFrame({"wave": x})
    r = TimeSeriesFeatureEngineer().seismic_features(df, sample_rate=fs)
    assert r.status == "ok"
    feats = r.data["features"].loc["wave"]
    assert feats["dominant_freq"] == pytest.approx(5.0, abs=0.5)
    assert feats["rms"] == pytest.approx(np.sqrt(0.5), abs=0.05)


def test_seismic_features_zero_cross_rate():
    # alternating ±1 → ZCR ≈ 1
    x = np.tile([1.0, -1.0], 50)
    df = pd.DataFrame({"w": x})
    r = TimeSeriesFeatureEngineer().seismic_features(df, sample_rate=100.0)
    assert r.data["features"].loc["w", "zero_cross_rate"] == pytest.approx(1.0, abs=0.05)


def test_seismic_features_band_energies_populated():
    fs = 100.0
    n = 1000
    t = np.arange(n) / fs
    x = np.sin(2 * np.pi * 3.0 * t)  # 3 Hz → in band [2, 5)
    df = pd.DataFrame({"w": x})
    r = TimeSeriesFeatureEngineer().seismic_features(df, sample_rate=fs,
                                                      bands_hz=(0.5, 2.0, 5.0, 10.0))
    feats = r.data["features"].loc["w"]
    assert feats["band_2_5Hz"] > feats["band_5_10Hz"]
    assert feats["band_2_5Hz"] > feats["band_0.5_2Hz"]


def test_seismic_features_rejects_bad_sample_rate():
    r = TimeSeriesFeatureEngineer().seismic_features(
        pd.DataFrame({"x": [1.0] * 10}), sample_rate=0,
    )
    assert r.status == "error"


def test_seismic_features_rejects_bad_bands():
    r = TimeSeriesFeatureEngineer().seismic_features(
        pd.DataFrame({"x": [1.0] * 10}), sample_rate=100.0, bands_hz=(1.0,),
    )
    assert r.status == "error"


def test_seismic_features_skips_short_columns():
    r = TimeSeriesFeatureEngineer().seismic_features(
        pd.DataFrame({"x": [1.0, 2.0]}), sample_rate=100.0,
    )
    assert r.status == "error"  # nothing usable


def test_seismic_features_column_filter():
    fs = 100.0
    df = pd.DataFrame({
        "keep": np.sin(np.linspace(0, 20, 500)),
        "skip": np.arange(500, dtype=float),
    })
    r = TimeSeriesFeatureEngineer().seismic_features(df, sample_rate=fs,
                                                      columns=["keep"])
    assert r.status == "ok"
    assert list(r.data["features"].index) == ["keep"]
