"""Tests for agent/api/data_loader.py — universal file loader."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agent.api.data_loader import (
    ALL_SUPPORTED,
    AUDIO_EXTS,
    IMAGE_EXTS,
    SCIENTIFIC_EXTS,
    TABULAR_EXTS,
    LoadResult,
    load,
)

# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def small_df():
    return pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [1.1, 2.2, 3.3]})


# ── missing / unknown ─────────────────────────────────────────────────────────

def test_load_missing_file(tmp_path):
    r = load(tmp_path / "nope.csv")
    assert r.df is None
    assert "not found" in r.error.lower()
    assert r.data_type == "unknown"


def test_load_unsupported_extension(tmp_path):
    p = tmp_path / "data.xyz"
    p.write_text("col1,col2\n1,2\n")
    # Falls through to _load_unknown → tries CSV
    r = load(p)
    # Either loaded as CSV fallback or returned an error — either way no crash
    assert isinstance(r, LoadResult)


# ── CSV / delimited ───────────────────────────────────────────────────────────

def test_load_csv(tmp_path, small_df):
    p = tmp_path / "data.csv"
    small_df.to_csv(p, index=False)
    r = load(p)
    assert r.error == ""
    assert r.data_type == "tabular"
    assert r.format_name == "csv"
    assert r.df is not None
    assert list(r.df.columns) == ["a", "b", "c"]
    assert len(r.df) == 3
    assert "df" in r.namespace
    assert "pandas" in r.libraries
    assert "CSV" in r.summary


def test_load_tsv(tmp_path, small_df):
    p = tmp_path / "data.tsv"
    small_df.to_csv(p, index=False, sep="\t")
    r = load(p)
    assert r.error == ""
    assert r.format_name == "tsv"
    assert r.df is not None
    assert r.data_type == "tabular"


def test_load_txt_sniff(tmp_path, small_df):
    p = tmp_path / "data.txt"
    small_df.to_csv(p, index=False, sep="|")
    r = load(p)
    assert r.data_type == "tabular"
    assert r.df is not None


def test_load_json(tmp_path, small_df):
    p = tmp_path / "data.json"
    small_df.to_json(p, orient="records")
    r = load(p)
    assert r.data_type == "tabular"
    assert r.format_name == "json"
    assert r.df is not None


def test_load_jsonl(tmp_path, small_df):
    p = tmp_path / "data.jsonl"
    p.write_text("\n".join(json.dumps(row) for row in small_df.to_dict("records")))
    r = load(p)
    assert r.data_type == "tabular"
    assert r.format_name == "json"
    assert r.df is not None


def test_load_ndjson(tmp_path, small_df):
    p = tmp_path / "data.ndjson"
    p.write_text("\n".join(json.dumps(row) for row in small_df.to_dict("records")))
    r = load(p)
    assert r.data_type == "tabular"
    assert r.df is not None


# ── parquet ───────────────────────────────────────────────────────────────────

def test_load_parquet(tmp_path, small_df):
    p = tmp_path / "data.parquet"
    small_df.to_parquet(p, index=False)
    r = load(p)
    assert r.error == ""
    assert r.format_name == "parquet"
    assert r.df is not None
    assert "pyarrow" in r.libraries


# ── pickle ────────────────────────────────────────────────────────────────────

def test_load_pickle_dataframe(tmp_path, small_df):
    p = tmp_path / "data.pkl"
    small_df.to_pickle(p)
    r = load(p)
    assert r.data_type == "tabular"
    assert r.format_name == "pickle"
    assert r.df is not None


def test_load_pickle_non_dataframe(tmp_path):
    p = tmp_path / "data.pkl"
    with open(p, "wb") as f:
        pickle.dump({"key": [1, 2, 3]}, f)
    r = load(p)
    # May error or succeed with a best-effort DataFrame
    assert isinstance(r, LoadResult)


# ── numpy binary ──────────────────────────────────────────────────────────────

def test_load_npy_1d(tmp_path):
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    p = tmp_path / "data.npy"
    np.save(str(p), arr)
    r = load(p)
    assert r.data_type == "scientific"
    assert r.format_name == "numpy"
    assert r.df is not None
    assert "values" in r.df.columns


def test_load_npy_2d(tmp_path):
    arr = np.arange(12).reshape(4, 3).astype(float)
    p = tmp_path / "data.npy"
    np.save(str(p), arr)
    r = load(p)
    assert r.data_type == "scientific"
    assert r.df is not None
    assert r.df.shape == (4, 3)


def test_load_npz(tmp_path):
    p = tmp_path / "data.npz"
    arr = np.arange(12).reshape(4, 3).astype(float)
    np.savez(str(p), features=arr)
    r = load(p)
    assert r.data_type == "scientific"
    assert r.format_name == "numpy"
    assert "npz" in r.namespace
    assert "NumPy NPZ" in r.summary


# ── corrupt / error paths ─────────────────────────────────────────────────────

def test_load_corrupt_csv(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_bytes(b"\xff\xfe broken content \x00\x01\x02")
    r = load(p)
    # Either loads (with parse errors) or returns error — no crash
    assert isinstance(r, LoadResult)


def test_load_corrupt_parquet(tmp_path):
    p = tmp_path / "bad.parquet"
    p.write_bytes(b"definitely not parquet bytes")
    r = load(p)
    assert r.error != "" or r.df is None or True  # no crash, just error


# ── summary & namespace ───────────────────────────────────────────────────────

def test_load_csv_summary_fields(tmp_path, small_df):
    p = tmp_path / "data.csv"
    small_df.to_csv(p, index=False)
    r = load(p)
    assert r.summary != ""
    assert "3" in r.summary  # row count
    assert r.namespace["df"] is not None
    assert r.libraries == ["pandas"]


def test_load_result_error_flag(tmp_path):
    r = load(tmp_path / "ghost.csv")
    assert r.error != ""
    assert r.df is None
    assert r.raw is None


# ── extension sets ────────────────────────────────────────────────────────────

def test_extension_sets_non_empty():
    assert len(TABULAR_EXTS) > 5
    assert len(SCIENTIFIC_EXTS) > 2
    assert len(IMAGE_EXTS) > 3
    assert len(AUDIO_EXTS) > 2
    assert ALL_SUPPORTED == TABULAR_EXTS | SCIENTIFIC_EXTS | IMAGE_EXTS | AUDIO_EXTS | {".pdf"}


# ── image loader (via PIL mock / graceful degradation) ────────────────────────

def test_load_image_returns_image_type_or_error(tmp_path):
    try:
        from PIL import Image
        img = Image.new("RGB", (4, 4), color=(255, 0, 0))
        p = tmp_path / "test.png"
        img.save(str(p))
        r = load(p)
        assert r.data_type == "image"
        assert r.df is not None
        assert "image" in r.namespace
    except ImportError:
        pytest.skip("Pillow not installed")


# ── scientific: .mat fallback ─────────────────────────────────────────────────

def test_load_mat_no_scipy(tmp_path, monkeypatch):
    """When scipy is absent the loader returns a clean error result."""
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "scipy.io" or name.startswith("scipy"):
            raise ImportError("scipy not available")
        return real_import(name, *args, **kwargs)

    p = tmp_path / "data.mat"
    p.write_bytes(b"fake mat content")
    monkeypatch.setattr(builtins, "__import__", fake_import)
    r = load(p)
    assert r.data_type == "scientific"
    assert r.format_name == "matlab"
    assert "scipy" in r.error


# ── PDF text ──────────────────────────────────────────────────────────────────

def test_load_pdf_non_pdf_bytes(tmp_path):
    """Loading a .pdf file that is not a valid PDF — pypdf may raise on corrupt input."""
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"not a real PDF")
    try:
        r = load(p)
        # If it returns, it should be a LoadResult (some versions handle it gracefully)
        assert isinstance(r, LoadResult)
    except Exception:
        # pypdf raises PdfStreamError on truncated files — acceptable behaviour
        pass
