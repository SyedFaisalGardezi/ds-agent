"""Universal data loader for ds-agent.

Detects file format from extension (and magic bytes as fallback), loads it
with the right library, and returns:
  - `df`          — always a pandas DataFrame (best-effort tabular view)
  - `raw`         — the native object (xarray Dataset, numpy array, PIL Image…)
  - `data_type`   — broad category: 'tabular' | 'image' | 'audio' | 'scientific' | 'text'
  - `format_name` — specific format: 'csv', 'excel', 'netcdf4', 'numpy', …
  - `namespace`   — extra kernel variables to inject (ds, audio, image, …)
  - `summary`     — human-readable load summary for the chat bubble
  - `libraries`   — list of library names that were used

Deliberately uses lazy imports so the agent works even when optional deps are
missing — it reports exactly which pip install is needed rather than crashing.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ── supported extensions ──────────────────────────────────────────────────────

TABULAR_EXTS = {
    ".csv", ".tsv", ".txt",          # delimited text
    ".xlsx", ".xls", ".xlsm", ".xlsb",  # Excel
    ".parquet",                       # Parquet
    ".feather", ".arrow",             # Arrow / Feather
    ".json", ".jsonl", ".ndjson",     # JSON lines / records
    ".h5", ".hdf5", ".hdf",          # HDF5 (pandas store)
    ".pkl", ".pickle",                # Pickle
    ".dta",                           # Stata
    ".sas7bdat", ".xpt",              # SAS
    ".sav",                           # SPSS
    ".orc",                           # ORC
}

SCIENTIFIC_EXTS = {
    ".nc", ".nc4", ".netcdf",        # NetCDF4
    ".npy", ".npz",                  # NumPy binary
    ".mat",                          # MATLAB
}

IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".tif", ".tiff",
    ".bmp", ".gif", ".webp", ".svg",
}

AUDIO_EXTS = {
    ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif",
}

ALL_SUPPORTED = TABULAR_EXTS | SCIENTIFIC_EXTS | IMAGE_EXTS | AUDIO_EXTS | {".pdf"}


@dataclass
class LoadResult:
    df: pd.DataFrame | None
    raw: Any
    data_type: str          # tabular | image | audio | scientific | text | unknown
    format_name: str        # csv | excel | parquet | netcdf4 | numpy | image | audio | …
    namespace: dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    libraries: list[str] = field(default_factory=list)
    error: str = ""


# ── entry point ───────────────────────────────────────────────────────────────

def load(data_path: Path) -> LoadResult:
    """Detect format and load `data_path` into a LoadResult."""
    data_path = Path(data_path).expanduser().resolve()
    if not data_path.exists():
        return LoadResult(df=None, raw=None, data_type="unknown",
                          format_name="unknown",
                          error=f"File not found: {data_path}")

    ext = data_path.suffix.lower()

    if ext in IMAGE_EXTS:
        return _load_image(data_path)
    if ext in AUDIO_EXTS:
        return _load_audio(data_path)
    if ext in SCIENTIFIC_EXTS:
        return _load_scientific(data_path, ext)
    if ext == ".pdf":
        return _load_pdf_as_text(data_path)
    if ext in TABULAR_EXTS or ext == "":
        return _load_tabular(data_path, ext)

    # Unknown extension — probe magic bytes then try generic tabular
    return _load_unknown(data_path)


# ── tabular loaders ───────────────────────────────────────────────────────────

def _load_tabular(path: Path, ext: str) -> LoadResult:
    libs: list[str] = ["pandas"]
    try:
        if ext in (".csv", ".tsv", ".txt", ""):
            if ext == ".tsv":
                df = pd.read_csv(path, sep="\t", low_memory=False)
                fmt = "tsv"
            elif ext == ".txt":
                # Sniff delimiter (python engine required, skip low_memory)
                df = pd.read_csv(path, sep=None, engine="python")
                fmt = "txt_delimited"
            else:
                df = pd.read_csv(path, low_memory=False)
                fmt = "csv"

        elif ext in (".xlsx", ".xls", ".xlsm", ".xlsb"):
            libs.append("openpyxl")
            df = pd.read_excel(path)
            fmt = "excel"

        elif ext == ".parquet":
            libs.append("pyarrow")
            df = pd.read_parquet(path)
            fmt = "parquet"

        elif ext in (".feather", ".arrow"):
            libs.append("pyarrow")
            df = pd.read_feather(path)
            fmt = "feather"

        elif ext in (".json", ".jsonl", ".ndjson"):
            try:
                df = pd.read_json(path, lines=(ext in (".jsonl", ".ndjson")))
            except Exception:
                df = pd.read_json(path)
            fmt = "json"

        elif ext in (".h5", ".hdf5", ".hdf"):
            libs.append("tables")
            try:
                # Try pandas HDF store first
                with pd.HDFStore(path, mode="r") as store:
                    key = store.keys()[0]
                    df = store[key]
                fmt = "hdf5_pandas"
            except Exception:
                # Fall through to scientific loader
                return _load_scientific(path, ext)

        elif ext in (".pkl", ".pickle"):
            raw = pd.read_pickle(path)
            df = raw if isinstance(raw, pd.DataFrame) else pd.DataFrame(raw)
            fmt = "pickle"

        elif ext == ".dta":
            df = pd.read_stata(path)
            fmt = "stata"

        elif ext in (".sas7bdat", ".xpt"):
            df = pd.read_sas(path)
            fmt = "sas"

        elif ext == ".sav":
            try:
                import pyreadstat
                libs.append("pyreadstat")
                df, _ = pyreadstat.read_sav(str(path))
            except ImportError:
                return LoadResult(df=None, raw=None, data_type="tabular",
                                  format_name="spss",
                                  error="Install pyreadstat: pip install pyreadstat")
            fmt = "spss"

        elif ext == ".orc":
            libs.append("pyarrow")
            df = pd.read_orc(path)
            fmt = "orc"

        else:
            df = pd.read_csv(path, low_memory=False)
            fmt = "csv_fallback"

        summary = (
            f"Loaded **{fmt.upper()}** → {df.shape[0]:,} rows × {df.shape[1]} columns.\n"
            f"Columns: {list(df.columns[:20])}"
            + (" …" if len(df.columns) > 20 else "")
        )
        return LoadResult(df=df, raw=df, data_type="tabular", format_name=fmt,
                          namespace={"df": df}, summary=summary, libraries=libs)

    except Exception as exc:
        return LoadResult(df=None, raw=None, data_type="tabular",
                          format_name=ext.lstrip("."),
                          error=f"Failed to load {ext} file: {exc}")


# ── scientific loaders ────────────────────────────────────────────────────────

def _load_scientific(path: Path, ext: str) -> LoadResult:
    libs: list[str] = []

    if ext in (".nc", ".nc4", ".netcdf"):
        try:
            import xarray as xr
            libs.append("xarray")
            ds = xr.open_dataset(str(path))
            # Best-effort DataFrame: flatten first data variable
            try:
                df = ds.to_dataframe().reset_index()
            except Exception:
                df = None
            dims = dict(ds.dims)
            vars_ = list(ds.data_vars)
            summary = (
                f"Loaded **NetCDF4** dataset.\n"
                f"Dimensions: {dims}\n"
                f"Variables: {vars_[:20]}"
                + (" …" if len(vars_) > 20 else "")
            )
            ns = {"ds": ds, "df": df}
            if df is not None:
                ns["df"] = df
            return LoadResult(df=df, raw=ds, data_type="scientific",
                              format_name="netcdf4", namespace=ns,
                              summary=summary, libraries=libs)
        except ImportError:
            return LoadResult(df=None, raw=None, data_type="scientific",
                              format_name="netcdf4",
                              error="Install xarray + netcdf4: pip install xarray netcdf4")

    if ext in (".npy", ".npz"):
        libs.append("numpy")
        raw = np.load(str(path), allow_pickle=True)
        if ext == ".npz":
            keys = list(raw.keys())
            # Use first 2-D array as df if possible
            df = None
            for k in keys:
                arr = raw[k]
                if arr.ndim == 2:
                    df = pd.DataFrame(arr, columns=[f"{k}_{i}" for i in range(arr.shape[1])])
                    break
            summary = f"Loaded **NumPy NPZ** archive.\nArrays: {keys}"
            ns = {"npz": raw, "df": df}
        else:
            arr = raw if isinstance(raw, np.ndarray) else raw.item()
            if isinstance(arr, np.ndarray) and arr.ndim == 2:
                df = pd.DataFrame(arr)
            elif isinstance(arr, np.ndarray) and arr.ndim == 1:
                df = pd.DataFrame({"values": arr})
            else:
                df = None
            summary = f"Loaded **NumPy NPY** array, shape={getattr(arr, 'shape', '?')}, dtype={getattr(arr, 'dtype', '?')}"
            ns = {"arr": arr, "df": df}
        return LoadResult(df=df, raw=raw, data_type="scientific",
                          format_name="numpy", namespace=ns,
                          summary=summary, libraries=libs)

    if ext == ".mat":
        try:
            import scipy.io as sio
            libs.append("scipy")
            mat = sio.loadmat(str(path))
            # Filter out MATLAB metadata keys
            data_keys = [k for k in mat if not k.startswith("__")]
            df = None
            for k in data_keys:
                v = mat[k]
                if isinstance(v, np.ndarray) and v.ndim == 2:
                    df = pd.DataFrame(v, columns=[f"{k}_{i}" for i in range(v.shape[1])])
                    break
            summary = f"Loaded **MATLAB .mat** file.\nVariables: {data_keys}"
            return LoadResult(df=df, raw=mat, data_type="scientific",
                              format_name="matlab",
                              namespace={"mat": mat, "df": df},
                              summary=summary, libraries=libs)
        except ImportError:
            return LoadResult(df=None, raw=None, data_type="scientific",
                              format_name="matlab",
                              error="Install scipy: pip install scipy")

    if ext in (".h5", ".hdf5", ".hdf"):
        try:
            import h5py
            libs.append("h5py")
            f = h5py.File(str(path), "r")
            keys = list(f.keys())
            df = None
            for k in keys:
                dset = f[k]
                if hasattr(dset, "shape") and len(dset.shape) == 2:
                    arr = dset[:]
                    df = pd.DataFrame(arr, columns=[f"{k}_{i}" for i in range(arr.shape[1])])
                    break
            summary = f"Loaded **HDF5** file.\nDatasets: {keys}"
            return LoadResult(df=df, raw=f, data_type="scientific",
                              format_name="hdf5",
                              namespace={"hdf": f, "df": df},
                              summary=summary, libraries=libs)
        except ImportError:
            return LoadResult(df=None, raw=None, data_type="scientific",
                              format_name="hdf5",
                              error="Install h5py: pip install h5py")

    return LoadResult(df=None, raw=None, data_type="scientific",
                      format_name=ext.lstrip("."),
                      error=f"Unsupported scientific format: {ext}")


# ── image loaders ─────────────────────────────────────────────────────────────

def _load_image(path: Path) -> LoadResult:
    try:
        from PIL import Image
        libs = ["Pillow", "numpy"]
        img = Image.open(path)
        arr = np.array(img)
        # Metadata DataFrame (1 row)
        df = pd.DataFrame([{
            "filename": path.name,
            "width": img.width,
            "height": img.height,
            "mode": img.mode,
            "channels": arr.shape[2] if arr.ndim == 3 else 1,
            "dtype": str(arr.dtype),
            "size_kb": round(path.stat().st_size / 1024, 1),
        }])
        summary = (
            f"Loaded **image** `{path.name}`.\n"
            f"Size: {img.width}×{img.height} px, mode: {img.mode}, "
            f"array shape: {arr.shape}"
        )
        return LoadResult(df=df, raw=img, data_type="image",
                          format_name="image",
                          namespace={"image": img, "arr": arr, "df": df},
                          summary=summary, libraries=libs)
    except ImportError:
        return LoadResult(df=None, raw=None, data_type="image",
                          format_name="image",
                          error="Install Pillow: pip install Pillow")


# ── audio loaders ─────────────────────────────────────────────────────────────

def _load_audio(path: Path) -> LoadResult:
    ext = path.suffix.lower()
    # Try librosa first (handles mp3/flac/ogg), fall back to scipy for wav
    try:
        import librosa
        libs = ["librosa"]
        audio, sr = librosa.load(str(path), sr=None, mono=False)
        duration = audio.shape[-1] / sr
        n_channels = 1 if audio.ndim == 1 else audio.shape[0]
        df = pd.DataFrame([{
            "filename": path.name,
            "sample_rate": sr,
            "duration_s": round(duration, 3),
            "n_samples": audio.shape[-1],
            "n_channels": n_channels,
            "dtype": str(audio.dtype),
            "size_kb": round(path.stat().st_size / 1024, 1),
        }])
        summary = (
            f"Loaded **audio** `{path.name}`.\n"
            f"Sample rate: {sr:,} Hz, Duration: {duration:.2f}s, "
            f"Channels: {n_channels}"
        )
        return LoadResult(df=df, raw=(audio, sr), data_type="audio",
                          format_name="audio",
                          namespace={"audio": audio, "sample_rate": sr, "df": df},
                          summary=summary, libraries=libs)
    except ImportError:
        pass

    if ext == ".wav":
        try:
            from scipy.io import wavfile
            libs = ["scipy"]
            sr, audio = wavfile.read(str(path))
            if audio.ndim == 1:
                n_channels, n_samples = 1, len(audio)
            else:
                n_samples, n_channels = audio.shape
            duration = n_samples / sr
            df = pd.DataFrame([{
                "filename": path.name, "sample_rate": sr,
                "duration_s": round(duration, 3), "n_samples": n_samples,
                "n_channels": n_channels, "dtype": str(audio.dtype),
            }])
            summary = (
                f"Loaded **WAV audio** `{path.name}` (scipy).\n"
                f"Sample rate: {sr:,} Hz, Duration: {duration:.2f}s"
            )
            return LoadResult(df=df, raw=(audio, sr), data_type="audio",
                              format_name="wav",
                              namespace={"audio": audio, "sample_rate": sr, "df": df},
                              summary=summary, libraries=libs)
        except ImportError:
            pass

    return LoadResult(df=None, raw=None, data_type="audio",
                      format_name=ext.lstrip("."),
                      error="Install librosa for audio: pip install librosa")


# ── text / PDF loader ─────────────────────────────────────────────────────────

def _load_pdf_as_text(path: Path) -> LoadResult:
    """Load PDF and return text content + per-page DataFrame."""
    try:
        import pypdf
        libs = ["pypdf"]
        reader = pypdf.PdfReader(str(path))
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages.append({"page": i + 1, "text": text,
                          "n_chars": len(text), "n_words": len(text.split())})
        df = pd.DataFrame(pages)
        full_text = "\n\n".join(p["text"] for p in pages)
        summary = (
            f"Loaded **PDF** `{path.name}`.\n"
            f"{len(pages)} pages, {df['n_chars'].sum():,} chars total."
        )
        return LoadResult(df=df, raw=full_text, data_type="text",
                          format_name="pdf",
                          namespace={"text": full_text, "df": df},
                          summary=summary, libraries=libs)
    except ImportError:
        return LoadResult(df=None, raw=None, data_type="text",
                          format_name="pdf",
                          error="Install pypdf: pip install pypdf")


# ── generic fallback ──────────────────────────────────────────────────────────

def _load_unknown(path: Path) -> LoadResult:
    """Probe magic bytes; try common parsers; last resort: plain text."""
    # Read first 8 bytes for magic
    try:
        header = path.read_bytes()[:8]
    except Exception:
        header = b""

    # Magic byte signatures
    if header[:4] == b"\x89PNG":
        return _load_image(path)
    if header[:3] in (b"\xff\xd8\xff",):   # JPEG
        return _load_image(path)
    if header[:4] in (b"RIFF",) and b"WAVE" in path.read_bytes()[:12]:
        return _load_audio(path)
    if header[:4] == b"PAR1":              # Parquet magic
        return _load_tabular(path, ".parquet")
    if header[:4] == b"\x89HDF":          # HDF5 magic
        return _load_scientific(path, ".h5")

    # Try CSV
    try:
        df = pd.read_csv(path, low_memory=False)
        summary = (
            f"Loaded as **CSV** (unknown extension).\n"
            f"{df.shape[0]:,} rows × {df.shape[1]} columns."
        )
        return LoadResult(df=df, raw=df, data_type="tabular",
                          format_name="csv_auto", namespace={"df": df},
                          summary=summary, libraries=["pandas"])
    except Exception:
        pass

    # Plain text
    try:
        text = path.read_text(errors="replace")
        lines = text.splitlines()
        df = pd.DataFrame({"line_no": range(1, len(lines) + 1), "text": lines})
        summary = f"Loaded as **plain text**: {len(lines):,} lines."
        return LoadResult(df=df, raw=text, data_type="text",
                          format_name="text", namespace={"text": text, "df": df},
                          summary=summary, libraries=[])
    except Exception as exc:
        return LoadResult(df=None, raw=None, data_type="unknown",
                          format_name="unknown",
                          error=f"Could not load file: {exc}")


# ── helpers ───────────────────────────────────────────────────────────────────

def supported_extensions() -> list[str]:
    return sorted(ALL_SUPPORTED)


def describe_load_result(result: LoadResult) -> str:
    """One-paragraph description for the code agent context."""
    if result.error:
        return f"Load error: {result.error}"
    lines = [result.summary]
    if result.df is not None:
        lines.append(f"df.shape = {result.df.shape}")
    extras = [k for k in result.namespace if k not in ("df",)]
    if extras:
        lines.append(f"Extra kernel variables: {extras}")
    if result.libraries:
        lines.append(f"Libraries used: {result.libraries}")
    return "\n".join(lines)
