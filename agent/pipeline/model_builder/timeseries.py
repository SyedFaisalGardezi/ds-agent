"""TimeSeriesModelBuilder — Prophet + four torch-based sequence models.

- ``fit_prophet`` — lazy import; clean error if the optional dep is missing.
- ``fit_lstm``    — single-layer LSTM regressor trained on sliding windows.
- ``fit_tcn``     — dilated 1-D conv "temporal convolutional network".
- ``fit_nbeats``  — tiny N-BEATS-style trend/seasonal basis stacks.
- ``fit_nhits``   — N-HiTS-style multi-rate pooling + interpolation stacks.

All four torch models share a ``_fit_windowed`` helper. Config defaults keep
training small so unit tests finish in seconds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from agent.core.types import ToolResult


def _err(explanation: str, error: str) -> ToolResult:
    return ToolResult(status="error", data=None, explanation=explanation,
                      math_trace="", error=error)


def _make_windows(y: np.ndarray, L: int, H: int) -> tuple[np.ndarray, np.ndarray]:
    """Sliding windows: (N, L) inputs, (N, H) horizons."""
    n = len(y) - L - H + 1
    if n <= 0:
        raise ValueError(f"series length {len(y)} too short for L={L}, H={H}")
    idx = np.arange(L)[None, :] + np.arange(n)[:, None]
    X = y[idx]
    Y = y[np.arange(H)[None, :] + np.arange(L, L + n)[:, None]]
    return X.astype(np.float32), Y.astype(np.float32)


class TimeSeriesModelBuilder:
    """Fit one of Prophet / LSTM / TCN / N-BEATS / N-HiTS."""

    # ---- Prophet ----------------------------------------------------- #

    def fit_prophet(self, df: pd.DataFrame, config: dict | None = None) -> ToolResult:
        try:
            from prophet import Prophet  # type: ignore
        except Exception as exc:  # noqa: BLE001
            return _err("prophet not installed", f"{type(exc).__name__}: {exc}")
        cfg = config or {}
        if not {"ds", "y"}.issubset(df.columns):
            return _err("prophet requires columns 'ds' and 'y'", "KeyError")
        m = Prophet(**{k: v for k, v in cfg.items() if k != "periods"})
        m.fit(df[["ds", "y"]])
        periods = int(cfg.get("periods", 10))
        future = m.make_future_dataframe(periods=periods)
        fc = m.predict(future)
        return ToolResult(
            status="ok",
            data={"model": m, "forecast": fc, "periods": periods},
            explanation=f"Prophet fit and forecast for {periods} step(s).",
            math_trace="y(t) = g(t) + s(t) + h(t) + ε_t.",
        )

    # ---- shared torch path ------------------------------------------ #

    def _fit_windowed(self, df: pd.DataFrame, config: dict,
                      build_model, name: str, math_trace: str) -> ToolResult:
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
        except Exception as exc:  # noqa: BLE001
            return _err("torch not installed", f"{type(exc).__name__}: {exc}")
        cfg = config or {}
        target = cfg.get("target", "y")
        if target not in df.columns:
            return _err(f"target {target!r} missing", "KeyError")
        y = df[target].astype(float).to_numpy()
        L = int(cfg.get("lookback", 10))
        H = int(cfg.get("horizon", 1))
        try:
            X, Y = _make_windows(y, L, H)
        except ValueError as exc:
            return _err(str(exc), "ValueError")
        epochs = int(cfg.get("epochs", 20))
        lr = float(cfg.get("lr", 1e-2))
        torch.manual_seed(int(cfg.get("seed", 0)))
        model = build_model(L=L, H=H, cfg=cfg)
        opt = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        X_t = torch.from_numpy(X)
        Y_t = torch.from_numpy(Y)
        losses: list[float] = []
        for _ in range(epochs):
            model.train()
            opt.zero_grad()
            out = model(X_t)
            loss = loss_fn(out, Y_t)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            pred = model(X_t).cpu().numpy()
        mse = float(((pred - Y) ** 2).mean())
        return ToolResult(
            status="ok",
            data={"model": model, "lookback": L, "horizon": H,
                  "loss_history": losses,
                  "metrics": {"train_mse": mse, "final_loss": losses[-1]},
                  "architecture": name},
            explanation=f"Trained {name} ({epochs} epochs); final_mse={mse:.6f}.",
            math_trace=math_trace,
        )

    # ---- LSTM -------------------------------------------------------- #

    def fit_lstm(self, df: pd.DataFrame, config: dict | None = None) -> ToolResult:
        import torch.nn as nn

        def _build(L: int, H: int, cfg: dict):
            hidden = int(cfg.get("hidden", 32))

            class LSTMNet(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.lstm = nn.LSTM(input_size=1, hidden_size=hidden,
                                        num_layers=1, batch_first=True)
                    self.head = nn.Linear(hidden, H)

                def forward(self, x):
                    x = x.unsqueeze(-1)  # (N, L, 1)
                    out, _ = self.lstm(x)
                    return self.head(out[:, -1, :])

            return LSTMNet()

        return self._fit_windowed(
            df, config or {}, _build, name="LSTM",
            math_trace=("h_t, c_t = LSTM(x_t, h_{t−1}, c_{t−1}); "
                        "ŷ_{t+1..t+H} = W h_L + b."),
        )

    # ---- TCN --------------------------------------------------------- #

    def fit_tcn(self, df: pd.DataFrame, config: dict | None = None) -> ToolResult:
        import torch.nn as nn

        def _build(L: int, H: int, cfg: dict):
            channels = int(cfg.get("channels", 16))
            kernel = 3

            class TCN(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.c1 = nn.Conv1d(1, channels, kernel, padding=1, dilation=1)
                    self.c2 = nn.Conv1d(channels, channels, kernel, padding=2, dilation=2)
                    self.c3 = nn.Conv1d(channels, channels, kernel, padding=4, dilation=4)
                    self.act = nn.ReLU()
                    self.head = nn.Linear(channels, H)

                def forward(self, x):
                    z = x.unsqueeze(1)  # (N, 1, L)
                    z = self.act(self.c1(z))
                    z = self.act(self.c2(z))
                    z = self.act(self.c3(z))
                    return self.head(z[:, :, -1])

            return TCN()

        return self._fit_windowed(
            df, config or {}, _build, name="TCN",
            math_trace=("y = Conv1D_{dilated}(x); receptive field grows "
                        "exponentially with depth."),
        )

    # ---- N-BEATS ----------------------------------------------------- #

    def fit_nbeats(self, df: pd.DataFrame, config: dict | None = None) -> ToolResult:
        import torch
        import torch.nn as nn

        def _build(L: int, H: int, cfg: dict):
            hidden = int(cfg.get("hidden", 32))
            stacks = int(cfg.get("stacks", 2))

            class Block(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.fc = nn.Sequential(
                        nn.Linear(L, hidden), nn.ReLU(),
                        nn.Linear(hidden, hidden), nn.ReLU(),
                    )
                    self.back = nn.Linear(hidden, L)
                    self.fore = nn.Linear(hidden, H)

                def forward(self, x):
                    h = self.fc(x)
                    return self.back(h), self.fore(h)

            class NBEATS(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.blocks = nn.ModuleList([Block() for _ in range(stacks)])

                def forward(self, x):
                    residual = x
                    forecast = torch.zeros(x.shape[0], H, device=x.device)
                    for b in self.blocks:
                        back, fore = b(residual)
                        residual = residual - back
                        forecast = forecast + fore
                    return forecast

            return NBEATS()

        return self._fit_windowed(
            df, config or {}, _build, name="N-BEATS",
            math_trace=("y_f = Σ_s f_s(residual_s); "
                        "residual_s = x − Σ_{s'<s} b_{s'}."),
        )

    # ---- N-HiTS ------------------------------------------------------ #

    def fit_nhits(self, df: pd.DataFrame, config: dict | None = None) -> ToolResult:
        import torch.nn as nn
        import torch.nn.functional as F

        def _build(L: int, H: int, cfg: dict):
            hidden = int(cfg.get("hidden", 32))
            pools = cfg.get("pool_sizes", (1, 2, 4))

            class Block(nn.Module):
                def __init__(self, pool: int) -> None:
                    super().__init__()
                    self.pool = pool
                    pooled_len = max(1, L // pool)
                    self.fc = nn.Sequential(
                        nn.Linear(pooled_len, hidden), nn.ReLU(),
                        nn.Linear(hidden, hidden), nn.ReLU(),
                    )
                    self.fore = nn.Linear(hidden, max(1, H // pool))

                def forward(self, x):
                    z = F.avg_pool1d(x.unsqueeze(1), kernel_size=self.pool,
                                     stride=self.pool, ceil_mode=True).squeeze(1)
                    h = self.fc(z)
                    coarse = self.fore(h)
                    up = F.interpolate(coarse.unsqueeze(1), size=H, mode="linear",
                                       align_corners=False).squeeze(1)
                    return up

            class NHITS(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.blocks = nn.ModuleList([Block(p) for p in pools])

                def forward(self, x):
                    out = 0
                    for b in self.blocks:
                        out = out + b(x)
                    return out

            return NHITS()

        return self._fit_windowed(
            df, config or {}, _build, name="N-HiTS",
            math_trace=("ŷ = Σ_r interp(MLP(pool_r(x))); "
                        "multi-rate frequency hierarchy."),
        )
