"""CPU-only Toto 2.0: load a 1D txt series, forecast, save txt."""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

warnings.filterwarnings(
    "ignore",
    message="Using `json`-module for json-handling",
    category=UserWarning,
)

MODEL_ID = "Datadog/Toto-2.0-22m"
MEDIAN_INDEX = 4
P10_INDEX = 0
P90_INDEX = 8
MIN_POINTS = 32
MAX_CONTEXT = 512
PATCH_SIZE = 32


def app_root() -> Path:
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        exe_dir = Path(sys.executable).resolve().parent
        if meipass:
            bundled = Path(meipass)
            if (bundled / "models" / "Toto-2.0-22m" / "config.json").exists():
                return bundled
        return exe_dir
    return Path(__file__).resolve().parent


def model_dir() -> Path:
    root = app_root()
    exe_dir = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else root
    candidates = [
        exe_dir / "models" / "Toto-2.0-22m",
        root / "models" / "Toto-2.0-22m",
        Path(r"e:\Work\factory-toto\models\Toto-2.0-22m"),
        Path(__file__).resolve().parent.parent / "factory-toto" / "models" / "Toto-2.0-22m",
    ]
    for path in candidates:
        if (path / "config.json").exists() and (
            list(path.glob("*.safetensors")) or list(path.glob("*.bin"))
        ):
            return path
    return root / "models" / "Toto-2.0-22m"


def has_local_weights(path: Path | None = None) -> bool:
    target = path or model_dir()
    if not (target / "config.json").exists():
        return False
    return bool(list(target.glob("*.safetensors")) or list(target.glob("*.bin")))


def load_series_txt(path: str | Path) -> np.ndarray:
    text = Path(path).read_text(encoding="utf-8-sig")
    values: list[float] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        for token in line.replace(";", " ").replace(",", " ").split():
            try:
                values.append(float(token))
            except ValueError:
                continue
    if len(values) < MIN_POINTS:
        raise ValueError(
            f"Need at least {MIN_POINTS} numbers in the txt file. Found {len(values)}."
        )
    arr = np.asarray(values, dtype=np.float32)
    if not np.isfinite(arr).all():
        raise ValueError("Series contains NaN or inf.")
    return arr


def save_series_txt(path: str | Path, values: np.ndarray, comments: list[str] | None = None) -> None:
    lines: list[str] = []
    for note in comments or []:
        lines.append("# " + note.lstrip("# ").rstrip())
    for value in np.asarray(values).reshape(-1):
        lines.append(f"{float(value):.6f}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


class TotoSimple:
    def __init__(self, weights: Path | None = None) -> None:
        import torch

        self.torch = torch
        self.device = torch.device("cpu")
        self.weights = Path(weights or model_dir())
        self.model = None

    def load(self) -> None:
        if self.model is not None:
            return
        if not has_local_weights(self.weights):
            raise FileNotFoundError(
                f"Local Toto weights not found at {self.weights}. "
                "This repo should already include models/Toto-2.0-22m "
                "(config.json and model.safetensors). No Hugging Face download."
            )
        from toto2 import Toto2Model

        self.model = Toto2Model.from_pretrained(
            str(self.weights),
            local_files_only=True,
            map_location="cpu",
        )
        self.model = self.model.to(self.device).eval()

    def forecast(self, series: np.ndarray, horizon: int, context: int = MAX_CONTEXT) -> dict:
        self.load()
        assert self.model is not None
        horizon = int(horizon)
        if horizon < 1:
            raise ValueError("Predict steps must be at least 1.")
        ctx = np.asarray(series, dtype=np.float32).reshape(-1)
        ctx = ctx[-int(context) :]
        values = ctx.reshape(1, -1)
        mask = np.ones_like(values, dtype=bool)
        patch_size = int(getattr(self.model.config, "patch_size", PATCH_SIZE))
        remainder = values.shape[-1] % patch_size
        if remainder:
            pad = patch_size - remainder
            values = np.pad(values, ((0, 0), (pad, 0)))
            mask = np.pad(mask, ((0, 0), (pad, 0)), constant_values=False)
        target = self.torch.from_numpy(values).unsqueeze(0).to(self.device)
        target_mask = self.torch.from_numpy(mask).unsqueeze(0).to(self.device)
        n_var = target.shape[1]
        series_ids = self.torch.zeros(1, n_var, dtype=self.torch.long, device=self.device)
        decode_block_size = None if horizon <= 512 else 768
        with self.torch.inference_mode():
            quantiles = self.model.forecast(
                {
                    "target": target,
                    "target_mask": target_mask,
                    "series_ids": series_ids,
                },
                horizon=horizon,
                decode_block_size=decode_block_size,
                has_missing_values=not bool(mask.all()),
            )
        q = quantiles[:, 0].detach().cpu().numpy()
        return {
            "p10": q[P10_INDEX, 0],
            "p50": q[MEDIAN_INDEX, 0],
            "p90": q[P90_INDEX, 0],
            "device": "cpu",
            "model": str(self.weights),
            "context": int(ctx.size),
            "horizon": horizon,
        }
