"""Verify local Toto-2.0-22m weights. No Hugging Face."""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "models" / "Toto-2.0-22m"
LOCAL_CANDIDATES = [
    ROOT.parent / "factory-toto" / "models" / "Toto-2.0-22m",
    Path(r"e:\Work\factory-toto\models\Toto-2.0-22m"),
]
NEEDED = ("config.json", "model.safetensors")


def _complete(path: Path) -> bool:
    return (path / "config.json").exists() and (
        list(path.glob("*.safetensors")) or list(path.glob("*.bin"))
    )


def _copy_weights(src: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name in NEEDED:
        file = src / name
        if file.exists():
            shutil.copy2(file, dest / name)


def write_sample_txt() -> Path:
    path = ROOT / "sample.txt"
    if path.exists():
        return path
    lines = ["# Toto Simple sample series. One number per line.", "# 256 points, sine + slow trend."]
    for i in range(256):
        value = 40.0 + 0.02 * i + 3.5 * math.sin(i / 18.0) + 0.4 * math.sin(i / 5.0)
        lines.append(f"{value:.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def download(force: bool = False) -> Path:
    TARGET.mkdir(parents=True, exist_ok=True)
    write_sample_txt()
    if _complete(TARGET) and not force:
        print(f"Local weights ready at {TARGET}. Offline. No Hugging Face.")
        return TARGET
    for src in LOCAL_CANDIDATES:
        if _complete(src):
            print(f"Copying local weights from {src}")
            _copy_weights(src, TARGET)
            print(f"Ready at {TARGET}")
            return TARGET
    raise FileNotFoundError(
        f"Toto weights not found at {TARGET}. "
        "This repo should include models/Toto-2.0-22m/config.json and model.safetensors. "
        "This app does not download from Hugging Face."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify local Toto weights (offline, no Hugging Face).")
    parser.add_argument("--force", action="store_true", help="Copy again from a local factory-toto folder if present.")
    args = parser.parse_args()
    download(force=args.force)
