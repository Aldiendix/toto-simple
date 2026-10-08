# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all, collect_submodules, copy_metadata

datas = [
    ("models/Toto-2.0-22m/config.json", "models/Toto-2.0-22m"),
    ("models/Toto-2.0-22m/model.safetensors", "models/Toto-2.0-22m"),
    ("sample.txt", "."),
]
binaries = []
hiddenimports = collect_submodules("toto2")
for pkg in (
    "toto2",
    "gluonts",
    "dd_unit_scaling",
    "einops",
    "jaxtyping",
    "safetensors",
    "lightning",
    "lightning_fabric",
    "lightning_utilities",
    "pytorch_lightning",
    "typeguard",
    "matplotlib",
):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        pass

for meta in (
    "typeguard",
    "jaxtyping",
    "toto-2",
    "toto-models",
    "lightning",
    "lightning-utilities",
    "gluonts",
    "einops",
    "huggingface_hub",
):
    try:
        datas += copy_metadata(meta)
    except Exception:
        pass

a = Analysis(
    ["app.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports
    + [
        "pandas",
        "numpy",
        "torch",
        "huggingface_hub",
        "lightning",
        "lightning.pytorch",
        "pytorch_lightning",
        "typeguard",
        "jaxtyping",
        "matplotlib",
        "matplotlib.pyplot",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=["pyi_rth_pkgmeta.py"],
    excludes=[
        "IPython",
        "notebook",
        "cuda",
        "nvidia",
        "triton",
        "flask",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="TotoSimple",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="TotoSimple",
)

import shutil
from pathlib import Path

src = Path("dist") / "TotoSimple"
dst = Path("release") / "TotoSimple"
if src.exists():
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print("Copied portable app to", dst.resolve())
    except OSError as exc:
        print("Skip release copy:", exc)

