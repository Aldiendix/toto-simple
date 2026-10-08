# Toto Simple

Local CPU widget for **Datadog Toto 2.0**. Load a `.txt` time series, predict several steps, save a `.txt`.

No GPU. No Hugging Face. Toto weights are in `models\Toto-2.0-22m`. Double-click `START_TOTO.bat` or `TotoSimple.exe` when it is built.

## First-time setup (once)

```bat
cd /d e:\Work\toto-simple
setup.bat
```

That creates a CPU-only venv, installs packages **without CUDA**, and checks local `Toto-2.0-22m` weights (~84 MB) in `models\`.

## Run the widget

```bat
run.bat
```

Or:

```bat
.venv\Scripts\python.exe app.py
```

1. **Load txt…** — one number per line (at least 32). Comma or space separated is also fine. Lines starting with `#` are ignored.
2. Set **Predict steps** (default 24).
3. **Predict** — Toto runs on CPU.
4. **Save prediction txt…** — writes one predicted median per line.

`sample.txt` loads automatically if present.

## Build the .exe

```bat
build_exe.bat
```

Then double-click:

`e:\Work\toto-simple\dist\TotoSimple\TotoSimple.exe`

Keep `_internal` next to the exe. That folder is the CPU runtime + the local model.

## Input txt format

```
# optional comment
40.1
40.3
40.2
```

Output is the same shape: comments, then one predicted value per line.
