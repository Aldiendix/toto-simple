"""Simple local Toto widget: load txt series, predict, save txt."""

from __future__ import annotations

import os
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from toto_core import (
    MIN_POINTS,
    TotoSimple,
    app_root,
    has_local_weights,
    load_series_txt,
    model_dir,
    save_series_txt,
)

BG = "#eef7f8"
CARD = "#ffffff"
TEAL = "#0d9488"
DARK = "#134e4a"
ORANGE = "#ea580c"
MUTED = "#64748b"


class TotoSimpleApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Toto Simple")
        self.configure(bg=BG)
        self.minsize(720, 560)
        self.geometry("860x640")

        self.engine: TotoSimple | None = None
        self.series = None
        self.result = None
        self.source_path: Path | None = None
        self.busy = False

        self._style()
        self._build()
        self.after(200, self._boot_status)

    def _style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background=BG)
        style.configure("Card.TFrame", background=CARD)
        style.configure("TLabel", background=BG, foreground=DARK, font=("Segoe UI", 10))
        style.configure("Title.TLabel", background=BG, foreground=DARK, font=("Segoe UI", 18, "bold"))
        style.configure("Hint.TLabel", background=BG, foreground=MUTED, font=("Segoe UI", 9))
        style.configure("Status.TLabel", background=BG, foreground=TEAL, font=("Segoe UI", 9))
        style.configure("Card.TLabel", background=CARD, foreground=DARK, font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=8)
        style.configure("Accent.TButton", background=TEAL, foreground="#ffffff")
        style.map("Accent.TButton", background=[("active", "#0f766e")])
        style.configure("TSpinbox", padding=4)

    def _build(self) -> None:
        pad = ttk.Frame(self)
        pad.pack(fill="both", expand=True, padx=22, pady=18)

        ttk.Label(pad, text="Toto Simple", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            pad,
            text="Local CPU · no internet · load a txt series, predict several steps, save txt",
            style="Hint.TLabel",
        ).pack(anchor="w", pady=(2, 14))

        row = ttk.Frame(pad)
        row.pack(fill="x")
        ttk.Button(row, text="Load txt…", command=self.load_txt).pack(side="left")
        self.file_var = tk.StringVar(value="No file loaded")
        ttk.Label(row, textvariable=self.file_var).pack(side="left", padx=12)

        meta = ttk.Frame(pad)
        meta.pack(fill="x", pady=(12, 8))
        self.points_var = tk.StringVar(value="Points: 0")
        ttk.Label(meta, textvariable=self.points_var).pack(side="left")
        ttk.Label(meta, text="Predict steps").pack(side="left", padx=(24, 8))
        self.horizon = tk.Spinbox(meta, from_=1, to=512, width=6, font=("Segoe UI", 11))
        self.horizon.delete(0, "end")
        self.horizon.insert(0, "24")
        self.horizon.pack(side="left")
        self.predict_btn = ttk.Button(meta, text="Predict", style="Accent.TButton", command=self.predict)
        self.predict_btn.pack(side="left", padx=16)
        self.save_btn = ttk.Button(meta, text="Save prediction txt…", command=self.save_txt, state="disabled")
        self.save_btn.pack(side="left")

        self.canvas = tk.Canvas(pad, bg=CARD, highlightthickness=1, highlightbackground="#99f6e4", height=280)
        self.canvas.pack(fill="both", expand=True, pady=(8, 8))
        self.canvas.bind("<Configure>", lambda _e: self.draw_chart())

        self.preview = tk.Text(
            pad,
            height=7,
            font=("Consolas", 10),
            bg=CARD,
            fg=DARK,
            relief="flat",
            highlightthickness=1,
            highlightbackground="#d1f0ea",
        )
        self.preview.pack(fill="x")
        self.preview.insert("1.0", "Load a .txt file with one number per line (at least 32 values).")
        self.preview.configure(state="disabled")

        self.status = tk.StringVar(value="Ready · CPU")
        ttk.Label(pad, textvariable=self.status, style="Status.TLabel").pack(anchor="w", pady=(10, 0))

    def _boot_status(self) -> None:
        path = model_dir()
        if has_local_weights(path):
            self.status.set(f"Model ready · CPU · {path}")
        else:
            self.status.set("Model missing. Run setup.bat / download_model.py once.")

    def set_preview(self, text: str) -> None:
        self.preview.configure(state="normal")
        self.preview.delete("1.0", "end")
        self.preview.insert("1.0", text)
        self.preview.configure(state="disabled")

    def load_txt(self) -> None:
        start = str(self.source_path.parent if self.source_path else app_root())
        path = filedialog.askopenfilename(
            title="Open series txt",
            initialdir=start,
            filetypes=[("Text", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            series = load_series_txt(path)
        except Exception as exc:
            messagebox.showerror("Load failed", str(exc))
            return
        self.series = series
        self.result = None
        self.source_path = Path(path)
        self.file_var.set(self.source_path.name)
        self.points_var.set(f"Points: {len(series)}")
        self.save_btn.configure(state="disabled")
        tail = ", ".join(f"{v:.3f}" for v in series[-8:])
        self.set_preview(f"Loaded {len(series)} values from {self.source_path.name}\nLast 8: {tail}")
        self.status.set("Series loaded. Set steps and press Predict.")
        self.draw_chart()

    def predict(self) -> None:
        if self.busy:
            return
        if self.series is None:
            messagebox.showinfo("Toto Simple", "Load a txt series first.")
            return
        try:
            horizon = int(self.horizon.get())
        except ValueError:
            messagebox.showerror("Toto Simple", "Predict steps must be a number.")
            return
        if horizon < 1:
            messagebox.showerror("Toto Simple", "Predict steps must be at least 1.")
            return
        self.busy = True
        self.predict_btn.configure(state="disabled")
        self.status.set("Loading Toto on CPU and predicting…")
        series = self.series.copy()
        threading.Thread(target=self._predict_worker, args=(series, horizon), daemon=True).start()

    def _predict_worker(self, series, horizon: int) -> None:
        try:
            if self.engine is None:
                self.engine = TotoSimple()
            result = self.engine.forecast(series, horizon=horizon)
            self.after(0, lambda r=result: self._predict_done(r, None))
        except Exception as exc:
            self.after(0, lambda e=exc: self._predict_done(None, e))

    def _predict_done(self, result, error) -> None:
        self.busy = False
        self.predict_btn.configure(state="normal")
        if error is not None:
            self.status.set("Predict failed.")
            messagebox.showerror("Predict failed", str(error))
            return
        self.result = result
        self.save_btn.configure(state="normal")
        p50 = result["p50"]
        preview = "Predicted median (p50), one value per future step:\n"
        preview += "\n".join(f"{i + 1:4d}  {v:.6f}" for i, v in enumerate(p50[:24]))
        if len(p50) > 24:
            preview += f"\n… {len(p50) - 24} more steps"
        self.set_preview(preview)
        self.status.set(
            f"Predicted {result['horizon']} steps on CPU · context {result['context']} · ready to save"
        )
        self.draw_chart()

    def save_txt(self) -> None:
        if self.result is None:
            messagebox.showinfo("Toto Simple", "Predict first, then save.")
            return
        initial = "prediction.txt"
        if self.source_path is not None:
            initial = f"{self.source_path.stem}_pred.txt"
        path = filedialog.asksaveasfilename(
            title="Save prediction txt",
            defaultextension=".txt",
            initialfile=initial,
            filetypes=[("Text", "*.txt")],
        )
        if not path:
            return
        try:
            save_series_txt(
                path,
                self.result["p50"],
                comments=[
                    "Toto Simple prediction (median p50)",
                    f"device=cpu  horizon={self.result['horizon']}  context={self.result['context']}",
                    f"model={self.result['model']}",
                ],
            )
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
            return
        self.status.set(f"Saved {path}")
        messagebox.showinfo("Saved", f"Wrote {len(self.result['p50'])} predicted values to:\n{path}")

    def draw_chart(self) -> None:
        c = self.canvas
        c.delete("all")
        w = max(c.winfo_width(), 40)
        h = max(c.winfo_height(), 40)
        c.create_rectangle(0, 0, w, h, fill=CARD, outline="")
        if self.series is None:
            c.create_text(w / 2, h / 2, text="Load a txt series to see the chart", fill=MUTED, font=("Segoe UI", 11))
            return
        hist = self.series.astype(float)
        future = self.result["p50"].astype(float) if self.result is not None else None
        values = hist if future is None else np.concatenate([hist, future])
        ymin, ymax = float(values.min()), float(values.max())
        if ymin == ymax:
            ymin -= 1
            ymax += 1
        pad_y = (ymax - ymin) * 0.08
        ymin -= pad_y
        ymax += pad_y
        left, right, top, bottom = 36, w - 16, 16, h - 22

        def xy(i: int, n: int, v: float) -> tuple[float, float]:
            x = left + (right - left) * (i / max(n - 1, 1))
            y = bottom - (bottom - top) * ((v - ymin) / (ymax - ymin))
            return x, y

        n_hist = len(hist)
        n_all = n_hist + (0 if future is None else len(future))
        hist_pts = [xy(i, n_all, hist[i]) for i in range(n_hist)]
        c.create_line(*[p for xy_ in hist_pts for p in xy_], fill=DARK, width=2, smooth=True)
        if future is not None and len(future):
            fut_pts = [xy(n_hist - 1, n_all, hist[-1])] + [
                xy(n_hist + i, n_all, future[i]) for i in range(len(future))
            ]
            c.create_line(*[p for xy_ in fut_pts for p in xy_], fill=ORANGE, width=2, dash=(7, 4))
            now_x, _ = xy(n_hist - 1, n_all, hist[-1])
            c.create_line(now_x, top, now_x, bottom, fill=ORANGE, dash=(2, 3))
            c.create_text(now_x + 8, top + 10, anchor="w", text="now", fill=ORANGE, font=("Segoe UI", 8))
        c.create_text(left, h - 8, anchor="w", text="input", fill=DARK, font=("Segoe UI", 8))
        if future is not None:
            c.create_text(left + 50, h - 8, anchor="w", text="prediction", fill=ORANGE, font=("Segoe UI", 8))


def main() -> None:
    app = TotoSimpleApp()
    sample = app_root() / "sample.txt"
    if sample.exists():
        try:
            app.series = load_series_txt(sample)
            app.source_path = sample
            app.file_var.set(sample.name)
            app.points_var.set(f"Points: {len(app.series)}")
            app.set_preview(
                f"Sample series loaded ({len(app.series)} points). Press Predict, then Save prediction txt."
            )
            app.draw_chart()
        except Exception:
            pass
    app.mainloop()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        dest = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
        log = dest / "toto_simple_error.log"
        try:
            import traceback

            log.write_text(traceback.format_exc(), encoding="utf-8")
        except Exception:
            pass
        try:
            messagebox.showerror("Toto Simple", f"{exc}\n\nSee {log}")
        except Exception:
            pass
        raise
