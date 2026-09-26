# filter_panel_by_year_and_minrows_csv.py
from __future__ import annotations
import sys
import re
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd

# ---------- helpers ----------

def guess_col(cols, patterns):
    lowmap = {c.lower(): c for c in cols}
    for pat in patterns:
        rx = re.compile(pat)
        for lc, orig in lowmap.items():
            if rx.search(lc):
                return orig
    return None

def normalize_quarter(series):
    s = series.copy().astype(str)
    # already 'Q1'..'Q4'
    mask_q = s.str.fullmatch(r"Q[1-4]", na=False)
    s.loc[mask_q] = s.loc[mask_q]
    # numeric 1..4
    mask_num = s.str.fullmatch(r"[1-4]", na=False)
    s.loc[mask_num] = "Q" + s.loc[mask_num]
    # embedded like 2021Q3, 2021-Q3
    extracted = s.str.extract(r"(?:^|\D)(Q[1-4])(?:\D|$)", expand=False)
    s = s.where(~extracted.notna(), extracted)
    return s

def full_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(how="any")

def load_csv(path: Path) -> pd.DataFrame:
    # sep=None + engine='python' will sniff commas/semicolons/tabs
    return pd.read_csv(path, sep=None, engine="python")

# ---------- app ----------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Panel Filter (CSV): Year Range & Min Rows")
        self.geometry("840x520")
        self.df_orig = None
        self.df_stats_view = None
        self.file_path: Path | None = None

        frm = tk.Frame(self, pady=8, padx=8); frm.pack(fill="x")
        self.btn_file = tk.Button(frm, text="1) Select CSV", command=self.on_select_file)
        self.btn_file.grid(row=0, column=0, padx=4, pady=4, sticky="w")

        tk.Label(frm, text="Start Year").grid(row=0, column=1, padx=4, sticky="e")
        self.ent_start = tk.Entry(frm, width=8); self.ent_start.grid(row=0, column=2, padx=4)
        tk.Label(frm, text="End Year").grid(row=0, column=3, padx=4, sticky="e")
        self.ent_end = tk.Entry(frm, width=8); self.ent_end.grid(row=0, column=4, padx=4)
        tk.Label(frm, text="Min rows per bank").grid(row=0, column=5, padx=4, sticky="e")
        self.ent_min = tk.Entry(frm, width=8); self.ent_min.grid(row=0, column=6, padx=4)

        self.btn_stats = tk.Button(frm, text="2) Compute stats", command=self.on_compute, state="disabled")
        self.btn_stats.grid(row=0, column=7, padx=6)
        self.btn_export = tk.Button(frm, text="3) Create filtered CSV/XLSX", command=self.on_export, state="disabled")
        self.btn_export.grid(row=0, column=8, padx=6)

        self.txt = tk.Text(self, height=24); self.txt.pack(fill="both", expand=True, padx=8, pady=8)
        self.txt.configure(state="disabled")
        self._log("Steps: Select CSV → enter years & min → Compute stats → Export.\n")

    def _log(self, s: str):
        self.txt.configure(state="normal")
        self.txt.insert("end", s + ("" if s.endswith("\n") else "\n"))
        self.txt.see("end")
        self.txt.configure(state="disabled")

    def on_select_file(self):
        path = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not path: return
        try:
            df = load_csv(Path(path))
        except Exception as e:
            messagebox.showerror("Load error", f"Could not read CSV:\n{e}")
            return
        self.df_orig = df
        self.file_path = Path(path)
        self.btn_stats.config(state="normal")
        self._log(f"Loaded: {self.file_path.name}  (rows={len(df)}, cols={len(df.columns)})")

    def _prepare(self):
        if self.df_orig is None:
            raise RuntimeError("Load a CSV first.")
        cols = list(self.df_orig.columns)
        year_col = guess_col(cols, [r"^year$", r"\byear\b"])
        qtr_col  = guess_col(cols, [r"^quarter$", r"\bquarter\b", r"\bqtr\b", r"\bq\b"])
        bank_col = guess_col(cols, [r"\bbank\b", r"\bbank[_ ]*name\b", r"\binstitution\b", r"\bentity\b"])
        missing = [name for name, col in [("Year", year_col), ("Quarter", qtr_col), ("Bank", bank_col)] if col is None]
        if missing:
            raise RuntimeError(f"Missing required columns: {', '.join(missing)}.\nFound: {cols}")
        df = self.df_orig.copy()
        df[year_col] = pd.to_numeric(df[year_col], errors="coerce").astype("Int64")
        df[qtr_col] = normalize_quarter(df[qtr_col])
        return df, year_col, qtr_col, bank_col

    def on_compute(self):
        try:
            start_y = int(self.ent_start.get()); end_y = int(self.ent_end.get()); min_rows = int(self.ent_min.get())
        except ValueError:
            messagebox.showwarning("Input", "Years and minimum must be integers."); return
        if self.df_orig is None:
            messagebox.showinfo("Info", "Please select a CSV first."); return
        if end_y < start_y:
            messagebox.showwarning("Input", "End year must be ≥ start year."); return

        try:
            df, year_col, qtr_col, bank_col = self._prepare()
        except Exception as e:
            messagebox.showerror("Error", str(e)); return

        df_time = df[(df[year_col].notna()) & (df[year_col] >= start_y) & (df[year_col] <= end_y)]
        df_full = df_time.dropna(how="any")
        cnt_per_bank = df_full.groupby(bank_col).size().rename("rows").reset_index()
        keep_banks = set(cnt_per_bank[cnt_per_bank["rows"] >= min_rows][bank_col])
        df_kept = df_full[df_full[bank_col].isin(keep_banks)]

        total_full_rows = len(df_full)
        bank_count_all  = df_full[bank_col].nunique()
        bank_count_kept = len(keep_banks)
        per_year = df_kept.groupby(year_col).size().sort_index()
        per_qtr  = df_kept.groupby(qtr_col).size().reindex(["Q1","Q2","Q3","Q4"]).fillna(0).astype(int)

        self.df_stats_view = {
            "params": (start_y, end_y, min_rows),
            "df_kept": df_kept,
            "year_col": year_col,
            "qtr_col": qtr_col,
            "bank_col": bank_col
        }

        self._log("---- SUMMARY (FULL rows only) ----")
        self._log(f"Year range: {start_y}-{end_y} | Min rows/bank: {min_rows}")
        self._log(f"Full rows in window (pre min-filter): {total_full_rows}")
        self._log(f"Unique banks in window (pre min-filter): {bank_count_all}")
        self._log(f"Kept banks (≥ {min_rows} full rows): {bank_count_kept}")
        self._log("\nFull rows per YEAR (kept banks):")
        for y, n in per_year.items(): self._log(f"  {int(y)}: {int(n)}")
        self._log("\nFull rows per QUARTER (kept banks):")
        for q, n in per_qtr.items(): self._log(f"  {q}: {int(n)}")
        self.btn_export.config(state="normal")

    def on_export(self):
        if not self.df_stats_view:
            messagebox.showinfo("Info", "Run 'Compute stats' first."); return
        df_kept = self.df_stats_view["df_kept"]
        start_y, end_y, min_rows = self.df_stats_view["params"]
        if df_kept.empty:
            messagebox.showwarning("No data", "No rows to export with current filters."); return

        out_dir = Path(__file__).parent if "__file__" in globals() else Path.cwd()
        base = f"filtered_{start_y}_{end_y}_min{min_rows}"
        out_csv = out_dir / f"{base}.csv"
        out_xlsx = out_dir / f"{base}.xlsx"

        try:
            df_kept.to_csv(out_csv, index=False)
            with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as xw:
                df_kept.to_excel(xw, index=False, sheet_name="Filtered")
        except Exception as e:
            messagebox.showerror("Export error", f"Failed to write files:\n{e}"); return

        self._log(f"\nExported:\n  {out_csv}\n  {out_xlsx}")
        messagebox.showinfo("Done", f"Created:\n{out_csv.name}\n{out_xlsx.name}")

if __name__ == "__main__":
    try:
        import pandas as pd  # ensure pandas is present
    except Exception:
        print("This script requires pandas. Install with: pip install pandas xlsxwriter")
        sys.exit(1)
    App().mainloop()
