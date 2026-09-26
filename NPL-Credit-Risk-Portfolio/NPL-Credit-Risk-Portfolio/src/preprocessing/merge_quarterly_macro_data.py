# batch_merge_by_year_quarter_skip_first_row.py
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import re

def read_csv_noheader(p: Path, skip_first_row: bool = False) -> pd.DataFrame:
    if skip_first_row:
        return pd.read_csv(p, header=None, dtype=str, skiprows=1)
    return pd.read_csv(p, header=None, dtype=str)

def parse_year_series(s: pd.Series) -> pd.Series:
    def _y(x):
        if x is None: return pd.NA
        t = str(x)
        m = re.search(r'(19|20)\d{2}', t)
        if m: return int(m.group(0))
        try:
            v = int(float(t))
            return v if 1800 <= v <= 2100 else pd.NA
        except:
            return pd.NA
    return s.map(_y).astype("Int64")

def parse_quarter_series(s: pd.Series) -> pd.Series:
    def _q(x):
        if x is None: return pd.NA
        t = str(x).strip()
        m = re.search(r'[Qq]?\s*([1-4])\b', t)
        if m: return int(m.group(1))
        try:
            v = int(float(t))
            return v if 1 <= v <= 4 else pd.NA
        except:
            return pd.NA
    return s.map(_q).astype("Int64")

def main():
    # Save where this script is
    try:
        script_dir = Path(__file__).resolve().parent
    except NameError:
        script_dir = Path.cwd()

    root = tk.Tk(); root.withdraw()

    # 1) Select MASTER CSV (Year in col 2, Quarter in col 3). DROP ITS FIRST ROW.
    master_fp = filedialog.askopenfilename(
        title="Select MASTER .csv (Year in col 2, Quarter in col 3)",
        filetypes=[("CSV","*.csv")]
    )
    if not master_fp: 
        return
    master_path = Path(master_fp)
    master = read_csv_noheader(master_path, skip_first_row=True)  # <-- drop header row
    if master.shape[1] < 3:
        raise ValueError("Master needs at least 3 columns (Year in col 2, Quarter in col 3).")

    m_year    = parse_year_series(master.iloc[:, 1])
    m_quarter = parse_quarter_series(master.iloc[:, 2])
    # Keep only rows with valid keys in master
    m_mask = m_year.notna() & m_quarter.notna()
    master = master.loc[m_mask].reset_index(drop=True)
    m_year = m_year[m_mask]
    m_quarter = m_quarter[m_mask]
    master_keys = pd.MultiIndex.from_arrays([m_year, m_quarter], names=["year","quarter"])

    # 2) Select top folder (contains subfolders with .csv files)
    top_dir = filedialog.askdirectory(title="Select the top folder with subfolders of CSVs")
    if not top_dir: 
        return
    top_path = Path(top_dir)

    csvs = sorted(top_path.rglob("*.csv"))
    if not csvs:
        print("No .csv files found under:", top_path)
        return

    results = []
    for fp in csvs:
        # skip if it's the same file as master
        try:
            if fp.resolve() == master_path.resolve():
                continue
        except Exception:
            pass

        try:
            # OTHER CSV: drop its first row (header), then parse
            other = read_csv_noheader(fp, skip_first_row=True)
            if other.shape[1] < 3:
                print(f"SKIP (too few cols): {fp}")
                continue

            # Year in col 1 (A), Quarter in col 2 (B), data from col 3→
            o_year    = parse_year_series(other.iloc[:, 0])
            o_quarter = parse_quarter_series(other.iloc[:, 1])
            mask = o_year.notna() & o_quarter.notna()
            other_vals = other.loc[mask, 2:].copy()
            o_year_v = o_year[mask]
            o_quarter_v = o_quarter[mask]

            # Index & dedup by (year, quarter)
            other_vals.index = pd.MultiIndex.from_arrays([o_year_v, o_quarter_v], names=["year","quarter"])
            other_vals = other_vals.sort_index().groupby(level=["year","quarter"]).first()

            # Align to master's (year, quarter) order (left join)
            other_aligned = other_vals.reindex(master_keys).reset_index(drop=True)

            # Per-file merged table: master + aligned other cols
            merged_one = pd.concat([master.reset_index(drop=True), other_aligned], axis=1)
            results.append(merged_one)
            print(f"OK: {fp}")
        except Exception as e:
            print(f"ERROR: {fp} -> {e}")

    if not results:
        print("No tables produced.")
        return

    # Stack all per-file tables one under another
    combined = pd.concat(results, axis=0, ignore_index=True)

    out_path = script_dir / f"{top_path.name}_batch_merged.csv"
    combined.to_csv(out_path, header=False, index=False, encoding="utf-8")
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
