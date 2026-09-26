# batch_clean_city_country_bank_transform.py
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import pandas as pd

def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, header=None, dtype=str)
    elif path.suffix.lower() in (".xlsx", ".xls"):
        # pip install openpyxl
        return pd.read_excel(path, header=None, dtype=str, engine="openpyxl")
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

def process_df(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure at least 27 rows (0..26) and 4 cols (index 3 = column D)
    if df.shape[0] < 27:
        df = df.reindex(range(27))
    if df.shape[1] < 4:
        df = df.reindex(columns=range(4))

    # B1 (bank name) = row 0, col 1
    bank_name = "" if df.shape[0] < 1 or df.shape[1] < 2 else (str(df.iat[0,1]) if df.iat[0,1] is not None else "")
    # B2 (city, country) = row 1, col 1 → take last comma piece
    b2 = "" if df.shape[0] < 2 or df.shape[1] < 2 else (str(df.iat[1,1]) if df.iat[1,1] is not None else "")
    parts = [p.strip() for p in b2.split(",") if str(p).strip()]
    country = parts[-1] if parts else ""

    # Fill row 26 (index 25) and row 27 (index 26) from column D (index 3) to end
    start_col = 3
    last_col = df.shape[1] - 1
    if last_col >= start_col:
        df.iloc[25, start_col:last_col+1] = country
        df.iloc[26, start_col:last_col+1] = bank_name

    # Drop columns B & C (1,2) and first two rows (0,1), then transpose
    df = df.drop(columns=[c for c in (1,2) if c < df.shape[1]], errors="ignore")
    df = df.drop(index=[i for i in (0,1) if i < len(df)], errors="ignore")
    return df.T

def main():
    # Where to save: folder of this script (fallback to CWD)
    try:
        script_dir = Path(__file__).resolve().parent
    except NameError:
        script_dir = Path.cwd()

    root = tk.Tk(); root.withdraw()
    top = filedialog.askdirectory(title="Select the top folder containing subfolders with files")
    if not top:
        print("No folder selected."); return
    top_path = Path(top)

    files = [p for p in top_path.rglob("*") if p.suffix.lower() in (".csv", ".xlsx", ".xls")]
    if not files:
        print("No .csv/.xlsx files found."); return

    print(f"Found {len(files)} file(s). Processing...")

    for fp in files:
        try:
            df = read_table(fp)
            out_df = process_df(df)

            # Determine output folder structure:
            # <script_dir>/<first_level_folder>_cleanedup/(preserve deeper subdirs)/<file>_cleanedup.csv
            rel = fp.relative_to(top_path)
            parts = rel.parts
            if len(parts) >= 2:
                first_level = parts[0]
                sub_rel_dirs = Path(*parts[1:-1]) if len(parts) > 2 else Path()
                out_dir = script_dir / f"{first_level}_cleanedup" / sub_rel_dirs
            else:
                # File directly under selected folder
                out_dir = script_dir / f"{top_path.name}_cleanedup"

            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"{fp.stem}_cleanedup.csv"
            out_df.to_csv(out_file, header=False, index=False, encoding="utf-8")
            print(f"OK: {fp} -> {out_file}")
        except Exception as e:
            print(f"ERROR: {fp} -> {e}")

    print("Done.")

if __name__ == "__main__":
    main()