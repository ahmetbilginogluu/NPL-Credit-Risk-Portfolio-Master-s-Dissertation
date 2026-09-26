#!/usr/bin/env python3
import argparse, os, sys
from typing import List, Tuple, Union
import pandas as pd

def _maybe_parse_time(s: pd.Series) -> pd.Series:
    # Try numeric
    try:
        return pd.to_numeric(s, errors="raise")
    except Exception:
        pass
    # Try datetime
    try:
        return pd.to_datetime(s, errors="raise", dayfirst=False)
    except Exception:
        pass
    # Fallback (string/object)
    return s

def split_by_time_period(
    df: pd.DataFrame, time_col: Union[int, str], ratios: List[float]
) -> Tuple[str, List[Tuple[float, pd.DataFrame, pd.DataFrame]]]:
    """Return (resolved_time_col_name, splits)."""
    # Resolve time column name
    if isinstance(time_col, int):
        time_col_name = df.columns[time_col]
    else:
        time_col_name = time_col

    if time_col_name not in df.columns:
        raise ValueError(f"Time column '{time_col_name}' not found in CSV.")

    # Sort chronologically by a robust key
    sortable = _maybe_parse_time(df[time_col_name])
    df_sorted = df.assign(_sortkey=sortable).sort_values("_sortkey", kind="mergesort")
    periods = pd.unique(df_sorted[time_col_name])

    if len(periods) < 2:
        raise ValueError("Not enough distinct time periods to split.")

    results: List[Tuple[float, pd.DataFrame, pd.DataFrame]] = []
    for r in ratios:
        if not (0.0 < r < 1.0):
            raise ValueError(f"Invalid split ratio {r}. Must be between 0 and 1.")

        cutoff = int(len(periods) * r)
        cutoff = max(1, min(cutoff, len(periods) - 1))  # avoid empty sides

        train_periods = set(periods[:cutoff])
        test_periods = set(periods[cutoff:])

        train_df = df_sorted[df_sorted[time_col_name].isin(train_periods)].drop(columns=["_sortkey"])
        test_df  = df_sorted[df_sorted[time_col_name].isin(test_periods)].drop(columns=["_sortkey"])

        results.append((r, train_df, test_df))

    return time_col_name, results

def save_splits(
    base_path: str,
    base_name: str,
    time_col_name: str,
    splits: List[Tuple[float, pd.DataFrame, pd.DataFrame]],
):
    for r, train_df, test_df in splits:
        pct = int(round(r * 100))
        out_train = os.path.join(base_path, f"{base_name}_{pct:02d}-{100-pct:02d}_train.csv")
        out_test  = os.path.join(base_path, f"{base_name}_{pct:02d}-{100-pct:02d}_test.csv")

        train_df.to_csv(out_train, index=False)
        test_df.to_csv(out_test, index=False)

        print(f"\nSplit {pct}:{100-pct} using time column '{time_col_name}':")
        print(f"  Periods -> train: {train_df[time_col_name].nunique()}, test: {test_df[time_col_name].nunique()}")
        print(f"  Rows    -> train: {len(train_df):,}, test: {len(test_df):,}")
        print(f"  Files   -> {out_train}\n            {out_test}")

def pick_file_with_dialog() -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return ""
    root = tk.Tk(); root.withdraw(); root.update()
    path = filedialog.askopenfilename(title="Select CSV file",
                                      filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
    root.destroy()
    return path or ""

def read_csv_safely(path: str) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path)

def main():
    parser = argparse.ArgumentParser(description="Create time-period-aware splits from CSV.")
    parser.add_argument("csv", nargs="?", help="Path to input CSV. If omitted, a file picker will open.")
    parser.add_argument("--time-col", default=0,
                        help="Time column (index or name). Default: 0 (first column / Column A).")
    args = parser.parse_args()

    csv_path = args.csv or pick_file_with_dialog()
    if not csv_path:
        print("No file selected. Exiting."); sys.exit(1)
    if not os.path.isfile(csv_path):
        print(f"File not found: {csv_path}"); sys.exit(1)

    # Interpret time column arg
    time_col_arg: Union[int, str]
    try:
        time_col_arg = int(args.time_col)
    except ValueError:
        time_col_arg = args.time_col  # name

    df = read_csv_safely(csv_path)
    if df.shape[1] < 1:
        print("CSV appears to be empty."); sys.exit(1)

    base_dir = os.path.dirname(csv_path)
    base_name = os.path.splitext(os.path.basename(csv_path))[0] + "_split"

    ratios = [0.5, 0.9, 0.75]
    time_col_name, splits = split_by_time_period(df, time_col=time_col_arg, ratios=ratios)
    save_splits(base_dir, base_name, time_col_name, splits)
    print("\nDone.")

if __name__ == "__main__":
    main()
