# batch_build_bank_metrics_from_csvs.py
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
import csv

BASE = 4  # first 4 rows kept; raw variables start at row 5 (0-based 4)
IDX = {
    "total_interest_income":       BASE + 0,
    "total_interest_expense":      BASE + 1,
    "total_income":                BASE + 2,
    "loan_loss_provisions":        BASE + 3,
    "operating_expenses":          BASE + 4,
    "operating_income":            BASE + 5,
    "net_income":                  BASE + 6,
    "gross_loans":                 BASE + 7,
    "total_loans":                 BASE + 8,
    "total_assets":                BASE + 9,
    "customer_deposits":           BASE +10,
    "short_term_liabilities":      BASE +11,
    "total_liabilities":           BASE +12,
    "total_equity":                BASE +13,
    "total_earning_assets":        BASE +14,
    "liquid_assets":               BASE +15,
    "risk_weighted_assets":        BASE +16,
    "tier1_capital":               BASE +17,
    "npls":                        BASE +18,
    "non_performing_assets":       BASE +19,
    "num_employees":               BASE +20,
}

def safe_num(s):
    return pd.to_numeric(s, errors="coerce")

def safe_div(a, b):
    return a.divide(b).where(b != 0)

def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    num = df.apply(safe_num)
    x = {k: num.iloc[i, :] for k, i in IDX.items()}
    non_interest_income = x["total_income"] - x["total_interest_income"]

    metrics = {
        "LDR (Loans/Deposits)":                    safe_div(x["total_loans"], x["customer_deposits"]),
        "ROA (NetIncome/TotalAssets)":             safe_div(x["net_income"], x["total_assets"]),
        "LNTA (ln Total Assets)":                  np.log(x["total_assets"]).where(x["total_assets"] > 0),
        "SR (Equity/Assets)":                      safe_div(x["total_equity"], x["total_assets"]),
        "INEFF (OpEx/OpInc)":                      safe_div(x["operating_expenses"], x["operating_income"]),
        "DIVER (Non-IntInc/TotalInc)":             safe_div(non_interest_income, x["total_income"]),
        "CQ (LLP/TotalLoans)":                     safe_div(x["loan_loss_provisions"], x["total_loans"]),
        # NIM removed
        "Equity Ratio (Equity/Assets)":            safe_div(x["total_equity"], x["total_assets"]),
        "Cost-to-Income (OpEx/OpInc)":             safe_div(x["operating_expenses"], x["operating_income"]),
        "Interest Income Ratio (IntInc/Assets)":   safe_div(x["total_interest_income"], x["total_assets"]),
        "NPA/Assets":                               safe_div(x["non_performing_assets"], x["total_assets"]),
        "NPA/Loans":                                safe_div(x["non_performing_assets"], x["total_loans"]),
        "Tier1 Ratio (Tier1/RWA)":                 safe_div(x["tier1_capital"], x["risk_weighted_assets"]),
        # Growth rates removed
        "Provisioning Ratio (LLP/GrossLoans)":     safe_div(x["loan_loss_provisions"], x["gross_loans"]),
        "Liquidity Ratio (Liquid/ST Liab)":        safe_div(x["liquid_assets"], x["short_term_liabilities"]),
        "ROE (NetIncome/Equity)":                  safe_div(x["net_income"], x["total_equity"]),
        "Assets per Employee":                     safe_div(x["total_assets"], x["num_employees"]),
        "NPL Ratio (NPLs/TotalLoans)":             safe_div(x["npls"], x["total_loans"]),
        "Cust Deposits / Liabilities":             safe_div(x["customer_deposits"], x["total_liabilities"]),
    }
    return pd.DataFrame(metrics).T  # rows=metrics, cols=original periods

def write_output(original_df: pd.DataFrame, metrics_df: pd.DataFrame, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # Keep the first 4 rows from the original
        for i in range(min(4, len(original_df))):
            w.writerow(original_df.iloc[i, :].tolist())
        # spacer
        w.writerow([])
        # Write metrics: name in column A, values start from column B (left-shift fix retained)
        for var_name, row in metrics_df.iterrows():
            w.writerow([var_name] + row.iloc[1:].tolist())

def main():
    # Determine script directory for creating output folders next to this file
    try:
        script_dir = Path(__file__).resolve().parent
    except NameError:
        script_dir = Path.cwd()

    # Pick top folder
    root = tk.Tk(); root.withdraw()
    top = filedialog.askdirectory(title="Select the top folder containing subfolders with CSVs")
    if not top:
        print("No folder selected."); return
    top_path = Path(top)

    csv_files = sorted(top_path.rglob("*.csv"))
    if not csv_files:
        print("No .csv files found under:", top_path)
        return

    print(f"Found {len(csv_files)} CSV file(s). Processing...")

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path, header=None, dtype=str)
            metrics_df = compute_metrics(df)

            # Decide output folder:
            # Map first-level subfolder "X" to "<script_dir>/X_calculated"
            rel = csv_path.relative_to(top_path)
            parts = rel.parts
            if len(parts) >= 2:
                first_level = parts[0]
                sub_rel_dirs = Path(*parts[1:-1]) if len(parts) > 2 else Path()
                out_dir = script_dir / f"{first_level}_calculated" / sub_rel_dirs
            else:
                # CSV directly under selected folder → put under "<SelectedName>_calculated"
                out_dir = script_dir / f"{top_path.name}_calculated"

            out_file = out_dir / (csv_path.stem + "_with_metrics.csv")
            write_output(df, metrics_df, out_file)
            print(f"OK: {csv_path} -> {out_file}")
        except Exception as e:
            print(f"ERROR: {csv_path} -> {e}")

    print("Done.")

if __name__ == "__main__":
    main()
