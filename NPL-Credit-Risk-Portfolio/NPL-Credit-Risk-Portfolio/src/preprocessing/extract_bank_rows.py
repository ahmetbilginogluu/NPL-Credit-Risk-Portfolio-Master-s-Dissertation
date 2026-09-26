import pandas as pd
from pathlib import Path
from tkinter import Tk, filedialog

# ---------- CONFIG ----------
TARGET_ROWS = [
    "∟ Gross loans & advances to customers",
    "Total impaired / Non-performing loans",
    "Customer deposits",
    "Return on average assets (ROAA)%",
    "Total assets",
    "Total liabilities",
    "Total equity",
    "Total liabilities and equity",
    "Total operating expenses",
    "Operating profit",
    "Operating revenues",
    "∟ Net interest income (expense)",
    "∟ Loans loss reserves",
    "Loan loss reserves / Impaired loans%",
    "Growth in gross customer loans &advances%",
    "Net impairment charges / Net interest income%",
    "Risk-weighted asset intensity (RWA / Total Assets)%"
]
OUTPUT_FILE = Path(__file__).with_name("selected_rows.xlsx")
# ----------------------------

def clean_text(text):
    if isinstance(text, str):
        return " ".join(text.strip().split())  # remove extra spaces/tabs
    return text

def main():
    Tk().withdraw()
    in_path = filedialog.askopenfilename(title="Select an Excel file",
                                         filetypes=[("Excel files", "*.xlsx *.xls")])
    if not in_path:
        print("No file selected.")
        return

    df = pd.read_excel(in_path, header=None, engine="openpyxl")

    # Drop leading fully empty columns
    df = df.loc[:, ~df.isna().all(axis=0)]

    # Normalize first column strings
    df.iloc[:, 0] = df.iloc[:, 0].map(clean_text)

    # Normalize target row labels too
    normalized_targets = [clean_text(row) for row in TARGET_ROWS]

    # Filter and reorder
    mask = df.iloc[:, 0].isin(normalized_targets)
    filtered = df[mask]

    ordered_rows = []
    for label in normalized_targets:
        matches = filtered[filtered.iloc[:, 0] == label]
        ordered_rows.append(matches)
    filtered = pd.concat(ordered_rows, ignore_index=True)

    filtered.to_excel(OUTPUT_FILE, index=False, header=False)
    print(f"Saved → {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
