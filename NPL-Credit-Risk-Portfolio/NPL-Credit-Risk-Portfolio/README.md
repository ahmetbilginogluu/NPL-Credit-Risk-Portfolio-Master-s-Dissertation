# Bank NPL Forecasting with Machine Learning

## Summary

This project was completed as part of my Banking and International Finance dissertation. It explores forecasting banks’ non-performing loan ratios using machine-learning models, bank financial indicators, and macroeconomic data. The code covers data preparation, chronological train/test splitting, and model comparison.

**Organization note:** The original dissertation folder was very cluttered, so I used ChatGPT to help clean and organize this GitHub version.

## Project overview

- **Target:** bank NPL ratio (non-performing loans relative to loans)
- **Inputs:** bank-level financial ratios and quarterly macroeconomic indicators
- **Models explored:** linear regression, regularized regression, decision trees, random forests, gradient boosting, support-vector regression, and neural networks
- **Tools:** Python, pandas, NumPy, scikit-learn, statsmodels

The repository contains a curated subset of the original working folder: the data-preparation steps, a chronological train/test splitter, and an interactive modeling workbench. Intermediate experiments, duplicate exports, draft documents, and generated results were removed.

## Data and reproducibility

The original project used bank financial exports from Bureau van Dijk’s BankFocus database, alongside macroeconomic series. Those source and processed data files are **not included**. BankFocus data may be subject to database licensing and redistribution restrictions; use only data you are authorized to access. The repository does not contain individual-level personal data or a sample of the licensed dataset.

As the inputs are omitted, the analysis cannot be reproduced from this repository alone. No performance figures are published here: the supplied dissertation document was explicitly marked as an early draft, and the archive contains multiple iterations with different samples and evaluation outputs. Final results should only be reported after confirming the final dataset, specification, and time-based evaluation.

## Repository structure

```text
src/
  preprocessing/
    extract_bank_rows.py            Select relevant rows from a BankFocus Excel export
    build_bank_metrics.py            Calculate bank-level ratios from raw CSV exports
    standardize_bank_exports.py      Standardize bank and country metadata in exports
    merge_quarterly_macro_data.py    Align macro data and bank exports by year/quarter
    filter_panel.py                  Filter and export panel observations
  modeling/
    split_by_time.py                 Create chronological train/test splits
    model_workbench.py               Interactive model comparison and diagnostics

data/                               Private local inputs; ignored by Git
```

## Setup

Python 3.10 or newer is recommended. The data preparation and modeling scripts use a file picker, so a working Tk installation is required for those screens.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

## Workflow

The scripts are standalone utilities because the original work evolved through several manual data-preparation stages. Run them on local, authorized files and check each output before moving to the next stage.

1. Extract selected financial statement rows with `python src/preprocessing/extract_bank_rows.py`.
2. Calculate bank ratios with `python src/preprocessing/build_bank_metrics.py`.
3. Standardize bank/country metadata and merge macroeconomic observations using the corresponding scripts in `src/preprocessing/`.
4. Filter the assembled panel, then create chronological splits. For example:

   ```bash
   python src/modeling/split_by_time.py path/to/panel.csv --time-col "TIME PERIOD"
   ```

5. Use `python src/modeling/model_workbench.py` to select prepared training and test CSV files and compare models.

The preprocessing scripts reflect the source export layouts used during the project and may need adjustment for different BankFocus exports or column names. The modeling workbench expects prepared train and test files; it does not orchestrate the complete pipeline.

## Method note

For forecasting, the test set should contain later quarters than the training set. The included splitter makes chronological splits by distinct time period. Because the data are a bank panel, evaluation should also check whether performance holds for banks and periods not represented in training. Report model metrics and feature importance only after those checks and after selecting one final specification.

## Status

This is a cleaned portfolio snapshot of an academic research project, not a production credit-risk system. The repository focuses on code and documents the data boundary; the source archive did not contain a final thesis report. The results and research write-up should be added only after the final analysis is verified and cleared for publication.

## License

No reuse license is granted in this repository. Add a license only after deciding how you want others to use the code. The source datasets remain subject to their own terms regardless of the code license.
