# Turbofan Engine RUL Prediction

Predicts Remaining Useful Life (RUL) of aircraft turbofan engines using the NASA C-MAPSS
run-to-failure dataset, comparing Random Forest, XGBoost, and a neural network (MLP).

## Files
- `data_utils.py` - data loading, piecewise-linear RUL target construction, feature
  selection, and the official PHM08 asymmetric scoring function
- `train_evaluate.py` - trains Random Forest, XGBoost, and an ANN (MLPRegressor) on each
  of the four C-MAPSS sub-datasets (FD001-FD004) and saves RMSE / MAE / PHM08 score results
- `make_figures.py` - generates all report figures from the saved results
- `figures/` - output plots (RMSE comparison, predicted-vs-true, feature importance, sensor trends)
- `outputs/` - results_summary.csv, per-subset feature importance CSVs, raw predictions

## Results (test set)

| Dataset | Conditions | Fault modes | Best model | RMSE (cycles) | MAE (cycles) |
|---|---|---|---|---|---|
| FD001 | 1 | 1 (HPC) | XGBoost | 18.02 | 13.24 |
| FD002 | 6 | 1 (HPC) | XGBoost | 29.43 | 20.82 |
| FD003 | 1 | 2 (HPC, Fan) | ANN | 20.88 | 15.48 |
| FD004 | 6 | 2 (HPC, Fan) | XGBoost | 30.29 | 22.56 |

## How to run
1. Download the C-MAPSS dataset (train_*.txt, test_*.txt, RUL_*.txt) and place the files
   in a `data/` folder next to these scripts.
2. Install dependencies: `pip install pandas numpy scikit-learn matplotlib xgboost`
3. `python train_evaluate.py`        # runs all 4 subsets (or pass e.g. `FD001 FD003` for a subset)
4. `python make_figures.py`          # regenerates the figures used in the report

## Requirements
pandas, numpy, scikit-learn, matplotlib, xgboost
