# Turbofan Engine RUL Prediction - Code

## Files
- `data_utils.py` - data loading, RUL target construction, feature selection, PHM08 scoring function
- `train_evaluate.py` - trains Random Forest, Gradient Boosting (HistGradientBoostingRegressor, used in place of XGBoost
  since this environment had no internet access to install the xgboost package), and an ANN (MLPRegressor) on each
  C-MAPSS sub-dataset and saves RMSE/MAE/PHM score results
- `make_figures.py` - generates all report figures from the saved results
- `figures/` - output plots (RMSE comparison, predicted-vs-true, feature importance, sensor trends)
- `results/` - results_summary.csv, per-subset feature importance CSVs, raw predictions

## How to run
1. Place the C-MAPSS `train_*.txt`, `test_*.txt`, and `RUL_*.txt` files in a `data/` folder next to these scripts.
2. `python train_evaluate.py`             # runs all 4 subsets (or pass e.g. `FD001 FD003` to run a subset)
3. `python make_figures.py`               # regenerates the figures used in the report

## Requirements
pandas, numpy, scikit-learn, matplotlib (no internet / GPU required)
