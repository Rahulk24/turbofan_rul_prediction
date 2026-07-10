import json
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

try:
    from xgboost import XGBRegressor
    HAVE_XGBOOST = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingRegressor
    HAVE_XGBOOST = False


from data_utils import (load_dataset, add_train_rul, get_test_last_cycle,
                         select_features, phm_score)

DATA_DIR = "data"
SUBSETS = sys.argv[1:] if len(sys.argv) > 1 else ["FD001", "FD002", "FD003", "FD004"]
RANDOM_STATE = 42

results_rows = []
importance_frames = {}
predictions_store = {}

for subset in SUBSETS:
    train, test, rul = load_dataset(DATA_DIR, subset)
    train = add_train_rul(train)
    test_last = get_test_last_cycle(test)
    test_last = test_last.merge(rul, on="unit")

    feat_cols = select_features(subset)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(train[feat_cols])
    y_train = train["RUL"].values

    X_test = scaler.transform(test_last[feat_cols])
    y_test = test_last["RUL"].values

    if HAVE_XGBOOST:
        gb_model = XGBRegressor(
            n_estimators=120, max_depth=5, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9,
            random_state=RANDOM_STATE, n_jobs=-1)
        gb_name = "XGBoost"
    else:
        gb_model = HistGradientBoostingRegressor(
            max_iter=120, max_depth=5, learning_rate=0.08,
            random_state=RANDOM_STATE)
        gb_name = "GradientBoosting"

    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=100, max_depth=8, min_samples_leaf=10,
            random_state=RANDOM_STATE, n_jobs=-1),
        gb_name: gb_model,
        "ANN": MLPRegressor(
            hidden_layer_sizes=(32, 16), activation="relu", solver="adam",
            alpha=1e-3, max_iter=300, early_stopping=True,
            random_state=RANDOM_STATE),
    }

    predictions_store[subset] = {"y_test": y_test}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred = np.clip(y_pred, 0, None)

        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        score = phm_score(y_test, y_pred)

        results_rows.append({
            "subset": subset, "model": name,
            "RMSE": rmse, "MAE": mae, "PHM_score": score,
        })
        predictions_store[subset][name] = y_pred

        print(f"{name:16s} RMSE={rmse:7.2f}  MAE={mae:7.2f}  PHM_score={score:10.1f}")

        if hasattr(model, "feature_importances_"):
            fi = pd.Series(model.feature_importances_, index=feat_cols).sort_values(ascending=False)
            importance_frames[(subset, name)] = fi

results_df = pd.DataFrame(results_rows)
out_csv = "outputs/results_summary.csv"
if os.path.exists(out_csv):
    prev = pd.read_csv(out_csv)
    prev = prev[~prev["subset"].isin(SUBSETS)]
    results_df = pd.concat([prev, results_df], ignore_index=True)
results_df.to_csv(out_csv, index=False)
print(f"\nSaved {out_csv}")

for (subset, name), fi in importance_frames.items():
    fi.to_csv(f"outputs/feature_importance_{subset}_{name}.csv", header=["importance"])

pred_json = "outputs/predictions_store.json"
all_preds = {}
if os.path.exists(pred_json):
    with open(pred_json) as f:
        all_preds = json.load(f)
all_preds.update({k: {kk: list(map(float, vv)) for kk, vv in v.items()}
                   for k, v in predictions_store.items()})
with open(pred_json, "w") as f:
    json.dump(all_preds, f)

print("Done.")
