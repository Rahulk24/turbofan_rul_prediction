import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_utils import load_dataset, add_train_rul, SENSOR_COLS

plt.rcParams.update({"figure.dpi": 150, "font.size": 10})

results = pd.read_csv("outputs/results_summary.csv")
with open("outputs/predictions_store.json") as f:
    preds = json.load(f)

fig, ax = plt.subplots(figsize=(7, 4.2))
pivot = results.pivot(index="subset", columns="model", values="RMSE")[["RandomForest", "GradientBoosting", "ANN"]]
pivot.plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452", "#55A868"])
ax.set_ylabel("RMSE (cycles)")
ax.set_title("Test-set RMSE by model and data subset")
ax.set_xlabel("")
plt.xticks(rotation=0)
plt.legend(title="Model")
plt.tight_layout()
plt.savefig("figures/rmse_comparison.png")
plt.close()

subset = "FD001"
best_model = results[results.subset == subset].sort_values("RMSE").iloc[0]["model"]
y_true = np.array(preds[subset]["y_test"])
y_pred = np.array(preds[subset][best_model])

fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.scatter(y_true, y_pred, alpha=0.6, edgecolor="k", linewidth=0.3)
lims = [0, max(y_true.max(), y_pred.max()) + 5]
ax.plot(lims, lims, "r--", label="Perfect prediction")
ax.set_xlabel("True RUL (cycles)")
ax.set_ylabel("Predicted RUL (cycles)")
ax.set_title(f"Predicted vs. True RUL - {subset} ({best_model})")
ax.legend()
plt.tight_layout()
plt.savefig("figures/pred_vs_true_FD001.png")
plt.close()

fi = pd.read_csv("outputs/feature_importance_FD001_RandomForest.csv", index_col=0).squeeze("columns")
fi = fi.sort_values(ascending=True).tail(12)
fig, ax = plt.subplots(figsize=(6.5, 5))
ax.barh(fi.index, fi.values, color="#4C72B0")
ax.set_xlabel("Relative importance")
ax.set_title("Top sensor importances - Random Forest (FD001)")
plt.tight_layout()
plt.savefig("figures/feature_importance_FD001.png")
plt.close()

train, _, _ = load_dataset("data", "FD001")
train = add_train_rul(train)
example_units = [1, 25, 50, 75]
sensor_to_plot = "sensor_11"  # static pressure at HPC outlet - known strong degradation indicator

fig, ax = plt.subplots(figsize=(7, 4.2))
for u in example_units:
    sub = train[train.unit == u]
    ax.plot(sub["cycle"], sub[sensor_to_plot], label=f"Engine {u}")
ax.set_xlabel("Operating cycle")
ax.set_ylabel(f"{sensor_to_plot} (Ps30, psia)")
ax.set_title("Sensor degradation trend across engine life (FD001)")
ax.legend()
plt.tight_layout()
plt.savefig("figures/sensor_trend_example.png")
plt.close()

sub = train[train.unit == 1]
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(sub["cycle"], sub["RUL"], color="#C44E52")
ax.set_xlabel("Operating cycle")
ax.set_ylabel("Target RUL (capped at 125)")
ax.set_title("Piecewise-linear RUL training target (Engine 1, FD001)")
plt.tight_layout()
plt.savefig("figures/rul_target_illustration.png")
plt.close()

print("All figures saved to figures/")
