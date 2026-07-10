import numpy as np
import pandas as pd

INDEX_COLS = ["unit", "cycle"]
SETTING_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
COLS = INDEX_COLS + SETTING_COLS + SENSOR_COLS

CONSTANT_SENSORS = ["sensor_1", "sensor_5", "sensor_6", "sensor_10",
                     "sensor_16", "sensor_18", "sensor_19"]

RUL_CAP = 125  # standard piecewise-linear RUL cap used throughout the CMAPSS literature

def load_dataset(data_dir, subset):
   
    train = pd.read_csv(f"{data_dir}/train_{subset}.txt", sep=r"\s+", header=None)
    test = pd.read_csv(f"{data_dir}/test_{subset}.txt", sep=r"\s+", header=None)
    rul = pd.read_csv(f"{data_dir}/RUL_{subset}.txt", sep=r"\s+", header=None)

    train.columns = COLS
    test.columns = COLS
    rul.columns = ["RUL"]
    rul["unit"] = np.arange(1, len(rul) + 1)
    return train, test, rul


def add_train_rul(train):
    max_cycle = train.groupby("unit")["cycle"].transform("max")
    rul = max_cycle - train["cycle"]
    train = train.copy()
    train["RUL"] = rul.clip(upper=RUL_CAP)
    return train


def get_test_last_cycle(test):
    return test.sort_values(["unit", "cycle"]).groupby("unit").tail(1).reset_index(drop=True)

def select_features(subset):
    if subset in ("FD001", "FD003"):
        sensors = [c for c in SENSOR_COLS if c not in CONSTANT_SENSORS]
        return SETTING_COLS + sensors
    else:
        return SETTING_COLS + SENSOR_COLS


def phm_score(y_true, y_pred, a1=10, a2=13):
    d = np.asarray(y_pred) - np.asarray(y_true)
    s = np.where(d < 0, np.exp(-d / a1) - 1, np.exp(d / a2) - 1)
    return float(np.sum(s))
