"""Step 7: baselines + models, scored on rows that have lag_yield.

Protocol (state this in the paper):
  - Validation (2022): models fit on train (2013-2021); a tiny grid picks hyperparameters.
  - Test (2023-24):    models refit on train+val with the chosen settings; test is never used for tuning.
  - Every model, including the last-year baseline, is scored on the SAME rows (has_lag == True).
  - Run twice: 'all' rows, and 'no_outliers' (rows flagged in step 1 removed from train AND eval).

Outputs (data/): results_main.csv, results_by_season.csv, predictions_test.csv
"""
import json
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from config import DATA_DIR

SEED = 42
try:
    from xgboost import XGBRegressor
    HAVE_XGB = True
except ImportError:
    HAVE_XGB = False
    print("xgboost not installed -> skipping it (pip install xgboost). HistGradientBoosting still runs.\n")

df = pd.read_csv(DATA_DIR / "model_ready.csv")
FEATURES = json.load(open(DATA_DIR / "features.json"))["features"]
TARGET = "yield"


def impute():  # fill lag_yield with TRAIN median and add a 'was missing' flag column
    return SimpleImputer(strategy="median", add_indicator=True)


def candidates():
    """name -> list of estimators (a small grid, picked on validation)."""
    c = {
        "Ridge": [make_pipeline(impute(), StandardScaler(), Ridge(alpha=a)) for a in (1, 10, 100)],
        "RandomForest": [make_pipeline(impute(), RandomForestRegressor(
            n_estimators=300, max_depth=d, min_samples_leaf=3, random_state=SEED, n_jobs=-1))
            for d in (6, 10, None)],
        "HistGradientBoosting": [HistGradientBoostingRegressor(
            max_depth=d, learning_rate=0.05, max_iter=300, l2_regularization=1.0, random_state=SEED)
            for d in (3, 5)],
    }
    if HAVE_XGB:
        c["XGBoost"] = [XGBRegressor(n_estimators=300, max_depth=d, learning_rate=0.05, subsample=0.8,
                                     colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, n_jobs=1)
                        for d in (3, 5)]
    return c


def metrics(y, p):
    return {"RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "MAE": float(mean_absolute_error(y, p)), "R2": float(r2_score(y, p))}


def district_season_mean(train, target_rows):
    key = ["district_name", "season"]
    m = train.groupby(key)[TARGET].mean().rename("m").reset_index()
    out = target_rows[key].merge(m, on=key, how="left")["m"].to_numpy()
    fallback = target_rows["season"].map(train.groupby("season")[TARGET].mean()).to_numpy()
    return np.where(np.isnan(out), fallback, out)


def run(setting, data):
    tr, va, te = (data[data.split == s] for s in ("train", "val", "test"))
    trva = pd.concat([tr, va])
    va_s, te_s = va[va.has_lag], te[te.has_lag]          # scored rows
    rows, preds = [], te_s[["district_name", "year_start", "season", TARGET]].copy()

    def log(name, v_pred, t_pred, note=""):
        for part, y, p in (("val", va_s[TARGET], v_pred), ("test", te_s[TARGET], t_pred)):
            rows.append({"setting": setting, "model": name, "split": part, "n": len(y), "note": note,
                         **metrics(y, p)})
        preds[name] = t_pred

    # baselines
    log("Baseline: last year", va_s["lag_yield"].to_numpy(), te_s["lag_yield"].to_numpy())
    log("Baseline: district-season mean", district_season_mean(tr, va_s), district_season_mean(trva, te_s))

    # models
    for name, grid in candidates().items():
        best, best_rmse = None, np.inf
        for est in grid:
            m = clone(est).fit(tr[FEATURES], tr[TARGET])
            r = metrics(va_s[TARGET], m.predict(va_s[FEATURES]))["RMSE"]
            if r < best_rmse:
                best, best_rmse = est, r
        v_pred = clone(best).fit(tr[FEATURES], tr[TARGET]).predict(va_s[FEATURES])
        final = clone(best).fit(trva[FEATURES], trva[TARGET])
        log(name, v_pred, final.predict(te_s[FEATURES]), note=f"val-RMSE {best_rmse:.0f}")
    return pd.DataFrame(rows), preds


raw = df.copy()                                   # original behaviour: nothing removed, raw lags
raw["lag_yield"] = raw["lag_yield_raw"]
raw["has_lag"] = raw["lag_yield"].notna()
clean = df[~df.suspect_fill]

SETTINGS = (("clean (MAIN)", clean),                          # suspected filled rows removed
            ("clean_no_outliers", clean[~clean.outlier_flag]),
            ("all_rows_raw", raw))                            # nothing removed (sensitivity only)
results, season_rows = [], []
for setting, data in SETTINGS:
    res, preds = run(setting, data)
    results.append(res)
    if setting == "clean (MAIN)":
        preds.to_csv(DATA_DIR / "predictions_test.csv", index=False)
        for mname in [c for c in preds.columns if c not in ("district_name", "year_start", "season", TARGET)]:
            for s, g in preds.groupby("season"):
                season_rows.append({"model": mname, "season": s, "n": len(g), **metrics(g[TARGET], g[mname])})

res = pd.concat(results)
res.to_csv(DATA_DIR / "results_main.csv", index=False)
pd.DataFrame(season_rows).to_csv(DATA_DIR / "results_by_season.csv", index=False)

pd.set_option("display.width", 200)
for setting, _ in SETTINGS:
    print(f"\n=== setting: {setting} (rows scored = those with lag_yield) ===")
    t = res[res.setting == setting].pivot(index="model", columns="split", values=["RMSE", "MAE", "R2"]).round(2)
    t[("n_test", "")] = res[(res.setting == setting) & (res.split == "test")].set_index("model")["n"]
    print(t.sort_values(("RMSE", "test")).to_string())
print("\nSaved data/results_main.csv, results_by_season.csv, predictions_test.csv")
