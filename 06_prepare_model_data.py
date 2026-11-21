"""Step 6: turn modelling_table.csv into a leak-free, model-ready table.

1. Drop the one broken yield value (Ramanathapuram Kharif 2016-17, yield = 2 kg/ha).
2. Keep only legitimate feature columns (no target leakage).
3. Flag rows with / without lag_yield so baselines are scored on the same rows.
"""
import json
import numpy as np
import pandas as pd
from config import DATA_DIR

m = pd.read_csv(DATA_DIR / "modelling_table.csv")

# ---- 1. broken value -------------------------------------------------------
bad = (m.district_name == "Ramanathapuram") & (m.year_start == 2016) & (m.season == "Kharif")
print(f"Dropping {int(bad.sum())} broken row(s); keeping {len(m) - int(bad.sum())}")
m = m[~bad].copy()
# The other 7 flagged outliers stay in. outlier_flag is kept ONLY for the sensitivity run
# (rerun the final model with: df[~df.outlier_flag]).

# ---- 1b. suspected state-average fill ---------------------------------------
# The source file seems to fill missing district values with one repeated number. The same yield
# appearing in >= 5 districts in the same year+season is flagged (3-4 matches could be chance).
SUSPECT_MIN_DISTRICTS = 5
same = m.groupby(["year_start", "season", "yield"])["district_name"].transform("count")
m["suspect_fill"] = same >= SUSPECT_MIN_DISTRICTS
print(f"Flagged suspect_fill rows: {int(m.suspect_fill.sum())} (rule: same yield in >= {SUSPECT_MIN_DISTRICTS} districts, same year+season)")

# A lag that was copied from a flagged row is also contaminated -> set to missing (raw copy kept).
prev = m[["district_name", "season", "year_start", "suspect_fill"]].copy()
prev["year_start"] += 1
prev = prev.rename(columns={"suspect_fill": "lag_suspect"})
m = m.merge(prev, on=["district_name", "season", "year_start"], how="left")
m["lag_suspect"] = m["lag_suspect"].fillna(False).astype(bool)
m["lag_yield_raw"] = m["lag_yield"]
m.loc[m["lag_suspect"], "lag_yield"] = np.nan
print(f"lag_yield blanked because it came from a flagged row: {int((m.lag_suspect & m.lag_yield_raw.notna()).sum())}")

# ---- 2. features (no production, robust_z, outlier_flag, yield_mismatch, ids, split...) ----
m["season_code"] = m["season"].map({"Kharif": 0, "Rabi": 1, "Summer": 2})
WEATHER = ["rain_total_mm", "rain_days", "tmax_mean", "tmin_mean", "tmean", "hot_days",
           "rh_mean", "solar_mean", "rain_early", "tmax_early", "rain_mid", "tmax_mid",
           "rain_late", "tmax_late"]
SOIL = ["soil_phh2o", "soil_soc", "soil_clay"]      # 37 distinct values only, so keep few
FEATURES = WEATHER + SOIL + ["lag_yield", "season_code"]
OPTIONAL = ["area"]                                  # test in an ablation, not in the main model

# ---- 3. lag handling -------------------------------------------------------
m["has_lag"] = m["lag_yield"].notna()

keep = ["district_name", "year_start", "season", "split", "yield", "has_lag", "outlier_flag",
        "suspect_fill", "lag_yield_raw"]
out = m[keep + FEATURES + OPTIONAL]
out.to_csv(DATA_DIR / "model_ready.csv", index=False)
json.dump({"features": FEATURES, "weather": WEATHER, "soil": SOIL, "optional": OPTIONAL,
           "target": "yield"}, open(DATA_DIR / "features.json", "w"), indent=2)

# ---- checks ----------------------------------------------------------------
print("\nSplit sizes:", out["split"].value_counts().to_dict())
print("Rows with lag_yield, by split:\n", out.groupby("split")["has_lag"].agg(["sum", "count"]).to_string())
na = out[FEATURES].isna().sum()
print("\nMissing values in features (only lag_yield should appear):", na[na > 0].to_dict())
print("Saved data/model_ready.csv and data/features.json")
