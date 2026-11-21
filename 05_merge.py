"""Step 5: build the modelling table = yield + seasonal weather features + soil + lagged yield.

Season windows (edit WINDOWS to match your crop calendar, and justify them in the paper):
  Kharif : 1 Jun - 31 Oct of the start year
  Rabi   : 1 Nov of the start year - 31 Mar of the next year
  Summer : 1 Apr - 30 Jun of the next year
"""
import numpy as np
import pandas as pd
from config import *

RAIN_DAY_MM, HOT_DAY_C = 2.5, 35.0


def window(year_start, season):
    y = int(year_start)
    return {
        "Kharif": (f"{y}-06-01", f"{y}-10-31"),
        "Rabi": (f"{y}-11-01", f"{y+1}-03-31"),
        "Summer": (f"{y+1}-04-01", f"{y+1}-06-30"),
    }[season]


yld = pd.read_csv(DATA_DIR / "yield_clean.csv")
yld = yld[yld["district_name"] != "Chennai"]
wx = pd.read_csv(DATA_DIR / "nasa_power_daily.csv", parse_dates=["date"])
soil = pd.read_csv(DATA_DIR / "soilgrids_district.csv")
wx = wx.sort_values(["district_name", "date"])
wx_by_district = {k: v.set_index("date") for k, v in wx.groupby("district_name")}


def season_features(district, year_start, season):
    if district not in wx_by_district:
        return None
    a, b = window(year_start, season)
    w = wx_by_district[district].loc[a:b]
    expected = (pd.Timestamp(b) - pd.Timestamp(a)).days + 1
    if len(w) == 0:
        return None
    f = {
        "wx_coverage": w["PRECTOTCORR"].notna().sum() / expected,
        "rain_total_mm": w["PRECTOTCORR"].sum(),
        "rain_days": (w["PRECTOTCORR"] >= RAIN_DAY_MM).sum(),
        "tmax_mean": w["T2M_MAX"].mean(),
        "tmin_mean": w["T2M_MIN"].mean(),
        "tmean": w["T2M"].mean(),
        "hot_days": (w["T2M_MAX"] > HOT_DAY_C).sum(),
        "rh_mean": w["RH2M"].mean(),
        "solar_mean": w["ALLSKY_SFC_SW_DWN"].mean(),
    }
    # Crude crop-stage proxy: split the season into thirds (early / mid / late).
    for tag, part in zip(["early", "mid", "late"], np.array_split(np.arange(len(w)), 3)):
        seg = w.iloc[part]
        f[f"rain_{tag}"] = seg["PRECTOTCORR"].sum()
        f[f"tmax_{tag}"] = seg["T2M_MAX"].mean()
    return f


feats = []
for r in yld.itertuples():
    feats.append(season_features(r.district_name, r.year_start, r.season) or {})
df = pd.concat([yld.reset_index(drop=True), pd.DataFrame(feats)], axis=1)

# Soil is static per district.
df = df.merge(soil, on="district_name", how="left")

# Lagged yield: same district and season, previous agricultural year.
prev = df[["district_name", "season", "year_start", "yield"]].copy()
prev["year_start"] += 1
prev = prev.rename(columns={"yield": "lag_yield"})
df = df.merge(prev, on=["district_name", "season", "year_start"], how="left")

# Temporal split (never shuffle across years).
df["split"] = np.where(df.year_start <= TRAIN_LAST_YEAR, "train",
                       np.where(df.year_start == VAL_YEAR, "val", "test"))

# Quality report and filters.
n0 = len(df)
no_wx = df["rain_total_mm"].isna().sum()
low_cov = (df["wx_coverage"] < 0.9).sum()
no_soil = df["soil_phh2o"].isna().sum()
df = df[(df["wx_coverage"] >= 0.9)]
print(f"rows before filters: {n0}")
print(f"  no weather match: {no_wx} | weather coverage < 90%: {low_cov} | no soil match: {no_soil}")
print(f"rows kept: {len(df)}")
print("split sizes:", df["split"].value_counts().to_dict())
print("lag_yield missing (expected for first year):", int(df["lag_yield"].isna().sum()))

df.to_csv(DATA_DIR / "modelling_table.csv", index=False)
print("Saved data/modelling_table.csv", df.shape)
print("Tip: compute any district-mean features from the TRAIN rows only, to avoid leakage.")
